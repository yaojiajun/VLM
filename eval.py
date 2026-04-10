import os
import argparse
import torch
import numpy as np
import json
import re
from tqdm import tqdm
from utils import load_pkl_dataset, compute_metric_cop
from datasets import load_dataset

# Vision-specific imports (loaded lazily to avoid breaking text-only runs)
def _load_vision_libs():
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from PIL import Image
    return AutoProcessor, Qwen2_5_VLForConditionalGeneration, Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Tester for solving combinatorial optimization problems')

    # Model and data parameters
    parser.add_argument('--model_id', type=str, default='saved_models', help='Model path')
    parser.add_argument('--problem', type=str, default='cvrp', help='Problem name')

    # Vision mode
    parser.add_argument('--vision', action='store_true', default=False,
                        help='Use vision model with image inputs (for VisionSolver)')
    parser.add_argument('--data_dir', type=str,
                        default='./data_sft/cvrp',
                        help='Root data dir containing eval/test.json and images_grid/')
    parser.add_argument('--images_grid_dir', type=str, default='images_grid',
                        help='Sub-dir under data_dir that holds the grid PNG images')
    parser.add_argument('--train_json', type=str, default='train_cvrp-001.json',
                        help='Training JSON used to resolve eval-image paths')

    # Evaluation method selection (text mode only)
    parser.add_argument('--eval_method', type=str, default='vanilla', choices=['vanilla', 'best_of_n'],
                        help='Evaluation method: vanilla or best_of_n (text mode only)')

    # Common parameters
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples to evaluate')

    # Best-of-N parameters (text mode only)
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size (best_of_n only)')
    parser.add_argument('--best_of_n', type=int, default=8, help='Candidates per prompt (best_of_n only)')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p (best_of_n only)')

    # Dataset loading method (text mode only)
    parser.add_argument('--dataset_method', type=str, default='auto',
                        choices=['auto', 'load_dataset', 'get_dataset'],
                        help='Dataset loading method (text mode only)')

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Vision evaluation
# ─────────────────────────────────────────────────────────────────────────────

def build_vision_instruction(vehicle_capacity: str) -> str:
    return (
        f"The image shows a Capacitated Vehicle Routing Problem (CVRP) instance. "
        f"Each blue circle is a customer (label = node id, size ∝ demand). "
        f"The green star is the depot (node 0). Vehicle capacity: {vehicle_capacity}. "
        f"Assign every customer to exactly one vehicle route so that each route "
        f"starts and ends at the depot and the total demand per route does not exceed "
        f"the vehicle capacity. Minimize the total travel distance.\n\n"
        f"Provide the solution in the following format:\n"
        f"Routes: [[0, ..., 0], [0, ..., 0], ...], Objective: <total distance>"
    )


def build_eval_image_index(data_dir: str, train_json: str, images_grid_dir: str):
    """
    Return a list of dicts (one per eval sample), each with:
        image_path, vehicle_capacity, output, num_nodes
    Images are resolved by matching eval inputs against the training JSON index.
    """
    train_json_path = os.path.join(data_dir, train_json)
    eval_json_path  = os.path.join(data_dir, 'eval', 'test.json')
    images_dir      = os.path.join(data_dir, images_grid_dir)

    with open(train_json_path) as f:
        train_data = json.load(f)
    with open(eval_json_path) as f:
        eval_data = json.load(f)

    # Build lookup: first 80 chars of input -> train index
    train_key_to_idx = {}
    for i, rec in enumerate(train_data):
        key = rec['input'][:80]
        if key not in train_key_to_idx:
            train_key_to_idx[key] = i

    # Detect eval-style naming: 5-digit (eval_00000) or 3-digit (eval_000)
    n0 = eval_data[0]['num_nodes'] if eval_data else 0
    if os.path.exists(os.path.join(images_dir, f"eval_00000_n{n0}.png")):
        eval_naming = '5digit'
    elif os.path.exists(os.path.join(images_dir, f"eval_000_n{n0}.png")):
        eval_naming = '3digit'
    else:
        eval_naming = None

    rows = []
    missing = 0
    for j, rec in enumerate(eval_data):
        if eval_naming == '5digit':
            img_path = os.path.join(images_dir, f"eval_{j:05d}_n{rec['num_nodes']}.png")
        elif eval_naming == '3digit':
            img_path = os.path.join(images_dir, f"eval_{j:03d}_n{rec['num_nodes']}.png")
        else:
            key = rec['input'][:80]
            train_idx = train_key_to_idx.get(key, -1)
            if train_idx >= 0:
                img_path = os.path.join(
                    images_dir,
                    f"instance_{train_idx + 1:06d}_n{rec['num_nodes']}.png"
                )
            else:
                img_path = os.path.join(images_dir, f"eval_{j:05d}_n{rec['num_nodes']}.png")

        if not os.path.exists(img_path):
            missing += 1
            print(f"  [warn] image not found for eval[{j}]: {img_path}")
            continue

        rows.append({
            'image_path':       img_path,
            'vehicle_capacity': str(rec['vehicle_capacity']),
            'output':           rec['output'],
            'num_nodes':        rec['num_nodes'],
        })

    if missing:
        print(f"  Skipped {missing} eval samples (image not found).")
    print(f"  Vision eval dataset: {len(rows)} samples.")
    return rows


def load_vision_model(model_id: str):
    AutoProcessor, Qwen2_5_VLForConditionalGeneration, _ = _load_vision_libs()
    print(f"Loading vision model from {model_id} ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()
    return model, processor


def run_vision_inference(model, processor, image_path: str, vehicle_capacity: str,
                         max_new_tokens: int = 1024) -> str:
    _, _, Image = _load_vision_libs()
    image = Image.open(image_path).convert("RGB")
    instruction = build_vision_instruction(vehicle_capacity)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": instruction},
            ],
        }
    ]

    # Apply chat template to get the text prompt
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Pass PIL image directly — no qwen_vl_utils needed
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    # Decode only the newly generated tokens
    input_len = inputs["input_ids"].shape[1]
    generated_ids_trimmed = generated_ids[:, input_len:]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return output_text


def run_vision_inference_n(model, processor, image_path, vehicle_capacity,
                           n=1, max_new_tokens=1024, temperature=0.7, top_p=0.9):
    """Generate n candidate solutions for one image. Returns list of raw strings."""
    _, _, Image = _load_vision_libs()
    image = Image.open(image_path).convert("RGB")
    instruction = build_vision_instruction(vehicle_capacity)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": instruction},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        if n == 1:
            generated_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
            )
        else:
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                num_return_sequences=n,
            )

    outputs = processor.batch_decode(
        generated_ids[:, input_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return outputs  # list of n strings


def evaluate_vision(args, model, processor, eval_rows, number_dataset):
    """Run vision evaluation: feed image + instruction, collect predictions."""
    labels      = [r['output'] for r in eval_rows]
    predictions = []
    raw_outputs = []   # store raw model text for saving

    n = args.best_of_n if args.eval_method == 'best_of_n' else 1

    for eval_idx, row in enumerate(tqdm(eval_rows[:args.num_samples])):
        candidates = run_vision_inference_n(
            model, processor,
            row['image_path'],
            row['vehicle_capacity'],
            n=n,
            max_new_tokens=1024,
            temperature=args.temperature,
            top_p=args.top_p,
        )

        if n == 1:
            best = candidates[0]
        else:
            # First filter feasible candidates, then pick lowest objective
            import ast as _ast
            def _is_feasible(text, demands, capacity):
                m = re.search(r'Routes:\s*\[\s*(.*)\]', text, re.DOTALL)
                if not m:
                    return False
                try:
                    import ast as _ast
                    routes = _ast.literal_eval(f'[{m.group(1).strip()}]')
                except Exception:
                    return False
                if not all(isinstance(r, list) for r in routes):
                    return False
                visited = set()
                for route in routes:
                    if not route or route[0] != 0 or route[-1] != 0:
                        return False
                    # Capacity check using instance demands
                    total = sum(demands[node] for node in route if node != 0)
                    if total > capacity:
                        return False
                    visited.update(route[1:-1])
                # All customers must be visited exactly once
                n_customers = len(demands)
                return visited == set(range(1, n_customers))

            inst_demands = number_dataset[eval_idx][1]
            inst_capacity = number_dataset[eval_idx][2]
            feasible = [(c, re.search(r'Objective:\s*([\d.]+)', c)) for c in candidates
                        if _is_feasible(c, inst_demands, inst_capacity)]

            if feasible:
                best = min(feasible, key=lambda x: float(x[1].group(1)) if x[1] else float('inf'))[0]
            else:
                # All infeasible: just pass the first candidate (will be counted as infeasible)
                best = candidates[0]

        raw_outputs.append(best)
        predictions.append("### Response:\n" + best)

    labels = labels[:args.num_samples]
    fea_rate, opt_gap, std_gap = compute_metric_cop(
        predictions, labels, number_dataset, problem=args.problem
    )

    # Save predictions to JSON for downstream visualization
    save_path = os.path.join(os.path.dirname(args.model_id) or '.', 'eval_predictions.json')
    records = []
    for i, (row, raw, pred) in enumerate(zip(eval_rows[:args.num_samples], raw_outputs, predictions)):
        records.append({
            'idx':              i,
            'image_path':       row['image_path'],
            'vehicle_capacity': row['vehicle_capacity'],
            'num_nodes':        row['num_nodes'],
            'prediction':       raw,
            'label':            row['output'],
        })
    with open(save_path, 'w') as f:
        json.dump(records, f, indent=2)
    print(f"\nPredictions saved to {save_path}")

    return fea_rate, opt_gap, std_gap


# ─────────────────────────────────────────────────────────────────────────────
# Text evaluation (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_id):
    from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return model, tokenizer, pipe


def load_datasets(problem, tokenizer, dataset_method='auto', eval_method='vanilla'):
    from rl_train import get_dataset
    if dataset_method == 'auto':
        dataset_method = 'load_dataset' if eval_method == 'vanilla' else 'get_dataset'

    if dataset_method == 'load_dataset':
        eval_dataset = load_dataset(f'./data/{problem}/eval', split="test")
    else:
        _, eval_dataset = get_dataset(problem, tokenizer, num_samples=None, train=False)

    number_dataset = load_pkl_dataset(f'./data/{problem}/instances.pkl')
    return eval_dataset, number_dataset


def get_generation_kwargs(tokenizer, eval_method, n=8, temperature=0.7, top_p=0.9):
    if eval_method == 'vanilla':
        return {
            "max_new_tokens": 5000,
            "do_sample": False,
            "temperature": 0.1,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        }
    else:
        return {
            "max_new_tokens": 5000,
            "do_sample": True,
            "temperature": temperature,
            "top_p": top_p,
            "num_return_sequences": n,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        }


def prepare_batch_prompts(eval_dataset, batch_indices):
    alpaca_prompt = """Below is an instruction describing a combinatorial optimization problem. It is paired with an input that provides the data of the instance.
    Your task is to produce a feasible solution that optimizes (minimizes or maximizes) the given objective.

    ### Instruction:{}

    ### Input:{}

    ### Response:"""
    batch_prompts = []
    for idx in batch_indices:
        instruction = eval_dataset[idx]['instruction']
        user_input  = eval_dataset[idx]['input']
        batch_prompts.append(alpaca_prompt.format(instruction, user_input))
    return batch_prompts


def evaluate_vanilla(args, pipe, eval_dataset, number_dataset):
    predictions = []
    labels = [item['output'] for item in eval_dataset]

    for idx in tqdm(range(args.num_samples)):
        prompt = prepare_batch_prompts(eval_dataset, [idx])[0]
        generation = pipe(prompt, **get_generation_kwargs(pipe.tokenizer, 'vanilla',
                                                          temperature=args.temperature))
        predictions.append(generation[0]['generated_text'])

    fea_rate, opt_gap, std_gap = compute_metric_cop(
        predictions, labels, number_dataset, problem=args.problem
    )
    return fea_rate, opt_gap, std_gap


def evaluate_best_of_n(args, pipe, eval_dataset, number_dataset):
    from Envs.eval_utils import (
        optimality_reward_func_op, optimality_reward_func_tsp,
        optimality_reward_func_mvc, optimality_reward_func_cvrp,
        optimality_reward_func_mis, optimality_reward_func_pfsp,
        optimality_reward_func_jssp,
    )

    batch_size = args.batch_size
    n          = args.best_of_n
    indices    = range(args.num_samples)
    generation_kwargs = get_generation_kwargs(
        pipe.tokenizer, 'best_of_n', n=n,
        temperature=args.temperature, top_p=args.top_p
    )

    labels       = [item['output'] for item in eval_dataset]
    predictions  = []
    ground_truth = eval_dataset["ground_truth"]

    for start_idx in tqdm(range(0, len(indices), batch_size)):
        end_idx      = min(start_idx + batch_size, len(indices))
        batch_indices = list(indices)[start_idx:end_idx]
        batch_prompts = prepare_batch_prompts(eval_dataset, batch_indices)
        raw_generations = pipe(batch_prompts, **generation_kwargs)

        grouped_generations = [
            raw_generations[i * n : (i + 1) * n]
            for i in range(len(batch_prompts))
        ]

        for i, idx in enumerate(batch_indices):
            completions = [gen["generated_text"] for gen in grouped_generations[i][0]]
            repeated_gt = [ground_truth[idx]] * n

            problem = args.problem
            if problem == 'cvrp':
                locs     = eval_dataset["instance_coords"][idx]
                demands  = eval_dataset["instance_demands"][idx]
                capacity = eval_dataset["instance_capacity"][idx]
                rewards  = optimality_reward_func_cvrp(
                    completions, repeated_gt,
                    [locs] * n, [demands] * n, [capacity] * n
                )
            else:
                raise NotImplementedError(f"best_of_n not implemented for {problem}")

            best_idx = int(np.argmax(rewards))
            predictions.append(completions[best_idx])

    fea_rate, opt_gap, std_gap = compute_metric_cop(
        predictions, labels, number_dataset, problem=args.problem
    )
    return fea_rate, opt_gap, std_gap


def evaluate_model(args):
    _, tokenizer, pipe = load_model_and_tokenizer(args.model_id)
    eval_dataset, number_dataset = load_datasets(
        args.problem, tokenizer, args.dataset_method, args.eval_method
    )
    if args.eval_method == 'vanilla':
        return evaluate_vanilla(args, pipe, eval_dataset, number_dataset)
    else:
        return evaluate_best_of_n(args, pipe, eval_dataset, number_dataset)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    if args.vision:
        print(f"Running VISION evaluation on {args.problem} problem...")
        print(f"Model: {args.model_id}")
        print(f"Number of samples: {args.num_samples}")

        # Build image index from eval JSON + train JSON
        eval_rows = build_eval_image_index(
            args.data_dir, args.train_json, args.images_grid_dir
        )
        # Load instances.pkl for compute_metric_cop
        number_dataset = load_pkl_dataset(
            os.path.join(args.data_dir, 'instances.pkl')
        )
        model, processor = load_vision_model(args.model_id)
        fea_rate, opt_gap, std_gap = evaluate_vision(
            args, model, processor, eval_rows, number_dataset
        )
    else:
        print(f"Running {args.eval_method} evaluation on {args.problem} problem...")
        print(f"Model: {args.model_id}")
        print(f"Number of samples: {args.num_samples}")
        if args.eval_method == 'best_of_n':
            print(f"Best-of-N: {args.best_of_n}")
            print(f"Batch size: {args.batch_size}")
            print(f"Temperature: {args.temperature}")
            print(f"Top-p: {args.top_p}")
        fea_rate, opt_gap, std_gap = evaluate_model(args)

    print(f"\nFeasibility Rate: {fea_rate:.4f}")
    print(f"Optimality Gap:   {opt_gap:.4f}")
    print(f"Std of Gap:       {std_gap:.4f}")


if __name__ == "__main__":
    main()
