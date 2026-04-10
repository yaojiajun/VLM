"""
POMO-style 8x augmentation evaluation.
For each instance, run inference on all 8 augmented images,
pick the best feasible solution among them.
"""
import os
import argparse
import torch
import numpy as np
import json
import re
from tqdm import tqdm
from utils import load_pkl_dataset, compute_metric_cop

def _load_vision_libs():
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from PIL import Image
    return AutoProcessor, Qwen2_5_VLForConditionalGeneration, Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='saved_models')
    parser.add_argument('--problem',  type=str, default='cvrp')
    parser.add_argument('--data_dir', type=str, default='./data_sft/cvrp')
    parser.add_argument('--images_grid8_dir', type=str, default='eval/images_grid8')
    parser.add_argument('--num_samples', type=int, default=100)
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    return parser.parse_args()


def build_vision_instruction(vehicle_capacity: str) -> str:
    return (
        f"The image shows a Capacitated Vehicle Routing Problem (CVRP) instance. "
        f"Each blue circle is a customer (label = node id, size \u221d demand). "
        f"The green star is the depot (node 0). Vehicle capacity: {vehicle_capacity}. "
        f"Assign every customer to exactly one vehicle route so that each route "
        f"starts and ends at the depot and the total demand per route does not exceed "
        f"the vehicle capacity. Minimize the total travel distance.\n\n"
        f"Provide the solution in the following format:\n"
        f"Routes: [[0, ..., 0], [0, ..., 0], ...], Objective: <total distance>"
    )


def load_eval_data(data_dir, images_grid8_dir, num_samples):
    eval_json = os.path.join(data_dir, 'eval', 'test.json')
    images_dir = os.path.join(data_dir, images_grid8_dir)
    with open(eval_json) as f:
        eval_data = json.load(f)

    rows = []
    for j, rec in enumerate(eval_data[:num_samples]):
        aug_paths = []
        for aug in range(8):
            p = os.path.join(images_dir, f"eval_{j:03d}_aug{aug}_n{rec['num_nodes']}.png")
            if os.path.exists(p):
                aug_paths.append(p)
        if not aug_paths:
            print(f"  [warn] no aug images found for eval[{j}]")
            continue
        rows.append({
            'aug_paths':        aug_paths,
            'vehicle_capacity': str(rec['vehicle_capacity']),
            'output':           rec['output'],
            'num_nodes':        rec['num_nodes'],
        })
    print(f"  Loaded {len(rows)} instances, each with up to 8 aug images.")
    return rows


def load_vision_model(model_id):
    AutoProcessor, Qwen2_5_VLForConditionalGeneration, _ = _load_vision_libs()
    print(f"Loading vision model from {model_id} ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()
    return model, processor


def run_inference_batch(model, processor, image_paths, vehicle_capacity, max_new_tokens=1024):
    """Run inference on a batch of images (all 8 augmentations) in one forward pass."""
    _, _, Image = _load_vision_libs()
    instruction = build_vision_instruction(vehicle_capacity)

    texts = []
    images = []
    for img_path in image_paths:
        image = Image.open(img_path).convert("RGB")
        images.append(image)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": instruction},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        texts.append(text)

    inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    return processor.batch_decode(
        generated_ids[:, input_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def is_feasible(text, demands, capacity):
    m = re.search(r'Routes:\s*\[\s*(.*)\]', text, re.DOTALL)
    if not m:
        return False
    try:
        import ast
        routes = ast.literal_eval(f'[{m.group(1).strip()}]')
    except Exception:
        return False
    if not all(isinstance(r, list) for r in routes):
        return False
    visited = set()
    for route in routes:
        if not route or route[0] != 0 or route[-1] != 0:
            return False
        total = sum(demands[node] for node in route if node != 0)
        if total > capacity:
            return False
        visited.update(n for n in route if n != 0)
    return visited == set(range(1, len(demands)))


def evaluate_pomo8(args, model, processor, eval_rows, number_dataset):
    labels      = [r['output'] for r in eval_rows]
    predictions = []
    raw_outputs = []

    for eval_idx, row in enumerate(tqdm(eval_rows)):
        inst_demands  = number_dataset[eval_idx][1]
        inst_capacity = number_dataset[eval_idx][2]

        # Batch inference: all 8 augmented images in one forward pass
        candidates = run_inference_batch(
            model, processor, row['aug_paths'],
            row['vehicle_capacity'], args.max_new_tokens
        )

        # Print each candidate result
        tqdm.write(f"\n--- Instance {eval_idx} (n={row['num_nodes']}, cap={row['vehicle_capacity']}) ---")
        feasible_count = 0
        for aug_i, c in enumerate(candidates):
            fea = is_feasible(c, inst_demands, inst_capacity)
            obj_m = re.search(r'Objective:\s*([\d.]+)', c)
            obj_str = obj_m.group(1) if obj_m else "N/A"
            status = "FEASIBLE" if fea else "infeasible"
            tqdm.write(f"  aug{aug_i}: {status}, obj={obj_str}")
            if fea:
                feasible_count += 1

        # Pick best feasible; fallback to first candidate
        best = None
        best_obj = float('inf')
        for c in candidates:
            if is_feasible(c, inst_demands, inst_capacity):
                m = re.search(r'Objective:\s*([\d.]+)', c)
                obj = float(m.group(1)) if m else float('inf')
                if obj < best_obj:
                    best_obj = obj
                    best = c
        if best is None:
            best = candidates[0]
            obj_fallback = re.search(r'Objective:\s*([\d.]+)', best)
            tqdm.write(f"  => no feasible solution, using aug0 (obj={obj_fallback.group(1) if obj_fallback else 'N/A'})")
        else:
            tqdm.write(f"  => best feasible obj={best_obj:.4f} (from {feasible_count} feasible)")

        raw_outputs.append(best)
        predictions.append("### Response:\n" + best)

    fea_rate, opt_gap, std_gap = compute_metric_cop(
        predictions, labels, number_dataset, problem=args.problem
    )

    save_path = os.path.join(os.path.dirname(args.model_id) or '.', 'eval_predictions_pomo8.json')
    records = []
    for i, (row, raw) in enumerate(zip(eval_rows, raw_outputs)):
        records.append({
            'idx':              i,
            'aug_paths':        row['aug_paths'],
            'vehicle_capacity': row['vehicle_capacity'],
            'num_nodes':        row['num_nodes'],
            'prediction':       raw,
            'label':            row['output'],
        })
    with open(save_path, 'w') as f:
        json.dump(records, f, indent=2)
    print(f"\nPredictions saved to {save_path}")
    return fea_rate, opt_gap, std_gap


def main():
    args = parse_args()
    print(f"Running POMO-8x evaluation on {args.problem} problem...")
    print(f"Model: {args.model_id}")
    print(f"Number of instances: {args.num_samples}")

    eval_rows = load_eval_data(args.data_dir, args.images_grid8_dir, args.num_samples)
    number_dataset = load_pkl_dataset(os.path.join(args.data_dir, 'instances.pkl'))
    model, processor = load_vision_model(args.model_id)

    fea_rate, opt_gap, std_gap = evaluate_pomo8(args, model, processor, eval_rows, number_dataset)

    print(f"\nFeasibility Rate: {fea_rate:.4f}")
    print(f"Optimality Gap:   {opt_gap:.4f}")
    print(f"Std of Gap:       {std_gap:.4f}")


if __name__ == "__main__":
    main()
