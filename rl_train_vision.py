"""
Vision GRPO RL fine-tuning for CVRP using Qwen2.5-VL.

Since trl's GRPOTrainer does not support vision models, this script
implements a lightweight GRPO training loop directly on top of
FastVisionModel.
"""

import os
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import argparse
import json
import random
import torch
import torch.nn.functional as F
import wandb
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from unsloth import FastVisionModel, is_bfloat16_supported
from unsloth.models.vision import process_vision_info
from rewards import optimality_reward_func_cvrp, feasibility_reward_func_cvrp


# ─────────────────────────────────────────────────────────────
def build_instruction(vehicle_capacity: str) -> str:
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str,
                        default='./output_vision_cvrp_vl7b/checkpoint-12500')
    parser.add_argument('--train_json', type=str,
                        default='./data_rl/cvrp/train/train_rl.json')
    parser.add_argument('--images_grid_dir', type=str,
                        default='./data_rl/cvrp/train/images_grid')
    parser.add_argument('--output_dir', type=str,
                        default='output_vision_rl_cvrp_vl7b')
    parser.add_argument('--max_seq_length', type=int, default=4096)
    parser.add_argument('--dtype', type=str, default='bfloat16')
    parser.add_argument('--load_in_4bit', action='store_true', default=False)
    # GRPO
    parser.add_argument('--num_generations', type=int, default=4,
                        help='Completions per prompt (G in GRPO)')
    parser.add_argument('--beta', type=float, default=0.05,
                        help='KL penalty coefficient')
    parser.add_argument('--max_new_tokens', type=int, default=512)
    # Training
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Prompts per gradient step')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-6)
    parser.add_argument('--save_steps', type=int, default=100)
    parser.add_argument('--logging_steps', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help='Path to RL checkpoint dir to resume from (e.g. output_vision_rl_cvrp_vl7b/checkpoint-1650)')
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────
def load_dataset(json_path, images_dir):
    with open(json_path) as f:
        records = json.load(f)
    rows = []
    missing = 0
    for idx, rec in enumerate(records):
        img_path = os.path.join(images_dir,
                                f"rl_train_{idx:05d}_n{rec['num_nodes']}.png")
        if not os.path.exists(img_path):
            missing += 1
            continue
        inst = rec["instance"]
        rows.append({
            "image_path":        img_path,
            "vehicle_capacity":  str(rec["vehicle_capacity"]),
            "output":            rec["output"],
            "instance_coords":   inst[0],
            "instance_demands":  inst[1],
            "instance_capacity": float(inst[2]) if len(inst) > 2 else float(rec["vehicle_capacity"]),
        })
    if missing:
        print(f"  Skipped {missing} records (image not found).")
    print(f"  Loaded {len(rows)} samples.")
    return rows


def build_messages(sample):
    image = Image.open(sample["image_path"]).convert("RGB")
    instruction = build_instruction(sample["vehicle_capacity"])
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": instruction},
            ],
        }
    ]


# ─────────────────────────────────────────────────────────────
def grpo_loss(model, processor, samples, args, device):
    """
    For each sample in the batch:
      1. Generate G completions
      2. Score with reward functions
      3. Compute GRPO advantage-weighted policy gradient loss
    Returns scalar loss.
    """
    total_loss = 0.0
    total_reward = 0.0
    n_valid = 0

    for sample in samples:
        messages = build_messages(sample)
        text_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
        ).to(device)

        prompt_len = inputs["input_ids"].shape[1]

        # ── Generate G completions ──────────────────────────
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.95,
                num_return_sequences=args.num_generations,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )

        # Decode completions (only the generated part)
        completions = processor.tokenizer.batch_decode(
            out[:, prompt_len:], skip_special_tokens=True
        )

        # ── Compute rewards ─────────────────────────────────
        coords    = [sample["instance_coords"]]    * args.num_generations
        demands   = [sample["instance_demands"]]   * args.num_generations
        capacity  = [sample["instance_capacity"]]  * args.num_generations
        gt        = [sample["output"]]             * args.num_generations

        r_opt  = optimality_reward_func_cvrp(completions, gt, coords, demands, capacity)
        r_feas = feasibility_reward_func_cvrp(completions, coords, demands, capacity)
        rewards = torch.tensor(
            [ro + rf for ro, rf in zip(r_opt, r_feas)],
            dtype=torch.float32, device=device
        )

        total_reward += rewards.mean().item()

        # GRPO advantage (group-normalised)
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)  # shape (G,)

        # ── Policy gradient loss over generated tokens ──────
        # Re-run forward pass for each completion to get log-probs
        sample_loss = torch.tensor(0.0, device=device)
        for g_idx in range(args.num_generations):
            gen_ids = out[g_idx].clone()  # clone to exit inference_mode tensor
            gen_ids = gen_ids.unsqueeze(0)  # (1, T)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16):
                logits = model(
                    input_ids=gen_ids,
                    pixel_values=inputs.get("pixel_values"),
                    image_grid_thw=inputs.get("image_grid_thw"),
                ).logits  # (1, T, V)

            # Log-probs of generated tokens only
            shift_logits = logits[0, prompt_len - 1:-1]      # (completion_len, V)
            shift_labels = gen_ids[0, prompt_len:]            # (completion_len,)
            log_probs = F.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_probs.gather(
                1, shift_labels.unsqueeze(1)
            ).squeeze(1)  # (completion_len,)

            # Mean log-prob weighted by advantage
            loss_g = -(adv[g_idx] * token_log_probs.mean())
            sample_loss = sample_loss + loss_g

        total_loss += sample_loss / args.num_generations
        n_valid += 1

    if n_valid == 0:
        return None, 0.0

    return total_loss / n_valid, total_reward / n_valid


# ─────────────────────────────────────────────────────────────
def train(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype  = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16

    wandb.init(project="Qwen2.5-VL-7B-Instruct_cvrp_vision_rl",
               name=args.output_dir)

    # ── Load model ─────────────────────────────────────────
    print(f"Loading model from {args.model_name} ...")
    model, processor = FastVisionModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=dtype,
        load_in_4bit=args.load_in_4bit,
    )
    FastVisionModel.for_training(model)
    model.to(device)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    # ── Resume from checkpoint ─────────────────────────────
    global_step = 0
    if args.resume_from_checkpoint and os.path.isdir(args.resume_from_checkpoint):
        ckpt = args.resume_from_checkpoint
        # Load LoRA weights
        from peft import set_peft_model_state_dict
        import safetensors.torch as st
        adapter_path = os.path.join(ckpt, "adapter_model.safetensors")
        if os.path.exists(adapter_path):
            state = st.load_file(adapter_path)
            set_peft_model_state_dict(model, state)
            print(f"  Loaded LoRA weights from {adapter_path}")
        # Load optimizer state
        opt_path = os.path.join(ckpt, "optimizer.pt")
        if os.path.exists(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location=device))
            print(f"  Loaded optimizer from {opt_path}")
        # Recover step count from dir name
        try:
            global_step = int(os.path.basename(ckpt).split("-")[-1])
            print(f"  Resuming from step {global_step}")
        except Exception:
            pass
    print("Loading dataset...")
    dataset = load_dataset(args.train_json, args.images_grid_dir)
    random.shuffle(dataset)

    # ── Training loop ──────────────────────────────────────
    global_step = 0
    optimizer.zero_grad()

    for epoch in range(args.num_epochs):
        print(f"\n=== Epoch {epoch + 1}/{args.num_epochs} ===")
        for i in range(0, len(dataset), args.batch_size):
            batch = dataset[i: i + args.batch_size]
            if not batch:
                continue

            loss, mean_reward = grpo_loss(model, processor, batch, args, device)
            if loss is None:
                continue

            (loss / args.gradient_accumulation_steps).backward()

            accum_idx = (global_step + 1) % args.gradient_accumulation_steps
            if accum_idx == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            global_step += 1

            if global_step % args.logging_steps == 0:
                log = {
                    "train/loss":        loss.item(),
                    "train/mean_reward": mean_reward,
                    "train/step":        global_step,
                }
                wandb.log(log, step=global_step)
                print(f"  step {global_step:5d} | loss {loss.item():.4f} | reward {mean_reward:.4f}")

            if global_step % args.save_steps == 0:
                ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                model.save_pretrained(ckpt_dir)
                processor.save_pretrained(ckpt_dir)
                torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pt"))
                print(f"  Saved checkpoint to {ckpt_dir}")

    # Final save
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print("Training complete.")
    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    train(args)
