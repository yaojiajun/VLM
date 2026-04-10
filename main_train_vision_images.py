import os
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import argparse
import torch
import wandb
from unsloth import FastVisionModel, is_bfloat16_supported
from unsloth.trainer import UnslothVisionDataCollator
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from transformers import TrainingArguments
import re
import os
import json
from PIL import Image
from utils import calculate_total_distance, compute_euclidean_distance_matrix, calculate_pfsp_makespan
import numpy as np


# ─────────────────────────────────────────────────────────────
# Prompt template
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Vision SFT trainer for CVRP (Qwen2.5-VL) using images/')

    # Model
    parser.add_argument('--model_name', type=str, default='/root/autodl-tmp/yao/models1/Qwen/Qwen2.5-VL-7B-Instruct',
                        help='HuggingFace model name (must be a VLM supported by Unsloth)')
    parser.add_argument('--max_seq_length', type=int, default=4096)
    parser.add_argument('--dtype', type=str, default='bfloat16', choices=['bfloat16', 'float16'])
    parser.add_argument('--load_in_4bit', action='store_true', default=False)

    # Data
    parser.add_argument('--data_dir', type=str,
                        default='./data_sft/cvrp',
                        help='Directory containing train JSON and images/')
    parser.add_argument('--train_json', type=str, default='train_cvrp-001.json')
    parser.add_argument('--images_dir', type=str, default='images',
                        help='Sub-directory (under data_dir) that holds cvrp_NNNNN.png images')
    parser.add_argument('--num_train_samples', type=int, default=100005,
                        help='How many training records to use (limited by available images)')

    # LoRA
    parser.add_argument('--lora_r', type=int, default=64)
    parser.add_argument('--lora_alpha', type=int, default=64)
    parser.add_argument('--bias', type=str, default='none', choices=['none', 'all', 'lora_only'])
    parser.add_argument('--use_gradient_checkpointing', type=str, default='unsloth')
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--use_rslora', action='store_true', default=False)
    parser.add_argument('--finetune_vision_layers', action='store_true', default=False,
                        help='Also train the vision encoder layers')
    parser.add_argument('--finetune_language_layers', action='store_true', default=True)
    parser.add_argument('--finetune_attention_modules', action='store_true', default=True)
    parser.add_argument('--finetune_mlp_modules', action='store_true', default=True)

    # Training
    parser.add_argument('--per_device_train_batch_size', type=int, default=2)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--warmup_steps', type=int, default=20)
    parser.add_argument('--num_train_epochs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--logging_steps', type=int, default=1)
    parser.add_argument('--optim', type=str, default='adamw_8bit')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--lr_scheduler_type', type=str, default='linear')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_total_limit', type=int, default=10)
    parser.add_argument('--save_steps', type=int, default=1000)

    # Output
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume training from')
    parser.add_argument('--cache_dir', type=str, default=None,
                        help='HuggingFace cache directory (optional)')

    return parser.parse_args()


# ─────────────────────────────────────────────────────────────
# Image filename lookup
# naming: cvrp_{idx:05d}.png  (0-based index)
# ─────────────────────────────────────────────────────────────
def get_image_path(images_dir: str, record_idx: int) -> str:
    """Return the expected image path for the i-th record (0-based)."""
    filename = f"cvrp_{record_idx:05d}.png"
    return os.path.join(images_dir, filename)


def load_vision_dataset(json_path: str, images_dir: str, max_samples: int) -> Dataset:
    """
    Load JSON records that have a matching image.
    Only stores image paths (not PIL Images) to avoid loading all images into memory.
    Images are loaded lazily during training via the collator.
    """
    with open(json_path) as f:
        records = json.load(f)

    records = records[:max_samples]
    rows = []
    missing = 0
    for idx, rec in enumerate(records):
        img_path = get_image_path(images_dir, idx)
        if not os.path.exists(img_path):
            missing += 1
            continue
        rows.append({
            "image_path": img_path,
            "vehicle_capacity": rec["vehicle_capacity"],
            "output": rec["output"],
        })

    if missing:
        print(f"  Skipped {missing} records (image not found).")
    print(f"  Loaded {len(rows)} vision training samples.")
    return Dataset.from_list(rows)


def make_conversation(sample: dict) -> dict:
    """Convert a dataset row (with image_path) to a messages conversation dict."""
    image = Image.open(sample["image_path"]).convert("RGB")
    instruction = build_instruction(sample["vehicle_capacity"])
    return {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text": instruction},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": sample["output"]}],
            },
        ]
    }


def train_model(args):
    # ── Output dir ──────────────────────────────────────────
    if args.output_dir is None:
        dir_out = (
            f"output_vision_images_alpha{args.lora_alpha}_r{args.lora_r}"
            f"_cvrp_seq{args.max_seq_length}"
            f"_b{args.per_device_train_batch_size}"
            f"_ep{args.num_train_epochs}"
        )
    else:
        dir_out = args.output_dir

    # ── W&B ─────────────────────────────────────────────────
    wandb.init(
        project=args.model_name.split('/')[-1] + "_cvrp_vision_sft",
        name=dir_out,
    )

    # ── Load model ──────────────────────────────────────────
    dtype = torch.bfloat16 if args.dtype == 'bfloat16' else torch.float16

    model, processor = FastVisionModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=dtype,
        load_in_4bit=args.load_in_4bit,
        **({"cache_dir": args.cache_dir} if args.cache_dir else {}),
    )

    # ── LoRA / PEFT ─────────────────────────────────────────
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=args.finetune_vision_layers,
        finetune_language_layers=args.finetune_language_layers,
        finetune_attention_modules=args.finetune_attention_modules,
        finetune_mlp_modules=args.finetune_mlp_modules,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0,
        bias=args.bias,
        random_state=args.random_state,
        use_rslora=args.use_rslora,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
    )

    # ── Dataset ─────────────────────────────────────────────
    images_dir = os.path.join(args.data_dir, args.images_dir)
    train_json_path = os.path.join(args.data_dir, args.train_json)

    print("Loading training dataset...")
    train_dataset = load_vision_dataset(train_json_path, images_dir, args.num_train_samples)
    train_dataset = train_dataset.shuffle(seed=args.seed)

    # ── Data collator ────────────────────────────────────────
    collator = UnslothVisionDataCollator(
        model=model,
        processor=processor,
        formatting_func=make_conversation,
    )

    # ── Trainer ──────────────────────────────────────────────
    FastVisionModel.for_training(model)

    trainer = SFTTrainer(
        model=model,
        tokenizer=processor,
        data_collator=collator,
        train_dataset=train_dataset,
        eval_dataset=None,
        args=SFTConfig(
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            num_train_epochs=args.num_train_epochs,
            learning_rate=args.learning_rate,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=args.logging_steps,
            optim=args.optim,
            weight_decay=args.weight_decay,
            lr_scheduler_type=args.lr_scheduler_type,
            seed=args.seed,
            output_dir=dir_out,
            report_to="wandb",
            eval_strategy="no",
            save_total_limit=args.save_total_limit,
            save_steps=args.save_steps,
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_seq_length=args.max_seq_length,
            packing=False,
        ),
    )

    # ── Train ────────────────────────────────────────────────
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    return trainer


if __name__ == "__main__":
    args = parse_args()
    train_model(args)
