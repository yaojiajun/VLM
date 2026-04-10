import os
import torch
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from unsloth import FastVisionModel

MODEL_DIR = "./output_vision_grid_uniform_cvrp_vl7b/checkpoint-6250"

print(f"Loading vision model from {MODEL_DIR} ...")
model, tokenizer = FastVisionModel.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float16,
    load_in_4bit=False,
    local_files_only=True,
)

print("Merging LoRA and saving to ./saved_models_grid_uniform ...")
model.save_pretrained_merged(
    "./saved_models_grid_uniform",
    tokenizer,
    save_method="merged_16bit",
)
print("Done.")
