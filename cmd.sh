#!/bin/bash

# Set the model directory path (vision model checkpoint)
MODEL_DIR="./output_vision_grid_uniform_cvrp_vl7b/checkpoint-6250"
SAVE_DIR="./saved_models_grid_uniform"

# Remove saved_models directory if it exists
if [ -d "${SAVE_DIR}" ]; then
    echo "Removing existing ${SAVE_DIR} directory..."
    rm -rf "${SAVE_DIR}"
fi

# Create a temporary Python script
cat > merge_model.py << 'EOF'
import os
import torch
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from unsloth import FastVisionModel

MODEL_DIR = "./output_vision_grid_uniform_cvrp_vl7b/checkpoint-6250"
SAVE_DIR  = "./saved_models_grid_uniform"

print(f"Loading vision model from {MODEL_DIR} ...")
model, tokenizer = FastVisionModel.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.float16,
    load_in_4bit=False,
    local_files_only=True,
)

# Merge LoRA adapters and save
print(f"Merging LoRA and saving to {SAVE_DIR} ...")
model.save_pretrained_merged(
    SAVE_DIR,
    tokenizer,
    save_method="merged_16bit",
)
print("Done.")
EOF

# Execute the Python script
echo "Merging LoRA and base model from ${MODEL_DIR}..."
if python merge_model.py; then
    echo "Model merging completed successfully!"
else
    echo "Error: Model merging failed!"
    rm merge_model.py
    exit 1
fi

# Clean up the temporary Python script
rm merge_model.py

echo "All operations completed successfully!"
