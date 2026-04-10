#!/bin/bash
echo "Waiting for image generation (PID 8868)..."
wait 8868
echo "Image generation done. Checking image count..."
COUNT=$(ls /root/autodl-tmp/yao/VisionSolver-main/data_sft/cvrp/images/*.png 2>/dev/null | wc -l)
echo "Generated $COUNT images."

echo "Starting SFT training..."
/root/autodl-tmp/yao/llm_co/bin/python main_train_vision_images.py \
    --num_train_samples 100000 \
    --output_dir output_vision_images_v2_cvrp_vl7b \
    > train_vision_images_v2.log 2>&1
echo "Training done."
