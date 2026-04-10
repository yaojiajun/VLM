#!/bin/bash
echo "Waiting for SFT training (PID 142445) to finish..."
while kill -0 142445 2>/dev/null; do
    sleep 60
done
echo "SFT done. Starting RL training at $(date)..."

cd /root/autodl-tmp/yao/VisionSolver-main
TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
nohup /root/autodl-tmp/yao/llm_co/bin/python rl_train_vision.py \
    --model_name ./output_vision_cvrp_vl7b/checkpoint-12500 \
    --output_dir ./output_vision_rl2_cvrp_vl7b \
    --save_steps 100 \
    --num_epochs 1 \
    --batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_generations 2 \
    --learning_rate 1e-6 \
    > rl2_train.log 2>&1
echo "RL training done at $(date)."
