#!/bin/bash
echo "Waiting for merge (PID 159180) to finish..."
while kill -0 159180 2>/dev/null; do
    sleep 15
done
echo "Merge done at $(date). Starting eval..."

TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 \
/root/autodl-tmp/yao/llm_co/bin/python eval.py \
    --vision \
    --model_id ./saved_models_grid_uniform \
    --problem cvrp \
    --data_dir ./data_sft/cvrp \
    --images_grid_dir eval/images_grid_uniform \
    --num_samples 100 \
    > eval_vision_grid_uniform.log 2>&1

echo "Eval done at $(date)."
