#!/bin/bash

# wandb setting
export WANDB_ENTITY=ivy_bcsh
export WANDB_PROJECT=qwen3vl_lgsc
export WANDB_MODE=online
export WANDB_START_METHOD="thread"

export HF_HOME=/mnt/ssd1/sh/hf_home

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NPROC_PER_NODE=4

# DeepSpeed configuration
deepspeed=./scripts/zero2.json

# Model configuration
llm=Qwen/Qwen3-VL-4B-Instruct
lr=1e-5
batch_size=1
num_train_epochs=3
entry_file=qwenvl/train/train_qwen_roi.py
datasets=refcoco

# Run configurations: (roi_lr grad_accum_steps)
declare -a configs=(
    "1e-3 32"
    "1e-4 32"
    "1e-3 16"
    "1e-4 16"
)

# NOTE: The first config "1e-3 32" is what was originally set.
# I am including it here so everything runs sequentially in one go.

for config in "${configs[@]}"; do
    # Parse config
    read -r roi_lr grad_accum_steps <<< "$config"
    
    # Generate dynamic run name and output dir
    run_name="qwen3vl_4b_roi_${datasets}_3E_${roi_lr}_${lr}_BS$((NPROC_PER_NODE * batch_size * grad_accum_steps))"
    export WANDB_NAME=$run_name 
    output_dir="/mnt/hdd_data2/qwen3vl_lgsc_ckpt/${run_name}"
    
    echo "=================================================================="
    echo "Starting training run: $run_name"
    echo "ROI LR: $roi_lr | LR: $lr | Grad Accum: $grad_accum_steps"
    echo "=================================================================="

    args="
    --deepspeed ${deepspeed} \
    --model_name_or_path ${llm} \
    --dataset_use ${datasets} \
    --data_flatten False \
    --tune_mm_vision False \
    --tune_mm_mlp False \
    --tune_mm_llm False \
    --tune_roi True \
    --max_region_num 100 \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs ${num_train_epochs} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 147456 \
    --min_pixels 784 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 300 \
    --save_total_limit 5 \
    --learning_rate ${lr} \
    --roi_lr ${roi_lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${WANDB_NAME} \
    --report_to wandb"

    # Launch training for the current configuration
    torchrun --nproc_per_node ${NPROC_PER_NODE} \
        --master_addr ${MASTER_ADDR} \
        --master_port ${MASTER_PORT} \
        ${entry_file} \
        ${args}
        
    echo "Completed training run: $run_name"
    echo "Proceeding to next config if available..."
    sleep 5 # short pause between runs
done

echo "All training configurations have completed successfully!"
