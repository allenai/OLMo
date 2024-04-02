#!/bin/bash

# get run name, we will use this as task name in gantry
RUN_NAME=$(cat $CONFIG_PATH | grep -ohP "^run_name\:\w*(.+)$" | sed 's/run_name:\s*//')
OUTPUT_DIR="${OUTPUT_DIR}/instruct_tune/runs/${RUN_NAME}"

deepspeed --module scripts.llava.train \
    --deepspeed configs/deepspeed/zero2.json \
    --model_name_or_path allenai/OLMo-7B-Instruct \
    --cache_dir ${DATA_DIR}/olmo \
    --config_file ${CONFIG_PATH} \
    --seed 6198 \
    --bf16 True \
    --output_dir ${OUTPUT_DIR} \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --report_to wandb