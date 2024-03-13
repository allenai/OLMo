#!/bin/bash

deepspeed --module scripts.llava.train \
    --deepspeed configs/deepspeed/zero2.json \
    --model_name_or_path allenai/OLMo-7B-Instruct \
    --cache_dir ${DATA_DIR}/olmo \
    --tune_mm_adapter True \
    --config_file configs/llava/pretrain/openai-vit-l-14-336-mlp2x_gelu-olmo-7b-instruct-hf-seq2048.yaml \
    --seed 6198 \
    --bf16 True \
    --output_dir ${OUTPUT_DIR}/runs/openai-vit-l-14-336-mlp2x_gelu-olmo-7b-instruct-hf-seq2048-test \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --report_to wandb
