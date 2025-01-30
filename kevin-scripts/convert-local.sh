#!/usr/bin/env bash


CHECKPOINT="/weka/oe-training-default/ai2-llm/checkpoints/Spicy-OLMo/peteish7-lr-0.00041414-anneal-from-477000-100B-lb-v0/latest"

python hf_olmo/convert_olmo_to_hf.py \
    --checkpoint-dir $CHECKPOINT \
    --destination-dir ${CHECKPOINT}-hf \
    --keep-olmo-artifact \
    --tokenizer allenai/dolma2-tokenizer






