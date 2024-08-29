#!/usr/bin/env bash

CHECKPOINT="s3://ai2-lucas-archival/checkpoints/peteish1-B34v0/latest"

python hf_olmo/convert_olmo_to_hf.py \
    --checkpoint-dir $CHECKPOINT \
    --destination-dir ${CHECKPOINT}-hf \
    --keep-olmo-artifact