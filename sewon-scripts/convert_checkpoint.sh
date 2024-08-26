#!/usr/bin/env bash


for c in peteish1-baseline peteish1-dclm-only peteish7-anneal-baseline ; do \
    CHECKPOINT="s3://ai2-lucas-archival/checkpoints/$c/latest" ; \
    python hf_olmo/convert_olmo_to_hf.py \
    --checkpoint-dir $CHECKPOINT \
    --destination-dir ${CHECKPOINT}-hf \
    --keep-olmo-artifact ; \
done