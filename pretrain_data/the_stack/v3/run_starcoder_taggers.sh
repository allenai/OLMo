# Run starcoder taggers

ai2_llm_filters  \
    -d stack-dedup/v3     \
    -n starcoder-v2 \
    -t code_starcoder_taggers_v2 \
    -p 64 \
    --reuse-existing s3://ai2-llm/pretraining-data/sources/stack-dedup/v3/attributes/starcoder-v2 \
    --skip-on-failure \
    --safe-mode
