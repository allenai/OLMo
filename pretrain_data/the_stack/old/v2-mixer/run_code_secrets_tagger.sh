# Run code secrets taggers

ai2_llm_filters  \
    -d stack-dedup/v2-mixer     \
    -n code_secrets \
    -t code_secrets_v1 \
    -p 64 \
    --reuse-existing s3://ai2-llm/pretraining-data/sources/stack-dedup/v2-mixer/attributes/code_secrets \
    --skip-on-failure
