ai2_llm_filters  \
    -d stack-dedup/v0     \
    -n copyright \
    -p 16 \
    --reuse-existing s3://ai2-llm/pretraining-data/sources/stack-dedup/v0/attributes/copyright \
    --skip-on-failure
