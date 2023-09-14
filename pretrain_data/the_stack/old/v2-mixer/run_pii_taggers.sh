# Run pii taggers

ai2_llm_filters  \
    -d stack-dedup/v2-mixer     \
    -n pii \
    -t pii_regex_with_counts_fast_v2 \
    -p 64 \
    --reuse-existing s3://ai2-llm/pretraining-data/sources/stack-dedup/v2-mixer/attributes/pii \
    --skip-on-failure
