# Run taggers for Red Pajama code filtering heuristics

ai2_llm_filters  \
    -d stack-dedup/v1-mixer     \
    -n rpj-heuristics \
    -t code_redpajama_taggers_v1 \
    -p 16 \
    --reuse-existing s3://ai2-llm/pretraining-data/sources/stack-dedup/v1-mixer/attributes/rpj-heuristics \
    --skip-on-failure
