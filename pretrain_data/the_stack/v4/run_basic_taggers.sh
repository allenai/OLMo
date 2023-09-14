# Run basic taggers

ai2_llm_filters  \
    -d stack-dedup/v4     \
    -n basic \
    -t random_number_v1 \
        whitespace_tokenizer_with_paragraphs_v1 \
	char_length_with_paragraphs_v1 \
    -p 64 \
    --reuse-existing s3://ai2-llm/pretraining-data/sources/stack-dedup/v4/attributes/basic \
    --skip-on-failure \
    --safe-mode
