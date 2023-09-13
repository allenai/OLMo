# Run basic taggers

ai2_llm_filters  \
    -d stack-dedup/v2-mixer     \
    -n basic \
    -t random_number_v1 \
        whitespace_tokenizer_with_paragraphs_v1 \
	char_length_with_paragraphs_v1 \
    -p 128 \
    --reuse-existing s3://ai2-llm/pretraining-data/sources/stack-dedup/v2-mixer/attributes/basic \
    --skip-on-failure
