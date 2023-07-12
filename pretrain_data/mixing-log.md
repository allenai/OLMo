# OLMO Mixing Log


Tagged Wikipedia using following command

```shell
ai2_llm_filters \
    -d 'wikipedia/v0' \
    -n olmo_mix_v1_taggers \
    -t \
        jigsaw_nsfw_sencence_v2 \
        jigsaw_hatespeech_sentence_v2 \
        pii_regex_with_counts_fast_v2 \
        gopher_v1 \
        ft_lang_id_en_paragraph_with_doc_score_v2 \
        uniseg_length_paragraphs_with_doc_length_v1 \
    -p 96 \
    --reuse-existing $HOME/wikipedia/meta \
    --local-read-cache $HOME/wikipedia/cache  \
    --skip-on-failure
```

Tagged C4 with the following. Using both `v0` and `v0-c4-cleaned`. The `c4-cleaned` shouldn't have much of a diff, but it's good for consistency.


```shell
ai2_llm_filters \
    -d 'c4/v0-c4-cleaned' \
    -n olmo_mix_v1_taggers \
    -t \
        jigsaw_nsfw_sencence_v2 \
        jigsaw_hatespeech_sentence_v2 \
        pii_regex_with_counts_v2 \
        gopher_v1 \
    -p 96 \
    --reuse-existing $HOME/c4-v0-c4-cleaned/meta \
    --local-read-cache $HOME/c4-v0-c4-cleaned/cache
```

```shell
ai2_llm_filters \
    -d 'c4/v0' \
    -n olmo_mix_v1_taggers \
    -t \
        jigsaw_nsfw_sencence_v2 \
        jigsaw_hatespeech_sentence_v2 \
        pii_regex_with_counts_v2 \
        gopher_v1 \
    -p 96 \
    --reuse-existing $HOME/c4-v0/meta \
    --local-read-cache $HOME/c4-v0/cache
```


Finally we tag books, both in `wikibooks` and `gutenberg`.

```shell
ai2_llm_filters \
    -d 'gutenberg/v0' \
    -n olmo_mix_v1_taggers \
    -t \
        jigsaw_nsfw_sencence_v2 \
        jigsaw_hatespeech_sentence_v2 \
        pii_regex_with_counts_v2 \
        gopher_v1 \
        ft_lang_id_en_paragraph_with_doc_score_v2 \
        uniseg_length_paragraphs_with_doc_length_v1 \
    -p 96 \
    --reuse-existing $HOME/gutemberg/meta \
    --local-read-cache $HOME/gutemberg/cache
```

```shell
ai2_llm_filters\
    -d 'wikibooks/v0'\
    -n olmo_mix_v1_taggers\
    -t\
        jigsaw_nsfw_sencence_v2\
        jigsaw_hatespeech_sentence_v2\
        pii_regex_with_counts_fast_v2\
        gopher_v1\
        ft_lang_id_en_paragraph_with_doc_score_v2\
        uniseg_length_paragraphs_with_doc_length_v1\
    -p 96\
    --reuse-existing $HOME/wikibooks/meta \
    --local-read-cache $HOME/wikibooks/cache
```

Running PII detection on v1-c4-small:

```shell
ai2_llm_filters \
    -d common-crawl/v1-c4-cleaned \
    -n pii_detection \
    -t pii_regex_with_counts_fast_v2 \
    -p 120 \
    --reuse-existing $HOME/v1-c4-cleaned-pii/cc_en_head \
    --skip-on-failure \
    --local-read-cache $HOME/v1-c4-cleaned/cc_en_head_download
```

Created configurations

```shell
python /Users/lucas/Code/LLM/pretrain_data/mixer/scripts/partition_deduper.py -w 100 -o pretrain_data
```

Output:

```text
pretrain_data/mixer/config/pdedup_c1_v1_c4-cleaned/0.json: 1105.45 GB, 756 files.
pretrain_data/mixer/config/pdedup_c1_v1_c4-cleaned/1.json: 1105.15 GB, 756 files.
pretrain_data/mixer/config/pdedup_c1_v1_c4-cleaned/2.json: 1104.69 GB, 756 files.
pretrain_data/mixer/config/pdedup_c1_v1_c4-cleaned/3.json: 1104.00 GB, 755 files.
pretrain_data/mixer/config/pdedup_c1_v1_c4-cleaned/4.json: 1103.65 GB, 755 files.
pretrain_data/mixer/config/pdedup_c1_v1_c4-cleaned/5.json: 1103.37 GB, 755 files.
pretrain_data/mixer/config/pdedup_c1_v1_c4-cleaned/6.json: 1103.09 GB, 755 files.
pretrain_data/mixer/config/pdedup_c1_v1_c4-cleaned/7.json: 1102.80 GB, 755 files.
```

Running paragraph deduplication on ~8 shards, each has about 1.1TB (see above).

```shell
~/target/release/deduper pretrain_data/mixer/config/pdedup_c1_v1_c4-cleaned/0.json
~/target/release/deduper pretrain_data/mixer/config/pdedup_c1_v1_c4-cleaned/1.json
```

```shell
~/target/release/deduper pretrain_data/mixer/config/pdedup_c1_v1_c4-cleaned/2.json
~/target/release/deduper pretrain_data/mixer/config/pdedup_c1_v1_c4-cleaned/3.json
```

```shell
~/target/release/deduper pretrain_data/mixer/config/pdedup_c1_v1_c4-cleaned/4.json
~/target/release/deduper pretrain_data/mixer/config/pdedup_c1_v1_c4-cleaned/5.json
```

```shell
~/target/release/deduper pretrain_data/mixer/config/pdedup_c1_v1_c4-cleaned/6.json
~/target/release/deduper pretrain_data/mixer/config/pdedup_c1_v1_c4-cleaned/7.json
```


After mixing, start tokenizing sources:

Books:

```shell
python scripts/prepare_memmap_dataset.py  \
    s3://ai2-llm/pretraining-data/sources/olmo-mix/v1/documents/books \
    --safe-mode \
    --output s3://ai2-llm/preprocessed/olmo-mix/v1/gpt-neox-20b-pii-special/books \
    --tokenizer "allenai/eleuther-ai-gpt-neox-20b-pii-special" \
    --workers 120
```

Semantic Scholar:

```shell
python scripts/prepare_memmap_dataset.py  \
    s3://ai2-llm/pretraining-data/sources/olmo-mix/v1/documents/s2 \
    --safe-mode \
    --output s3://ai2-llm/preprocessed/olmo-mix/v1/gpt-neox-20b-pii-special/s2 \
    --tokenizer "allenai/eleuther-ai-gpt-neox-20b-pii-special" \
    --workers 120
```

Wikipedia:

```shell
python scripts/prepare_memmap_dataset.py  \
    s3://ai2-llm/pretraining-data/sources/olmo-mix/v1/documents/wiki \
    --safe-mode \
    --output s3://ai2-llm/preprocessed/olmo-mix/v1/gpt-neox-20b-pii-special/wiki \
    --tokenizer "allenai/eleuther-ai-gpt-neox-20b-pii-special" \
    --workers 120
```

C4:

```shell
python scripts/prepare_memmap_dataset.py  \
    s3://ai2-llm/pretraining-data/sources/olmo-mix/v1/documents/c4 \
    --safe-mode \
    --output s3://ai2-llm/preprocessed/olmo-mix/v1/gpt-neox-20b-pii-special/c4 \
    --tokenizer "allenai/eleuther-ai-gpt-neox-20b-pii-special" \
    --workers 120 \
    --cache-dir /data2/llm-preprocessed
```

Stack:

```shell
python scripts/prepare_memmap_dataset.py  \
    s3://ai2-llm/pretraining-data/sources/olmo-mix/v1/documents/stack \
    --safe-mode \
    --output s3://ai2-llm/preprocessed/olmo-mix/v1/gpt-neox-20b-pii-special/stack \
    --tokenizer "allenai/eleuther-ai-gpt-neox-20b-pii-special" \
    --workers 120 \
    --cache-dir /data2/llm-preprocessed
```


## Calculating Size

- Wiki: 6.21 GB -> 3,635,728,771 tokens (585m tokens per GB)
- Books: 7.08 GB -> 4,755,860,202 tokens (672m tokens per GB)
- S2: 160.01 GB -> 56,783,583,427 tokens (355m tokens per GB)
- C4: 323.95 GB -> 174,398,315,760 tokens (538m tokens per GB)
- Stack: 724.28 GB -> 430,067,843,952 tokens (593m tokens per GB)


Sampled

- Wiki: 6.21 GB -> 3,635,728,771 tokens (585m tokens per GB)
- Books: 7.08 GB -> 4,755,860,202 tokens (672m tokens per GB)
- S2: 160.01 GB -> 56,783,583,427 tokens (355m tokens per GB)
- C4: 323.95 GB -> 174,398,315,760 tokens (538m tokens per GB)
- Stack: 593.4 GB -> 430,067,843,952 tokens (593m tokens per GB)
- Common Crawl: 2.61 TB -> 1,500,000,000,000 tokens (574m tokens per GB)


Over a single cc

2.87 GB -> 1.76 GB (61.3% reduction)

9.48 TB -> 4.89 TB -> 2.861T tokens


Random tagger:

```shell
ai2_llm_filters \
    -d 'olmo-mix/v1' \
    -n random \
    -t random_number_v1 \
    -p 120 \
    --reuse-existing $HOME/olmo_mix/meta \
    --files-regex-pattern 'stack-v2-mxixer-train' \
    --local-read-cache $HOME/olmo_mix/cache
```

```shell
ai2_llm_filters \
    -d 'olmo-mix/v1' \
    -n random \
    -t random_number_v1 \
    -p 120 \
    --reuse-existing /data2/olmo_mix/meta \
    --files-regex-pattern 'cc_en_head' \
    --local-read-cache /data2/olmo_mix/cache
```

```shell
ai2_llm_filters \
    -d 'olmo-mix/v1' \
    -n random \
    -t random_number_v1 \
    -p 120 \
    --reuse-existing /tmp/olmo_mix/meta \
    --files-regex-pattern 'cc_en_middle' \
    --local-read-cache /tmp/olmo_mix/cache
```

```shell
ai2_llm_filters \
    -d 'olmo-mix/v1' \
    -n random \
    -t random_number_v1 \
    -p 120 \
    --reuse-existing /tmp/olmo_mix/meta \
    --files-regex-pattern 'reddit-ablation-base' \
    --local-read-cache /tmp/olmo_mix/cache
```


```shell
python scripts/prepare_memmap_dataset.py  \
    s3://ai2-llm/pretraining-data/sources/olmo-mix/v1-sample/documents/stack \
    --safe-mode \
    --output s3://ai2-llm/preprocessed/olmo-mix/v1-sample/gpt-neox-20b-pii-special/stack \
    --tokenizer "allenai/eleuther-ai-gpt-neox-20b-pii-special" \
    --workers 120 \
    --cache-dir /tmp/llm-preprocessed
```


```shell
python scripts/prepare_memmap_dataset.py  \
    s3://ai2-llm/pretraining-data/sources/olmo-mix/v1-sample/documents/cc_en_head \
    --safe-mode \
    --output s3://ai2-llm/preprocessed/olmo-mix/v1-sample/gpt-neox-20b-pii-special/common-crawl/cc_en_head \
    --tokenizer "allenai/eleuther-ai-gpt-neox-20b-pii-special" \
    --workers 120 \
    --cache-dir /tmp/llm-preprocessed/cc_en_head
```

```shell
python scripts/prepare_memmap_dataset.py  \
    s3://ai2-llm/pretraining-data/sources/olmo-mix/v1-sample/documents/cc_en_middle \
    --safe-mode \
    --output s3://ai2-llm/preprocessed/olmo-mix/v1-sample/gpt-neox-20b-pii-special/common-crawl/cc_en_middle \
    --tokenizer "allenai/eleuther-ai-gpt-neox-20b-pii-special" \
    --workers 120 \
    --cache-dir /tmp/llm-preprocessed
```

```shell
python scripts/prepare_memmap_dataset.py  \
    s3://ai2-llm/pretraining-data/sources/olmo-mix/v1-sample/documents/cc_en_tail \
    --safe-mode \
    --output s3://ai2-llm/preprocessed/olmo-mix/v1-sample/gpt-neox-20b-pii-special/common-crawl/cc_en_tail \
    --tokenizer "allenai/eleuther-ai-gpt-neox-20b-pii-special" \
    --workers 120 \
    --cache-dir /tmp/llm-preprocessed
```


```shell
~/target/release/mixer pretrain_data/mixer/config/olmo-train/common-crawl-tail.json && ai2_llm_filters -d 'olmo-mix/v1' -n random -t random_number_v1 -p 120 --reuse-existing /tmp/olmo_mix/meta --files-regex-pattern 'cc_en_tail' --local-read-cache /tmp/olmo_mix/cache && ~/target/release/mixer pretrain_data/mixer/config/olmo-train-sample/common-crawl-tail.json
```

## Gopher-like


```shell
python scripts/prepare_memmap_dataset.py  \
    s3://ai2-llm/pretraining-data/sources/olmo-mix/v1-sample-small/documents/stack \
    --safe-mode \
    --output s3://ai2-llm/preprocessed/olmo-mix/v1-sample-small/gpt-neox-20b-pii-special/stack \
    --tokenizer "allenai/eleuther-ai-gpt-neox-20b-pii-special" \
    --workers 120 \
    --cache-dir /tmp/llm-preprocessed
```

```shell
python scripts/prepare_memmap_dataset.py  \
    s3://ai2-llm/pretraining-data/sources/olmo-mix/v1-no-removal/documents/common-crawl/cc_en_head \
    --safe-mode \
    --output s3://ai2-llm/preprocessed/olmo-mix/v1-no-removal/gpt-neox-20b-pii-special/common-crawl/cc_en_head \
    --tokenizer "allenai/eleuther-ai-gpt-neox-20b-pii-special" \
    --workers 120 \
    --cache-dir /tmp/llm-preprocessed/cc_en_head
```

```shell
python scripts/prepare_memmap_dataset.py  \
    s3://ai2-llm/pretraining-data/sources/olmo-mix/v1-no-removal/documents/common-crawl/cc_en_middle \
    --safe-mode \
    --output s3://ai2-llm/preprocessed/olmo-mix/v1-no-removal/gpt-neox-20b-pii-special/common-crawl/cc_en_middle \
    --tokenizer "allenai/eleuther-ai-gpt-neox-20b-pii-special" \
    --workers 120 \
    --cache-dir /tmp/llm-preprocessed
```

```shell
python scripts/prepare_memmap_dataset.py  \
    s3://ai2-llm/pretraining-data/sources/olmo-mix/v1-no-removal/documents/common-crawl/cc_en_tail \
    --safe-mode \
    --output s3://ai2-llm/preprocessed/olmo-mix/v1-no-removal/gpt-neox-20b-pii-special/common-crawl/cc_en_tail \
    --tokenizer "allenai/eleuther-ai-gpt-neox-20b-pii-special" \
    --workers 120 \
    --cache-dir /tmp/llm-preprocessed
```
