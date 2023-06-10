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