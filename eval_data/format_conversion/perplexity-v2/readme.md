The code in this directory reproduces the v2 perplexity eval suite. This is at least intended for use with data ablations.

run `run_subsetter.sh` to produce this subset from the v0 full eval suite as follows,

```
bash subset_v2_eval.sh /path/to/eval-data/perplexity/v2 /path/to/eval-data/perplexity/v0
```

This produces a subset with 
* 105,949,524 gpt2 tokens
* 459,482 documents

## decontamination against this eval data

1) Setup bff at this exact commit ([repo at specific commit](https://github.com/allenai/bff/tree/27e30eb64dea98796a4877ea80ea37de4cbe442f))
2) get a bloom-filter of the eval data to decontaminate against

    Either download the bloom-filter from `$llm-bucket/eval-data/perplexity/blocklists/eval_subset_v2/decontaminating_filter_full_paragraph.bff` for `$llm-bucket=<path to shared project bucket>`
    <details><summary> Or rebuild the filter from the eval data</summary>
        ```
        bff --bloom-filter-file path/to/decontaminating_filter_full_paragraph.bff --bloom-filter-size 8388608 --expected-ngram-count 4751287 --output-directory trash --filtering-threshold 1.0 --min-ngram-size 13  --whole-paragraphs --annotate-attribute-only `find /path/to/ai2-llm/eval-data/perplexity/v2/ -name *.gz`
        ```
    </details>

3) Now run decontamination against the bloom-filter over some training data
    ```
    bff --bloom-filter-file path/to/decontaminating_filter_full_paragraph.bff --bloom-filter-size 8388608 --expected-ngram-count 4751287 --output-directory path/to/output --no-update-bloom-filter --min-ngram-size 13  --whole-paragraphs --annotate-attribute-only /path/to/training/data/*.gz
    ```

    The output will json lines for each document as follows:
    ```
    {
        "bff_duplicate_spans"  : [[start, end]. ... ] # byte indices of contaminated paragraphs
        "bff_contained_ngram_count" : int # count of ngrams overlapping with training corpus.
        "id": str # orginal id from input data
        "source": str # original source from input data
    }
    ```

At this time we propose just removing any document with any spans in bff_duplicate_spans.


### decontamination results on C4
* takes about 4 mins to run on a machine with ~200 cpus (so about 640 million tokens per wall clock second).
* removes 0.02% of tokens and 0.01% of documents


## v2 small

We also provide a script to create a smaller version that has at most 1 million tokens per domain. To create this run `subset_v2_smal_eval.sh`. It is a subset of the v2 data, so decontaminating against the full v2 subset will also cover v2 small.