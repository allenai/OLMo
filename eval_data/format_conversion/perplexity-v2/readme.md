The code in this directory reproduces the v2 perplexity eval suite. This is at least intended for use with data ablations.

run `run_subsetter.sh` to produce this subset from the v0 full eval suite as follows,

```
bash subset_v2_eval.sh /path/to/eval-data/perplexity/v2 /path/to/eval-data/perplexity/v0
```

This produces a subset with 
* 105,949,524 gpt2 tokens
* 459,482 documents

## decontamination against this eval data

1) Setup bff ([repo](https://github.com/allenai/bff))
2) build a bloom-filter on the eval data
```
bff --bloom-filter-file perplexity_suite.bff --bloom-filter-size 268435456 --expected-ngram-count 88440012 --output-directory trash --filtering-threshold 1.0 --annotate-attribute-only `find /path/to/ai2-llm/eval-data/perplexity/v2/ -name *.gz`
```
3) Either for each data source or for each finished ablation dataset, run decontamination against the bloom-filter
```
bff --bloom-filter-file perplexity_suite.bff --bloom-filter-size 268435456 --expected-ngram-count 88440012 --output-directory path/to/output --no-update-bloom-filter --annotate-only /path/to/training/data/*.gz
```


The output will be the original data with two new json fields:
* "bff_duplicate_spans"  : [[start, end]. ... ] # byte indices of contaminated paragraphs
* "bff_contained_ngram_count" : int #count of ngrams overlapping with training corpus.

We propose just removing any document with any spans in bff_duplicate_spans. You can also just get the new fields without the old data by using `--annotate-attribute-only` instead of `--annotate-only`  in the second command.


### decontamination results on C4 (using )
* takes about ~30 mins to run on a machine with ~200 cpus.
* if you then remove any doc with any paragraph marked as contaminated
    * removes 0.24% data
    * removes 0.18% docs
* if you just removed contaminated paragraphs
    * removes 0.07% data


## v2 small

We also provide a script to create a smaller version that has at most 1 million tokens per domain. To create this run `subset_v2_smal_eval.sh`.