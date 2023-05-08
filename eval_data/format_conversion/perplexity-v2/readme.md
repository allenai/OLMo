The code in this directory reproduces the v2 perplexity eval suite. This is at least intended for use with data ablations.

run `run_subsetter.sh` to produce this subset from the v0 full eval suite as follows,

```
bash subset_v2_eval.sh /path/to/eval-data/perplexity/v2 /path/to/eval-data/perplexity/v0
```

This produces a subset with 
* 108,110,181 gpt2 tokens
* 459,536 documents