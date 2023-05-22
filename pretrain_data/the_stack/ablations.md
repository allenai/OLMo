# The Stack Ablations

Proposal on what ablations to run for the stack. May change based on discussion.

### Overview of current versions

|  | Processing details | # whitespace tokens | # unicode tokens |
| --- | --- | --- | --- |
| v0 | raw | 195B | 514B |
| v1 | remove copyright lines (RedPajama) | 129B | 334B |
| v2 | line-based-heuristics (RedPajama) | 101B | 217B |

### Proposed ablations

1. Stack-only ablations:
    1. Ablation 1: Does further deduplication help? (We already start with minhash-LSH deduped Stack). Train model on v0-sampled and v0-sampled-deduped.
        1. Compare perplexity on held-out stack test set.
        2. Compare performance on code-eval datasets (HumanEval/MBPP). 
    2. Ablation 2: Does Redpajama-filtering help? Train model on v2-sampled. Compare with v0-sampled.
        1. Compare perplexity on held-out stack test set.
        2. Compare performance on code-eval datasets (HumanEval/MBPP). 
       
2. Stack + C4 ablations: These are to test whether training on code improves performance on certain downstream tasks.
    1. Ablation 1: Take the best-performing set from Stack-only ablations, and add it to c4. Compare with c4-only model.
        1. Compare performance on code-eval datasets (HumanEval/MBPP).
        2. Compare performance on chain-of-thought reasoning datasets (TODO).

### Questions 

1. Should we already start with pii-filtering? Or remove it for the final data?
2. We will sample 150B tokens for training for ablations. What should the size of the held-out eval set be for the stack?
3. When running C4 + Stack ablations, should the The Stack and C4 be downsampled further so that the total size is still 150B tokens?
4. Number of tokens varies quite a bit for code documents; should we be considering 150 B olmo-tokenizer tokens?


### Summary of models to be trained

| Model | Eval |
| --- | --- |
| v0-sampled | * perplexity on held-out stack eval set. <br /> * performance on code-eval datasets (HumanEval/MBPP). |
| v0-sampled-deduped | * perplexity on held-out stack eval set. <br /> * performance on code-eval datasets (HumanEval/MBPP). |
| v2-sampled | * perplexity on held-out stack eval set. <br /> * performance on code-eval datasets (HumanEval/MBPP). |
| selected from above + c4 | * performance on code-eval datasets (HumanEval/MBPP). <br /> * performance on chain-of-thought reasoning tasks. |
