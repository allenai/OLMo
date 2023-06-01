# The Stack Ablations

### Overview of current versions

|  | Processing details | # whitespace tokens | # documents |
| --- | --- | --- | --- |
| v0 | raw | 448 B | 272455312 |
| v1 | remove copyright lines (RedPajama) | 334 B |  |
| v2 | line-based-heuristics (RedPajama) | 296 B | 255064041 |

### Evaluation-set preparation

1. Code completion eval: `openai_humaneval` and `mbpp` datasets as evaluation sets. Both of these are python programming based datasets and contain a prompt, expected code generation, and test cases. `openai_humaneval` contains function definition (and docstring) as the prompt, whereas `mbpp`’s prompt is text-based (i.e., write a python function to print numbers from 1 to 10). All of our downstream evaluation tasks are evaluated using ranked-classification, which is not appropriate for code generation. Hence, the **datasets are converted to our ai2-format, where the “text” field will be “prompt” + “code”, and evaluated on perplexity**.
    
    Note: Code datasets should really be evaluated on the generated code (we have test cases as part of the datasets). However, 1B models are unlikely to produce runnable code, so this type of evaluation can be left for later models, and is not included in ablations. We only evaluate perplexity.
    
2. Held-out stack eval: A held-out set is created from the “cleaned” `v2-mixer` version. Since we are not ablating on the filtering itself, using this version to create the held-out set should be ok. 
3. Logical reasoning eval: There is some prior work (citation needed) which suggests that adding code to the training data improves model performance on downstream tasks that require chain-of-thought reasoning, etc. Again, this ability is unlikely to show up in 1B models, and is thus, not included in ablations.

### Sanity check

Run the 1B model on `v2-mixer-sampled` (150B tokens) to confirm that the loss/perplexity is going down (at least on the code eval sets).

### Ablations

1. Ablation 0 (Cancelled): Does RedPajama-filtering help? Going on prior work (RedPajama) and concurrent work (Suchin and other UW people, who are working on training code models; we do not have a paper reference for this), filtering code files by line-based heuristics is essential, and thus, **ablating this is not necessary**. This will be done as part of the preprocessing (Version: `v2-mixer`).
2. Ablation 1: Does further deduplication help? (We already started with minhash-LSH deduped Stack).
    1. Document-based deduplication on `v2-mixer` only removed 0.5% of the documents. Thus, **ablating this is also not necessary**.
    2. Paragraph-based deduplication: mark as duplicate if 80% of document is already present in the bloom filter (scored using sum of characters present in duplicate paragraphs).
3. Ablation 2: Mixing with C4. How much code can we tolerate without degrading performance? How much code should we add to improve performance on specific tasks?
    1. 5% stack, 95% c4.
    2. 10% stack, 90% c4.
    3. 15% stack, 85% c4.

### Other considerations

1. PII: we currently do not have ablations that indicate if PII-removal degrades performance or not (and how to remove PII in the first place). Hence, for 1B ablations, we do not remove PII (the c4 baseline also does not remove PII).

### Summary of models to be trained

These are compared against the c4 150B baseline.

| Model | Eval |
| --- | --- |
| v2-mixer-sampled (150B) | * confirm that loss / perplexity goes down. |
| v2-mixer-sampled-5p + c4-sampled-95p | * perplexity on v2-mixer-held-out. <br>* perplexity on openai_eval. <br>* perplexity on mbpp. |
| v2-mixer-sampled-10p + c4-sampled-90p | * perplexity on v2-mixer-held-out. <br>* perplexity on openai_eval. <br>* perplexity on mbpp. |
| v2-mixer-sampled-15p + c4-sampled-85p | * perplexity on v2-mixer-held-out. <br> * perplexity on openai_eval. <br>* perplexity on mbpp. |



---



### Proposed ablations (Old)

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
| v0-sampled | * perplexity on held-out stack eval set.* performance on code-eval datasets (HumanEval/MBPP). |
| v0-sampled-deduped | * perplexity on held-out stack eval set.* performance on code-eval datasets (HumanEval/MBPP). |
| v2-sampled | * perplexity on held-out stack eval set.* performance on code-eval datasets (HumanEval/MBPP). |
| selected from above + c4 | * performance on code-eval datasets (HumanEval/MBPP).* performance on chain-of-thought reasoning tasks. |
