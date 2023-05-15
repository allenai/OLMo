# Datasets for Ablation Experiments

A list of ablation experiments and the datasets used for each.

## Doc-level deduplication by URL
 
### Control

Uniform sampling of [v0](./Notes.md#v0).

### Treatment

Uniform sampling of [v1](./Notes.md#v1).

### Status

 - [ ] Train model

## Full-corpus paragraph-level deduplication

### Control

Uniform sampling of `cc_en_head` and `cc_en_middle` from [v1](./NOTES.md#v1).

### Treatment

Uniform sampling of `cc_en_head` and `cc_en_middle` from [v2](./NOTES.md#v2).


### Status

 - [ ] Train model

## C4 text-modification rules

### Control

Uniform sampling of [v1](./Notes.md#v1). 

### Treatment

Uniform sampling of `vX`

### Status

 - [ ] Create `vX` dataset by post-processing `v1` to replace `text` with `modified_text` from [c4](../filtering_heuristics/c4.py) 
 - [ ] Train model


## C4 filtering rules

### Control

Uniform sampling of [v1](./Notes.md#v1).

### Treatment

Uniform sampling of `vX`

### Status

 - [ ] Create `vX` dataset by post-processing `v1` to drop documents that don't pass the [c4](../filtering_heuristics/c4.py) filter

## Gopher filtering rules

### Control

Uniform sampling of [v1](./Notes.md#v1).

### Treatment

Uniform sampling of `vX`

### Status

 - [ ] Create `vX` dataset by post-processing `v1` to drop documents that don't pass the [gopher](../filtering_heuristics/gopher.py) filter
 - [ ] Train model


### Perplexity Quality Filter

### Control

Uniform sampling of [v1](./Notes.md#v1).

### Treatments

Different biased samplings of `v1`, so that the token contributions to the final dataset are:

| cc_en_head | cc_en_middle | cc_en_tail |
|------------|--------------|------------|
| 100% | 0%           | 0%         |
 | 50% | 50%          | 0%         |
 | 50% | 25%          | 25%        |
 | 66% | 33%          | 0%         |
 | 66% | 16%          | 16%        |
 | 75% | 25%          | 0%         |
 | 75% | 12.5%        | 12.5%      |

### Status

- [ ] Train model
