Author: Akshita Bhagia @akshitab

# Overview

[The Stack](https://huggingface.co/datasets/bigcode/the-stack) is a 6 TB dataset of code, containing 358 programming languages. This is too large for our purpose, and research has shown that deduplication improves model performance. 
We use the [deduplicated version](https://huggingface.co/datasets/bigcode/the-stack) of The Stack, which contains 3 TB of data.

# Version updates

* 2023-03-17: [v0](v0/README.md)
* 2023-05-10: [v1](v1/README.md)
* 2023-05-10: [v2](v2/README.md)

## Notes / Comments

1. The Stack allows Github users to opt out of being part of the dataset. We timestamp the version we are downloading, and should also allow users to opt-out when publishing the final dataset.

2. Notes about The Stack's deduplication methods:

   - Remove exact duplicates
   - Remove near duplicates (MinHash + LSH). https://github.com/bigcode-project/bigcode-analysis/tree/main/data_analysis
