# Overview

[The Stack](https://huggingface.co/datasets/bigcode/the-stack) is a 6 TB dataset of code, containing 30 programming languages. This is too large for our purpose, and research has shown that deduplication improves model performance. 
We use the [deduplicated version](https://huggingface.co/datasets/bigcode/the-stack) of The Stack, which contains 3 TB of data.

**In Progress**: The dataset is being downloaded on lm-datasets.

## Notes / Comments

1. The Stack allows Github users to opt out of being part of the dataset. We timestamp the version we are downloading, is there something else that we should take care of?

2. Notes about the deduplication performed:

  - Remove exact duplicates
  - Remove near duplicates (MinHash + LSH). https://github.com/bigcode-project/bigcode-analysis/tree/main/data_analysis
