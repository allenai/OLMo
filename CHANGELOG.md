# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Added

- Added clipping fix to `Optimizer` class to make it work with FSDP `no_shard` and DDP.
- Added tests to compare grad norm differences between torch optimizer and clipping and OLMo optimizer and clipping on both CPU and GPU.
- Expose memmap dtype in data config 
- Added support for DDP training.
- Added caching to disk of HF datasets used in downstream evals
- Added FLOPs logging
- Added configs for OLMo tiny set of models

### Changed

- Added original legacy unsharding implementation back, as the default. The new
shared memory implementation can be used by passing `use_legacy_shared_mem_impl` to `unshard.py`.
- Refactor weight initialization. IMPORTANT: this does not maintain backwards-compatibility with older configs; the jobs will still run, but may produce different outputs.

### Fixed

- Changed from `ignored_index` to `ignore_index` for `cross_entropy_loss` when `flash-attn>=2.5.8`.
- Make `hf_olmo` support `AutoModelForCasualLM` and similar HF methods again.

## [v0.3.0](https://github.com/allenai/OLMo/releases/tag/v0.3.0) - 2024-04-25

### Added

- Added support for Grouped Query Attention.
- Added commonsense_qa and social_iqa downstream evaluation tasks
- Added ce_loss metric, with TriviaQA and NaturalQuestions tasks
- Makes it possible to read from http/https the same way we read from s3/r2.
- Added MMLU multiple choice (A/B/C/D) 5-shot variant downstream tasks
- Tokenizer patch
- Added option to specify number of model replicas when using hybrid sharding.

### Changed

- Rename `Olmo` to `OLMo` everywhere in the codebase
- Disabled automatic garbage collection during training, instead we run manually at regular intervals to avoid ranks getting out-of-sync with their own gc.

### Removed

- Removed `AMDLayerNorm`, since the original layer norm bug has been fixed and we don't need this workaround anymore.
- Removed `OLMoParallelBlock`.

### Fixed

- Don't log garbage on nodes that aren't rank 0
- Don't crash in the HF code when we are referring to a tokenizer in a local file
- Point official training scripts to publicly available URLs
- Corrected the `resize_token_embeddings` method in the `OLMoForCausalLM` class to properly update the token embeddings when resizing the vocabulary.
- Changed `tie_weights` method to a no-op as weight tying is handled in olmo/model.py
- Fixed the size calculation for qk layer norm
- Fixed pipeline test failure that occurs due to a bug in transformers version 4.39.1
- Make `hf_olmo` compatible with transformers versions >=4.40.0

## [v0.2.5](https://github.com/allenai/OLMo/releases/tag/v0.2.5) - 2024-03-06

### Fixed

- Fixed default value of `--tokenizer` argument to `scripts/prepare_tulu_data.py` to be an absolute path, not relative path, the script can be run from other directories.
- Added the option to directly pass input embeddings to `OLMo` and `OLMoForCausalLM`.
- Added support for Python 3.8.
- Added code to throw an error if `output_attentions` is set to `True` in forward call to `OLMoForCausalLM`. This functionality hasn't been implemented yet.
- Correct scheme displayed in error messages that come from R2
- Fixed running with multiple data loading workers in LUMI
- Minor bug fix: uninitialized prompts variable

### Added
- Added `output_hidden_states` argument and associated functionality to `OLMo` and `OLMoForCausalLM` to return model intermediate hidden states.
- Ability to read from R2 like we read from S3
- Added MMLU downstream evaluation tasks, with prompt variations.
- Added support for PyTorch v2.2.
- Added ability to show logs from all ranks
- Added option for QKV clipping.
- Added basic_arithmetic downstream evaluation task

### Changed

- Changed legacy checkpoint unsharding to use processes and shared memory instead of threads


## [v0.2.4](https://github.com/allenai/OLMo/releases/tag/v0.2.4) - 2024-02-02

### Fixed

- Fixed an issue with the HuggingFace integration where we were inadvertently using a feature that was introduced in Python 3.10, causing an error for older Python versions.

## [v0.2.3](https://github.com/allenai/OLMo/releases/tag/v0.2.3) - 2024-01-31

## [v0.2.2](https://github.com/allenai/LLM/releases/tag/v0.2.2) - 2023-12-10

## [v0.2.1](https://github.com/allenai/LLM/releases/tag/v0.2.1) - 2023-12-10

## [v0.2.0](https://github.com/allenai/LLM/releases/tag/v0.2.0) - 2023-12-08

### Added

- GPT-based model.
- Tokenizer and data pre-processing pipeline.
- training script.
- Triton-based FlashAttention.
