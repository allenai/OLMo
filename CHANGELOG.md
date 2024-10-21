# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

## [v0.5.1](https://github.com/allenai/OLMo/releases/tag/v0.5.1) - 2024-10-17

### Added

- Added ability to try loading latest checkpoint from save folder using `--try_load_latest_save`.
- Added support for flash attention and gradient checkpointing to `hf_olmo`.
- Added `effective_n_kv_heads` to OLMoConfig for hacky VLLM support.

## [v0.5.0](https://github.com/allenai/OLMo/releases/tag/v0.5.0) - 2024-08-26

- Fixed conversion to HuggingFace model for DDP-trained models.
- Added support for remote source and destination for HuggingFace model conversion.

### Added

- Added support for document masking via flash-attn during training with `--data.generate_doc_lengths`.
- Added config options for `model.norm_after`, `model.scale_emb_init`, and `auxiliary_loss_multiplier` (used with zloss).
- Added scripts for running experiments on qk_norm, norm reordering, and zloss.
- Added `model.rope_theta` configuration option.
- Added `model.embedding_layer_norm` configuration option for adding a LN to the embeddings.
- Added `model.emb_init_std` configuration option to override the standard deviation used to initialize the embeddings.
- Added downstream eval task for requests dumped from oe-eval tasks
- Added `CosLinearEnvelope` scheduler, which is a pointwise product of a cosine schedule and a linear decay.
- Added ability to save outputs of submodules for debugging purposes.
- Added a number of tasks from oe-eval to the downstream eval tasks.
- Version dolma flan change in named_data_mix.py

### Changed

- Changed default distributed training strategy from single-GPU to FSDP
- Fixed behavior of `effective_memmap_dtype` to prevent unrecognized dtypes to be parsed as `uint16`.

### Fixed

- Fixed restarting a training run in later epochs so that we no longer need to set the flag `--epoch=INT`.
- Swapped in correct flan data mix.
- Fix bug where the attention norm, when applied before the attention block, was modifying the residual stream.
- Fixed `OLMo.from_checkpoint()` so that it correctly loads `olmo_core` and `torch_new` style checkpoints.
- Fixed `preserve_rng_state` being incorrectly set to False when doing gradient checkpointing with dropout 


## [v0.4.0](https://github.com/allenai/OLMo/releases/tag/v0.4.0) - 2024-07-11

### Added

- Added clipping fix to `Optimizer` class to make it work with FSDP `no_shard` and DDP.
- Added tests to compare grad norm differences between torch optimizer and clipping and OLMo optimizer and clipping on both CPU and GPU.
- Expose memmap dtype in data config
- Added support for DDP training.
- Added caching to disk of HF datasets used in downstream evals
- Added FLOPs logging
- Added configs for OLMo tiny set of models
- Added configuration field `optimizer.record_update_metrics`, which defaults to `False`, but when set to `True` will trigger AdamW to collect the step size norm and absolute max for each parameter.
- Added configuration field `optimizer.selective_updates`, which defaults to `False`, but when set to `True` will tell the optimizer to skip updating the parameter and state when the corresponding gradient is 0.
- Added configuration field `optimizer.record_update_metrics`, which defaults to `False`, but when set to True will trigger AdamW to collect the step size norm and absolute max for each parameter.
- Added `olmo_data`, a package holding data files like tokenizers.
- Added ability to load tokenizers from `olmo_data` package data.
- Added a script that can run a series of models with predictable scaling properties.

### Changed

- Added original legacy unsharding implementation back, as the default. The new
shared memory implementation can be used by passing `use_legacy_shared_mem_impl` to `unshard.py`.
- Refactor weight initialization. IMPORTANT: this does not maintain backwards-compatibility with older configs; the jobs will still run, but may produce different outputs.
- Changed the behavior of the Lion optimizer to only record the update cosine similarity when `optimizer.record_update_metrics` is `True` in order to be consistent with the API.
- Added HF datasets into `olmo_data`, and changed downstream eval to load from the package.

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
