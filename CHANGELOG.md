# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

### Fixed

- Fixed default value of `--tokenizer` argument to `scripts/prepare_tulu_data.py` to be an absolute path, not relative path, the script can be run from other directories.
- Added the option to directly pass input embeddings to `OLMo` and `OLMoForCausalLM`.
- Added support for Python 3.8.
- Added code to throw an error if `output_attentions` is set to `True` in forward call to `OLMoForCausalLM`. This functionality hasn't been implemented yet.
- Fixed running with data loading workers on LUMI
- Minor bug fix: uninitialized prompts variable

### Added
- Added `output_hidden_states` argument and associated functionality to `OLMo` and `OLMoForCausalLM` to return model intermediate hidden states.
- Added MMLU downstream evaluation tasks, with prompt variations.
- Added support for PyTorch v2.2.
- Added ability to show logs from all ranks



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
