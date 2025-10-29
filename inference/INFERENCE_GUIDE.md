# OLMo Inference Guide

This guide explains how to run inference on your trained OLMo models, including STU (Spectral Transform Unit) variants.

## Quick Start

### Simple Inference

Run inference on a single model with one or more prompts:

```bash
# Single prompt
python inference/simple_inference.py \
    --checkpoint-dir /scratch/gpfs/EHAZAN/tharuntk/OLMo-data/checkpoints/OLMo-150M-Ablations/OLMo-STU-All-150M/step0-unsharded \
    --prompt "Once upon a time" \
    --max-tokens 50

# Multiple prompts
python inference/simple_inference.py \
    --checkpoint-dir /scratch/gpfs/EHAZAN/tharuntk/OLMo-data/checkpoints/OLMo-150M-Ablations/OLMo-STU-All-150M/step0-unsharded \
    --prompts "The capital of France is" "Machine learning is" "Hello, world!" \
    --max-tokens 30
```

### Batch Inference (Compare Multiple Models)

Compare multiple models on the same prompts:

```bash
python inference/batch_inference.py \
    --checkpoint-dirs \
        /scratch/gpfs/EHAZAN/tharuntk/OLMo-data/checkpoints/OLMo-150M-Ablations/OLMo-STU-All-150M/step0-unsharded \
        /scratch/gpfs/EHAZAN/tharuntk/OLMo-data/checkpoints/OLMo-150M-Ablations/OLMo-STU-Alternating-150M/step0-unsharded \
    --prompts "The capital of France is" "Once upon a time" \
    --output comparison_results.json \
    --max-tokens 50
```

## Available Models

Your STU ablation models are located at:
```
/scratch/gpfs/EHAZAN/tharuntk/OLMo-data/checkpoints/OLMo-150M-Ablations/
```

Available models:
- `OLMo-150M` - Baseline transformer model (12 layers)
- `OLMo-STU-All-150M` - All layers use STU (15 layers)
- `OLMo-STU-Alternating-150M` - Alternating attention and STU layers (14 layers)
- `OLMo-STU-Sandwich-All-150M` - Sandwich architecture with all STU
- `OLMo-STU-Sandwich-Alternating-150M` - Sandwich architecture with alternating STU

Each model directory contains checkpoints like:
- `step0-unsharded/` - Initial checkpoint (before/minimal training)
- `stepN-unsharded/` - Checkpoint at step N
- `latest-unsharded` - Symlink to the most recent checkpoint

## Command Line Options

### simple_inference.py

```
--checkpoint-dir PATH        Path to checkpoint directory (required)
--prompt TEXT               Single prompt text
--prompts TEXT [TEXT ...]   Multiple prompts
--max-tokens INT            Maximum tokens to generate (default: 50)
--temperature FLOAT         Sampling temperature (default: 1.0)
--top-k INT                 Top-k sampling (default: 50)
--top-p FLOAT              Top-p/nucleus sampling (default: 0.95)
--device {cuda,cpu}         Device to use (default: cuda if available)
```

### batch_inference.py

```
--checkpoint-dirs PATH [PATH ...]  Paths to checkpoint directories (required)
--prompts TEXT [TEXT ...]          List of prompts (required)
--output PATH                      Output JSON file (default: inference_results.json)
--max-tokens INT                   Maximum tokens to generate (default: 50)
--top-p FLOAT                     Top-p sampling (default: 0.95)
--device {cuda,cpu}               Device to use (default: cuda if available)
```

## Sampling Parameters

- **temperature**: Controls randomness. Lower values (< 1.0) make output more focused, higher values (> 1.0) make it more random.
- **top-k**: Only sample from the top k most likely tokens.
- **top-p**: Nucleus sampling - sample from smallest set of tokens whose cumulative probability exceeds p.

Examples:
```bash
# Greedy decoding (deterministic)
--top-p 1.0 --temperature 1.0

# More focused/deterministic output
--top-p 0.9 --temperature 0.7

# More diverse output
--top-p 0.95 --temperature 1.2

# Top-k sampling
--top-k 40 --top-p 1.0
```

## Notes

### STU Model Support

The inference scripts include fixes for STU (Spectral Transform Unit) models:

1. **Device placement**: STU's `phi` tensor is automatically moved to the correct device (CUDA/CPU)
2. **KV cache handling**: STU layers don't use key-value caching, so `None` values are properly handled
3. **Past key values**: All accesses to past_key_values check for `None` entries from STU layers

### Checkpoint Format

The scripts work with **unsharded checkpoints** which contain:
- `model.pt` - Model weights
- `config.yaml` - Model configuration
- `optim.pt` - Optimizer state (not used for inference)
- `train.pt` - Training state (not used for inference)

### Output Quality

Initial checkpoints (step0) will produce random/gibberish output since the model hasn't been trained yet. Use later checkpoints (e.g., `latest-unsharded`) for better quality generations after training.

## Troubleshooting

### ModuleNotFoundError: No module named 'torch'

Make sure you've activated the virtual environment:
```bash
source OLMo-base/bin/activate
```

### CUDA out of memory

Try using CPU instead:
```bash
--device cpu
```

Or reduce batch size by generating prompts one at a time.

### Model loading errors

Ensure you're pointing to a checkpoint subdirectory (e.g., `step0-unsharded`), not the parent directory.

## Code Fixes Applied

The following fixes were applied to support STU model inference:

### 1. `olmo/model.py`
- Fixed cache handling for STU layers (lines 1462-1468)
- Fixed `flatten_past_key_values` to skip None entries (lines 1714-1720)
- Fixed `unflatten_past_key_values` to handle missing keys (lines 1726-1735)
- Fixed `past_length` calculation to skip None entries (lines 1357-1362)
- Fixed `mask_len` calculation in attention bias (lines 1420-1426)

### 2. `inference/simple_inference.py`
- Added device placement fix for STU's phi tensor (lines 52-58)
- Fixed sampler instantiation to use proper classes (lines 105-118)
- Fixed token decoding to handle correct tensor dimensions (lines 129-132)

### 3. `inference/batch_inference.py`
- Applied same fixes as simple_inference.py

These fixes ensure that STU models (which have layers that don't use attention and return None for cache) work correctly with the beam search generation code.

