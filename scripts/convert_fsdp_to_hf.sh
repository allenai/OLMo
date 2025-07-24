#!/bin/bash

# Usage: ./convert_fsdp_to_hf.sh <input_checkpoint_dir>

set -e

if [ $# -ne 1 ]; then
    echo "Usage: $0 <input_checkpoint_dir>"
    echo "Example: $0 ../step1000002-c"
    exit 1
fi

INPUT_DIR="$1"
# OUTPUT_DIR="$2"
# MAX_SEQ_LEN="$3"

echo "Converting FSDP checkpoint to HuggingFace format..."
echo "Input: $INPUT_DIR"
# echo "Output: $OUTPUT_DIR"
# echo "Max sequence length: $MAX_SEQ_LEN"

if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory $INPUT_DIR does not exist"
    exit 1
fi

if [ ! -d "$INPUT_DIR/model_and_optim" ]; then
    echo "Error: $INPUT_DIR/model_and_optim directory not found"
    exit 1
fi

# if [ ! -f "$INPUT_DIR/config.json" ]; then
#     echo "Error: $INPUT_DIR/config.json not found"
#     exit 1
# fi

if ls "$INPUT_DIR/model_and_optim"/*.distcp 1> /dev/null 2>&1; then
    echo "Step 1: Backing up original FSDP checkpoint..."
    if [ ! -d "$INPUT_DIR/model_and_optim_fsdp_backup" ]; then
        cp -r "$INPUT_DIR/model_and_optim" "$INPUT_DIR/model_and_optim_fsdp_backup"
        echo "Original FSDP checkpoint backed up to model_and_optim_fsdp_backup"
    else
        echo "Backup already exists, skipping..."
    fi

    echo "Step 2: Unsharding FSDP checkpoint..."
    python -c "
from olmo_core.distributed.checkpoint import unshard_checkpoint
import os

output_dir = '$INPUT_DIR/model_and_optim_unsharded'
os.makedirs(output_dir, exist_ok=True)

try:
    model_path, optim_path = unshard_checkpoint(
        '$INPUT_DIR/model_and_optim',
        output_dir,
        optim=False  # Skip optimizer state
    )
    print(f'Successfully unsharded checkpoint to: {model_path}')
except Exception as e:
    print(f'Error unsharding checkpoint: {e}')
    exit(1)
"

    echo "Step 3: Replacing FSDP checkpoint with unsharded version..."
    rm -rf "$INPUT_DIR/model_and_optim"
    mv "$INPUT_DIR/model_and_optim_unsharded" "$INPUT_DIR/model_and_optim"
    echo "Replaced with unsharded checkpoint"
else
    echo "No .distcp files found, assuming checkpoint is already unsharded"
fi

# echo "Step 4: Converting to HuggingFace format..."
# mkdir -p "$OUTPUT_DIR"

# DEVICE_ARG=""
# if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
#     echo "CUDA detected, using GPU"
#     DEVICE_ARG="--device cuda"
# else
#     echo "CUDA not available, using CPU"
#     DEVICE_ARG="--device cpu"
# fi

# python scripts/convert_checkpoint_to_hf.py \
#     -i "$INPUT_DIR" \
#     -o "$OUTPUT_DIR" \
#     -s "$MAX_SEQ_LEN" \
#     $DEVICE_ARG \
#     --skip-validation

echo "Unsharding completed successfully!"
# echo "HuggingFace model saved to: $OUTPUT_DIR"
# echo ""
# echo "You can now use the model with:"
# echo "  from transformers import AutoModel, AutoTokenizer"
# echo "  model = AutoModel.from_pretrained('$OUTPUT_DIR')"
# echo ""
# echo "To restore the original FSDP checkpoint:"
# echo "  rm -rf $INPUT_DIR/model_and_optim"
# echo "  mv $INPUT_DIR/model_and_optim_fsdp_backup $INPUT_DIR/model_and_optim"