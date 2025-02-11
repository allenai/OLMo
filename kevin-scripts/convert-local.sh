#!/usr/bin/env bash

# Function to check if local path exists
check_local_path() {
  [ -d "$1" ] && return 0 || return 1
}

# might need to export path here
CHECKPOINT="/weka/oe-training-default/kevinf/checkpoints/private-olmo/peteish7-medlr-anneal-from-477000-10B-mjdata/latest"

# Check if the provided path exists
if check_local_path "$CHECKPOINT"; then
    echo "Path exists: $CHECKPOINT"
else
    # Raise error and exit if neither path exists
    echo "Error: Path does not exist: $CHECKPOINT"
    exit 1
fi

python hf_olmo/convert_olmo_to_hf.py \
    --checkpoint-dir $CHECKPOINT \
    --destination-dir ${CHECKPOINT}-hf \
    --keep-olmo-artifact \
    --tokenizer allenai/dolma2-tokenizer