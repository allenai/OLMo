#!/usr/bin/env bash

set -exuo pipefail
IFS=$'\n\t'

# Setup Python environment.
conda shell.bash activate base

# Install flash-attn
pip install awscli
pip install '.[train]'
pip freeze

# Move AWS credentials from env to relevant files
mkdir -p ~/.aws
printenv AWS_CONFIG > ~/.aws/config
printenv AWS_CREDENTIALS > ~/.aws/credentials

checkpoint=$1
shift

echo "Unsharding..."
python scripts/unshard.py --safe-tensors "$@" "/weka/oe-training-default/${checkpoint}" \
    "/weka/oe-training-default/${checkpoint}-unsharded"

echo "Uploading to S3..."
aws s3 cp --no-progress --recursive --profile=S3 \
    "/weka/oe-training-default/${checkpoint}-unsharded" \
    "s3://${checkpoint}-unsharded"

echo "Done!"
