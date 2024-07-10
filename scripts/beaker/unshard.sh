#!/usr/bin/env bash

set -exuo pipefail
IFS=$'\n\t'

# Setup Python environment.
conda shell.bash activate base

# Install flash-attn
# pip install awscli
pip install '.[train]'
pip freeze

# Move AWS credentials from env to relevant files
mkdir -p ~/.aws
printenv AWS_CONFIG > ~/.aws/config
printenv AWS_CREDENTIALS > ~/.aws/credentials

checkpoint=/weka/oe-training-default/ai2-llm/checkpoints/OLMo-medium/amberish7/step477850

# python scripts/unshard.py "${checkpoint}" "${checkpoint}-unsharded" --type=olmo_core
python scripts/unshard.py --help
echo "Done!"
