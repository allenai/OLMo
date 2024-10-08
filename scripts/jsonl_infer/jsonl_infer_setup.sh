#!/usr/bin/env bash

set -exuo pipefail
IFS=$'\n\t'

BEAKER_LEADER_REPLICA_HOSTNAME=$1
shift

NUM_NODES=$1
shift

BEAKER_REPLICA_RANK=$1
shift

INPUT_DIR=$1
shift

OUTPUT_DIR=$1
shift

NUM_PARTS=$1
shift

PART=$1
shift


# Setup Python environment.
conda shell.bash activate base

# Install flash-attn
#conda install -y -c nvidia cuda-python
pip install packaging ninja
export FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE
pip install flash-attn==2.5.9.post1 --no-build-isolation
# pip install awscli
pip install --upgrade transformers
pip freeze

# Install AWS CLI 
apt-get install unzip 
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
./aws/install

# Log in to hugging face 
pip install huggingface-hub 
huggingface-cli login --token $HF_TOKEN

# Force processes to synchronize at init_process_group
export TORCH_DIST_INIT_BARRIER=1

# Tell OLMo all ranks share the same filesystem for checkpoints.
export OLMO_SHARED_FS=1

# export NCCL_IB_GID_INDEX=0


python scripts/jsonl_infer/infer_text.py \
--input-dir $INPUT_DIR \
--output-dir $OUTPUT_DIR \
--part $PART \
--num-parts $NUM_PARTS 