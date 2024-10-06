#!/bin/bash

set -euxo pipefail

NUM_NODES=$1
shift

source ~/venv/OLMo/bin/activate

cd ~/OLMo
NCCL_LIB_DIR=/var/lib/tcpxo/lib64 source /var/lib/tcpxo/lib64/nccl-env-profile.sh
export NCCL_NET=FasTrak
HOST_NODE_ADDR=augusta-vms-0001:12345
torchrun --nproc_per_node=8 --nnodes=$NUM_NODES --rdzv-backend=c10d --rdzv-endpoint=$HOST_NODE_ADDR scripts/train.py "$@"
