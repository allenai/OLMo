#!/bin/bash

NCCL_LIB_DIR=/var/lib/tcpxo/lib64 source /var/lib/tcpxo/lib64/nccl-env-profile.sh

set -euxo pipefail

HOST_VARS=$(sed 's/ \{1,\}/ -x /g' <<<"${!NCCL*} LD_LIBRARY_PATH")
FIRST_HOST=$(( echo "$1" && echo "$2" ) | sort | head -1)

mpirun \
  --mca btl self,tcp \
  --mca btl_tcp_if_include enp0s12 \
  --mca orte_base_help_aggregate 0 \
  -H $1,$2 \
  -np 2 \
  --bind-to none \
  -npernode 1 \
  -tag-output \
  -x ${HOST_VARS} \
  -x NCCL_NET=FasTrak \
  -x GLOO_SOCKET_IFNAME=enp0s12 \
  -x CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -x OMP_NUM_THREADS=16 \
  bash -c "source ~/venv/OLMo/bin/activate && torchrun --nproc_per_node 8 --nnodes=2 --rdzv-backend=c10d --rdzv-endpoint=$FIRST_HOST ~/OLMo/scripts/augusta/all_reduce_bench.py"
