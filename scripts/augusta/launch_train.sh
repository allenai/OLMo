#!/bin/bash

# This runs only on the first host.

NCCL_LIB_DIR=/var/lib/tcpxo/lib64 source /var/lib/tcpxo/lib64/nccl-env-profile.sh

set -euxo pipefail

NUM_NODES=$1
FIRST_HOST=$(head -1 ~/hostfile | cut -f 1 -d" ")

HOST_VARS=$(sed 's/ \{1,\}/ -x /g' <<<"${!NCCL*} LD_LIBRARY_PATH")
mpirun \
  --mca btl self,tcp \
  --mca btl_tcp_if_include enp0s12 \
  --hostfile ~/hostfile \
  -np $NUM_NODES \
  -npernode 1 \
  -x ${HOST_VARS} \
  -x WANDB_ENTITY \
  -x WANDB_API_KEY \
  -x S3_ACCESS_KEY_ID \
  -x S3_SECRET_ACCESS_KEY \
  -x AWS_ACCESS_KEY_ID \
  -x AWS_SECRET_ACCESS_KEY \
  bash ~/OLMo/scripts/augusta/run_with_environment.sh $FIRST_HOST "$@"
