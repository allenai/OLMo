#!/bin/bash

# This runs only on the first host.

NCCL_LIB_DIR=/var/lib/tcpxo/lib64 source /var/lib/tcpxo/lib64/nccl-env-profile.sh

set -euxo pipefail

HOST_VARS=$(sed 's/ \{1,\}/ -x /g' <<<"${!NCCL*} LD_LIBRARY_PATH")
mpirun \
  --mca btl self,tcp \
  --mca btl_tcp_if_include enp0s12 \
  --hostfile ~/hostfile \
  -N 2 \
  -npernode 1 \
  -x ${HOST_VARS} \
  -x WANDB_ENTITY \
  -x S3_ACCESS_KEY_ID \
  -x S3_SECRET_ACCESS_KEY \
  -x AWS_ACCESS_KEY_ID \
  -x AWS_SECRET_ACCESS_KEY \
  bash ~/OLMo/scripts/augusta/run_with_environment.sh "$@"
