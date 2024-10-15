#!/bin/bash

NCCL_LIB_DIR=/var/lib/tcpxo/lib64 source /var/lib/tcpxo/lib64/nccl-env-profile.sh

set -euxo pipefail

HOSTS=$1
shift

HOSTS=$(echo "$HOSTS" | tr ',' '\n' | awk '{print $0":8"}' | paste -sd,)
HOST_VARS=$(sed 's/ \{1,\}/ -x /g' <<<"${!NCCL*} LD_LIBRARY_PATH")

mpirun \
  --mca btl self,tcp \
  --mca btl_tcp_if_include enp0s12 \
  --bind-to none \
  -H $HOSTS \
  -npernode 8 \
  -x ${HOST_VARS} \
  -x WANDB_ENTITY \
  -x WANDB_API_KEY \
  -x S3_ACCESS_KEY_ID \
  -x S3_SECRET_ACCESS_KEY \
  -x AWS_ACCESS_KEY_ID \
  -x AWS_SECRET_ACCESS_KEY \
  -x NCCL_DEBUG=WARN \
  bash ~/OLMo/scripts/augusta/run_with_environment_mpi.sh "$@"
