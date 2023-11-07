#!/bin/bash

# Note: This script does not run inside the container. It runs on the bare compute node.

set -euo pipefail

export NODENAME=$(hostname -s)
export MASTER_ADDR=$(scontrol show hostnames | head -n 1)
export MASTER_PORT=39591
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export FS_LOCAL_RANK=$SLURM_PROCID
export LOCAL_WORLD_SIZE=$SLURM_NTASKS_PER_NODE
export LOCAL_RANK=$SLURM_LOCALID
export NODE_RANK=$((($RANK - $LOCAL_RANK) / $LOCAL_WORLD_SIZE))

# Redirect stdout and stderr so that we get a prefix with the node name
exec > >(trap "" INT TERM; sed -u "s/^/$NODENAME:$LOCAL_RANK out: /")
exec 2> >(trap "" INT TERM; sed -u "s/^/$NODENAME:$LOCAL_RANK err: /" >&2)

if [ $SLURM_LOCALID -eq 0 ] ; then
  if command -v rocm-smi &> /dev/null ; then
    rm -rf /dev/shm/* || true
    rocm-smi || true	# rocm-smi returns exit code 2 even when it succeeds
  fi
else
  sleep 2
fi

exec $*
