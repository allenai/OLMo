#!/bin/bash
set -euo pipefail

export NODENAME=$(hostname -s)
export MASTER_ADDR=$1
export MASTER_PORT=39591
export WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
export RANK=$OMPI_COMM_WORLD_RANK
export LOCAL_WORLD_SIZE=$OMPI_COMM_WORLD_LOCAL_SIZE
export LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK
export NODE_RANK=$((($RANK - $LOCAL_RANK) / $LOCAL_WORLD_SIZE))

# Redirect stdout and stderr so that we get a prefix with the node name
exec > >(trap "" INT TERM; sed -u "s/^/$NODENAME:$LOCAL_RANK out: /")
exec 2> >(trap "" INT TERM; sed -u "s/^/$NODENAME:$LOCAL_RANK err: /" >&2)

if [ $LOCAL_RANK -eq 0 ] ; then
  nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill
else
  sleep 2
fi

exec python scripts/train.py "$@"
