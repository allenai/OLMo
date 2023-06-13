#!/bin/bash
# This script serves as the entrypoint to the Docker/singularity image built from
# 'Dockerfile.lumi_light'.

# Load PyTorch environment.
source /opt/miniconda3/bin/activate pytorch

# Additional stuff from Samuel Antao. We might want to move these lines to `run_with_environment.sh`
# but I'm not sure if the other image needs this stuff.
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3
export MIOPEN_USER_DB_PATH="/tmp/$(whoami)-miopen-cache-$SLURM_NODEID"
export MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_USER_DB_PATH
# Set MIOpen cache out of the home folder.
if [ $SLURM_LOCALID -eq 0 ] ; then
  rm -rf $MIOPEN_USER_DB_PATH
  mkdir -p $MIOPEN_USER_DB_PATH
fi
sleep 5

# Execute arguments to this script as commands themselves.
exec "$@"
