#!/usr/bin/env bash
# Takes a sharded checkpoint and turns it into an unsharded checkpoint.
#
# This should be launched with slurm or 'torchrun --no-python ...' in the same way that the training
# script is, with the same number of processes.
#
# This script takes one argument: the path to a sharded checkpoint. The unsharded
# checkpoint will be saved alongside the sharded checkpoint.

checkpoint_dir=$1
save_folder=$(dirname $checkpoint_dir)

python scripts/train.py "${checkpoint_dir}/config.yaml" \
    --wandb=null \
    --compile=null \
    --save_folder=${save_folder} \
    --dry_run \
    --force_save_unsharded
