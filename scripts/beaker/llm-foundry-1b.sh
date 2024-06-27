#!/usr/bin/env bash
set -exuo pipefail
IFS=$'\n\t'

BEAKER_LEADER_REPLICA_HOSTNAME=$1
shift

NUM_NODES=$1
shift

BEAKER_REPLICA_RANK=$1
shift

# Move AWS credentials from env to relevant files
mkdir -p ~/.aws
printenv AWS_CONFIG > ~/.aws/config
printenv AWS_CREDENTIALS > ~/.aws/credentials

export EXPERIMENT=llm-foundry-1b

export NCCL_SOCKET_IFNAME=ib
export NCCL_IB_HCA="^=mlx5_bond_0"
export NCCL_DEBUG=INFO


composer \
  --world_size $(($NUM_NODES * 8)) \
  --node_rank $BEAKER_REPLICA_RANK \
  --master_addr $BEAKER_LEADER_REPLICA_HOSTNAME \
  --master_port 29400 \
  train/train.py \
    train/yamls/pretrain/mpt-1b.yaml \
    variables.data_local=/weka/oe-training-default/shanea/mds_data/dolma \
    train_loader.dataset.split=train \
    eval_loader=null \
    eval_interval=0 \
    train_loader.dataset.shuffle=False \
    save_folder=/weka/oe-training-default/shanea/checkpoints/mpt-1b
