#!/usr/bin/env bash

"""
This script has been modified from the original peteish1.sh for the purpose of resuming training from step0 through 20k, saving every 1k.
  - Loading in checkpoint at step 0 (https://olmo-checkpoints.org/ai2-llm/peteish1/step0-unsharded)
  - Training will stop at 10k, then will resume either with the backfill 10k checkpoint or the original, through 20k
      --stop_at 10000
  - Using 1 node instead of 8 (hardcoded --nnodes 1)
  - Subbing in peteish1-weka.yaml (augusta not available)
  - Saving directly as unsharded
      --save_interval_ephemeral=null
      --save_interval_unsharded=1000
      --sharding_strategy=FULL_SHARD
      --sharded_checkpointer=torch
  - Checkpoints are named with "-backfill" suffix to avoid overwriting
      - remove --save_overwrite, for now
"""

set -exuo pipefail
IFS=$'\n\t'
BEAKER_LEADER_REPLICA_HOSTNAME=$1
shift
BEAKER_REPLICA_RANK=$1
shift

# augusta specific environment
export LD_LIBRARY_PATH="/var/lib/tcpxo/lib64:${LD_LIBRARY_PATH}"
export NCCL_CROSS_NIC=0
export NCCL_ALGO=Ring,Tree
export NCCL_PROTO=Simple
export NCCL_MIN_NCHANNELS=4
export NCCL_P2P_NET_CHUNKSIZE=524288
export NCCL_P2P_PCI_CHUNKSIZE=524288
export NCCL_P2P_NVL_CHUNKSIZE=1048576
export NCCL_FASTRAK_NUM_FLOWS=2
export NCCL_FASTRAK_ENABLE_CONTROL_CHANNEL=0
export NCCL_BUFFSIZE=8388608
export NCCL_FASTRAK_USE_SNAP=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_NET_GDR_LEVEL=PIX
export NCCL_FASTRAK_ENABLE_HOTPATH_LOGGING=0
export NCCL_TUNER_PLUGIN=libnccl-tuner.so
export NCCL_TUNER_CONFIG_PATH=/var/lib/tcpxo/lib64/a3plus_tuner_config.textproto
export NCCL_SHIMNET_GUEST_CONFIG_CHECKER_CONFIG_FILE=/var/lib/tcpxo/lib64/a3plus_guest_config.textproto
export NCCL_FASTRAK_PLUGIN_ACCEPT_TIMEOUT_MS=600000
export NCCL_NVLS_ENABLE=0
export NCCL_DEBUG=WARN
export NCCL_FASTRAK_CTRL_DEV=enp0s12
export NCCL_FASTRAK_IFNAME=enp6s0,enp7s0,enp13s0,enp14s0,enp134s0,enp135s0,enp141s0,enp142s0
export NCCL_SOCKET_IFNAME=enp0s12
export NCCL_USE_SNAP=1
export NCCL_FASTRAK_USE_LLCM=1
export NCCL_FASTRAK_LLCM_DEVICE_DIRECTORY=/dev/aperture_devices

pip install '.[train]'
pip freeze

export TORCH_DIST_INIT_BARRIER=1
export PYTHONFAULTHANDLER=1

NAME=${GANTRY_TASK_NAME// /_}
RUN_NAME=$NAME-backfill-$(date -u +"%Y%m%d_%H%M%S")
SAVE_FOLDER=/data/$RUN_NAME
mkdir -p $SAVE_FOLDER

torchrun \
  --nnodes 1 \
  --nproc-per-node 8 \
  --rdzv_id 12348 \
  --rdzv_backend static \
  --rdzv_endpoint "${BEAKER_LEADER_REPLICA_HOSTNAME}:29400" \
  --node_rank ${BEAKER_REPLICA_RANK} \
  --rdzv_conf 'read_timeout=420' \
  scripts/train.py \
    configs/peteish1-weka.yaml \
      --run_name=$RUN_NAME \
      --wandb.group=$NAME \
      --save_interval_ephemeral=null \
      --save_interval_unsharded=1000 \
      --eval_interval=1000 \
      --fsdp.sharding_strategy=FULL_SHARD \
      --fsdp.wrapping_strategy=by_block_and_size \
      --try_load_latest_save \
      --save_folder=/weka/oe-training-default/ai2-llm/checkpoints/peteish1-backfill \
      --remote_save_folder=gs://ai2-llm/checkpoints/OLMo-medium/peteish1-backfill/ \
      --load_path=https://olmo-checkpoints.org/ai2-llm/peteish1/step0-unsharded/ \
      --sharded_checkpointer=torch \
      --device_train_microbatch_size=4 \
      --device_eval_batch_size=8 \
      --compile.fullgraph=false \
      --fused_loss=false \
      --model.flash_attention=false \
      --data.num_workers=8 \
      --optimizer.metrics_log_interval=10 \
      --data.prefetch_factor=8 \
      --stop_at=10000
