#!/usr/bin/env bash

set -exuo pipefail
IFS=$'\n\t'

BEAKER_LEADER_REPLICA_HOSTNAME=$1
shift

BEAKER_REPLICA_RANK=$1
shift

ORIGINAL_WANDB_RUN_ID=$1
shift

START_STEP=$1
shift

LENGTH=$1
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

# Install flash-attn
#conda install -y pytorch-cuda==12.4 packaging ninja cccl cuda-nvcc libcusolver-dev cuda-profiler-api libcusparse-dev libcublas-dev -c pytorch -c nvidia
#pip install flash-attn==2.5.9.post1 --no-build-isolation
pip install '.[train]'
pip freeze

# Force processes to synchronize at init_process_group
export TORCH_DIST_INIT_BARRIER=1
# Better error handling from Python
export PYTHONFAULTHANDLER=1

NAME=${GANTRY_TASK_NAME// /_}
RUN_NAME=$NAME-$(date -u +"%Y%m%d_%H%M%S")
STEPS_TO_TRAIN=$(python -c "print(round($LENGTH / (1024 * 4096)) + $START_STEP)")
SAVE_FOLDER=/data/$RUN_NAME
mkdir -p $SAVE_FOLDER

torchrun \
  --nnodes "${BEAKER_REPLICA_COUNT}:${BEAKER_REPLICA_COUNT}" \
  --nproc-per-node 8 \
  --rdzv_id 12348 \
  --rdzv_backend static \
  --rdzv_endpoint "${BEAKER_LEADER_REPLICA_HOSTNAME}:29400" \
  --node_rank "${BEAKER_REPLICA_RANK}" \
  --rdzv_conf 'read_timeout=420' \
  scripts/train.py \
    configs/peteish7-google.yaml \
      --run_name=$RUN_NAME \
      --wandb.group=$NAME \
      --save_interval_ephemeral=1000 \
      --eval_interval=1000 \
      --fsdp.sharding_strategy=HYBRID_SHARD \
      --fsdp.hybrid_sharding_num_model_replicas="${BEAKER_REPLICA_COUNT}" \
      --save_folder=$SAVE_FOLDER \
      --remote_save_folder="gs://ai2-llm/checkpoints/OLMo-medium/$NAME/" \
      --save_overwrite \
      --load_path=gs://ai2-llm/checkpoints/OLMo-medium/$(python ./scripts/group_name_from_wandb.py $ORIGINAL_WANDB_RUN_ID)/step$START_STEP \
      '--load_path=${path.last_checkpoint:${remote_save_folder}}' \
      --sharded_checkpointer=olmo_core \
      --device_train_microbatch_size=2 \
      --activation_checkpointing=one_in_four \
      --compile.fullgraph=false \
      --fused_loss=false \
      --model.flash_attention=false \
      --data.num_workers=8 \
      --optimizer.learning_rate=$(python ./scripts/learning_rate_at_step_from_wandb.py $ORIGINAL_WANDB_RUN_ID $START_STEP) \
      --scheduler.units=steps \
      --scheduler.name=linear_with_warmup \
      --scheduler.t_warmup=$START_STEP \
      --scheduler.t_max=null \
      --scheduler.alpha_f=0 \
      --max_duration=$STEPS_TO_TRAIN \
      --stop_at=$(($STEPS_TO_TRAIN + 10)) \
      --no_pre_train_checkpoint \
      --optimizer.metrics_log_interval=10 \
      --data.prefetch_factor=8
