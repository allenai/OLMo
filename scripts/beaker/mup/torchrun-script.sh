#!/usr/bin/env bash
set -exuo pipefail
IFS=$'\n\t'

BEAKER_LEADER_REPLICA_HOSTNAME=$1
shift

NUM_NODES=$1
shift

BEAKER_REPLICA_RANK=$1
shift

WIDTH=$1
shift

LR=$1
shift

# Setup Python environment.
conda shell.bash activate base

# Install flash-attn
#conda install -y -c nvidia cuda-python
pip install packaging ninja
export FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE
pip install flash-attn==2.5.9.post1 --no-build-isolation
# pip install awscli
pip install '.[train]'
pip install '.[scaling]'

pip freeze

export HF_DATASETS_OFFLINE=1

# Move AWS credentials from env to relevant files
mkdir -p ~/.aws
printenv AWS_CONFIG > ~/.aws/config
printenv AWS_CREDENTIALS > ~/.aws/credentials

# Force processes to synchronize at init_process_group
export TORCH_DIST_INIT_BARRIER=1

# Tell OLMo all ranks share the same filesystem for checkpoints.
export OLMO_SHARED_FS=1

export NCCL_DEBUG=INFO
export NCCL_IB_HCA="^=mlx5_bond_0"
export NCCL_SOCKET_IFNAME=ib
# export NCCL_IB_GID_INDEX=0

torchrun \
  --nnodes ${NUM_NODES}:${NUM_NODES} \
  --nproc-per-node 8 \
  --rdzv_id=101 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=$BEAKER_LEADER_REPLICA_HOSTNAME:29400 \
  --node_rank=$BEAKER_REPLICA_RANK \
  scripts/train.py \
    configs/mup/base-olmo-cosine.yaml \
      --run_name="mup_lr_search_${WIDTH}_${LR}" \
      --wandb.name="mup_lr_search_${WIDTH}_${LR}" \
      --wandb.group="mup_lr_search" \
      --wandb.project=olmo-mup \
      --stop_at=1000 \
      --model.use_mup \
      --model.mup_query_zero_init=false \
      --model.mup_base_shapes=scripts/beaker/mup/lr_search_base_shapes_300m.bsh \
      --model.d_model=$WIDTH \
      --optimizer.learning_rate=$LR \
      --scheduler.t_warmup=50 \
      --distributed_strategy=fsdp \
      --device_train_microbatch_size=4 \
      --try_load_latest_save \
      --save_overwrite

#for WIDTH in 128 256 512 1024 2048 4096;
#do
#  torchrun \
#    --nnodes ${NUM_NODES}:${NUM_NODES} \
#    --nproc-per-node 8 \
#    --rdzv_id=101 \
#    --rdzv_backend=c10d \
#    --rdzv_endpoint=$BEAKER_LEADER_REPLICA_HOSTNAME:29400 \
#    --node_rank=$BEAKER_REPLICA_RANK \
#    scripts/train.py \
#      configs/mup/base-olmo.yaml \
#        --run_name="sp_olmo_$WIDTH" \
#        --wandb.name="sp_olmo_$WIDTH" \
#        --wandb.group="sp_olmo" \
#        --wandb.project=olmo-mup \
#        --save_overwrite \
#        --stop_at=1000 \
#        --model.d_model=$WIDTH
#done
