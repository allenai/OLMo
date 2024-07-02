#!/usr/bin/env bash
set -exuo pipefail
IFS=$'\n\t'

BEAKER_LEADER_REPLICA_HOSTNAME=$1
shift

NUM_NODES=$1
shift

BEAKER_REPLICA_RANK=$1
shift

# Warm HF cache
mkdir -p /root/.cache
pushd /root/.cache
curl "https://storage.googleapis.com/hf-cache/huggingface_cache_v4.tar.gz" | tar --keep-newer-files -xzf -
popd
export HF_DATASETS_OFFLINE=1

# Move AWS credentials from env to relevant files
mkdir -p ~/.aws
printenv AWS_CONFIG > ~/.aws/config
printenv AWS_CREDENTIALS > ~/.aws/credentials

# Temporary, since it is not part of the image yet.
pip install mup@git+https://github.com/microsoft/mup#egg=19814971934ef91dd546f88e913fc963e096d11c

for WIDTH in 128 256 512 1024 2048 4096;
do
  torchrun \
    --nnodes ${NUM_NODES}:${NUM_NODES} \
    --nproc-per-node 8 \
    --rdzv_id=101 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$BEAKER_LEADER_REPLICA_HOSTNAME:29400 \
    --node_rank=$BEAKER_REPLICA_RANK \
    scripts/train.py \
      configs/mup/base-olmo.yaml \
        --run_name="new_mup_olmo_$WIDTH" \
        --wandb.name="new_mup_olmo_$WIDTH" \
        --wandb.group="new_mup_olmo" \
        --wandb.project=olmo-mup \
        --stop_at=1000 \
        --model.use_mup \
	      --model.mup_base_shapes=scripts/beaker/mup/base_olmo_shapes.bsh \
        --model.d_model=$WIDTH
done

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
