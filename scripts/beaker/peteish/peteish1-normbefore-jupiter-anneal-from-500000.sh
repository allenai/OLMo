#!/usr/bin/env bash

set -ex

NUM_NODES=8

gantry run \
  --allow-dirty \
  --workspace ai2/OLMo-tiny \
  --task-name peteish1-normbefore-jupiter-anneal-from-500000 \
  --description "Pete-ish 1B, norm before, run on jupiter, anneal from 500000" \
  --priority high \
  --beaker-image petew/olmo-torch23-gantry \
  --cluster ai2/jupiter-cirrascale-2 \
  --gpus 8 \
  --replicas "${NUM_NODES}" \
  --leader-selection \
  --host-networking \
  --budget ai2/oe-training \
  --no-nfs \
  --weka oe-training-default:/weka/oe-training-default \
  --propagate-failure \
  --propagate-preemption \
  --synchronized-start-timeout 90m \
  --no-python \
  --env LOG_FILTER_TYPE=local_rank0_only \
  --env OMP_NUM_THREADS=8 \
  --env OLMO_TASK=model \
  --env R2_PROFILE=R2 \
  --env S3_PROFILE=S3 \
  --env WEKA_PROFILE=WEKA \
  --env-secret AWS_CONFIG=PETEW_AWS_CONFIG \
  --env-secret AWS_CREDENTIALS=PETEW_AWS_CREDENTIALS \
  --env-secret R2_ENDPOINT_URL=R2_ENDPOINT_URL \
  --env-secret WEKA_ENDPOINT_URL=WEKA_ENDPOINT_URL \
  --env-secret WANDB_API_KEY=JIACHENGL_WANDB_API_KEY \
  --shared-memory 10GiB \
  --yes \
  --timeout=0 \
  -- /bin/bash -c "\
    set -exuo pipefail; \
    IFS=$'\n\t'; \
    conda shell.bash activate base; \
    pip install packaging ninja; export FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE; pip install flash-attn==2.5.9.post1 --no-build-isolation; pip install '.[train]'; pip freeze; \
    mkdir -p ~/.aws; printenv AWS_CONFIG > ~/.aws/config; printenv AWS_CREDENTIALS > ~/.aws/credentials; \
    export TORCH_DIST_INIT_BARRIER=1; \
    export OLMO_SHARED_FS=1; \
    export NCCL_DEBUG=INFO; \
    export NCCL_IB_HCA="^=mlx5_bond_0"; \
    export NCCL_SOCKET_IFNAME=ib; \
    torchrun \
        --nnodes "${NUM_NODES}:${NUM_NODES}" \
        --nproc-per-node 8 \
        --rdzv_id 12347 \
        --rdzv_backend static \
        --rdzv_endpoint "\${BEAKER_LEADER_REPLICA_HOSTNAME}:29400" \
        --node_rank "\${BEAKER_REPLICA_RANK}" \
        --rdzv_conf 'read_timeout=420' \
        scripts/train.py \
            configs/annealing/peteish1-anneal.yaml \
            --run_name="peteish1-normbefore-jupiter-anneal-from-500000" \
            --save_interval_ephemeral=null \
            --save_overwrite \
            --load_path=/weka/oe-training-default/ai2-llm/checkpoints/OLMo-small/peteish1-normbefore-jupiter/step500000 \
            --optimizer.learning_rate=0.00036279 \
            --model.norm_after=false \
    "
