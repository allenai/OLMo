#!/usr/bin/env bash

set -ex

CONFIG_PATH=configs/mitchish7-s3.yaml
NUM_NODES=8
ARGS='--run_name=mitchish7 --wandb.name=mitchish7 --model.flash_attention=true --fsdp.wrapping_strategy=by_block_and_size --fsdp.sharding_strategy=SHARD_GRAD_OP --save_folder=runs/ --device_train_microbatch_size=2 --global_train_batch_size=1024 --save_overwrite'

gantry run \
  --workspace ai2/dirkg \
  --task-name mitchish7 \
  --description "OLMo medium - 7B" \
  --priority high \
  --stop-preemptible \
  --beaker-image petew/olmo-torch2-gantry \
  --cluster ai2/pluto-cirrascale \
  --gpus 8 \
  --replicas "${NUM_NODES}" \
  --leader-selection \
  --host-networking \
  --budget ai2/oe-training \
  --nfs \
  --mount /net/nfs.cirrascale/allennlp/petew/cache:/root/.cache \
  --env LOG_FILTER_TYPE=local_rank0_only \
  --env OMP_NUM_THREADS=8 \
  --env OLMO_TASK=model \
  --env-secret WANDB_API_KEY=WANDB_API_KEY \
  --env-secret AWS_ACCESS_KEY_ID=AWS_ACCESS_KEY_ID \
  --env-secret AWS_SECRET_ACCESS_KEY=AWS_SECRET_ACCESS_KEY \
  --shared-memory 10GiB \
  --venv base \
  --yes \
  --timeout=-1 \
  -- /bin/bash -c "source scripts/beaker/warm_hf_cache.sh && sysctl net.ipv6.conf.all.disable_ipv6=1 && torchrun --nnodes ${NUM_NODES}:${NUM_NODES} --nproc-per-node 8 --rdzv_id=101 --rdzv_backend=c10d --rdzv_endpoint=\$BEAKER_LEADER_REPLICA_HOSTNAME:29400 scripts/train.py ${CONFIG_PATH} ${ARGS}"
