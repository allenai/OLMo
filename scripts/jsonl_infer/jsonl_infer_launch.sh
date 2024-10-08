#!/usr/bin/env bash


# Args are: 
# 1: input dir
# 2: output dir 
# 3: num parts 
# 4: part 
# 5: model
set -ex

NUM_NODES=1
gantry run \
  --workspace ai2/mattj \
  --task-name mattj_toyexp \
  --description "mattj's custom batch inference" \
  --priority normal \
  --preemptible \
  --beaker-image petew/olmo-torch23-gantry \
  --cluster ai2/saturn-cirrascale \
  --gpus 1 \
  --replicas "${NUM_NODES}" \
  --leader-selection \
  --host-networking \
  --budget ai2/oe-data \
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
  --env-secret AWS_CONFIG=MATTJ_AWS_CONFIG \
  --env-secret AWS_ACCESS_KEY_ID=MATTJ_HALODEV_AWS_ID \
  --env-secret AWS_SECRET_ACCESS_KEY=MATTJ_HALODEV_AWS_KEY \
  --env-secret R2_ENDPOINT_URL=R2_ENDPOINT_URL \
  --env-secret WEKA_ENDPOINT_URL=WEKA_ENDPOINT_URL \
  --env-secret WANDB_API_KEY=MATTJ_WANDB_API_KEY \
  --env-secret HF_TOKEN=MATTJ_HF_TOKEN \
  --shared-memory 10GiB \
  --yes \
  --timeout=-1 \
  -- /bin/bash -c "scripts/jsonl_infer/jsonl_infer_setup.sh \$BEAKER_LEADER_REPLICA_HOSTNAME ${NUM_NODES} \$BEAKER_REPLICA_RANK \$1 \$2 \$3 \$4"