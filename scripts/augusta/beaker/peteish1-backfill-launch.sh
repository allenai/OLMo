#!/usr/bin/env bash

# This script has been modified from the original peteish1-launch.sh for the purpose of resuming training from step0 through 10k, saving every 1k.
# It currently lives in /augusta/beaker but should eventually be moved to scripts/beaker/peteish1.


set -ex
NUM_NODES=$1
shift
gantry run \
  --workspace ai2/OLMo-pretraining-stability \
  --task-name peteish1 \
  --description "Pete-ish 1B backfill every 1k steps up to 20k" \
  --priority normal \
  --preemptible \
  --beaker-image michalg/cuda11.8-ubuntu20.04-arb \
  --cluster ai2/augusta-google-1 \
  --gpus 8 \
  --replicas "${NUM_NODES}" \
  --leader-selection \
  --host-networking \
  --budget ai2/oe-training \
  --propagate-failure \
  --propagate-preemption \
  --synchronized-start-timeout 15m \
  --no-python \
  --env LOG_FILTER_TYPE=local_rank0_only \
  --env OMP_NUM_THREADS=8 \
  --env OLMO_TASK=model \
  --env R2_PROFILE=R2 \
  --env S3_PROFILE=S3 \
  --env WEKA_PROFILE=WEKA \
  --env-secret AWS_CONFIG=BAILEYK_AWS_CONFIG \
  --env-secret AWS_CREDENTIALS=BAILEYK_AWS_CREDENTIALS \
  --env-secret R2_ENDPOINT_URL=R2_ENDPOINT_URL \
  --env-secret WEKA_ENDPOINT_URL=WEKA_ENDPOINT_URL \
  --env-secret WANDB_API_KEY=BAILEYK_WANDB_API_KEY \
  --shared-memory 10GiB \
  --yes \
  --timeout=-1 \
  --allow-dirty \
  --retries 10 \
  -- /bin/bash -c "scripts/augusta/beaker/peteish1-backfill.sh \$BEAKER_LEADER_REPLICA_HOSTNAME \$BEAKER_REPLICA_RANK"
