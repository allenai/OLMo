#!/usr/bin/env bash
set -ex
NUM_NODES=$1
shift
gantry run \
  --workspace ai2/olmo2-1b-backfill-1k \
  --task-name peteish1 \
  --description "Pete-ish 1B backfill every 1k steps to 20k" \
  --priority normal \
  --preemptible \
  --beaker-image michalg/cuda11.8-ubuntu20.04-arb \
  --cluster ai2/augusta-google-1 \
  --gpus 8 \
  --replicas "${NUM_NODES}" \
  --leader-selection \
  --host-networking \
  --budget ai2/oe-training \
  --no-nfs \
  --propagate-failure \
  --propagate-preemption \
  --synchronized-start-timeout 15m \
  --no-python \
  --env LOG_FILTER_TYPE=local_rank0_only \
  --env OMP_NUM_THREADS=8 \
  --env OLMO_TASK=model \
  --env-secret WANDB_API_KEY=BAILEYK_WANDB_API_KEY \
  --env-secret AWS_ACCESS_KEY_ID=BAILEYK_AWS_ACCESS_KEY_ID \
  --env-secret AWS_SECRET_ACCESS_KEY=BAILEYK_AWS_SECRET_ACCESS_KEY \
  --shared-memory 10GiB \
  --yes \
  --timeout=-1 \
  --allow-dirty \
  --retries 10 \
  -- /bin/bash -c "scripts/augusta/peteish1-backfill.sh \$BEAKER_LEADER_REPLICA_HOSTNAME \$BEAKER_REPLICA_RANK"
