#!/usr/bin/env bash

set -ex

gantry run \
  --workspace ai2/ianm_secrets \
  --task-name sync-and-prep-checkpoints \
  --description "Sync all checpoints from s3 path and as needed unshard and hf_olmo convert" \
  --priority normal \
  --preemptible \
  --beaker-image petew/olmo-torch23-gantry \
  --venv base \
  --cluster ai2/jupiter-cirrascale-2 \
  --gpus 0 \
  --budget ai2/oe-eval \
  --no-nfs \
  --weka oe-eval-default:/weka/oe-eval-default \
  --env-secret AWS_ACCESS_KEY_ID=aws-key \
  --env-secret AWS_SECRET_ACCESS_KEY=aws-secret \
  --shared-memory 10GiB \
  --yes \
  --timeout=-1 \
  -- /bin/bash -c "scripts/beaker/sync-and-prep-checkpoints.sh $*"
