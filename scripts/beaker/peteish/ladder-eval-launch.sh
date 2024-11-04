#!/usr/bin/env bash

set -ex

NUM_NODES=1
NUM_GPUS=8
SUFFIX=backfill2

# checkpoints=(
#     peteish-150M-1xC
#     peteish-190M-10xC
#     peteish-190M-1xC
#     peteish-190M-2xC
#     peteish-190M-5xC
#     peteish-1B-1xC
#     peteish-1B-2xC
#     peteish-1B-5xC
#     peteish-370M-10xC
#     peteish-370M-1xC
#     peteish-370M-5xC
#     peteish-600M-10xC
#     peteish-600M-1xC
#     peteish-600M-2xC
#     peteish-600M-5xC
#     peteish-760M-10xC
#     peteish-760M-1xC
#     peteish-760M-2xC
#     peteish-760M-5xC
# )

# checkpoints=(
#     peteish-const-190M-10xC
#     peteish-const-1B-10xC
#     peteish-const-370M-10xC
#     peteish-const-600M-10xC
#     peteish-const-760M-10xC
# )

checkpoints=(
    peteish-final-190M-10xC
    peteish-final-190M-1xC
    peteish-final-190M-2xC
    peteish-final-190M-5xC
    peteish-final-1B-10xC
    peteish-final-1B-1xC
    peteish-final-1B-2xC
    peteish-final-1B-5xC
    peteish-final-370M-10xC
    peteish-final-370M-1xC
    peteish-final-370M-2xC
    peteish-final-370M-5xC
    peteish-final-600M-10xC
    peteish-final-600M-1xC
    peteish-final-600M-2xC
    peteish-final-600M-5xC
    peteish-final-760M-10xC
    peteish-final-760M-1xC
    peteish-final-760M-2xC
    peteish-final-760M-5xC
)

for checkpoint in ${checkpoints[@]}; do
    gantry run \
    --allow-dirty \
    --name ladder-eval-${checkpoint} \
    --workspace ai2/alexw \
    --task-name ladder-eval-${checkpoint} \
    --description "Ladder eval backfill for ${checkpoint}" \
    --priority normal \
    --preemptible \
    --beaker-image petew/olmo-torch23-gantry \
    --cluster ai2/jupiter-cirrascale-2 \
    --gpus "${NUM_GPUS}" \
    --replicas "${NUM_NODES}" \
    --leader-selection \
    --host-networking \
    --propagate-failure \
    --propagate-preemption \
    --budget ai2/oe-training \
    --no-nfs \
    --weka oe-training-default:/weka/oe-training-default \
    --no-python \
    --env LOG_FILTER_TYPE=local_rank0_only \
    --env OMP_NUM_THREADS=8 \
    --env OLMO_TASK=model \
    --env-secret WANDB_API_KEY=WANDB_API_KEY \
    --shared-memory 10GiB \
    --yes \
    --timeout=-1 \
    -- /bin/bash -c "scripts/beaker/peteish/ladder-eval.sh \$BEAKER_LEADER_REPLICA_HOSTNAME ${NUM_NODES} ${NUM_GPUS} \$BEAKER_REPLICA_RANK ${checkpoint} ${SUFFIX}" &

done