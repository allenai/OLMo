#!/usr/bin/env bash

set -ex

NUM_NODES=1
NUM_GPUS=8
BACKFILL_SUFFIX=backfill_last

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
    --workspace ai2/OLMo-tiny \
    --task-name ladder-eval-${checkpoint} \
    --description "Ladder eval backfill for ${checkpoint}" \
    --priority high \
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
    --env OMP_NUM_THREADS=${NUM_GPUS} \
    --env OLMO_TASK=model \
    --env-secret WANDB_API_KEY=WANDB_API_KEY \
    --env BACKFILL_SUFFIX=${BACKFILL_SUFFIX} \
    --env CHECKPOINT=/weka/oe-training-default/ai2-llm/checkpoints/OLMo-ladder/${checkpoint} \
    --env NUM_CHECKPOINTS=1 \
    --shared-memory 10GiB \
    --yes \
    --timeout=-1 \
    -- /bin/bash -c "scripts/beaker/peteish/ladder-eval.sh \$BEAKER_LEADER_REPLICA_HOSTNAME \$BEAKER_REPLICA_RANK ${NUM_NODES} ${NUM_GPUS}" &
done

checkpoint=peteish7
gantry run \
    --allow-dirty \
    --name ladder-eval-${checkpoint} \
    --workspace ai2/OLMo-tiny \
    --task-name ladder-eval-${checkpoint} \
    --description "Ladder eval backfill for ${checkpoint}" \
    --priority high \
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
    --env OMP_NUM_THREADS=${NUM_GPUS} \
    --env OLMO_TASK=model \
    --env-secret WANDB_API_KEY=WANDB_API_KEY \
    --env BACKFILL_SUFFIX=${BACKFILL_SUFFIX} \
    --env CHECKPOINT=/weka/oe-training-default/ai2-llm/checkpoints/OLMo-medium/${checkpoint} \
    --env CONFIG=configs/peteish7-weka.yaml \
    --env NUM_CHECKPOINTS=1 \
    --shared-memory 10GiB \
    --yes \
    --timeout=-1 \
    -- /bin/bash -c "scripts/beaker/peteish/ladder-eval.sh \$BEAKER_LEADER_REPLICA_HOSTNAME \$BEAKER_REPLICA_RANK ${NUM_NODES} ${NUM_GPUS}" &
