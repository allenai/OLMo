#!/usr/bin/env bash

$CHECKPOINT="s3://lucas-archival/checkpoints/peteish1-baseline/step19000"

command="pip install awscli && pip install -e '.[all]' && python hf_olmo/convert_olmo_to_hf.py \
--checkpoint-dir '$CHECKPOINT' \
--destination-dir '${CHECKPOINT}-hf' \
--keep-olmo-artifact"

gantry run \
    --description "Converting ${CHECKPOINT}" \
    --no-python \
    --workspace ai2/oe-training \
    --priority normal \
    --gpus 0 \
    --preemptible \
    --beaker-image petew/olmo-torch23-gantry \
    --cluster 'ai2/*' \
    --budget ai2/oe-training \
    --env-secret AWS_CONFIG=SEWONM_AWS_CONFIG \
    --env-secret AWS_CREDENTIALS=SEWONM_AWS_CREDENTIALS \
    --shared-memory 10GiB \
    --yes \
    -- /bin/bash -c "${command}"