#!/usr/bin/env bash

# To be run at the top of the root of OLMo repository.
#  Converts s3 checkpoints into WEKA

# ASSUMPTIONS
# - INPUT must be on s3
# - OUTPUT is weka with the same path name as s3 + "-hf" suffix appended to the path
# - Budget for oe-eval
# - Experiments saved to ai2/cheap-decisions
# - Assumes tokenizer allenai/gpt-neox-olmo-dolma-v1_5.json

# NOTES
# - saves metrics.json
# - allows for wildcard (*)

# TODOs
# - Make consistent with Luca's code
# - Code allows for a txt file with a list of checkpoint paths, sh needs to allow this
#

CHECKPOINT_PATH=$1


gantry run \
    --description "checkpoint conv; eval for cons ranking" \
    --allow-dirty \
    --workspace ai2/cheap-decisions  \
    --priority normal \
    --gpus 0 \
    --preemptible \
    --cluster ai2/jupiter-cirrascale-2 \
    --budget ai2/oe-eval \
    --env-secret AWS_ACCESS_KEY_ID=JENA_AWS_ACCESS_KEY_ID \
    --env-secret AWS_SECRET_ACCESS_KEY=JENA_AWS_SECRET_ACCESS_KEY \
    --shared-memory 10GiB \
    --weka=oe-eval-default:/data/input \
    --yes \
    -- /bin/bash -c "python scripts/convert_checkpoints_batch.py --checkpoint-path $CHECKPOINT_PATH --weka-load-dir '/data/input/'"

