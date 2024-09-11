#!/usr/bin/env bash

# To be run at the top of the root of OLMo repository.
#  Converts s3 checkpoints into WEKA
#
# Example use:
# sh scripts/convert_checkpoints.sh s3://ai2-llm/checkpoints/cheap_decisions/dolma-v1-6-and-sources-baseline-3x-code-1B-N-1T-D-mitchish1-001/step9*
#
# This will convert all models in the directory
# and save them to their respective directories under
#
# /weka/ai2-llm/checkpoints/cheap_decisions/dolma-v1-6-and-sources-baseline-3x-code-1B-N-1T-D-mitchish1-001/step9*
#
# It will first, though, check that the weka directory doesn't exist AND that s3 doesn't have a corresponding directory (so as not to replicate what conversions already made)

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
# - Output is not saving. But it prints to the log. Fix this.
# - Make tokenizer updatable

CHECKPOINT_PATH=$1
DESCRIPTION=$2


gantry run \
    --description $DESCRIPTION \
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
    -- /bin/bash -c "python scripts/convert_checkpoints_batch.py --checkpoint-path $CHECKPOINT_PATH --weka-load-dir '/data/input'"

