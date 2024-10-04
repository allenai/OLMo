#!/usr/bin/env bash

# Converts s3 checkpoints into WEKA
# To be run at the top of the root of OLMo repository.
# Script requires the use of GANTRY and AWS access to WEKA
#
# Example use:
# Run:
# sh scripts/convert_checkpoints.sh s3://ai2-llm/checkpoints/cheap_decisions/dolma-v1-6-and-sources-baseline-3x-code-1B-N-1T-D-mitchish1-001/step9*
# This will convert all models in the directory and save them to:
# weka://oe-eval-default/ai2-llm/checkpoints/cheap_decisions/dolma-v1-6-and-sources-baseline-3x-code-1B-N-1T-D-mitchish1-001-hf/step9*
#
# It will first, though, check that the weka directory doesn't exist AND that s3 doesn't have a corresponding directory (so as not to replicate what conversions already made)
#
# ASSUMPTIONS
# - INPUT must be on s3. Multiple wildcards allowed
# - OUTPUT to weka is saved to the path as found on s3 with "-hf" suffix appended to the path
# - Assumes tokenizer allenai/gpt-neox-olmo-dolma-v1_5.json
#
# OUTPUT logs
# - saves log.jsonl
# - saves model_checkpoints.jsonl: this is input file is formatted for oe-eval-internal experiments
#
# SH run SPECIFICATION DEFAULTS:
# - Budget for oe-eval (see below)
# - Loading for weka weka://oe-eval-default/ (see below)
# - Gantry experiments saved to beaker://ai2/cheap-decisions
# - Weka prefix is used for model_checkpoints.jsonl
#
# TODOs
# - Make tokenizer updatable

CHECKPOINT_PATH=$1

gantry run \
    --description "Converting $CHECKPOINT_PATH" \
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
    -- /bin/bash -c "python scripts/convert_checkpoints_batch.py --checkpoint-path $CHECKPOINT_PATH --weka-load-dir '/data/input' --weka-prefix 'weka://oe-eval-default' --save-to-weka"

