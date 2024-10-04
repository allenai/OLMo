#!/usr/bin/env bash

# Converts s3 checkpoints into WEKA
# To be run at the top of the root of OLMo repository.
# Script requires the use of GANTRY and AWS access to WEKA
#
# Usage: scripts/convert_checkpoints.sh <s3 checkpoint to process> [-s]
#  -s if converted checkpoint is found in s3, then save to weka
#  -c sanity check; don't actually do conversion. just go through the motions and print stuff
#
# calls: convert_checkpoints_batch.py
# usage: convert_checkpoints_batch.py [-h]
#                                    (--checkpoint-path CHECKPOINT_PATH | --checkpoint-path-file CHECKPOINT_PATH_FILE)
#                                    [--weka-load-dir WEKA_LOAD_DIR]
#                                    [--weka-prefix WEKA_PREFIX]
#                                    [--sanity-check] [--save-to-weka]
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
# - saves log.jsonl. For every checkpoint found given input:
#   - "unprocessed_path" :=  checkpoint path to convert
#   - "converted_path" := checkpoint converted path
#   - "conversion_status" := [new | existing (already in weka) | existing-downloaded (from s3) ]
#   - "date" := datestamp
#   - "error" := error if any conversions didn't pan out for any reason
# - saves model_checkpoints.jsonl: this is input file is formatted for oe-eval-internal experiments
# - example log files for the following run:
#   > sh scripts/convert_checkpoints.sh s3://ai2-llm/checkpoints/OLMo-ladder/baseline-300M-1xC/step91*6-unsharded
#   log.jsonl:
#     {"unprocessed_path": "s3://ai2-llm/checkpoints/OLMo-ladder/baseline-300M-1xC/step9176-unsharded", "converted_path": "weka://oe-eval-default/ianm/ai2-llm/checkpoints/OLMo-ladder/baseline-300M-1xC/step9176-unsharded-hf", "conversion": "existing", "date_time": "Oct-04-2024_2012", "error": ""}
#     {"unprocessed_path": "s3://ai2-llm/checkpoints/OLMo-ladder/baseline-300M-1xC/step9166-unsharded", "converted_path": "weka://oe-eval-default/ianm/ai2-llm/checkpoints/OLMo-ladder/baseline-300M-1xC/step9166-unsharded-hf", "conversion": "existing", "date_time": "Oct-04-2024_2012", "error": ""}
#     {"unprocessed_path": "s3://ai2-llm/checkpoints/OLMo-ladder/baseline-300M-1xC/step9186-unsharded", "converted_path": "weka://oe-eval-default/ianm/ai2-llm/checkpoints/OLMo-ladder/baseline-300M-1xC/step9186-unsharded-hf", "conversion": "existing", "date_time": "Oct-04-2024_2012", "error": ""}
#   model_checkpoints.jsonl:
#     {"model_name": "baseline-300M-1xC", "checkpoints_location": "weka://oe-eval-default/ianm/ai2-llm/checkpoints/OLMo-ladder/baseline-300M-1xC", "revisions": ["step9176-unsharded-hf", "step9166-unsharded-hf", "step9186-unsharded-hf"]}
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
SAVE_TO_WEKA=""
SANITY_CHECK=""
shift

usage() {
  echo "Usage: $0 <s3 checkpoint to process> [-s]"
  echo "  -s --save-to-weka"
  echo "  -c --sanity-check"
  exit 1;
}

while getopts "sc" opt;
do
  case $opt in
    s) SAVE_TO_WEKA="--save-to-weka" ;;
    c) SANITY_CHECK="--sanity-check" ;;  # mostly useful for local test runs - it will stop from doing any copying or conversions.
    *) usage ;;
  esac
done

#echo  "python scripts/convert_checkpoints_batch.py --checkpoint-path $CHECKPOINT_PATH --weka-load-dir '/data/input' --weka-prefix 'weka://oe-eval-default' $SAVE_TO_WEKA $SANITY_CHECK"

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
    -- /bin/bash -c "python scripts/convert_checkpoints_batch.py --checkpoint-path $CHECKPOINT_PATH --weka-load-dir '/data/input' --weka-prefix 'weka://oe-eval-default' $SAVE_TO_WEKA $SANITY_CHECK"

