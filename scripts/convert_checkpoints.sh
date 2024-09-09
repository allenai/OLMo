##!/usr/bin/env bash
#
## RUN AT THE TOP OF THE OLMo root
#
#CHECKPOINT_PATH=$1
#shift
#
#SUFFIX="hf"
#WORKSPACE=""
#BUDGET=""
#PRIORITY="normal"
#
#while getopts "s:w:b:p:t:" opt; do
#    case $opt in
#        s)
#            SUFFIX="$OPTARG"
#            ;;
#        w)
#            WORKSPACE="$OPTARG"
#            ;;
#        b)
#            BUDGET="$OPTARG"
#            ;;
#        p)
#            PRIORITY="$OPTARG"
#            ;;
#        t)
#            CUSTOM_TOKENIZER="--tokenizer $OPTARG"
#            ;;
#        \?)
#            echo "Invalid option: -$OPTARG" >&2
#            exit 1
#            ;;
#    esac
#done
#
## Set default values if not specified
#if [ -z "$WORKSPACE" ]; then
#    WORKSPACE="ai2/oe-data"
#fi
#
#if [ -z "$BUDGET" ]; then
#    BUDGET="$WORKSPACE"
#fi
#
## Verify that a path has been provided
#if [ -z "$CHECKPOINT_PATH" ]; then
#  echo "Error: No path provided."
#  exit 1
#fi
#
## Check if CHECKPOINT_PATH is an s3:// path or an absolute path
#if [[ ! "$CHECKPOINT_PATH" =~ ^s3:// ]] && [[ ! "$CHECKPOINT_PATH" =~ ^/ ]]; then
#    echo "Error: CHECKPOINT_PATH must be an s3:// path or an absolute path."
#    exit 1
#fi
#
#
## Extract weka_mountpoint if checkpoint path starts with specific directories
#CLUSTERS="ai2/*"
#for dir in climate-default mosaic-default nora-default oe-adapt-default oe-data-default oe-eval-default oe-training-default prior-default reviz-default skylight-default; do
#    if [[ $CHECKPOINT_PATH == "/$dir"* ]]; then
#        WEKA_MOUNTPOINTS=" --weka=${dir}:/${dir}"
#        # Override clusters to use only jupiter-cirrascale-2
#        CLUSTERS="ai2/jupiter-cirrascale-2"
#        break
#    fi
#done
#
#
## Function to check if S3 path exists
#check_s3_path() {
#  aws s3 ls "$1" > /dev/null 2>&1
#  return $?
#}
#
## Check if the provided path exists (only for S3 paths)
#if [[ "$CHECKPOINT_PATH" =~ ^s3:// ]]; then
#    if check_s3_path "$CHECKPOINT_PATH"; then
#        echo "S3 path exists: $CHECKPOINT_PATH"
#    else
#        echo "Error: S3 path does not exist: $CHECKPOINT_PATH"
#        exit 1
#    fi
#else
#    echo "Skipping existence check for non-S3 path: $CHECKPOINT_PATH"
#fi
#
#commands=(
#    "pip install awscli"
#    "git clone https://github.com/allenai/OLMo.git"
#    "cd OLMo"
#    "pip install -e '.[all]'"
#    "if [ ! -d '${CHECKPOINT_PATH}-${SUFFIX}' ]; then python hf_olmo/convert_olmo_to_hf.py --checkpoint-dir '$CHECKPOINT_PATH'  --destination-dir '${CHECKPOINT_PATH}-${SUFFIX}' --keep-olmo-artifacts ${CUSTOM_TOKENIZER}; else echo 'Destination directory already exists. Skipping conversion.'; fi"
#)
#
#
#for cmd in "${commands[@]}"; do
#  if [ -z "$joined_commands" ]; then
#    joined_commands="$cmd"
#  else
#    joined_commands="$joined_commands && $cmd"
#  fi
#done
#
#gantry run \
#    --description "Converting ${CHECKPOINT_PATH}" \
#    --allow-dirty \
#    --no-python \
#    --workspace ${WORKSPACE} \
#    --priority ${PRIORITY} \
#    --gpus 0 \
#    --preemptible \
#    --cluster ${CLUSTERS} \
#    --budget ${BUDGET} \
#    --env-secret AWS_ACCESS_KEY_ID=S2_AWS_ACCESS_KEY_ID \
#    --env-secret AWS_SECRET_ACCESS_KEY=S2_AWS_SECRET_ACCESS_KEY \
#    --shared-memory 10GiB \
#    ${WEKA_MOUNTPOINTS} \
#    --yes \
#    -- /bin/bash -c "${joined_commands}"

gantry run \
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
    --pip requirements.txt \
    --yes \
    -- /bin/bash -c "python convert_checkpoints_batch.pyq --checkpoint-dir 's3://ai2-llm/checkpoints/cheap_decisions/dolma-v1-6-and-sources-baseline-3x-code-1B-N-1T-D-mitchish1-001/step351000-unsharded' --weka-load-dir '/data/input/'"

#    --install install_torch.sh \
