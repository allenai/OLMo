#!/bin/bash

set -eo pipefail

#local_data_path=/net/nfs/allennlp/matthewp/llm-data
local_data_path=/home/matthewp/llm-data
s3_bucket=s3://ai2-llm
relative_raw_data_path=pretraining-data/sources/c4/raw/en/train/
tokenizer=EleutherAI/gpt-neox-20b

local_full_raw_data_path=${local_data_path}/${relative_raw_data_path}

# Copy files from S3.
mkdir -p $local_full_data_path
aws s3 sync $s3_bucket/$relative_raw_data_path ${local_full_raw_data_path}/ \
  --exclude "*" \
  --include "*.json.gz" 


#if [ -d "/net/nfs/allennlp/" ]; then
#  output_dir=/net/nfs/allennlp/llm-data/${data_path}
#elif [ -v "$SCRATCH_DIR" ] & [ -d "$SCRATCH_DIR" ]; then
#  output_dir=$SCRATCH_DIR/${data_path}
#fi

#output_dir=/net/nfs/allennlp/matthewp/llm-data/${data_path}

# Preprocess, creating numpy memmap files of token IDs.
for (( i = 0 ; i < 10 ; i++ ));
do
    python scripts/prepare_memmap_dataset.py --tokenizer ${tokenizer} -o ${local_full_raw_data_path}/c4-train.00${i}00-00${i}99.npy ${local_full_raw_data_path}/c4-train.00${i}*.json.gz
done

python scripts/prepare_memmap_dataset.py --tokenizer ${tokenizer} -o ${local_full_raw_data_path}/c4-train.01000-01023.npy ${local_full_raw_data_path}/c4-train.01*.json.gz

# upload to S3
aws s3 sync ${output_dir}/ s3://ai2-llm/eval-data/perplexity/v2_small_gptneox20b/ \
  --exclude "*" \
  --include "*/${eval_set}.npy"

