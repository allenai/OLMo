#!/bin/bash

set -eo pipefail

tokenizer=EleutherAI/gpt-neox-20b

s3_bucket=s3://ai2-llm
relative_raw_data_path=pretraining-data/sources/c4/raw/en/train/

##local_data_path=/net/nfs/allennlp/matthewp/llm-data
#local_data_path=/home/matthewp/llm-data
#local_full_raw_data_path=${local_data_path}/${relative_raw_data_path}

local_data_path=/mnt/tank/c4/en
local_full_raw_data_path=$local_data_path

# Preprocess, creating numpy memmap files of token IDs.
conda activate LLM
mkdir -p ${local_full_raw_data_path}/gpt-neox-20b/

for (( i = 1 ; i < 10 ; i++ ));
do
    # python -u == unbuffered stdout/stderr
    python -u scripts/prepare_memmap_dataset.py --validate --tokenizer ${tokenizer} -o ${local_full_raw_data_path}/gpt-neox-20b/c4-train.00${i}00-00${i}99.npy ${local_full_raw_data_path}/c4-train.00${i}*.json.gz 2>&1 > log_${i}.txt
done

python -u scripts/prepare_memmap_dataset.py --validate --tokenizer ${tokenizer} -o ${local_full_raw_data_path}/gpt-neox-20b/c4-train.01000-01023.npy ${local_full_raw_data_path}/c4-train.01*.json.gz 2>&1 > log_10.txt
