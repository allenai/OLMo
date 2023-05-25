#!/bin/bash

set -eo pipefail

local_data_path=/net/nfs/allennlp/matthewp/llm-data
#local_data_path=/home/matthewp/llm-data
s3_bucket=s3://ai2-llm
relative_raw_data_path=pretraining-data/sources/c4/raw/en/train/
tokenizer=EleutherAI/gpt-neox-20b

local_full_raw_data_path=${local_data_path}/${relative_raw_data_path}

# Copy files from S3.
mkdir -p $local_full_data_path
aws s3 sync $s3_bucket/$relative_raw_data_path ${local_full_raw_data_path}/ \
  --exclude "*" \
  --include "*.json.gz" 


# Preprocess, creating numpy memmap files of token IDs.
for (( i = 0 ; i < 10 ; i++ ));
do
    # python -u == unbuffered stdout/stderr
    cmd="python -u scripts/prepare_memmap_dataset.py --validate --tokenizer ${tokenizer} -o ${local_full_raw_data_path}/c4-train.00${i}00-00${i}99.npy ${local_full_raw_data_path}/c4-train.00${i}*.json.gz 2>&1 > log_${i}.txt &"
    cmd_with_conda="source ~/.bashrc; conda activate LLM; ${cmd}"
    # the -i is necessary to run ~/.bashrc, which initializes conda
    #/bin/bash -i -c "$cmd_with_conda 2>&1 | tee log_${i}.txt"
    /bin/bash -c "$cmd_with_conda"
done

conda activate LLM
python -u scripts/prepare_memmap_dataset.py --validate --tokenizer ${tokenizer} -o ${local_full_raw_data_path}/c4-train.01000-01023.npy ${local_full_raw_data_path}/c4-train.01*.json.gz 2>&1 > log_10.txt


python -u scripts/prepare_memmap_dataset.py --validate --tokenizer ${tokenizer} -o ${local_full_raw_data_path}/tt.npy ${local_full_raw_data_path}/c4-train.00000-of-01024.json.gz ${local_full_raw_data_path}/c4-train.00001-of-01024.json.gz ${local_full_raw_data_path}/c4-train.00002-of-01024.json.gz

## Wait here for tokenizing to finish

# upload to S3
aws s3 sync ${output_dir}/ s3://ai2-llm/eval-data/perplexity/v2_small_gptneox20b/ \
  --exclude "*" \
  --include "*/${eval_set}.npy"

