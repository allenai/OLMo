#!/bin/bash

set -eo pipefail

data_path=eval-data/perplexity/v2_small
output_dir=/net/nfs/allennlp/llm-data/${data_path}
eval_set=val
tokenizer=gpt2

# Copy files from S3.
mkdir -p ${output_dir}
aws s3 sync s3://ai2-llm/${data_path}/ ${output_dir}/ \
  --exclude "*" \
  --include "*/${eval_set}/*.jsonl.gz"

# Show what was downloaded from S3.
for dir in ${output_dir}/*/${eval_set}; do
  echo "Downloaded $(du -h -d 0 ${dir} | cut -d$'\t' -f1) to ${dir}"
done
echo "$(du -h -d 0 ${output_dir} | cut -d$'\t' -f1) total"

# Preprocess, creating numpy memmap files of token IDs.
for dir in ${output_dir}/*/${eval_set}; do
  output_file=${dir}.npy
  echo "Preparing ${dir}..."
  python scripts/prepare_memmap_dataset.py --tokenizer ${tokenizer} -o ${output_file} ${dir}/*.jsonl.gz
  echo -e "Wrote $(du -h ${output_file} | cut -d$'\t' -f1) to ${output_file}\n"
done
