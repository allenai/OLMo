set -Euo pipefail

# a script that takes a path that has a bunch of unsharded checkpoints and then converts any that haven't been converted to hf
# usage: ./convert_all_checkpoints_and_upload.sh /path/to/checkpoints

path=$1

checkpoints_needing_conversion=$(python scripts/find_all_checkpoints_needing_conversion.py --path $path)
echo "Converting the following checkpoints: $checkpoints_needing_conversion"

len=${#checkpoints_needing_conversion[@]}
progress=0
for checkpoint in $checkpoints_needing_conversion; do
    echo "Converting $progress / $len $checkpoint"
    scripts/convert_olmo_to_hf_new.py --input_dir ${checkpoint} --output_dir ${checkpoint}-hf --tokenizer_json_path tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json
    progress=$((progress+1))
    break
done