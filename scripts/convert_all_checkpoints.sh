set -Euo pipefail

# a script that takes a path that has a bunch of unsharded checkpoints and then converts any that haven't been converted to hf
# usage: ./convert_all_checkpoints_and_upload.sh /path/to/checkpoints

path=$1

pip install accelerate

checkpoints_needing_conversion=($(python scripts/find_all_checkpoints_needing_conversion.py --path $path))
echo "Converting the following checkpoints: $checkpoints_needing_conversion"

len=${#checkpoints_needing_conversion[@]}
progress=0
for checkpoint in $checkpoints_needing_conversion; do
    echo "Converting $progress / $len $checkpoint"
    python scripts/convert_olmo_to_hf_new.py --input_dir $path${checkpoint} --output_dir $path${checkpoint}-hf --tokenizer_json_path tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json
    progress=$((progress+1))
done