set -Euo pipefail

# a script that takes a path that has a bunch of unsharded checkpoints and then converts any that haven't been converted to hf
# usage: ./convert_all_checkpoints_and_upload.sh /path/to/checkpoints

path=$1

pip install accelerate

checkpoints_needing_conversion=($(python scripts/find_all_checkpoints_needing_conversion.py --path $path))
echo "Converting the following checkpoints: $checkpoints_needing_conversion"

len=${#checkpoints_needing_conversion[@]}
progress=0
for checkpoint in "${checkpoints_needing_conversion[@]}"; do
    echo "Converting $progress / $len $checkpoint"
    cp -r "$path$checkpoint" "$path$checkpoint-hf"
    python hf_olmo/convert_olmo_to_hf.py --ignore-olmo-compatibility --checkpoint-dir "$path$checkpoint"
    progress=$((progress+1))
done