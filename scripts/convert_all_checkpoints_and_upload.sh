set -Euo pipefail

# a script that takes a s3 path that has a bunch of unsharded checkpoints and then converts them one at a time and uploads them again

# usage: ./convert_all_checkpoints_and_upload.sh s3://path/to/checkpoints

# get the s3 path
s3_path=$1

# get the list of unsharded checkpoints, eg step38000-unsharded
checkpoints=$(aws s3 ls $s3_path | grep unsharded  | awk '{print $2}')

# echo checkpoints: $checkpoints

# for checkpoint in $checkpoints; do
#     echo checkpoint: $checkpoint
#     echo " ${checkpoint}-hf "
# done

# get the list of already converted checkpoints, eg step38000-unsharded-hf
converted_checkpoints=$(aws s3 ls $s3_path | grep hf  | awk '{print $2}')

echo here
# # make a list of just the checkpoints that have no hf version
# checkpoints_to_convert=()
# for checkpoint in $checkpoints; do
#   echo " ${converted_checkpoints[@]} "
#   echo " ${checkpoint}-hf "
# #   if [[ ! " ${converted_checkpoints[@]} " =~ " ${checkpoint}-hf " ]]; then
# #     echo $checkpoint has not been converted
# #     checkpoints_to_convert+=($checkpoint)
# #   fi
# done

# echo checkpoints to convert: ${checkpoints_to_convert[@]}