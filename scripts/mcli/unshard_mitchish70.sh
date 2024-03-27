#!/bin/bash

set -euo pipefail

remote_sharded_checkpoint=$(python -c "from olmo.util import find_latest_checkpoint; print(find_latest_checkpoint('s3://ai2-llm/checkpoints/OLMo-large/mitchish70-002'))")
local_folder=~/checkpoints

mkdir -p ${local_folder}

local_sharded_checkpoint="${local_folder}/$(basename ${remote_sharded_checkpoint})"
remote_unsharded_checkpoint="${remote_sharded_checkpoint}-unsharded"
local_unsharded_checkpoint="${local_sharded_checkpoint}-unsharded"

echo "Downloading '${remote_sharded_checkpoint}' to '${local_sharded_checkpoint}'..."
aws s3 cp --recursive ${remote_sharded_checkpoint} ${local_sharded_checkpoint}

echo "Unsharding '${local_sharded_checkpoint}' to '${local_unsharded_checkpoint}'..."
python scripts/unshard.py ${local_sharded_checkpoint} ${local_unsharded_checkpoint} --safe-tensors --type=local

echo "Uploading '${local_unsharded_checkpoint}' to '${remote_unsharded_checkpoint}'..."
aws s3 cp --recursive ${local_unsharded_checkpoint} ${remote_unsharded_checkpoint}
