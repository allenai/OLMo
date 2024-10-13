#!/usr/bin/env bash

# Function to check if S3 path exists
check_s3_path() {
  aws s3 ls "$1" > /dev/null 2>&1
  return $?
}

base_dir="s3://ai2-lucas-archival/checkpoints"

#for m in peteish7-anneal-B3x50 peteish7-init ; do \
#for m in peteish7-anneal-dclmx1_booksx50 peteish7-anneal-dclmx1_mathx30 peteish7-anneal-dclmx1_codex20 ; do \

for m in peteish7-anneal-from-1T-fin peteish7-anneal-from-1T-tutorial ; do \

	CHECKPOINT="${base_dir}/$m/latest"

	# Check if the provided path exists
	if check_s3_path "$CHECKPOINT"; then
	  echo "Path exists: $CHECKPOINT"
	else
	    # Raise error and exit if neither path exists
	    echo "Error: Path does not exist: $CHECKPOINT"
	    exit 1
	fi

	python hf_olmo/convert_olmo_to_hf.py \
	    --checkpoint-dir $CHECKPOINT \
	    --destination-dir ${CHECKPOINT}-hf \
	    --keep-olmo-artifact \
	    --tokenizer allenai/dolma2-tokenizer

done

#cd ../oe-eval-internal
#bash sewon-scripts/run.sh



