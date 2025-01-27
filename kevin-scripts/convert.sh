#!/usr/bin/env bash

# Function to check if S3 path exists
check_s3_path() {
  aws s3 ls "$1" > /dev/null 2>&1
  return $?
}
# might need to export path here
base_dir="s3://ai2-kevinfarhat/checkpoints"

#for m in peteish7-anneal-B3x50 peteish7-init ; do \
#for m in peteish7-anneal-dclmx1_booksx50 peteish7-anneal-dclmx1_mathx30 peteish7-anneal-dclmx1_codex20 ; do \

for m in peteish7-lr-0.00041414-anneal-from-477000-100B-lb-v1p0 ; do \

	CHECKPOINT="${base_dir}/$m/step23852"

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



# #!/usr/bin/env bash
# export PYTHONPATH=/weka/oe-training-default/kevinf/OLMo:$PYTHONPATH

# # Function to check if S3 path exists
# check_s3_path() {
#   aws s3 ls "$1" > /dev/null 2>&1
#   return $?
# }

# base_dir="s3://ai2-kevinfarhat/checkpoints"

# #for m in peteish7-anneal-B3x50 peteish7-init ; do \
# #for m in peteish7-anneal-dclmx1_booksx50 peteish7-anneal-dclmx1_mathx30 peteish7-anneal-dclmx1_codex20 ; do \

# for m in peteish7-lr-0.00041414-anneal-from-477000-50B-dclm_baseline ; do \

# 	CHECKPOINT="${base_dir}/$m/step11931"

# 	# Check if the provided path exists
# 	if check_s3_path "$CHECKPOINT"; then
# 	  echo "Path exists: $CHECKPOINT"
# 	else
# 	    # Raise error and exit if neither path exists
# 	    echo "Error: Path does not exist: $CHECKPOINT"
# 	    exit 1
# 	fi

# 	python -m hf_olmo.convert_olmo_to_hf\
# 	    --checkpoint-dir $CHECKPOINT \
# 	    --destination-dir ${CHECKPOINT}-hf \
# 	    --keep-olmo-artifact \
# 	    --tokenizer allenai/dolma2-tokenizer

# done

# #cd ../oe-eval-internal
# #bash sewon-scripts/run.sh






