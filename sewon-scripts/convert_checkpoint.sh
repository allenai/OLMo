#!/usr/bin/env bash

# Function to check if S3 path exists
check_s3_path() {
  aws s3 ls "$1" > /dev/null 2>&1
  return $?
}

#for m in dense dense_wo_paywall dense_wo_unk ; do \
for m in dense_wo_paywall_wo_tos ; do \
	
	CHECKPOINT="s3://ai2-llm/ds-olmo/checkpoints/baselines/$m"
	
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
	    --keep-olmo-artifact

done

cd ../oe-eval-internal

#for m in dense dense_wo_paywall dense_wo_unk ; do \
for m in dense_wo_paywall_wo_tos ; do \

	CHECKPOINT="s3://ai2-llm/ds-olmo/checkpoints/baselines/$m-hf"

	bash sewon-scripts/eval_checkpoint.sh $CHECKPOINT

done


