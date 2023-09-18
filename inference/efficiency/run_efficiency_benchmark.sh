PRETRAINED_MODEL=$1
QUANTIZED_MODEL=$2

efficiency-pentathlon run \
	--task wikitext-prompt \
	--scenario single_stream \
	--limit 100 \
  -- python general_purpose_efficiency_benchmark.py \
       --pretrained-model-dir $PRETRAINED_MODEL \
       --quantized-model-dir $QUANTIZED_MODEL
