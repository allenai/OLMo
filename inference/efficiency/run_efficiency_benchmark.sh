PRETRAINED_MODEL=$1
QUANTIZED_MODEL=$2

efficiency-pentathlon run \
	--task wikitext-prompt \
	--scenario single_stream \
	--limit 10 \
  -- python run_efficiency_benchmark.py \
       --pretrained-model-dir $PRETRAINED_MODEL \
       --quantized-model-dir $QUANTIZED_MODEL #\
       #--vllm
