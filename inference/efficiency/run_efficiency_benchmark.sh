
PRETRAINED_MODEL=$1
QUANTIZED_MODEL=$2

efficiency-benchmark run \
	--task wikitext-prompt \
      	--scenario single_stream \
	--limit 100 \
	-- python run_efficiency_benchmark.py --pretrained_model_dir $PRETRAINED_MODEL --quantized_model_dir $QUANTIZED_MODEL
