efficiency-pentathlon run \
	--task wikitext-prompt \
	--scenario single_stream \
	--limit 100 \
       -- python general_purpose_efficiency_benchmark.py --pretrained-model-dir facebook/opt-125m
#       --quantized-model-dir opt-125m-4bit
