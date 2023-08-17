
#PRETRAINED_MODEL= 'facebook/opt-125m'
#QUANTIZED_MODEL='/home/pranjalib/LLM/inference/quantized_opt125m'

efficiency-pentathlon run \
	--task wikitext-prompt \
	--scenario single_stream \
	--limit 1 \
       -- python efficiency_report.py --pretrained-model-dir 'facebook/opt-125m'
