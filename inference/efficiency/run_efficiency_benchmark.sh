
#PRETRAINED_MODEL= 'facebook/opt-125m'
#QUANTIZED_MODEL='opt-125m-4bit'

efficiency-pentathlon run \
	--task wikitext-prompt \
	--scenario single_stream \
	--limit 1 \
       -- python efficiency_report.py --pretrained-model-dir 'facebook/opt-125m'
#       --quantized-model-dir 'None'
