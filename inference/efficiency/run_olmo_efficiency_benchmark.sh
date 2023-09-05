PRETRAINED_MODEL='/net/nfs.cirrascale/allennlp/akshitab/olmo-models/olmo-1b'
QUANTIZED_MODEL='/home/pranjalib/LLM/inference/compression/quantized_olmo-1b'

efficiency-pentathlon run \
	--task wikitext-prompt \
	--scenario single_stream \
	--limit 10 \
  -- python olmo_efficiency_benchmark.py \
  --pretrained-model-dir $PRETRAINED_MODEL \
  --quantized-model-dir $QUANTIZED_MODEL
