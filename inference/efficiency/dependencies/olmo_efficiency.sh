
PRETRAINED_MODEL=$1

efficiency-benchmark run \
	--task wikitext-prompt \
      	--scenario single_stream \
	--limit 10 \
	-- python olmo_efficiency.py --pretrained-model $PRETRAINED_MODEL

