PRETRAINED_MODEL='/net/nfs.cirrascale/allennlp/akshitab/olmo-models/olmo-1b'
QUANTIZED_MODEL_DIR='quantized_olmo-1b'

python run_quantization.py \
    --pretrained-model $PRETRAINED_MODEL \
    --quantized-model-dir $QUANTIZED_MODEL_DIR \
    --n-samples 128