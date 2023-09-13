PRETRAINED_MODEL=$1
QUANTIZED_MODEL_DIR=$2

python run_quantization.py \
    --pretrained-model $PRETRAINED_MODEL \
    --quantized-model-dir $QUANTIZED_MODEL_DIR \
    --n-samples 128



