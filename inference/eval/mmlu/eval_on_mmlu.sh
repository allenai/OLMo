MODEL_DIR=$1
TOKENIZER=$2
DATA_DIR=$3
SAVE_DIR=$4

python -m eval.mmlu.run_eval \
    --ntrain 0 \
    --data_dir $DATA_DIR \
    --save_dir $SAVE_DIR \
    --model_name_or_path $MODEL_DIR \
    --tokenizer_name_or_path $TOKENIZER \
    --eval_batch_size 8 \
    --gptq
