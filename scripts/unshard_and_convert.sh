MODEL_NAME="v2.7_v2.5_vera_no-infgram"
CKPT="step11234"
CKPT_NAME="${MODEL_NAME}_${CKPT}"

aws s3 --profile weka-aus cp --recursive s3://oe-training-default/wolf/ckpt/${MODEL_NAME}/${CKPT} ./ckpt_sharded/${CKPT_NAME}
python scripts/unshard.py ./ckpt_sharded/${CKPT_NAME} ./ckpt_unsharded/${CKPT_NAME} --model-only
python hf_olmo/convert_olmo_to_hf.py --checkpoint-dir ./ckpt_unsharded/${CKPT_NAME}
python scripts/convert_olmo_to_hf_new.py --input_dir ./ckpt_unsharded/${CKPT_NAME} --output_dir ./ckpt_transformers/${CKPT_NAME} --tokenizer_json_path olmo_data/tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json
