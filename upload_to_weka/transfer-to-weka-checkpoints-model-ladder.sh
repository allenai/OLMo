export BUDGET=ai2/oe-eval
export S3_BUCKET=ai2-llm
export S3_PREFIX=checkpoints/OLMo-ladder/baseline-1B-2xC/
export WEKA_BUCKET=oe-eval-default
export WEKA_PREFIX=ianm/ai2-llm/checkpoints/OLMo-ladder/baseline-1B-2xC/
export SYNC_INCLUDE="*unsharded/model.pt"
export SYNC_INCLUDE_2="*unsharded/config.yaml"