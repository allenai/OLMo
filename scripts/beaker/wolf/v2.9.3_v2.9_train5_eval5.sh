#!/usr/bin/env bash

set -ex

CONFIG_PATH=configs/amberish1-weka.yaml
NUM_NODES=1
RUN_NAME="v2.9.3_v2.9_train5_eval5"

gantry run \
  --allow-dirty \
  --name ${RUN_NAME} \
  --task-name ${RUN_NAME} \
  --description ${RUN_NAME} \
  --workspace ai2/hb-wolf-olmo \
  --budget ai2/oe-training \
  --beaker-image petew/olmo-torch23-gantry \
  --cluster ai2/jupiter-cirrascale-2 \
  --priority high \
  --preemptible \
  --no-nfs \
  --weka oe-training-default:/weka/oe-training-default \
  --gpus 8 \
  --shared-memory 10GiB \
  --no-python \
  --env LOG_FILTER_TYPE=local_rank0_only \
  --env OMP_NUM_THREADS=8 \
  --env OLMO_TASK=model \
  --env WANDB__SERVICE_WAIT=300 \
  --env WANDB_HTTP_TIMEOUT=60 \
  --env-secret WANDB_API_KEY=WANDB_API_KEY \
  --env-secret AWS_ACCESS_KEY_ID=AWS_ACCESS_KEY_ID \
  --env-secret AWS_SECRET_ACCESS_KEY=AWS_SECRET_ACCESS_KEY \
  --yes \
  -- /bin/bash -c "\
    set -exuo pipefail; \
    IFS=$'\n\t'; \
    conda shell.bash activate base; \
    conda install gxx -c conda-forge; \
    pip install packaging ninja; export FLASH_ATTENTION_SKIP_CUDA_BUILD=TRUE; pip install flash-attn==2.5.9.post1 --no-build-isolation; pip install '.[train]'; pip freeze; \
    mkdir -p /root/.cache; pushd /root/.cache; curl "https://storage.googleapis.com/hf-cache/huggingface_cache_v4.tar.gz" | tar --keep-newer-files -xzf -; popd; export HF_DATASETS_OFFLINE=1; \
    export TORCH_DIST_INIT_BARRIER=1; \
    export OLMO_SHARED_FS=1; \
    export NCCL_DEBUG=INFO; \
    export NCCL_IB_HCA="^=mlx5_bond_0"; \
    export NCCL_IB_TIMEOUT=22; \
    export PYTHONPATH=.; \
    torchrun --nnodes ${NUM_NODES}:${NUM_NODES} --nproc-per-node 8 \
        scripts/train.py ${CONFIG_PATH} \
        --run_name=${RUN_NAME} \
        --wandb.project=hb-wolf-olmo-2 --wandb.entity=liujch1998 \
        --save_folder=/weka/oe-training-default/wolf/ckpt/${RUN_NAME} --save_overwrite=true --load_path=\\\${path.last_checkpoint:/weka/oe-training-default/wolf/ckpt/${RUN_NAME}} \
        --device_train_microbatch_size=4 \
        --max_duration=11224 \
        --infgram.index_dir=/weka/oe-training-default/wolf/index/v5_dolma-v1_7_olmo/0 --infgram.prefetch=true --infgram.separate_wte=true --infgram.method_train=5 --infgram.method_eval=5 \
    "
