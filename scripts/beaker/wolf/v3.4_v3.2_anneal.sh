#!/usr/bin/env bash

set -ex

CONFIG_PATH=configs/annealing/peteish1-anneal_legal-whammy-2.yaml
NUM_NODES=1
RUN_NAME="v3.4_v3.2_anneal"

gantry run \
  --allow-dirty \
  --name ${RUN_NAME} \
  --task-name ${RUN_NAME} \
  --description ${RUN_NAME} \
  --workspace ai2/infini-llm \
  --budget ai2/oe-training \
  --beaker-image petew/olmo-torch23-gantry \
  --cluster ai2/jupiter-cirrascale-2 \
  --priority high \
  --not-preemptible \
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
    export TORCH_DIST_INIT_BARRIER=1; \
    export OLMO_SHARED_FS=1; \
    export NCCL_DEBUG=INFO; \
    export NCCL_IB_HCA="^=mlx5_bond_0"; \
    export NCCL_IB_TIMEOUT=22; \
    export PYTHONPATH=.; \
    mkdir -p /weka/oe-training-default/jiachengl/hb-wolf/ckpt/${RUN_NAME}; \
    cp -r /weka/oe-training-default/ai2-llm/checkpoints/OLMo-small/peteish1/step1907359 /weka/oe-training-default/jiachengl/hb-wolf/ckpt/${RUN_NAME}/step0; \
    torchrun --nnodes ${NUM_NODES}:${NUM_NODES} --nproc-per-node 8 \
        scripts/train.py ${CONFIG_PATH} \
        --run_name=${RUN_NAME} \
        --wandb.project=hb-wolf-olmo-3 --wandb.entity=liujch1998 \
        --save_folder=/weka/oe-training-default/jiachengl/hb-wolf/ckpt/${RUN_NAME} --save_overwrite=true --load_path=\\\${path.last_checkpoint:/weka/oe-training-default/jiachengl/hb-wolf/ckpt/${RUN_NAME}} \
        --global_train_batch_size=1024 --device_train_microbatch_size=4
    "
