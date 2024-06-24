#!/usr/bin/env bash
set -ex

#CONFIG_PATH=configs/olmoe/OLMoE-8x1B-NOSHARD-S3.yml
#CONFIG_PATH=configs/olmoe17/olmoe17-8x1b-fullshard-swiglu-wrapb.yml
#CONFIG_PATH=configs/olmoe17/olmoe17-8x1b-fullshard-swiglu-wrapb-s1k1.yml
#CONFIG_PATH=configs/olmoe17/olmoe17-32x1b-fullshard-swiglu-wrapb-s1k3.yml
#CONFIG_PATH=configs/olmoe17/olmoe17-32x1b-fullshard-wrapb-s1k3.yml
#CONFIG_PATH=configs/olmoe17/olmoe17-8x1b-fullshard-swiglu-wrapb-lblfp32.yml
#CONFIG_PATH=configs/olmoe17/olmoe17-32x1b-fullshard-swiglu-wrapb-s1k3-ec.yml
#CONFIG_PATH=configs/olmoe17/olmoe17-8x1b-fullshard-swiglu-wrapb-ec.yml
#CONFIG_PATH=configs/olmoe17/olmoe17-8x1b-fullshard-swiglu-wrapb-ec.yml

#CONFIG_PATH=configs/olmoe17/olmoe17-8x1b-fullshard-swiglu-wrapb-k2-compile.yml
#CONFIG_PATH=configs/olmoe17/olmoe17-8x1b-fullshard-wrapb-k2-compile.yml
#CONFIG_PATH=configs/olmoe17/olmoe17-8x1b-fullshard-swiglu-wrapb-k2-lblfp32.yml
#CONFIG_PATH=configs/olmoe17/olmoe17-16x1b-fullshard-swiglu-wrapb-s1k1.yml
#CONFIG_PATH=configs/olmoe17/olmoe17-16x1b-fullshard-swiglu-wrapb-k2.yml
#CONFIG_PATH=configs/olmoe17/olmoe17-8x1b-fullshard-swiglu-wrapb-k2.yml
#CONFIG_PATH=configs/olmoe17/olmoe17-8x1b-fullshard-swiglu-wrapb-k2-init.yml
#ARGS='--run_name=olmoe17-8x1b-fullshard-swiglu-wrapb-k2-init'

#CONFIG_PATH=configs/olmoe17/olmoe17-s128x1b-fullshard-swiglu-wrapb-k2.yml
#ARGS='--run_name=olmoe17-s128x1b-fullshard-swiglu-wrapb-k2 --device_train_microbatch_size=1'

#CONFIG_PATH=configs/olmoe17/olmoe17-8x1b-fullshard-swiglu-wrapb-k2-datafix.yml
#CONFIG_PATH=configs/olmoe17/olmoe17-8x1b-fullshard-swiglu-wrapb-k2.yml
#ARGS='--run_name=olmoe17-8x1b-fullshard-swiglu-wrapb-k2-datafix-scratch'

#CONFIG_PATH=configs/olmoe17/olmoe17-8x1b-fullshard-swiglu-wrapb-k2-zloss.yml
#ARGS='--run_name=olmoe17-8x1b-fullshard-swiglu-wrapb-k2-zloss-scratch'

# --gen1_gc_interval=32'

# --activation_checkpointing=fine_grained' # --gen1_gc_interval=32
#--load_path=s3://ai2-llm/checkpoints/OLMoE/olmoe17-8x1b-fullshard-swiglu-wrapb-k2-init-qknorm/step155000/

#CONFIG_PATH=configs/olmoe17/olmoe17-8x1b-fullshard-swiglu-wrapb-ec-k2-il.yml

#CONFIG_PATH=configs/olmoe17/olmoe17-8x1b-fullshard-swiglu-wrapb-ecg-k2-il.yml
#ARGS='--load_path=s3://ai2-llm/checkpoints/OLMoE/8x1b-954000-il-unsharded/ --reset_optimizer_state=True --reset_trainer_state=True --run_name=olmoe17-8x1b-fullshard-swiglu-wrapb-ecg-k2-il --gen1_gc_interval=32 --device_train_microbatch_size=4 --fused_loss=true'

#CONFIG_PATH=configs/olmoe17/olmoe17-8x1b-fullshard-swiglu-wrapb-k2-zloss.yml
#ARGS='--run_name=olmoe17-8x1b-fullshard-swiglu-wrapb-k2-zloss-scratch --load_path=s3://ai2-llm/checkpoints/OLMoE/olmoe17-8x1b-fullshard-swiglu-wrapb-k2-zloss-scratch/step45000/ --save_overwrite'

#CONFIG_PATH=configs/olmoe17/olmoe17-8x1b-fullshard-swiglu-wrapb-k2.yml
#ARGS='--load_path=s3://ai2-llm/checkpoints/OLMoE/olmoe17-8x1b-fullshard-swiglu-wrapb-k2-05noise/step350000/ --run_name=olmoe17-8x1b-fullshard-swiglu-wrapb-k2-05noise'

#CONFIG_PATH=configs/olmoe17/olmoe17-8x1b-fullshard-swiglu-wrapb-k2-init-qknorm.yml
#ARGS='--load_path=s3://ai2-llm/checkpoints/OLMoE/olmoe17-8x1b-fullshard-swiglu-wrapb-k2-init-qknorm/step155000/ --run_name=olmoe17-8x1b-fullshard-swiglu-wrapb-k2-init-qknorm'

#CONFIG_PATH=configs/olmoe17/olmoe17-8x1b-fullshard-swiglu-wrapb-k2-qknorm-zloss.yml
#ARGS='--run_name=olmoe17-8x1b-fullshard-swiglu-wrapb-k2-qknorm-zloss-scratch --save_overwrite'


#CONFIG_PATH=configs/olmoe17/olmoe17-8x1b-fullshard-swiglu-wrapb-k2-init-zloss.yml
#ARGS='--run_name=olmoe17-8x1b-fullshard-swiglu-wrapb-k2-init-zloss-scratch --save_overwrite'

#CONFIG_PATH=configs/olmoe17/olmoe17-8x1b-fullshard-swiglu-wrapb-k2-init-zloss-paths.yml
#ARGS='--run_name=olmoe17-8x1b-fullshard-swiglu-wrapb-k2-init-zloss-scratch-paths --save_overwrite'


CONFIG_PATH=configs/olmoe17/olmoe17-8x1b-final.yml
ARGS='--run_name=olmoe17-8x1b-final-norm --save_overwrite --device_train_microbatch_size=2 --load_path=s3://ai2-llm/checkpoints/OLMoE/olmoe17-8x1b-final-norm/step35000/'


CONFIG_PATH=configs/olmoe17/olmoe17-8x1b-final-nodecnorm.yml
ARGS='--run_name=olmoe17-8x1b-final-nodecnorm --save_overwrite --device_train_microbatch_size=4'

CONFIG_PATH=configs/olmoe17/olmoe17-8x1b-final-decemb.yml
ARGS='--run_name=olmoe17-8x1b-final-decemb --save_overwrite --device_train_microbatch_size=4'


CONFIG_PATH=configs/olmoe17/olmoe17-8x1b-fullshard-swiglu-wrapb-k2-zloss.yml
ARGS='--run_name=olmoe17-8x1b-fullshard-swiglu-wrapb-k2-zloss --load_path=s3://ai2-llm/checkpoints/OLMoE/olmoe17-8x1b-fullshard-swiglu-wrapb-k2-zloss/step115000/ --save_overwrite'


# --activation_checkpointing=fine_grained'

# --activation_checkpointing=fine_grained


#CONFIG_PATH=configs/olmoe17/olmoe17-8x7b-final.yml
#ARGS='--run_name=olmoe17-8x7b-final'


#CONFIG_PATH=configs/olmoe17/olmoe17-8x1b-fullshard-swiglu-wrapb-k2-zloss.yml
#ARGS='--run_name=olmoe17-8x1b-fullshard-swiglu-wrapb-k2-zloss --load_path=s3://ai2-llm/checkpoints/OLMoE/olmoe17-8x1b-fullshard-swiglu-wrapb-k2-zloss/step40000/ --save_overwrite'

#CONFIG_PATH=configs/olmoe17/olmoe17-8x1b-fullshard-swiglu-wrapb-k2-zloss-final.yml

#ARGS='--run_name=olmoe17-8x1b-final --save_overwrite --fsdp.sharding_strategy=SHARD_GRAD_OP --fused_loss=true --activation_checkpointing=fine_grained'
#ARGS='--run_name=olmoe17-8x1b-final --save_overwrite'
# --fused_loss=true --activation_checkpointing=fine_grained'
# --fused_loss=true --activation_checkpointing=fine_grained'
# --fsdp.sharding_strategy=SHARD_GRAD_OP'

NUM_NODES=16
BEAKER_REPLICA_RANK=0

#ARGS='--load_path=s3://ai2-llm/checkpoints/OLMoE/8x1b-954000-unsharded/ --reset_optimizer_state=True --reset_trainer_state=True --run_name=olmoe17-8x1b-fullshard-swiglu-wrapb-s1k1'
#ARGS='--load_path=s3://ai2-llm/checkpoints/OLMoE/8x1b-954000-unsharded/ --reset_optimizer_state=True --reset_trainer_state=True --run_name=olmoe17-8x1b-fullshard-swiglu-wrapb-noscaling'
#ARGS='--load_path=s3://ai2-llm/checkpoints/OLMoE/8x1b-954000-unsharded/ --reset_optimizer_state=True --reset_trainer_state=True --run_name=olmoe17-8x1b-fullshard-swiglu-wrapb-lblfp32'
#ARGS='--load_path=s3://ai2-llm/checkpoints/OLMoE/32x1b-954000-s1k3-unsharded/ --reset_optimizer_state=True --reset_trainer_state=True --run_name=olmoe17-32x1b-fullshard-swiglu-wrapb-s1k3'
#ARGS='--load_path=s3://ai2-llm/checkpoints/OLMoE/olmoe17-32x1b-fullshard-swiglu-wrapb-s1k3/step5000/ --run_name=olmoe17-32x1b-fullshard-swiglu-wrapb-s1k3 --fused_loss=true'
#ARGS='--load_path=s3://ai2-llm/checkpoints/OLMoE/olmoe17-32x1b-fullshard-swiglu-wrapb-s1k3/step5000/ --run_name=olmoe17-32x1b-fullshard-swiglu-wrapb-s1k3 --fsdp.sharding_strategy=SHARD_GRAD_OP'
#ARGS='--run_name=olmoe17-32x1b-fullshard-swiglu-wrapb-s1k3 --fsdp.sharding_strategy=HYBRID_SHARD'
#ARGS='--run_name=olmoe17-32x1b-fullshard-wrapb-s1k3'
#ARGS='--load_path=s3://ai2-llm/checkpoints/OLMoE/32x1b-954000-s1k3-unsharded/ --reset_optimizer_state=True --reset_trainer_state=True --run_name=olmoe17-8x1b-fullshard-swiglu-wrapb-ec'

#ARGS='--load_path=s3://ai2-llm/checkpoints/OLMoE/olmoe17-8x1b-fullshard-swiglu-wrapb-k2/step330000/ --run_name=olmoe17-8x1b-fullshard-swiglu-wrapb-k2'
#ARGS='--load_path=s3://ai2-llm/checkpoints/OLMoE/olmoe17-8x1b-fullshard-swiglu-wrapb-k2-lblfp32/step15000/ --run_name=olmoe17-8x1b-fullshard-swiglu-wrapb-k2-lblfp32'
#ARGS='--load_path=s3://ai2-llm/checkpoints/OLMoE/8x1b-954000-unsharded/ --reset_optimizer_state=True --reset_trainer_state=True --run_name=olmoe17-8x1b-fullshard-swiglu-wrapb-k2-lblfp32 --device_train_microbatch_size=2'
#ARGS='--load_path=s3://ai2-llm/checkpoints/OLMoE/8x1b-954000-unsharded/ --reset_optimizer_state=True --reset_trainer_state=True --run_name=olmoe17-8x1b-fullshard-swiglu-wrapb-k2-lblfp32 --device_train_microbatch_size=2'
#ARGS='--load_path=s3://ai2-llm/checkpoints/OLMoE/olmoe17-8x1b-fullshard-swiglu-wrapb-lblfp32/step40000/ --run_name=olmoe17-8x1b-fullshard-swiglu-wrapb-lblfp32'
#ARGS='--load_path=s3://ai2-llm/checkpoints/OLMoE/8x1b-954000-05noise-unsharded/ --reset_optimizer_state=True --reset_trainer_state=True --run_name=olmoe17-8x1b-fullshard-swiglu-wrapb-k2-05noise'
#ARGS='--load_path=s3://ai2-llm/checkpoints/OLMoE/olmoe17-8x1b-fullshard-swiglu-wrapb-k2-05noise/step200000/ --run_name=olmoe17-8x1b-fullshard-swiglu-wrapb-k2-05noise'
#ARGS='--run_name=olmoe17-8x1b-fullshard-wrapb-k2-scratch --gen1_gc_interval=32'
#ARGS='--load_path=s3://ai2-llm/checkpoints/OLMoE/8x1b-954000-il-unsharded/ --reset_optimizer_state=True --reset_trainer_state=True --run_name=olmoe17-8x1b-fullshard-swiglu-wrapb-ec-k2 --gen1_gc_interval=32 --device_train_microbatch_size=8 --fused_loss=true --activation_checkpointing=fine_grained'
#ARGS='--load_path=s3://ai2-llm/checkpoints/OLMoE/8x1b-954000-il-unsharded/ --reset_optimizer_state=True --reset_trainer_state=True --run_name=olmoe17-8x1b-fullshard-swiglu-wrapb-ec-k2 --gen1_gc_interval=32 --device_train_microbatch_size=4 --fused_loss=true'
#--fused_loss=true --activation_checkpointing=fine_grained'


#ARGS='--load_path=s3://ai2-llm/checkpoints/OLMoE/16x1b-954000-unsharded/ --reset_optimizer_state=True --reset_trainer_state=True --run_name=olmoe17-16x1b-fullshard-swiglu-wrapb-k2'
#ARGS='--load_path=s3://ai2-llm/checkpoints/OLMoE/16x1b-954000-s1k1-unsharded/ --reset_optimizer_state=True --reset_trainer_state=True --run_name=olmoe17-16x1b-fullshard-swiglu-wrapb-k2'

# Add fast_forward_batches to ARGS for when loading and starting from scratch
#ARGS="${ARGS} --fast_forward_batches=136153"

# --evaluators=[]
# s3://ai2-llm/checkpoints/OLMoE/8x1b-954000-unsharded/



# Warm HF cache
#mkdir -p /root/.cache
#pushd /root/.cache
#curl "https://storage.googleapis.com/dirkgr-public/huggingface_cache_v3.tar.gz" | tar --keep-newer-files -xzf -
#popd
#export HF_DATASETS_OFFLINE=1

#shanea/olmo-torch2.3-gantry
#shanea/olmo-torch2.2-gantry
#petew/olmo-torch2-gantry
#  --priority normal \
#  --preemptible \
#  --priority normal \
#--cluster ai2/jupiter-cirrascale \


#shanea/olmo-torch2.3-gantry
#petew/olmo-torch23-gantry

# change your paths from s3://ai2-llm/... to /weka/oe-training-default/ai2-llm/....
#  --weka oe-training-default:/weka/oe-training-default \

gantry run \
  --allow-dirty \
  --priority high \
  --preemptible \
  --workspace ai2/olmoe \
  --task-name olmoe \
  --description olmoe \
  --beaker-image shanea/olmo-torch2.3-gantry \
  --budget ai2/oe-training \
  --cluster ai2/jupiter-cirrascale-2 \
  --gpus 8 \
  --replicas "${NUM_NODES}" \
  --env-secret WANDB_API_KEY=WANDB_API_KEY \
  --env-secret AWS_ACCESS_KEY_ID=AWS_ACCESS_KEY_ID \
  --env-secret AWS_SECRET_ACCESS_KEY=AWS_SECRET_ACCESS_KEY \
  --env-secret R2_ENDPOINT_URL=R2_ENDPOINT_URL \
  --leader-selection \
  --host-networking \
  --env LOG_FILTER_TYPE=local_rank0_only \
  --env OMP_NUM_THREADS=8 \
  --env OLMO_TASK=model \
  --shared-memory 10GiB \
  --venv base \
  --yes \
  --synchronized-start-timeout 30m \
  -- /bin/bash -c "pip install --upgrade torch==2.3.0; pip install git+https://github.com/Muennighoff/megablocks.git@zloss; mkdir -p /root/.cache; pushd /root/.cache; curl "https://storage.googleapis.com/dirkgr-public/huggingface_cache_v3.tar.gz" | tar --keep-newer-files -xzf -; popd; export HF_DATASETS_OFFLINE=1; SLURM_JOB_ID=${BEAKER_JOB_ID} torchrun --nnodes ${NUM_NODES}:${NUM_NODES} --nproc-per-node 8 --rdzv_id=12347 --rdzv_backend=c10d --rdzv_conf='read_timeout=420' --rdzv_endpoint=\$BEAKER_LEADER_REPLICA_HOSTNAME:29400 scripts/train.py ${CONFIG_PATH} ${ARGS}"

#conda install nvidia/label/cuda-11.8.0::cuda; 
#pip install --upgrade torch; 
#; export NCCL_DEBUG=TRACE
# pip install git+https://github.com/Muennighoff/megablocks.git
# pip install git+https://github.com/Muennighoff/megablocks.git@zloss
# pip install git+https://github.com/Muennighoff/megablocks.git@expertchoice

#  --synchronized-start-timeout 30m \

#  -- /bin/bash -c "mkdir -p /root/.cache; pushd /root/.cache; curl "https://storage.googleapis.com/dirkgr-public/huggingface_cache_v3.tar.gz" | tar --keep-newer-files -xzf -; popd; export HF_DATASETS_OFFLINE=1; \
#    torchrun --nnodes ${NUM_NODES}:${NUM_NODES} --nproc-per-node 8 --rdzv_id=101 --rdzv_backend=c10d --rdzv_conf='read_timeout=420' --rdzv_endpoint=\$BEAKER_LEADER_REPLICA_HOSTNAME:29400 scripts/train.py ${CONFIG_PATH} ${ARGS}"
# pip install git+https://github.com/Muennighoff/megablocks.git
# pip install git+https://github.com/Muennighoff/megablocks.git@noscaling
# --synchronized-start-timeout 30m \
# --no-deps # -> Does not work
#export NCCL_DEBUG=INFO;

# Single node:
#--rdzv_endpoint=\$BEAKER_NODE_HOSTNAME:29400
# Multinode:
#--rdzv_endpoint=\$BEAKER_LEADER_REPLICA_HOSTNAME:29400
#  --mount /net/nfs.cirrascale/allennlp/petew/cache:/root/.cache \
#--node_rank=$BEAKER_REPLICA_RANK
#  --nfs \