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


#CONFIG_PATH=configs/olmoe17/olmoe17-8x1b-final-nodecln.yml
#ARGS='--run_name=olmoe17-8x1b-final-nodecln --save_overwrite --device_train_microbatch_size=2'
# --activation_checkpointing=fine_grained'

#CONFIG_PATH=configs/olmoe17/olmoe17-8x1b-final-decemb.yml
#ARGS='--run_name=olmoe17-8x1b-final-decemb --save_overwrite --device_train_microbatch_size=2 --load_path=s3://ai2-llm/checkpoints/OLMoE/olmoe17-8x1b-final-decemb/step5000/'


#CONFIG_PATH=configs/olmoe17/olmoe17-8x1b-final-eps.yml
#ARGS='--run_name=olmoe17-8x1b-final-eps --save_overwrite --device_train_microbatch_size=2 --load_path=s3://ai2-llm/checkpoints/OLMoE/olmoe17-8x1b-final-eps/step5000/ --save_overwrite'

#CONFIG_PATH=configs/olmoe17/olmoe17-8x1b-final-fine.yml
#ARGS='--run_name=olmoe17-8x1b-final-fine --save_overwrite --device_train_microbatch_size=2'
# --fused_loss=true --activation_checkpointing=fine_grained'

#CONFIG_PATH=configs/olmoe17/olmoe17-8x1b-fullshard-swiglu-wrapb-k2-zloss.yml
#ARGS='--run_name=olmoe17-8x1b-fullshard-swiglu-wrapb-k2-zloss --load_path=s3://ai2-llm/checkpoints/OLMoE/olmoe17-8x1b-fullshard-swiglu-wrapb-k2-zloss/step120000/ --save_overwrite'


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

#CONFIG_PATH=configs/olmoe17/olmoe17-8x1b-final-normdc.yml
#ARGS='--run_name=olmoe17-8x1b-final-normdc --save_overwrite --device_train_microbatch_size=2'

#CONFIG_PATH=configs/olmoe17/olmoe17-8x1b-final-normdc.yml
#ARGS='--run_name=olmoe17-8x1b-final-normdc --save_overwrite --device_train_microbatch_size=2'

#CONFIG_PATH=configs/olmoe17/olmoe-8x1b-newhp-oldds.yml
#ARGS='--run_name=olmoe-8x1b-newhp-oldds --save_overwrite --device_train_microbatch_size=2'

#CONFIG_PATH=configs/olmoe17/olmoe17-8x1b-final-eps-noqk.yml
#ARGS='--run_name=olmoe17-8x1b-final-eps-noqk --save_overwrite --device_train_microbatch_size=2'




#CONFIG_PATH=configs/olmoe17/olmoe17-8x1b-final-eps-fine.yml
#ARGS='--run_name=olmoe17-8x1b-final-eps-fine --save_overwrite --device_train_microbatch_size=2 --fsdp.sharding_strategy=HYBRID_SHARD'

#CONFIG_PATH=configs/olmoe17/olmoe-8x1b-newhp-newds-s3.yml
#ARGS='--run_name=olmoe-8x1b-newhp-newds --save_overwrite --fsdp.sharding_strategy=HYBRID_SHARD'

#CONFIG_PATH=configs/olmoe17/olmoe-8x1b-newhp-newds-cx5-fine.yml
#ARGS='--run_name=olmoe-8x1b-newhp-newds-cx5-fine --save_overwrite --fsdp.sharding_strategy=HYBRID_SHARD --device_train_microbatch_size=4 --load_path=s3://ai2-llm/checkpoints/OLMoE/olmoe-8x1b-newhp-newds-cx5-fine/step20000/'

#CONFIG_PATH=configs/olmoe17/olmo-1b-newhp-newds-cx5-reddit.yml
#ARGS='--run_name=olmo-1b-newhp-newds-cx5-reddit --save_overwrite --fsdp.sharding_strategy=HYBRID_SHARD --device_train_microbatch_size=4 --load_path=s3://ai2-llm/checkpoints/OLMoE/olmo-1b-newhp-newds-cx5-reddit/step5000/'

CONFIG_PATH=configs/olmoe17/olmo-1b-newhp-newds-cx5-flan.yml
ARGS='--run_name=olmo-1b-newhp-newds-cx5-flan --save_overwrite --fsdp.sharding_strategy=FULL_SHARD --device_train_microbatch_size=4 --load_path=s3://ai2-llm/checkpoints/OLMoE/olmo-1b-newhp-newds-cx5-flan/step20000/'

#CONFIG_PATH=configs/olmoe17/olmo-1b-newhp-newds-cx5-datafix.yml
#ARGS='--run_name=olmo-1b-newhp-newds-cx5-datafix --save_overwrite --fsdp.sharding_strategy=HYBRID_SHARD --device_train_microbatch_size=4'

#CONFIG_PATH=configs/olmoe17/olmoe-8x1b-newhp-newds-s3-cx5.yml
#ARGS='--run_name=olmoe-8x1b-newhp-newds-cx5 --save_overwrite --fsdp.sharding_strategy=HYBRID_SHARD --device_train_microbatch_size=4'

#CONFIG_PATH=configs/olmoe17/olmo-1b-newhp-oldds-cx5.yml
#ARGS='--run_name=olmo-1b-newhp-oldds-cx5 --save_overwrite --fsdp.sharding_strategy=HYBRID_SHARD --device_train_microbatch_size=4'

#CONFIG_PATH=configs/olmoe17/olmo-1b-newhp-newds-cx5.yml
#ARGS='--run_name=olmo-1b-newhp-newds-cx5 --save_overwrite --fsdp.sharding_strategy=HYBRID_SHARD --device_train_microbatch_size=4 --load_path=s3://ai2-llm/checkpoints/OLMoE/olmo-1b-newhp-newds-cx5/step10000/'

#CONFIG_PATH=configs/olmoe17/olmoe-8x1b-newhp-newds-cx5-k2.yml
#ARGS='--run_name=olmoe-8x1b-newhp-newds-cx5-k2 --save_overwrite --fsdp.sharding_strategy=HYBRID_SHARD --device_train_microbatch_size=4 --fused_loss=true --activation_checkpointing=fine_grained'

#CONFIG_PATH=configs/olmoe17/olmoe-8x1b-newhp-newds-cx5-k2-fine.yml
#ARGS='--run_name=olmoe-8x1b-newhp-newds-cx5-k2-fine --save_overwrite --fsdp.sharding_strategy=HYBRID_SHARD --device_train_microbatch_size=2'
#CONFIG_PATH=configs/olmoe17/olmoe-8x1b-newhp-newds-cx5-k2-fine-s3.yml
#ARGS='--run_name=olmoe-8x1b-newhp-newds-cx5-k2-fine --save_overwrite --fsdp.sharding_strategy=HYBRID_SHARD --device_train_microbatch_size=2 --load_path=s3://ai2-llm/checkpoints/OLMoE/olmoe-8x1b-newhp-newds-cx5-k2-fine/step10000/'

CONFIG_PATH=configs/olmoe17/olmoe-8x1b-newhp-newds-final.yml
ARGS='--run_name=olmoe-8x1b-newhp-newds-final --save-overwrite --fsdp.sharding_strategy=FULL_SHARD --device_train_microbatch_size=4 --canceled_check_interval=9999999 --load_path=s3://ai2-llm/checkpoints/OLMoE/olmoe-8x1b-newhp-newds-final/step1110000/ --epoch=1'

CONFIG_PATH=configs/olmoe17/olmoe-8x7b.yml
ARGS='--run_name=olmoe-8x7b.yml --save_overwrite --fsdp.sharding_strategy=FULL_SHARD'
# --activation_checkpointing=fine_grained --fused_loss=true'

CONFIG_PATH=configs/olmoe17/olmoe-8x7b-A7B.yml
ARGS='--run_name=olmoe-8x7b-A7B --save_overwrite --fsdp.sharding_strategy=FULL_SHARD --device_train_microbatch_size=2'

CONFIG_PATH=configs/olmoe17/olmoe-8x1b-newhp-newds-final-double.yml
ARGS='--run_name=olmoe-8x1b-newhp-newds-final-double --save_overwrite --device_train_microbatch_size=2'

CONFIG_PATH=configs/olmoe17/olmoe-8x1b-newhp-newds-final-double-alt.yml
ARGS='--run_name=olmoe-8x1b-newhp-newds-final-double-alt --save_overwrite --device_train_microbatch_size=2'

CONFIG_PATH=configs/olmoe17/olmoe-8x1b-newhp-newds-cx5-fine1-datafix.yml
ARGS='--run_name=olmoe-8x1b-newhp-newds-cx5-fine1-datafix --save_overwrite --device_train_microbatch_size=4'

CONFIG_PATH=configs/olmoe17/olmoe-8x1b-newhp-newds-cx5-fine1-newtok.yml
ARGS='--run_name=olmoe-8x1b-newhp-newds-cx5-fine1-newtok --save_overwrite --device_train_microbatch_size=2'

CONFIG_PATH=configs/olmoe17/olmoe-8x1b-newhp-newds-cx5-fine1-newtok.yml
ARGS='--run_name=olmoe-8x1b-newhp-newds-cx5-fine1-newtok --save_overwrite --device_train_microbatch_size=4 --fsdp.wrapping_strategy=by_block_and_size'

CONFIG_PATH=configs/olmoe17/olmoe-8x1b-newhp-newds-cx5-fine1-newtok.yml
ARGS='--run_name=olmoe-8x1b-newhp-newds-cx5-fine1-newtok --save_overwrite --device_train_microbatch_size=4 --fsdp.wrapping_strategy=by_block_and_size'

CONFIG_PATH=configs/olmoe17/olmoe-8x1b-newhp-newds-cx5-fine1-newtok.yml
ARGS='--run_name=olmoe-8x1b-newhp-newds-cx5-fine1-newtok --save_overwrite --device_train_microbatch_size=4 --activation_checkpointing=fine_grained --fused_loss=true'

CONFIG_PATH=configs/olmoe17/olmoe-8x1b-newhp-newds-cx5-fine1-newtok.yml
ARGS='--run_name=olmoe-8x1b-newhp-newds-cx5-fine1-newtok --save_overwrite --device_train_microbatch_size=4 --activation_checkpointing=fine_grained --fused_loss=true --fsdp.wrapping_strategy=by_block_and_size'


#CONFIG_PATH=configs/olmoe17/olmoe-8x1b-newhp-newds-cx5-fine1-normreorder.yml
#ARGS='--run_name=olmoe-8x1b-newhp-newds-cx5-fine1-normreorder --save_overwrite --device_train_microbatch_size=4'

#CONFIG_PATH=configs/olmoe17/olmoe-8x1b-newhp-newds-cx5-fine1-docmask.yml
#ARGS='--run_name=olmoe-8x1b-newhp-newds-cx5-fine1-docmask --save_overwrite --device_train_microbatch_size=4'

CONFIG_PATH=configs/olmoe17/olmoe-8x1b-newhp-newds-cx5-fine1-docmask-8k.yml
ARGS='--run_name=olmoe-8x1b-newhp-newds-cx5-fine1-docmask-8k --save_overwrite --device_train_microbatch_size=2'

# --activation_checkpointing=fine_grained --fused_loss=true'

#configs/olmoe17/olmoe-8x1b-newhp-newds-final-double.yml
#CONFIG_PATH=configs/olmoe17/olmoe-8x1b-newhp-newds-final-anneal.yml
#ARGS='--run_name=olmoe-8x1b-newhp-newds-final-annealFrom1200000 --save-overwrite --fsdp.sharding_strategy=FULL_SHARD --canceled_check_interval=9999999 --load_path=s3://ai2-llm/checkpoints/OLMoE/olmoe-8x1b-newhp-newds-final/step1200000/'

# --load_path=s3://ai2-llm/checkpoints/OLMoE/olmoe-8x1b-newhp-newds-final/step20000/'
#--activation_checkpointing=fine_grained'

#CONFIG_PATH=configs/olmoe17/olmoe-8x1b-newhp-newds-final-densecomp.yml
#ARGS='--run_name=olmoe-8x1b-newhp-newds-final-densecomp --save_overwrite --fsdp.sharding_strategy=FULL_SHARD --device_train_microbatch_size=8 --canceled_check_interval=200' # --activation_checkpointing=fine_grained

# --fused_loss=true --activation_checkpointing=fine_grained'

#CONFIG_PATH=configs/olmoe17/olmoe-8x1b-newhp-newds-cx5-fine-shared-s3.yml
#ARGS='--run_name=olmoe-8x1b-newhp-newds-cx5-fine-shared --save_overwrite --fsdp.sharding_strategy=FULL_SHARD --device_train_microbatch_size=4 --load_path=s3://ai2-llm/checkpoints/OLMoE/olmoe-8x1b-newhp-newds-cx5-fine-shared/step20000/'

#CONFIG_PATH=configs/olmoe17/olmoe-8x1b-newhp-newds-cx5-fine05.yml
#ARGS='--run_name=olmoe-8x1b-newhp-newds-cx5-fine05 --save_overwrite --fsdp.sharding_strategy=HYBRID_SHARD --device_train_microbatch_size=4 --fused_loss=true --activation_checkpointing=fine_grained'

#CONFIG_PATH=configs/olmoe17/olmoe-8x1b-newhp-newds-cx5-fine1.yml
#ARGS='--run_name=olmoe-8x1b-newhp-newds-cx5-fine1 --save_overwrite --fsdp.sharding_strategy=HYBRID_SHARD --device_train_microbatch_size=4 --fused_loss=true --activation_checkpointing=fine_grained --load_path=s3://ai2-llm/checkpoints/OLMoE/olmoe-8x1b-newhp-newds-cx5-fine1/step5000/'

CONFIG_PATH=configs/olmoe17/olmoe-8x1b-newhp-newds-final-v2.yml
ARGS='--run_name=olmoe-8x1b-newhp-newds-final-v2 --fused_loss=true --activation_checkpointing=fine_grained --fsdp.wrapping_strategy=by_block_and_size  --load_path=s3://ai2-llm/checkpoints/OLMoE/olmoe-8x1b-newhp-newds-final-v2/step300000/'
#--activation_checkpointing=fine_grained 
#NUM_NODES=1
#NUM_NODES=16
NUM_NODES=32
#NUM_NODES=8
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


#shanea/olmo-torch23-gantry
#petew/olmo-torch23-gantry

# change your paths from s3://ai2-llm/... to /weka/oe-training-default/ai2-llm/....
#  --weka oe-training-default:/weka/oe-training-default \

# export NCCL_IB_HCA="^=mlx5_bond_0" 

#  --preemptible \

#  --weka oe-training-default:/weka/oe-training-default \

gantry run \
  --weka oe-training-default:/weka/oe-training-default \
  --allow-dirty \
  --preemptible \
  --priority urgent \
  --workspace ai2/olmoe \
  --task-name olmoe \
  --description olmoe \
  --beaker-image shanea/olmo-torch2.2-gantry \
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
  --synchronized-start-timeout 60m \
  -- /bin/bash -c "pip install --upgrade torch==2.3.0; pip install --upgrade flash-attn --no-build-isolation; pip install git+https://github.com/Muennighoff/megablocks.git@zloss; mkdir -p /root/.cache; pushd /root/.cache; curl "https://storage.googleapis.com/dirkgr-public/huggingface_cache_v3.tar.gz" | tar --keep-newer-files -xzf -; popd; export HF_DATASETS_OFFLINE=1; export NCCL_IB_HCA=^=mlx5_bond_0; SLURM_JOB_ID=${BEAKER_JOB_ID} torchrun --nnodes ${NUM_NODES}:${NUM_NODES} --node_rank ${BEAKER_REPLICA_RANK} --nproc-per-node 8 --rdzv_id=12347 --rdzv_backend=c10d --rdzv_conf='read_timeout=420' --rdzv_endpoint=\$BEAKER_LEADER_REPLICA_HOSTNAME:29400 scripts/train.py ${CONFIG_PATH} ${ARGS}"

#sleep 60000; 
#  --synchronized-start-timeout 60m \

#export NCCL_DEBUG=INFO
#pip install --upgrade flash-attn --no-build-isolation
#conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia; 
# conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
# conda install -y -c conda-forge cudatoolkit-dev
#pip install --upgrade torch==2.3.0; conda install -y cuda-nvcc cuda-python -c pytorch -c nvidia; 
#conda install -y cuda-nvcc cuda-python -c pytorch -c nvidia
#; conda install -c conda-forge cudatoolkit-dev
#conda install -c conda-forge cudatoolkit-dev
#export TORCH_DIST_INIT_BARRIER=1; export OLMO_SHARED_FS=1; export NCCL_IB_HCA=^=mlx5_bond_0
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