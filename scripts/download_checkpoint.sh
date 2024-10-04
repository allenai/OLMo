# target_root=/home1/09636/zyliu/work/shared_resources/models
target_root=/home1/09636/zyliu/work/OLMo
# model_name=OLMo-7B-final
model_name=
target_model_dir=${target_root}/${model_name}

# mkdir -p ${target_model_dir}

# remote_url=https://olmo-checkpoints.org/ai2-llm/olmo-small/g4g72enr/step738020-unsharded
remote_url=https://olmo-checkpoints.org/ai2-llm/olmo-medium/p067ktg9/step558223-unsharded

for fname in config.yaml # model.pt optim.pt train.pt
do 

    if [ -f $target_model_dir/$fname ]; then
        echo "File found! -- ${target_model_dir}/${fname}"
    fi
    wget -P ${target_model_dir} ${remote_url}/${fname}

done