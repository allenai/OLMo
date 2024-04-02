# Visual Instruction Tune

The goal of this repo is to build large multimodal models (LMMs) by integrating visual backbones with large languag models (LLMs) based on the OLMo codebase. We follow the training protocol of [LLaVA](https://github.com/haotian-liu/LLaVA) (Large Language and Vision Assistant).


## Pretraining: vision-language alignment

Train a vision-language connector to align visual features with the LLM's embedding space.

### Download model and data

Download `blip_laion_cc_sbu_558k.json` and `images.zip` from [LLaVA-Pretrain](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain).
Then, organize them in `${DATA_DIR}` as follows (unzip `images.zip` under `${DATA_DIR}/llava/pretrain/images`).

```
${DATA_DIR}
└── llava
    └── pretrain
        └── images
        └── blip_laion_cc_sbu_558k.json
```

We use [OLMo-7B-Instruct](https://huggingface.co/allenai/OLMo-7B-Instruct). Download the model as follows
```python
from hf_olmo import *
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-7B-Instruct", cache_dir="${DATA_DIR}/olmo", model_max_length=2048)
olmo_model = AutoModelForCausalLM.from_pretrained("allenai/OLMo-7B-Instruct", cache_dir="${DATA_DIR}/olmo")
torch.save(olmo_model.model.transformer.state_dict(), "${DATA_DIR}/olmo/olmo-7b-instruct-hf-transformer.pt")
```

### Preprocess data

```bash
DATA_DIR={path_to_data_dir} PROJECT_DIR={path_to_project_dir} python3 scripts/llava/preprocess/preprocess_llava_data.py scripts/llava/preprocess/pretrain-olmo-7b-instruct-hf.yaml
```

### With HF Trainer + DeepSpeed

for beaker interactives
```bash
DATA_DIR={path_to_data_dir} OUTPUT_DIR={path_to_output_dir} WANDB_ENTITY={wandb_entity} CONFIG_PATH={config_path} bash scripts/llava/pretrain.sh
```

for beaker experiments
```bash
WORKSPACE={your_beaker_workspace} \
DATA_DIR={path_to_data_dir} \
OUTPUT_DIR={path_to_output_dir} \
NUM_GPUS={num_gpus} \
CONFIG_PATH={config_path} \
WANDB_ENTITY={wandb_entity} \
bash scripts/beaker/hfds-mm-olmo-pretrain.sh
```

### With Olmo Trainer + FSDP

for beaker interactives
```bash
DATA_DIR={path_to_data_dir} OUTPUT_DIR={path_to_output_dir} WANDB_ENTITY={wandb_entity} torchrun -m --nproc-per-node {num_gpus} scripts.train {config_path}
```

for beaker experiments
```bash
WORKSPACE={your_beaker_workspace} \
DATA_DIR={path_to_data_dir} \
OUTPUT_DIR={path_to_output_dir} \
NUM_GPUS={num_gpus} \
CONFIG_PATH={config_path} \
WANDB_ENTITY={wandb_entity} \
bash scripts/beaker/mm-olmo-train.sh
```


## Visual instruct-tuning

Train the vision-language connector as well as the language model on visual instruct-tuning data.

### Download data

Download `llava_v1_5_mix665k.json` and images from [LLaVA](https://github.com/haotian-liu/LLaVA).
After downloading all of them, organize the data in `${DATA_DIR}` as follows:

```
${DATA_DIR}
└── llava
    └── instruct_tune
        └── llava_v1_5_mix665k.json
        └── coco
            └── train2017
        └── gqa
            └── images
        └── ocr_vqa
            └── images
        └── textvqa
            └── train_images
        └── vqa
            └── VG_100K
            └── VG_100K_2
```

### Preprocess data

```bash
# olmo-7b-instruct, without system message
DATA_DIR={path_to_data_dir} PROJECT_DIR={path_to_project_dir} python3 scripts/llava/preprocess/preprocess_llava_data.py scripts/llava/preprocess/olmo-7b-instruct-hf-no-system.yaml
```

### With HF Trainer + DeepSpeed

for beaker interactives
```bash
DATA_DIR={path_to_data_dir} OUTPUT_DIR={path_to_output_dir} WANDB_ENTITY={wandb_entity} CONFIG_PATH={config_path} bash scripts/llava/finetune.sh
```

for beaker experiments
```bash
WORKSPACE={your_beaker_workspace} \
DATA_DIR={path_to_data_dir} \
OUTPUT_DIR={path_to_output_dir} \
NUM_GPUS={num_gpus} \
CONFIG_PATH={config_path} \
WANDB_ENTITY={wandb_entity} \
bash scripts/beaker/hfds-mm-olmo-finetune.sh
```


### With Olmo Trainer + FSDP

for beaker interactives
```bash
DATA_DIR={path_to_data_dir} OUTPUT_DIR={path_to_output_dir} WANDB_ENTITY={wandb_entity} torchrun -m --nproc-per-node {num_gpus} scripts.train {config_path} --load_path={checkpoint_path}
```

for beaker experiments
```bash
WORKSPACE={your_beaker_workspace} \
DATA_DIR={path_to_data_dir} \
OUTPUT_DIR={path_to_output_dir} \
NUM_GPUS={num_gpus} \
CONFIG_PATH={config_path} \
LOAD_PATH={checkpoint_path} \
WANDB_ENTITY={wandb_entity} \
bash scripts/beaker/mm-olmo-train.sh
```