# Visual Instruction Tune

The goal of this repo is to build a large multimodal models (LMMs) by integrating visual backbones with large languag models (LLMs) based on the OLMo codebase. We follow the training protocol of [LLaVA](https://github.com/haotian-liu/LLaVA) (Large Language and Vision Assistant).

## Pretraining: vision-language alignment

Train a vision-language connector to align visual features with the LLM's embedding space.

### Download model and data

Download `blip_laion_cc_sbu_558k.json` and `images.zip` from [LLaVA-Pretrain](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain).
Then, organize them as follows in `${DATA_DIR}` (unzip `images.zip` under `${DATA_DIR}/llava/pretrain/images`).

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
DATA_DIR={path_to_data_dir} PROJECT_DIR={path_to_project_dir} python3 scripts/llava/preprocess/preprocess_llava_pretrain_data_olmo.py scripts/llava/preprocess/olmo-7b-instruct-hf.yaml
```


### With HF Trainer + DeepSpeed

```bash
DATA_DIR={path_to_data_dir} OUTPUT_DIR={path_to_output_dir} WANDB_ENTITY={wandb_entity} bash scripts/llava/pretrain.sh
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
bash scripts/beaker/mm-olmo-pretrain.sh
```
