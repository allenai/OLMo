<div align="center">
  <!-- <img src="https://github.com/allenai/OLMo/assets/8812459/774ac485-a535-4768-8f7c-db7be20f5cc3" width="300"/> -->
  <img src="https://allenai.org/olmo/olmo-7b-animation.gif" alt="OLMo Logo" width="800" style="margin-left:'auto' margin-right:'auto' display:'block'"/>
  <br>
  <br>
  <h1>OLMo: Open Language Model</h1>
</div>
<p align="center">
  <a href="https://github.com/allenai/OLMo/blob/main/LICENSE">
    <img alt="GitHub License" src="https://img.shields.io/github/license/allenai/OLMo">
  </a>
  <a href="https://github.com/allenai/OLMo/releases">
    <img alt="GitHub release" src="https://img.shields.io/github/release/allenai/OLMo.svg">
  </a>
  <a href="https://arxiv.org/pdf/2402.00838.pdf">
    <img alt="Paper URL" src="https://img.shields.io/badge/arxiv-2402.00838-blue">
  </a>
</p>

OLMo is a repository for training and using AI2's state-of-the-art open language models. It is designed by scientists, for scientists.

## Installation

First, install [PyTorch](https://pytorch.org) following the instructions specific to your operating system.

For training and fine-tuning, we recommend installing from source:

```bash
git clone https://github.com/allenai/OLMo.git
cd OLMo
pip install -e .[all]
```
You can also install from PyPI with:
```bash
pip install ai2-olmo
```

## Models

### Overview

The core models in the OLMo family released so far are (all trained on the [Dolma dataset](https://huggingface.co/datasets/allenai/dolma)): 
| Model | Training Tokens | Context Length | Training Config | W&B Logs | Data Order File(s) ☨ |
|-------|-----------------|:--------------:|-----------------|----------|--------------------|
| [OLMo 1B](https://huggingface.co/allenai/OLMo-1B) | 3 Trillion | 2048 | [configs/official/OLMo-1B.yaml](https://github.com/allenai/OLMo/blob/main/configs/official/OLMo-1B.yaml) | [wandb.ai/…/OLMo-1B](https://wandb.ai/ai2-llm/OLMo-1B/reports/OLMo-1B--Vmlldzo2NzY1Njk1) | [epoch 1](https://olmo-checkpoints.org/ai2-llm/olmo-small/46zc5fly/train_data/global_indices.npy) |
| [OLMo 7B](https://huggingface.co/allenai/OLMo-7B) | 2.5 Trillion | 2048 | [configs/official/OLMo-7B.yaml](https://github.com/allenai/OLMo/blob/main/configs/official/OLMo-7B.yaml) | [wandb.ai/…/OLMo-7B](https://wandb.ai/ai2-llm/OLMo-7B/reports/OLMo-7B--Vmlldzo2NzQyMzk5) | [epoch 1](https://olmo-checkpoints.org/ai2-llm/olmo-medium/wvc30anm/train_data/global_indices.npy), [epoch 2](https://olmo-checkpoints.org/ai2-llm/olmo-medium/wd2gxrza/train_data/global_indices.npy) |
| [OLMo 7B Twin 2T](https://huggingface.co/allenai/OLMo-7B-Twin-2T) | 2 Trillion  | 2048 | [configs/official/OLMo-7B.yaml](https://github.com/allenai/OLMo/blob/main/configs/official/OLMo-7B.yaml) | [wandb.ai/…/OLMo-7B-Twin-2T](https://wandb.ai/ai2-llm/OLMo-7B/reports/OLMo-7B-Twin-2T--Vmlldzo2NzU0NTIz) | [epoch 1](https://olmo-checkpoints.org/ai2-llm/olmo-medium/wvc30anm/train_data/global_indices.npy) |
| [OLMo 7B April 2024](https://huggingface.co/allenai/OLMo-7B-0424-hf) | 2.05 Trillion  | 4096 | [configs/official/OLMo-7B-0424.yaml](https://github.com/allenai/OLMo/blob/main/configs/official/OLMo-7B-0424.yaml) | *Coming soon* | *Coming soon* |
| [OLMo 7B July 2024](https://huggingface.co/allenai/OLMo-7B-0724-hf) | 2.75 Trillion  | 4096 | [configs/official/OLMo-7B-0724.yaml](https://github.com/allenai/OLMo/blob/main/configs/official/OLMo-7B-0724.yaml) | *Coming soon* | *Coming soon* |

> ☨ *See [Inspecting training data](#inspecting-training-data) below for usage.*

### Checkpoints

URLs to checkpoints at intermediate steps of the models' trainings can be found in the csv files under [`checkpoints/official/`](https://github.com/allenai/OLMo/blob/main/checkpoints/official). These 'directory' URLs cannot currently be directly accessed, but files within the directory are publicly accessible. These URLs can also be provided to the training script to resume training from the checkpoint (see [Training](#training)). Each checkpoint directory consists of:

- `config.yaml`: the config at that training step.
- `model.safetensors`, `optim.safetensors`, `train.pt`: model, optimizer and training state at that training step.

Details about the other types of OLMo checkpoints (including OLMo HF Transformers checkpoints) can be found in [Checkpoints.md](https://github.com/allenai/OLMo/blob/main/docs/Checkpoints.md).

## Inference

You can utilize our Hugging Face integration to run inference on the OLMo Transformers checkpoints:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

olmo = AutoModelForCausalLM.from_pretrained("allenai/OLMo-7B-0724-hf")
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-7B-0724-hf")

message = ["Language modeling is "]
inputs = tokenizer(message, return_tensors='pt', return_token_type_ids=False)
response = olmo.generate(**inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
print(tokenizer.batch_decode(response, skip_special_tokens=True)[0])
```

Alternatively, with the Hugging Face pipeline abstraction:

```python
from transformers import pipeline
olmo_pipe = pipeline("text-generation", model="allenai/OLMo-7B-0724-hf")
print(olmo_pipe("Language modeling is"))
```

### Inference on finetuned checkpoints
After fine-tuning the model using the code in the [Fine-tuning](#fine-tuning) section, you can use the conversion script to convert a native OLMo checkpoint to a HuggingFace-compatible format.

```bash
python scripts/convert_olmo_to_hf_new.py --input_dir /path/to/olmo/checkpoint --output_dir /path/to/hf/checkpoint/ --tokenizer_json_path tokenizers/allenai_gpt-neox-olmo-dolma-v1_5.json
```

### Quantization

```python
olmo = AutoModelForCausalLM.from_pretrained("allenai/OLMo-7B-0724-hf", torch_dtype=torch.float16, load_in_8bit=True)  # requires bitsandbytes
```

The quantized model is sensitive to input types and CUDA handling. To avoid potential issues, we recommend explicitly converting input IDs to CUDA using: `inputs.input_ids.to('cuda')`

## Reproducibility
## Training

Install required packages:
```bash
pip3 install ai2-olmo wandb datasets torchmetrics scikit-learn
```

### Training from a Checkpoint

To continue training from a specific checkpoint:

1. Download the checkpoint using the provided script. Checkpoints are listed in CSV files under `checkpoints/official/`:
```bash
python scripts/download_checkpoints.py [PATH_TO_CSV] --save-dir [SAVE_PATH] --step [STEP]
```

Example: To download checkpoint at step 2000:
```bash
python scripts/download_checkpoints.py checkpoints/official/OLMo-1B.csv --save-dir ./checkpoints/ --step 2000
```
**Note**: All checkpoints in `checkpoints/official/` are unsharded files.

2. Resume training using the downloaded checkpoint. You can specify either a local path or URL using the --load_path argument: For example, to resume training from step 2000 of the OLMo 1B run:

```bash
torchrun --nproc_per_node=8 scripts/train.py configs/official/OLMo-1B.yaml     --load_path=checkpoints/step2000 --save_folder=./new_checkpoints --run_name=olmo_test --save_overwrite
```
The command above:
- Loads the checkpoint from `checkpoints/step2000`
- Saves new checkpoints to `./new_checkpoints`
- Names the training run `olmo_test` in wandb.
- Overwrites existing checkpoints in the save folder.

### Inspecting training data

To inspect the exact tokens used in training batches for OLMo models, first download the training data. If you don't have an R2 API key, use the public HTTP URLs and update your configuration file with the local data paths. After completing this setup, you can use the inspection tools to examine the training batches.

Find the data order file URL in the [Models Overview](#models-overview) table. For example, the OLMo-7B model's first epoch data order file is located at [https://olmo-checkpoints.org/ai2-llm/olmo-medium/wvc30anm/train_data/global_indices.npy](https://olmo-checkpoints.org/ai2-llm/olmo-small/46zc5fly/train_data/global_indices.npy).
Once you have that you can use this snippet to inspect the data within a particular batch:

```python
import numpy as np
from cached_path import cached_path

from olmo.config import TrainConfig
from olmo.data import build_memmap_dataset

# Update these paths to what you want:
data_order_file_path = cached_path("https://olmo-checkpoints.org/ai2-llm/olmo-medium/wvc30anm/train_data/global_indices.npy")
train_config_path = "configs/official/OLMo-7B.yaml"


cfg = TrainConfig.load(train_config_path)
dataset = build_memmap_dataset(cfg, cfg.data)
batch_size = cfg.global_train_batch_size
global_indices = np.memmap(data_order_file_path, mode="r+", dtype=np.uint32)


def get_batch_instances(batch_idx: int) -> list[list[int]]:
    batch_start = batch_idx * batch_size
    batch_end = (batch_idx + 1) * batch_size
    batch_indices = global_indices[batch_start:batch_end]
    batch_instances = []
    for index in batch_indices:
        token_ids = dataset[index]["input_ids"].tolist()
        batch_instances.append(token_ids)
    return batch_instances


# Get all 2048 x 2048 token IDs in the first batch.
get_batch_instances(0)
```


## Fine-tuning

To fine-tune an OLMo model using our trainer you'll first need to prepare your dataset by tokenizing it and saving the tokens IDs to a flat numpy memory-mapped array. See [`scripts/prepare_tulu_data.py`](./scripts/prepare_tulu_data.py) for an example with the Tulu V2 dataset, which can be easily modified for other datasets.

Next, prepare your training config. There are many examples in the [`configs/`](https://github.com/allenai/OLMo/blob/main/configs) directory that you can use as a starting point. The most important thing is to make sure the model parameters (the `model` field in the config) match up with the checkpoint you're starting from. To be safe you can always start from the config that comes with the model checkpoint. At a minimum you'll need to make the following changes to the config or provide the corresponding overrides from the command line:

- Update `load_path` to point to the checkpoint you want to start from.
- Set `reset_trainer_state` to `true`.
- Update `data.paths` to point to the `token_ids.npy` file you generated.
- Optionally update `data.label_mask_paths` to point to the `label_mask.npy` file you generated, unless you don't need special masking for the loss.
- Update `evaluators` to add/remove in-loop evaluations.

Once you're satisfied with your training config, you can launch the training job via `torchrun`. For example:

```
torchrun --nproc_per_node=8 scripts/train.py {path_to_train_config} \
    --data.paths=[{path_to_data}/input_ids.npy] \
    --data.label_mask_paths=[{path_to_data}/label_mask.npy] \
    --load_path={path_to_checkpoint} \
    --reset_trainer_state
```

Note: passing CLI overrides like `--reset_trainer_state` is only necessary if you didn't update those fields in your config.

## Evaluation

Additional tools for evaluating OLMo models are available at the [OLMo Eval](https://github.com/allenai/ai2-olmo-eval) repo.

## Debugging

See [Debugging](https://github.com/allenai/OLMo/blob/main/docs/NOTES.md#debugging).

## Citing

```bibtex
@article{OLMo,
  title={OLMo: Accelerating the Science of Language Models},
  author={Dirk Groeneveld and Iz Beltagy and Pete Walsh and Akshita Bhagia and Rodney Kinney and Oyvind Tafjord and A. Jha and Hamish Ivison and Ian Magnusson and Yizhong Wang and Shane Arora and David Atkinson and Russell Authur and Khyathi Raghavi Chandu and Arman Cohan and Jennifer Dumas and Yanai Elazar and Yuling Gu and Jack Hessel and Tushar Khot and William Merrill and Jacob Daniel Morrison and Niklas Muennighoff and Aakanksha Naik and Crystal Nam and Matthew E. Peters and Valentina Pyatkin and Abhilasha Ravichander and Dustin Schwenk and Saurabh Shah and Will Smith and Emma Strubell and Nishant Subramani and Mitchell Wortsman and Pradeep Dasigi and Nathan Lambert and Kyle Richardson and Luke Zettlemoyer and Jesse Dodge and Kyle Lo and Luca Soldaini and Noah A. Smith and Hanna Hajishirzi},
  year={2024},
  url={https://api.semanticscholar.org/CorpusID:267365485},
  journal={arXiv preprint},
}
```
