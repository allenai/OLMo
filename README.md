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
  <a href="https://arxiv.org/pdf/2501.00656.pdf">
    <img alt="Paper URL" src="https://img.shields.io/badge/arxiv-2402.00838-blue">
  </a>
  <a href="https://playground.allenai.org">
    <img alt="Playground" src="https://img.shields.io/badge/Ai2-Playground-F0529C">
  </a>
  <a href="https://discord.gg/sZq3jTNVNG">
    <img alt="Discord" src="https://img.shields.io/badge/Discord%20-%20blue?style=flat&logo=discord&label=Ai2&color=%235B65E9">
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

## Pretraining

OLMo pretraining follows a two-stage training procedure.
In the first stage, we train on large amounts of mostly web-based data: [OLMo-mix-1124](https://huggingface.co/datasets/allenai/olmo-mix-1124)
In the second stage, we train on a smaller amount of high-quality, targeted data: [Dolmino-mix-1124](https://huggingface.co/datasets/allenai/dolmino-mix-1124)

You can find *all* the checkpoints, at minimum every 1000 training steps in OLMo core and Hugging Face format:


| Variant         | OLMo Format (Stage 1)                                                                                         | OLMo Format (Stage 2) | Hugging Face Format                                                               |
|----------------|-----------------------------------------------------------------------------------------------------|--------|----------------------------------------------------------------------------------|
| **OLMo-2 7B**  | [OLMo-2 7B](https://github.com/allenai/OLMo/blob/main/configs/official-1124/OLMo-2-1124-7B.csv)     | [OLMo-2 7B](https://github.com/allenai/OLMo/blob/main/configs/official-1124/OLMo-2-1124-7B-stage2.csv)      | [Hugging Face for the 7B variant](https://huggingface.co/allenai/OLMo-2-1124-7B)  |
| **OLMo-2 13B** | [OLMo-2 13B](https://github.com/allenai/OLMo/blob/main/configs/official-1124/OLMo-2-1124-13B.csv)   | [OLMo-2 13B](https://github.com/allenai/OLMo/blob/main/configs/official-1124/OLMo-2-1124-13B-stage2.csv)       | [Hugging Face for the 13B variant](https://huggingface.co/allenai/OLMo-2-1124-13B) |
| **OLMo-2 32B** | [OLMo-2 32B](https://github.com/allenai/OLMo-core/blob/main/src/scripts/official/OLMo2-0325-32B.csv)   | [OLMo-2 32B](https://github.com/allenai/OLMo-core/blob/main/src/scripts/official/OLMo-2-0325-32B-stage2.csv) | [Hugging Face for the 32B variant](https://huggingface.co/allenai/OLMo-2-0325-32B) |

> Note: The 32B variant was trained on our new trainer. To train or fine-tune OLMo-2 32B, visit [OLMo-core](https://github.com/allenai/OLMo-core).

### Steps to reproduce

To reproduce any of the training processes described below, run this:

```bash
torchrun --nproc_per_node=8 scripts/train.py {path_to_train_config}
```

For the training config, use any of the configs listed below.

If you want to override any of the settings in the training config without having to write a new config every time,
you can do this:

```bash
torchrun --nproc_per_node=8 scripts/train.py {path_to_train_config} \
  --setting1=value \
  --setting2=value \
  --setting3.subsetting1=value
```

The training configs below refer to training data that gets streamed in live over HTTP.
To reproduce at large scale, we recommend downloading the files locally and changing the paths to point to your
local file system.

#### To run on Mac silicon devices:
```bash
python scripts/train.py {path_to_train_config}
```
Example:
```bash
python scripts/train.py configs/tiny/OLMo-20M.yaml --save_overwrite
```
> Note: You need to upgrade PyTorch to 2.5.x to run.

### Stage 1

Stage 1 is the biggest stage, where we train on 4T or 5T tokens on largely web-based data. 

|                 | OLMo2 7B                                                                                                          | OLMo2 13B                                                                                                          |
|-----------------|-------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| Number of tokens| 4 Trillion                                                                                                        | 5 Trillion                                                                                                         |
| Checkpoint      | [stage1-step928646-tokens3896B](https://huggingface.co/allenai/OLMo-2-1124-7B/tree/stage1-step928646-tokens3896B) | [stage1-step596057-tokens5001B](https://huggingface.co/allenai/OLMo-2-1124-13B/tree/stage1-step596057-tokens5001B) |
| Training config | [OLMo2-7B-stage1.yaml](configs/official-1124/OLMo2-7B-stage1.yaml)                                                | [OLMo2-13B-stage1.yaml](configs/official-1124/OLMo2-13B-stage1.yaml)                                               |                                              |
| WandB           | [wandb.ai/OLMo2-7B](https://wandb.ai/ai2-llm/OLMo-2-1124-7B/reports/OLMo-2-7B-Nov-2024--VmlldzoxMDUzMzE1OA)       | [wandb.ai/OLMo2-13B](https://wandb.ai/ai2-llm/OLMo-2-1124-13B/reports/OLMo-2-13B-Nov-2024--VmlldzoxMDUzMjQxNg) |


### Stage 2 for the 7B

For the 7B model, we train three times with different data order on 50B high quality tokens, and then average ("soup") the models.

|                        | Checkpoint                                                                                                                          | Training config                                                                        | WandB       |
|------------------------|-------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------|-------------|
| random seed 42         | [stage2-ingredient1-step11931-tokens50B](https://huggingface.co/allenai/OLMo-2-1124-7B/tree/stage2-ingredient1-step11931-tokens50B) | [OLMo2-7B-stage2-seed42.yaml](configs/official-1124/OLMo2-7B-stage2-seed42.yaml)       | [wandb.ai/OLMo2-7B](https://wandb.ai/ai2-llm/OLMo-2-1124-7B/reports/) |
| random seed 42069      | [stage2-ingredient2-step11931-tokens50B](https://huggingface.co/allenai/OLMo-2-1124-7B/tree/stage2-ingredient2-step11931-tokens50B) | [OLMo2-7B-stage2-seed42069.yaml](configs/official-1124/OLMo2-7B-stage2-seed42069.yaml) | [wandb.ai/OLMo2-7B](https://wandb.ai/ai2-llm/OLMo-2-1124-7B/reports/) |
| random seed 666        | [stage2-ingredient3-step11931-tokens50B](https://huggingface.co/allenai/OLMo-2-1124-7B/tree/stage2-ingredient3-step11931-tokens50B) | [OLMo2-7B-stage2-seed666.yaml](configs/official-1124/OLMo2-7B-stage2-seed666.yaml)     | [wandb.ai/OLMo2-7B](https://wandb.ai/ai2-llm/OLMo-2-1124-7B/reports/) |
| **final souped model** | [main](https://huggingface.co/allenai/OLMo-2-1124-7B/tree/main) | no config, we just averaged the weights in Python                                      | |

The training configs linked here are set up to download the latest checkpoint after stage 1, and start training from there.

### Stage 2 for the 13B

For the 13B model, we train three times with different data order on 100B high quality tokens, and one more time
on 300B high quality tokens. Then we average ("soup") the models.

|                        | Checkpoint                                                                                                                             | Training config                                                                                  | WandB       |
|------------------------|----------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------|-------------|
| random seed 1110, 100B | [stage2-ingredient1-step11931-tokens100B](https://huggingface.co/allenai/OLMo-2-1124-13B/tree/stage2-ingredient1-step11931-tokens100B) | [OLMo2-13B-stage2-seed1110-100B.yaml](configs/official-1124/OLMo2-13B-stage2-seed1110-100B.yaml) | [wandb.ai/OLMo2-13B](https://wandb.ai/ai2-llm/OLMo-2-1124-13B/reports/OLMo-2-13B-Nov-2024--VmlldzoxMDUzMjQxNg) |
| random seed 2662, 100B | [stage2-ingredient2-step11931-tokens100B](https://huggingface.co/allenai/OLMo-2-1124-13B/tree/stage2-ingredient2-step11931-tokens100B) | [OLMo2-13B-stage2-seed2662-100B.yaml](configs/official-1124/OLMo2-13B-stage2-seed2662-100B.yaml) | [wandb.ai/OLMo2-13B](https://wandb.ai/ai2-llm/OLMo-2-1124-13B/reports/OLMo-2-13B-Nov-2024--VmlldzoxMDUzMjQxNg) |
| random seed 6209, 100B | [stage2-ingredient3-step11931-tokens100B](https://huggingface.co/allenai/OLMo-2-1124-13B/tree/stage2-ingredient3-step11931-tokens100B) | [OLMo2-13B-stage2-seed6209-100B.yaml](configs/official-1124/OLMo2-13B-stage2-seed6209-100B.yaml) | [wandb.ai/OLMo2-13B](https://wandb.ai/ai2-llm/OLMo-2-1124-13B/reports/OLMo-2-13B-Nov-2024--VmlldzoxMDUzMjQxNg) |
| random seed 2662, 300B | [stage2-ingredient4-step11931-tokens300B](https://huggingface.co/allenai/OLMo-2-1124-13B/tree/stage2-ingredient4-step35773-tokens300B) | [OLMo2-13B-stage2-seed2662-300B.yaml](configs/official-1124/OLMo2-13B-stage2-seed2662-300B.yaml) | [wandb.ai/OLMo2-13B](https://wandb.ai/ai2-llm/OLMo-2-1124-13B/reports/OLMo-2-13B-Nov-2024--VmlldzoxMDUzMjQxNg) |
| **final souped model** | [main](https://huggingface.co/allenai/OLMo-2-1124-13B/tree/main)                                                                       | no config, we just averaged the weights in Python                                                | |

The training configs linked here are set up to download the latest checkpoint after stage 1, and start training from there.

> Note: You can find all the information about the 32B in the [OLMo-core](https://github.com/allenai/OLMo-core) repository.

## Instruction tuned variants

For instruction tuned variants of these models, go to
 * [OLMo2 7B Instruct](https://huggingface.co/allenai/OLMo-2-1124-7B-Instruct)
 * [OLMo2 13B Instruct](https://huggingface.co/allenai/OLMo-2-1124-13B-Instruct)
 * [OLMo2 32B Instruct](https://huggingface.co/allenai/OLMo-2-0325-32B-Instruct)

## Inference

You can use our Hugging Face integration to run inference on the OLMo Transformers checkpoints:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
olmo = AutoModelForCausalLM.from_pretrained("allenai/OLMo-2-1124-7B")
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-1124-7B")
message = ["Language modeling is "]
inputs = tokenizer(message, return_tensors='pt', return_token_type_ids=False)
# optional verifying cuda
# inputs = {k: v.to('cuda') for k,v in inputs.items()}
# olmo = olmo.to('cuda')
response = olmo.generate(**inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
print(tokenizer.batch_decode(response, skip_special_tokens=True)[0])
```

Alternatively, with the Hugging Face pipeline abstraction:

```python
from transformers import pipeline
olmo_pipe = pipeline("text-generation", model="allenai/OLMo-2-1124-7B")
print(olmo_pipe("Language modeling is"))
```

### Quantization

```python
olmo = AutoModelForCausalLM.from_pretrained("allenai/OLMo-2-1124-7B", torch_dtype=torch.float16, load_in_8bit=True)  # requires bitsandbytes
```

The quantized model is sensitive to input types and CUDA handling. To avoid potential issues, we recommend explicitly converting input IDs to CUDA using: `inputs.input_ids.to('cuda')`

## Evaluation

Additional tools for evaluating OLMo models are available at the [OLMo Eval](https://github.com/allenai/OLMo-eval) and [olmes](https://github.com/allenai/olmes) repositories.

## Modal.com Hosting

An example script is provided for hosting an OLMo 2 model on Modal.com using the OpenAI API in `./scripts/olmo2_modal_openai.py`.
To run that:

1. Follow the instructions under Getting Started in [the Modal.com Guide](https://modal.com/docs/guide) to install
the Modal library and command line tools.</li>
2. Follow the instructions under [Secrets](https://modal.com/docs/guide/secrets) in the Modal.com Guide to create a Modal secret named "example-secret-token"
that defines a value for the variable MODAL_TOKEN for your server.</li>
3. Then run
```bash
modal deploy ./scripts/olmo2_modal_openai.py
```

You can check your endpoint using curl similar to the following:
```bash
curl -X POST \
  -H "Authorization: Bearer [the secret token from above]" \
  -H "Content-Type: application/json" \
  -d @body.json \
  https://[the web endpoint modal creates above]/v1/chat/completions
```

where `body.json` is of the form:
```
{
    "model": "OLMo-2-1124-13B-Instruct",
    "messages": [
        {
            "role": "user",
            "content": "Who was Alan Turing?"
        }
      ],
    "max_tokens": 100,
    "temperature": 0.9,
    "stream": true
}
```


## Citing

```bibtex
@misc{olmo20242olmo2furious,
      title={2 OLMo 2 Furious}, 
      author={Team OLMo and Pete Walsh and Luca Soldaini and Dirk Groeneveld and Kyle Lo and Shane Arora and Akshita Bhagia and Yuling Gu and Shengyi Huang and Matt Jordan and Nathan Lambert and Dustin Schwenk and Oyvind Tafjord and Taira Anderson and David Atkinson and Faeze Brahman and Christopher Clark and Pradeep Dasigi and Nouha Dziri and Michal Guerquin and Hamish Ivison and Pang Wei Koh and Jiacheng Liu and Saumya Malik and William Merrill and Lester James V. Miranda and Jacob Morrison and Tyler Murray and Crystal Nam and Valentina Pyatkin and Aman Rangapur and Michael Schmitz and Sam Skjonsberg and David Wadden and Christopher Wilhelm and Michael Wilson and Luke Zettlemoyer and Ali Farhadi and Noah A. Smith and Hannaneh Hajishirzi},
      year={2024},
      eprint={2501.00656},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2501.00656}, 
}
```