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

The core models in the OLMo family released are: 
| Model | Training Tokens | Context Length | Training Config | W&B Logs |
|-------|-----------------|:--------------:|-----------------|----------|
| [OLMo2 7B](https://huggingface.co/allenai/OLMo-2-1124-7B) | 4 Trillion | 4096 | [configs/official-1124/OLMo2-7B-stage1.yaml](https://github.com/allenai/OLMo/blob/main/configs/official-1124/OLMo2-7B-stage1.yaml) | wandb.ai/…/OLMo2-7B (link to come)
| [OLMo2 13B](https://huggingface.co/allenai/OLMo-2-1124-13B) | 5 Trillion | 4096 | [configs/official-1124/OLMo2-12B-stage1.yaml](https://github.com/allenai/OLMo/blob/main/configs/official-1124/OLMo2-13B-stage1.yaml) | wandb.ai/…/OLMo2-13B (link to come)

For instruction tuned variants of these models, go to
 * [OLMo2 7B Instruct](https://huggingface.co/allenai/OLMo-2-1124-7B-Instruct)
 * [OLMo2 13B Instruct](https://huggingface.co/allenai/OLMo-2-1124-13B-Instruct)

> ☨ *See [Inspecting training data](#inspecting-training-data) below for usage.*

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

Additional tools for evaluating OLMo models are available at the [OLMo Eval](https://github.com/allenai/OLMo-eval) repo.

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
