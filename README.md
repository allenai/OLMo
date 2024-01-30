<div align="center">
  <img src="https://github.com/allenai/OLMo/assets/8812459/774ac485-a535-4768-8f7c-db7be20f5cc3" width="300"/>
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
</p>

OLMo is a repository for training and using AI2's state-of-the-art open language models. 
It is built by scientists, for scientists.

## Installation

First install [PyTorch](https://pytorch.org) according to the instructions specific to your operating system.

To install from source (recommended for training/fine-tuning) run:

```bash
git clone https://github.com/allenai/OLMo.git
cd OLMo
pip install -e .
```

Otherwise you can install the model code by itself directly from PyPI with:

```bash
pip install ai2-olmo
```

## Models overview

The core models in the OLMo family released so far are (all trained on the [Dolma dataset](https://huggingface.co/datasets/allenai/dolma)): 
| Model | Training Tokens | Context Length |
|------|--------|---------|
| [OLMo 1B](https://huggingface.co/allenai/OLMo-1B)   | 3 Trillion | 2048  |
| [OLMo 7B](https://huggingface.co/allenai/OLMo-7B) | 2.5 Trillion   |  2048  |
| [OLMo 7B Twin 2T](https://huggingface.co/allenai/OLMo-7B-Twin-2T) | 2 Trillion  |   2048  |


## Fine-tuning

To fine-tune an OLMo model using our trainer you'll first need to prepare your dataset by tokenizing it and saving the tokens IDs to a flat numpy memory-mapped array. See [`scripts/prepare_tulu_data.py`](./scripts/prepare_tulu_data.py) for an example with the Tulu V2 dataset, which can be easily modified for other datasets.

Next, prepare your training config. There are many examples in the [`configs/`](./configs) directory that you can use as a starting point. The most important thing is to make sure the model parameters (the `model` field in the config) match up with the checkpoint you're starting from. To be safe you can always start from the config that comes with the model checkpoint. At a minimum you'll need to make the following changes to the config or provide the corresponding overrides from the command line:

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


## Inference

You can utilize our HuggingFace integration to run inference on the olmo checkpoints:

```python
from hf_olmo import * # registers the Auto* classes

from transformers import AutoModelForCausalLM, AutoTokenizer

olmo = AutoModelForCausalLM.from_pretrained("allenai/OLMo-7B")
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-7B")

message = ["Language modeling is "]
inputs = tokenizer(message, return_tensors='pt', return_token_type_ids=False)
response = olmo.generate(**inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
print(tokenizer.batch_decode(response, skip_special_tokens=True)[0])
```

Alternatively, with the huggingface pipeline abstraction:

```python
from transformers import pipeline
olmo_pipe = pipeline("text-generation", model="allenai/OLMo-7B")
print(olmo_pipe("Language modeling is"))
```


### Inference on finetuned checkpoints

If you finetune the model using the code above, you can use the conversion script to convert a native OLMo checkpoint to a HuggingFace-compatible checkpoint

```bash
python hf_olmo/convert_olmo_to_hf.py --checkpoint-dir /path/to/checkpoint
```

### Quantization

```python
olmo = AutoModelForCausalLM.from_pretrained("allenai/OLMo-7B", torch_dtype=torch.float16, load_in_8bit=True)  # requires bitsandbytes
```

The quantized model is more sensitive to typing / cuda, so it is recommended to pass the inputs as inputs.input_ids.to('cuda') to avoid potential issues.


## Evaluation

Additional tools for evaluating OLMo models are available at the [OLMo Eval](https://github.com/allenai/ai2-olmo-eval) repo.
