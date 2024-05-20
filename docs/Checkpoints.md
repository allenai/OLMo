Checkpoints
===

There are 3 types of OLMo checkpoints.

1. OLMo (standard) checkpoints. These checkpoints can be produced and used by the code in this repo. "OLMo checkpoints" will typically refer to these checkpoints.
2. Transformers checkpoints. These checkpoints can be produced and used via the OLMo implementation in the [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) library. As we continue to develop and improve OLMo, our implementation in this repo may temporariy become incompatible with the implementation in the Transformer library.
3. HF OLMo checkpoints. These checkpoints can be produced and used via the `hf_olmo` package. The `hf_olmo` package provides basic Transformers functionality while always staying compatible with the OLMo library.

OLMo (standard) checkpoints
---

There are 2 categories of OLMo checkpoints:
- unsharded: a complete checkpoint in a standard form;
- sharded: a checkpoint that has been broken down into smaller components, for easier use in our multi-node training.

Unless otherwise specified, an OLMo checkpoint is assumed to be unsharded. OLMo sharded and unsharded checkpoints can be used with the pretraining/fine-tuning script provided in this repo.

#### Sharded OLMo Checkpoints

There are currently 4 types of sharded checkpoints:
- torch_legacy,
- torch_new,
- local,
- olmo_core.

We are still working on improving sharded checkpointing and thus do not have any guidelines for using them at present. A sharded checkpoint can be converted to an unsharded checkpoint using [unshard.py](https://github.com/allenai/OLMo/blob/main/scripts/unshard.py).

Transformers Checkpoints
---

These checkpoints can be used with the OLMo implementation in the Transformers library. Since the OLMo implementation is integrated into the library, OLMo models support most Transformers model functionality. These checkpoints cannot run the pretraining or fine-tuning provided in this repo.

Transformers checkpoints can be found in HF Hub repos that end in `-hf` (e.g. [OLMo-1.7-7B-hf](https://huggingface.co/allenai/OLMo-1.7-7B-hf)). An OLMo checkpoint can be converted into its Transformers equivalent using [convert_olmo_to_hf_new.py](https://github.com/allenai/OLMo/blob/main/scripts/convert_olmo_to_hf_new.py).

*Warning*: As we continue to develop and improve OLMo, our implementation in this repo may become incompatible with the implementation in the Transformer library. During these periods, OLMo checkpoints may not be convertible to Transformers checkpoint. At present, all OLMo checkpoints of our officially released models are convertible to Transformers checkpoints.

HF OLMo checkpoints
---

These checkpoints can be used with the Transformers-style OLMo implementation in the `hf_olmo` package. This implementation has only partial support for Transformers functionality. Consequently, we recommend using Transformers checkpoints over these if available. These checkpoints cannot run the pretraining or fine-tuning provided in this repo.

The following checkpoints on HF Hub are HF OLMo checkpoints:
- [OLMo-1.7-7B](https://huggingface.co/allenai/OLMo-1.7-7B)
- [OLMo-1B](https://huggingface.co/allenai/OLMo-1B)
- [OLMo-7B](https://huggingface.co/allenai/OLMo-7B)
- [OLMo-7B-Twin-2T](https://huggingface.co/allenai/OLMo-7B-Twin-2T)

An OLMo checkpoint can be converted into its HF OLMo equivalent using [convert_olmo_to_hf.py](https://github.com/allenai/OLMo/blob/main/hf_olmo/convert_olmo_to_hf.py).