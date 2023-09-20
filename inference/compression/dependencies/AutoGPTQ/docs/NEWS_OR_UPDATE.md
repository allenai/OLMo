## <center>News or Update</center>

- 2023-08-23 - (News) - 🤗 Transformers, optimum and peft have integrated `auto-gptq`, so now running and training GPTQ models can be more available to everyone! See [this blog](https://huggingface.co/blog/gptq-integration) and it's resources for more details!
- 2023-08-21 - (News) - Team of Qwen officially released 4bit quantized version of Qwen-7B based on `auto-gptq`, and provided [a detailed benchmark results](https://huggingface.co/Qwen/Qwen-7B-Chat-Int4#%E9%87%8F%E5%8C%96-quantization)
- 2023-08-06 - (Update) - Support exllama's q4 CUDA kernel to have at least 1.3x speed up for int4 quantized models when doing inference.
- 2023-08-04 - (Update) - Support RoCm so that AMD GPU users can use auto-gptq with CUDA extensions.
- 2023-07-26 - (Update) - An elegant [PPL benchmark script](examples/benchmark/perplexity.py) to get results that can be fairly compared with other libraries such as `llama.cpp`.
- 2023-06-05 - (Update) - Integrate with 🤗 peft to use gptq quantized model to train adapters, support LoRA, AdaLoRA, AdaptionPrompt, etc.
- 2023-05-30 - (Update) - support download/upload quantized model from/to 🤗 Hub.
- 2023-05-27 - (Update) - Support quantization and inference for `gpt_bigcode`, `codegen` and `RefineWeb/RefineWebModel`(falcon) model types.
- 2023-05-04 - (Update) - Support using faster cuda kernel when `not desc_act or group_size == -1`
- 2023-04-29 - (Update) - Support loading quantized model from arbitrary quantize_config and model_basename.
- 2023-04-28 - (Update) - Support CPU offload and quantize/inference on multiple devices, support `gpt2` type models.
- 2023-04-26 - (Update) - Using `triton` to speed up inference is now supported.
- 2023-04-25 - (News&Update) - [MOSS](https://github.com/OpenLMLab/MOSS) is an open-source tool-augmented conversational language model from Fudan University, quantization is now supported in AutoGPTQ.
- 2023-04-23 - (Update) - Support evaluation on multiple (down-stream) tasks such as: language-modeling, text-classification, text-summarization.
- 2023-04-22 - (News) - qwopqwop200's [AutoGPTQ-triton](https://github.com/qwopqwop200/AutoGPTQ-triton) provides faster speed to integrate with quantized model, for everyone who can access to triton, try and enjoy yourself!
- 2023-04-20 - (News) - AutoGPTQ is automatically compatible with Stability-AI's newly released `gpt_neox` type model family [StableLM](https://github.com/Stability-AI/StableLM).
- 2023-04-16 - (Update) - Support quantization and inference for `bloom`, `gpt_neox`, `gptj`, `llama` and `opt`.