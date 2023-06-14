
## Goal

Apply model-specific methods to 
1. Serve a model on a single GPU
2. Easily scale to multiple GPUs
3. Improve latency


## Methods

### Current (ongoing)

* 4-bit quantization using AutoGPTQ

### Future (maybe)

* Quantization-aware training.
* Using `torch.compile`.
* `hidet`
* `qLoRA`
* Caching key/value matrices? Backend server trickery?
* ONYX Runtime


## Scale to multiple gpus

TODO

### References


