from ._base import *


class QwenGPTQForCausalLM(BaseGPTQForCausalLM):
    layer_type = "QWenBlock"
    layers_block_name = "transformer.h"
    outside_layer_modules = ["transformer.wte", "transformer.wpe", "transformer.ln_f"]
    inside_layer_modules = [
        ["attn.c_attn"],
        ["attn.c_proj"],
        ["mlp.w1", "mlp.w2"],
        ["mlp.c_proj"]
    ]


__all__ = ["QwenGPTQForCausalLM"]
