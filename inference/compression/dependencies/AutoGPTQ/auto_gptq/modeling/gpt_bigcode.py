from auto_gptq.modeling import BaseGPTQForCausalLM


class GPTBigCodeGPTQForCausalLM(BaseGPTQForCausalLM):
    layer_type = "GPTBigCodeBlock"
    layers_block_name = "transformer.h"
    outside_layer_modules = [
        "transformer.wpe", "transformer.wte", "transformer.ln_f"
    ]
    inside_layer_modules = [
        ["attn.c_attn"],
        ["attn.c_proj"],
        ["mlp.c_fc"],
        ["mlp.c_proj"]
    ]

__all__ = ["GPTBigCodeGPTQForCausalLM"]