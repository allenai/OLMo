from ._base import *


class BaiChuanGPTQForCausalLM(BaseGPTQForCausalLM):
    layer_type = "DecoderLayer"
    layers_block_name = "model.layers"
    outside_layer_modules = ["model.embed_tokens", "model.norm"]
    inside_layer_modules = [
        ["self_attn.W_pack"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"]
    ]


__all__ = ["BaiChuanGPTQForCausalLM"]
