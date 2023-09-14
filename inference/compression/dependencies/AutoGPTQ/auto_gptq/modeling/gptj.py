from ..nn_modules.fused_gptj_attn import FusedGPTJAttentionForQuantizedModel
from ._base import *


class GPTJGPTQForCausalLM(BaseGPTQForCausalLM):
    layer_type = "GPTJBlock"
    layers_block_name = "transformer.h"
    outside_layer_modules = ["transformer.wte", "transformer.ln_f"]
    inside_layer_modules = [
        ["attn.k_proj", "attn.v_proj", "attn.q_proj"],
        ["attn.out_proj"],
        ["mlp.fc_in"],
        ["mlp.fc_out"],
    ]

    fused_attn_module_type = FusedGPTJAttentionForQuantizedModel


__all__ = ["GPTJGPTQForCausalLM"]
