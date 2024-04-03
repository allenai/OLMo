from ._base import *


class OLMoGPTQForCausalLM(BaseGPTQForCausalLM):
    # Attribute name of Transformer layer block.
    layers_block_name = "model.transformer.blocks"

    # Excludes `transformer.emb_drop`, which has no parameters; this is consistent with
    # GPT-J.
    outside_layer_modules = ["model.transformer.wte", "model.transformer.ln_f"]

    # Attribute names of linear layers in the transformer layer module.
    # These should be ordered as they are executed, which is usually:
    # - Attention Q / K / V projection
    # - Attention output projection
    # - MLP projection
    # - MLP output

    inside_layer_modules = [["att_proj"], ["attn_out"], ["ff_proj"], ["ff_out"]]


__all__ = ["OLMoGPTQForCausalLM"]
