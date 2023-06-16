from auto_gptq.modeling._base import BaseGPTQForCausalLM

# NOTE: In progress; may change if OLMo model is updated.


class OlmoGPTQForCausalLM(BaseGPTQForCausalLM):
    # Attribute name of Transformer layer block.
    layers_block_name = "transformer.blocks"  # NOTE(wadden) Correct

    # Attribute names of other modules in the same level as transformer layer block.
    # Excludes `transformer.emb_drop`, which has no parameters; this is consistent with
    # GPT-J.

    # TODO(wadden) Figure out if I need wpe
    outside_layer_modules = ["transformer.wte", "transformer.ln_f", "transformer.wpe"]

    # Attribute names of linear layers in the transformer layer module.
    # These should be ordered as they are executed, which is usually:
    # - Attention Q / K / V projection
    # - Attention output projection
    # - MLP projection
    # - MLP output

    # NOTE(wadden) For other models, layer norm, dropout, and activation functions are
    # not included; I do the same here.
    # TODO deal with case of fused attention.
    inside_layer_modules = [
        ["transformer.blocks.att_proj"],
        ["transformer.blocks.att_out"],
        ["transformer.blocks.ff_proj"],
        ["transformer.blocks.ff_out"],
    ]


__all__ = ["OlmoGPTQForCausalLM"]
