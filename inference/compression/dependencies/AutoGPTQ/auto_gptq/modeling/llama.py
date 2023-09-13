from logging import getLogger

from ._base import *
from ..utils.import_utils import compare_transformers_version

if compare_transformers_version("v4.28.0", op="ge"):
    from ..nn_modules.fused_llama_attn import FusedLlamaAttentionForQuantizedModel
    from ..nn_modules.fused_llama_mlp import FusedLlamaMLPForQuantizedModel
else:
    FusedLlamaAttentionForQuantizedModel = None
    FusedLlamaMLPForQuantizedModel = None

logger = getLogger(__name__)


class LlamaGPTQForCausalLM(BaseGPTQForCausalLM):
    layer_type = "LlamaDecoderLayer"
    layers_block_name = "model.layers"
    outside_layer_modules = ["model.embed_tokens", "model.norm"]
    inside_layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"]
    ]

    fused_attn_module_type = FusedLlamaAttentionForQuantizedModel
    fused_mlp_module_type = FusedLlamaMLPForQuantizedModel


__all__ = ["LlamaGPTQForCausalLM"]
