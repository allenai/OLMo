import math

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    apply_rotary_pos_emb,
)

from ..utils.import_utils import compare_pytorch_version, dynamically_import_QuantLinear
from ._fused_base import FusedBaseAttentionModule


class FusedLlamaAttentionForQuantizedModel(FusedBaseAttentionModule):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        hidden_size,
        num_heads,
        qkv_proj,
        o_proj,
        rotary_emb,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        if self.head_dim * num_heads != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {num_heads})."
            )
        self.qkv_proj = qkv_proj
        self.o_proj = o_proj
        self.rotary_emb = rotary_emb

    def _shape(self, tensor, seq_len, bsz):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states,
        past_key_value=None,
        attention_mask=None,
        position_ids=None,
        output_attentions=False,
        use_cache=False,
        **kwargs,
    ):
        """Input shape: Batch x Time x Channel"""

        bsz, q_len, _ = hidden_states.size()

        qkv_states = self.qkv_proj(hidden_states)
        query_states, key_states, value_states = torch.split(qkv_states, self.hidden_size, dim=2)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]

        is_causal = past_key_value is None
        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        if use_cache:
            # Since qkv_proj is fused, query_states etc will hold a reference to the original qkv_states tensor
            # which can cause excessive memory usage by the cache. `contiguous` is a convenient way to workaround this.
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        past_key_value = (key_states, value_states) if use_cache else None

        if compare_pytorch_version("v2.0.0", op="ge"):
            attn_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=None if is_causal else attention_mask,
                is_causal=is_causal,
            )
            attn_weights = None
        else:
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz * self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )

            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                    )
                attn_weights = attn_weights + attention_mask
                attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value

    @classmethod
    def inject_to_model(
        cls,
        model,
        use_triton=False,
        group_size=-1,
        use_cuda_fp16=True,
        desc_act=False,
        trainable=False,
        bits: int = 4,
        disable_exllama=False,
        **kwargs,
    ):
        """
        Replace all LlamaAttention modules with QuantLlamaAttention modules, fusing the q, k, v projections.
        """
        QuantLinear = dynamically_import_QuantLinear(
            use_triton=use_triton,
            desc_act=desc_act,
            group_size=group_size,
            bits=bits,
            disable_exllama=disable_exllama,
        )

        for name, m in model.named_modules():
            if not isinstance(m, LlamaAttention):
                continue

            q_proj = m.q_proj
            k_proj = m.k_proj
            v_proj = m.v_proj

            qweights = torch.cat([q_proj.qweight, k_proj.qweight, v_proj.qweight], dim=1)
            qzeros = torch.cat([q_proj.qzeros, k_proj.qzeros, v_proj.qzeros], dim=1)
            scales = torch.cat([q_proj.scales, k_proj.scales, v_proj.scales], dim=1)

            if QuantLinear.QUANT_TYPE == "exllama":
                if desc_act:
                    # TODO: support it. The issue lies maybe in the line:
                    # int groups = qzeros.size(0);
                    # in exllama_ext.cpp
                    raise ValueError(
                        "Exllama kernel does not support query/key/value fusion with act-order. Please either use inject_fused_attention=False or disable_exllama=True."
                    )
                else:
                    g_idx = None
            else:
                g_idx = torch.cat([q_proj.g_idx, k_proj.g_idx, v_proj.g_idx], dim=0)

            bias = torch.cat([q_proj.bias, k_proj.bias, v_proj.bias], dim=0) if q_proj.bias is not None else None

            qlinear_args = (
                q_proj.bits,
                q_proj.group_size,
                q_proj.infeatures,
                q_proj.outfeatures + k_proj.outfeatures + v_proj.outfeatures,
                True if q_proj.bias is not None else False,
            )
            qlinear_kwargs = {"trainable": trainable}
            if (not desc_act or group_size == -1) and not use_triton:
                qlinear_kwargs["use_cuda_fp16"] = use_cuda_fp16
            qkv_layer = QuantLinear(*qlinear_args, **qlinear_kwargs)
            qkv_layer.qweight = qweights
            qkv_layer.qzeros = qzeros
            qkv_layer.scales = scales
            qkv_layer.g_idx = g_idx
            qkv_layer.bias = bias

            attn = cls(m.hidden_size, m.num_heads, qkv_layer, m.o_proj, m.rotary_emb)

            if "." in name:
                parent_name = name.rsplit(".", 1)[0]
                child_name = name[len(parent_name) + 1 :]
                parent = model.get_submodule(parent_name)
            else:
                parent_name = ""
                parent = model
                child_name = name

            setattr(parent, child_name, attn)


__all__ = ["FusedLlamaAttentionForQuantizedModel"]
