from typing import *

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers.models.gptj.modeling_gptj import GPTJAttention

from ._fused_base import FusedBaseAttentionModule
from ..utils.import_utils import compare_pytorch_version, dynamically_import_QuantLinear


def fixed_pos_embedding(x, seq_dim=1, seq_len=None):
    dim = x.shape[-1]
    if seq_len is None:
        seq_len = x.shape[seq_dim]
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))
    sinusoid_inp = (
        torch.einsum("i , j -> i j", torch.arange(seq_len, dtype=torch.float), inv_freq).to(x.device).float()
    )
    return torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)


def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')


def duplicate_interleave(m):
    """
    A simple version of `torch.repeat_interleave` for duplicating a matrix while interleaving the copy.
    """
    dim0 = m.shape[0]
    m = m.view(-1, 1)  # flatten the matrix
    m = m.repeat(1, 2)  # repeat all elements into the 2nd dimension
    m = m.view(dim0, -1)  # reshape into a matrix, interleaving the copy
    return m


def apply_rotary_pos_emb(x, sincos, offset=0):
    sin, cos = (duplicate_interleave(t)[None, offset : x.shape[1] + offset, None, :] for t in sincos)
    # einsum notation for lambda t: repeat(t[offset:x.shape[1]+offset,:], "n d -> () n () (d j)", j=2)
    return (x * cos) + (rotate_every_two(x) * sin)


class FusedGPTJAttentionForQuantizedModel(FusedBaseAttentionModule):
    def __init__(self, config):
        super().__init__()

        max_positions = config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e9))

        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.attn_dropout_p = config.attn_pdrop
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.embed_dim = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_attention_heads
        if self.head_dim * self.num_attention_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_attention_heads (got `embed_dim`: {self.embed_dim} and"
                f" `num_attention_heads`: {self.num_attention_heads})."
            )
        self.scale_attn = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)).to(torch.get_default_dtype())

        self.qkv_proj = nn.Linear(self.embed_dim, self.embed_dim * 3, bias=False)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.rotary_dim = config.rotary_dim

    def _split_heads(self, qkv):
        """
        Splits hidden dim into attn_head_size and num_attention_heads
        """
        new_shape = qkv.size()[:-1] + (3, self.num_attention_heads, self.head_dim)
        qkv = qkv.view(new_shape)  # (batch, seq_length, 3, head, head_features)
        query = qkv[:, :, 0]
        key = qkv[:, :, 1]
        value = qkv[:, :, 2]

        return query, key, value

    def _merge_heads(self, tensor, num_attention_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into hidden dim
        """
        if len(tensor.shape) == 5:
            tensor = tensor.permute(0, 1, 3, 2, 4).contiguous()
        elif len(tensor.shape) == 4:
            tensor = tensor.permute(0, 2, 1, 3).contiguous()
        else:
            raise ValueError(f"Input tensor rank should be one of [4, 5], but is: {len(tensor.shape)}")
        new_shape = tensor.size()[:-2] + (num_attention_heads * attn_head_size,)
        return tensor.view(new_shape)

    def _attn(
        self,
        query,
        key,
        value,
        attention_mask=None,
        head_mask=None,
    ):
        # compute causal mask from causal mask buffer
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[:, :, key_length - query_length: key_length, :key_length]

        # Keep the attention weights computation in fp32 to avoid overflow issues
        query = query.to(torch.float32)
        key = key.to(torch.float32)

        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        mask_value = torch.finfo(attn_weights.dtype).min
        # Need to be a tensor, otherwise we get error: `RuntimeError: expected scalar type float but found double`.
        # Need to be on the same device, otherwise `RuntimeError: ..., x and y to be on the same device`
        mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)
        attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        attn_weights = attn_weights / self.scale_attn

        if attention_mask is not None:
            # Apply the attention mask
            attn_weights = attn_weights + attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.to(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[
        Tuple[torch.Tensor, Tuple[torch.Tensor]],
        Optional[Tuple[torch.Tensor, Tuple[torch.Tensor], Tuple[torch.Tensor, ...]]],
    ]:
        query, key, value = self._split_heads(self.qkv_proj(hidden_states))

        seq_len = key.shape[1]
        offset = 0

        if layer_past is not None:
            offset = layer_past[0].shape[-2]
            seq_len += offset

        if self.rotary_dim is not None:
            k_rot = key[:, :, :, : self.rotary_dim]
            k_pass = key[:, :, :, self.rotary_dim:]

            q_rot = query[:, :, :, : self.rotary_dim]
            q_pass = query[:, :, :, self.rotary_dim:]

            sincos = fixed_pos_embedding(k_rot, 1, seq_len=seq_len)
            k_rot = apply_rotary_pos_emb(k_rot, sincos, offset=offset)
            q_rot = apply_rotary_pos_emb(q_rot, sincos, offset=offset)

            key = torch.cat([k_rot, k_pass], dim=-1)
            query = torch.cat([q_rot, q_pass], dim=-1)
        else:
            sincos = fixed_pos_embedding(key, 1, seq_len=seq_len)
            key = apply_rotary_pos_emb(key, sincos, offset=offset)
            query = apply_rotary_pos_emb(query, sincos, offset=offset)

        key = key.permute(0, 2, 1, 3)
        query = query.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)

        is_causal = layer_past is None
        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()
            present = (key, value)
        else:
            present = None

        # compute self-attention: V x Softmax(QK^T)
        if compare_pytorch_version("v2.0.0", op="ge"):
            attn_output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                attn_mask=None if is_causal else attention_mask,
                dropout_p=self.attn_dropout_p,
                is_causal=is_causal
            )
            attn_weights = None
        else:
            attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, present, (attentions)

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
        **kwargs
    ):
        config = model.config
        QuantLinear = dynamically_import_QuantLinear(use_triton=use_triton, desc_act=desc_act, group_size=group_size, bits=bits, disable_exllama=disable_exllama)

        for name, m in model.named_modules():
            if not isinstance(m, GPTJAttention):
                continue

            attn = cls(config).to(device=next(m.buffers()).device)

            q_proj = m.q_proj
            k_proj = m.k_proj
            v_proj = m.v_proj

            qweights = torch.cat([q_proj.qweight, k_proj.qweight, v_proj.qweight], dim=1)
            qzeros = torch.cat([q_proj.qzeros, k_proj.qzeros, v_proj.qzeros], dim=1)
            scales = torch.cat([q_proj.scales, k_proj.scales, v_proj.scales], dim=1)

            if QuantLinear.QUANT_TYPE == "exllama":
                if desc_act:
                    # See fused_llama_attn.py comment
                    raise ValueError("Exllama kernel does not support query/key/value fusion with act-order. Please either use inject_fused_attention=False or disable_exllama=True.")
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
            qkv_proj = QuantLinear(*qlinear_args, **qlinear_kwargs)
            qkv_proj.qweight = qweights
            qkv_proj.qzeros = qzeros
            qkv_proj.scales = scales
            qkv_proj.g_idx = g_idx
            qkv_proj.bias = bias

            if '.' in name:
                parent_name = name.rsplit('.', 1)[0]
                child_name = name[len(parent_name) + 1:]
                parent = model.get_submodule(parent_name)
            else:
                parent_name = ''
                parent = model
                child_name = name

            attn.qkv_proj = qkv_proj
            attn.out_proj = m.out_proj

            setattr(parent, child_name, attn)
            del m


__all__ = ["FusedGPTJAttentionForQuantizedModel"]
