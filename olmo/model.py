"""
Adapted from
[MosaiclML](https://github.com/mosaicml/examples.git) and
[minGPT](https://github.com/karpathy/minGPT.git)
"""

from __future__ import annotations

import logging
import math
import os
from abc import abstractmethod
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple, cast

import torch
import torch.backends.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

from .aliases import PathOrStr
from .beam_search import BeamSearch, Constraint, FinalSequenceScorer, Sampler
from .config import ActivationType, BlockType, LayerNormType, ModelConfig
from .exceptions import OlmoConfigurationError
from .initialization import init_weights

__all__ = [
    "LayerNormBase",
    "LayerNorm",
    "RMSLayerNorm",
    "RotaryEmbedding",
    "Activation",
    "GELU",
    "ReLU",
    "SwiGLU",
    "OlmoBlock",
    "OlmoSequentialBlock",
    "OlmoParallelBlock",
    "Olmo",
    "OlmoOutput",
    "OlmoGenerateOutput",
]


log = logging.getLogger(__name__)


class LayerNormBase(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def build(cls, config: ModelConfig, size: Optional[int] = None, **kwargs) -> LayerNormBase:
        if config.layer_norm_type == LayerNormType.default:
            return LayerNorm(config, size=size, low_precision=False, **kwargs)
        elif config.layer_norm_type == LayerNormType.low_precision:
            return LayerNorm(config, size=size, low_precision=True, **kwargs)
        elif config.layer_norm_type == LayerNormType.rms:
            return RMSLayerNorm(config, size=size, low_precision=False, **kwargs)
        elif config.layer_norm_type == LayerNormType.low_precision_rms:
            return RMSLayerNorm(config, size=size, low_precision=True, **kwargs)
        else:
            raise NotImplementedError(f"Not sure how to handle '{config.layer_norm_type}' LayerNorm type")

    def _cast_if_autocast_enabled(self, tensor: torch.Tensor) -> torch.Tensor:
        if torch.is_autocast_enabled():
            if tensor.device.type == "cuda":
                dtype = torch.get_autocast_gpu_dtype()
            elif tensor.device.type == "cpu":
                dtype = torch.get_autocast_cpu_dtype()
            else:
                raise NotImplementedError()
            return tensor.to(dtype=dtype)
        return tensor

    def reset_parameters(self):
        if self.weight is not None:
            torch.nn.init.ones_(self.weight)  # type: ignore
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)  # type: ignore


class LayerNorm(LayerNormBase):
    """
    The default :class:`LayerNorm` implementation which can optionally run in low precision.
    """

    def __init__(
        self,
        config: ModelConfig,
        size: Optional[int] = None,
        low_precision: bool = False,
        elementwise_affine: Optional[bool] = None,
    ):
        super().__init__(config)
        self.normalized_shape = (size or config.d_model,)
        self.eps = 1e-05
        self.low_precision = low_precision

        # We always have weight and bias even if they are turned off/set to 1 and 0, because ROCm has a
        # bug where F.layer_norm() crashes during the backwards pass when no bias was given.
        # When they are turned off, they need to be buffers, because FSDP can't handle the situation
        # where some parameters don't require gradients.

        if elementwise_affine is None:
            elementwise_affine = self.config.layer_norm_with_affine
        weight = torch.ones(self.normalized_shape, device=config.init_device)
        if elementwise_affine:
            self.register_parameter("weight", nn.Parameter(weight))
        else:
            self.register_buffer("weight", weight, persistent=False)

        needs_bias = elementwise_affine and self.config.include_bias
        bias = torch.zeros(self.normalized_shape, device=config.init_device)
        if needs_bias:
            self.register_parameter("bias", nn.Parameter(bias))
        else:
            self.register_buffer("bias", bias, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.low_precision:
            module_device = x.device
            downcast_x = self._cast_if_autocast_enabled(x)
            downcast_weight = (
                self._cast_if_autocast_enabled(self.weight) if self.weight is not None else self.weight
            )
            downcast_bias = self._cast_if_autocast_enabled(self.bias) if self.bias is not None else self.bias
            with torch.autocast(enabled=False, device_type=module_device.type):
                return F.layer_norm(
                    downcast_x, self.normalized_shape, weight=downcast_weight, bias=downcast_bias, eps=self.eps
                )
        else:
            return F.layer_norm(x, self.normalized_shape, weight=self.weight, bias=self.bias, eps=self.eps)


class RMSLayerNorm(LayerNorm):
    """
    RMS layer norm, a simplified :class:`LayerNorm` implementation that can optionally run
    in low-precision.
    """

    def __init__(
        self,
        config: ModelConfig,
        size: Optional[int] = None,
        low_precision: bool = False,
        elementwise_affine: Optional[bool] = None,
    ):
        super().__init__(config)
        self.eps = 1e-08
        self.size = size or config.d_model

        if elementwise_affine is None:
            elementwise_affine = self.config.layer_norm_with_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.config.d_model))
            if self.config.include_bias:
                self.bias = nn.Parameter(torch.zeros(self.config.d_model))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.low_precision = low_precision

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.low_precision:
            module_device = x.device
            downcast_x = self._cast_if_autocast_enabled(x)
            downcast_weight = self._cast_if_autocast_enabled(self.weight)
            downcast_bias = self._cast_if_autocast_enabled(self.bias) if self.config.include_bias else None
            with torch.autocast(enabled=False, device_type=module_device.type):
                return self.rms_norm(downcast_x, downcast_weight, downcast_bias)
        else:
            return self.rms_norm(x, self.weight, self.bias if self.config.include_bias else None)

    def rms_norm(
        self, x: torch.Tensor, weight: Optional[torch.Tensor], bias: Optional[torch.Tensor]
    ) -> torch.Tensor:
        norm_x = x.norm(2, dim=-1, keepdim=True)

        rms_x = norm_x * self.size ** (-1.0 / 2)
        x_normed = x / (rms_x + self.eps)

        if weight is not None:
            if bias is not None:
                return weight * x_normed + self.bias
            else:
                return weight * x_normed


class RotaryEmbedding(nn.Module):
    """
    [Rotary positional embeddings (RoPE)](https://arxiv.org/abs/2104.09864).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        dim = config.d_model // config.n_heads
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=config.init_device).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, max_seq_len, *, device):
        seq = torch.arange(max_seq_len, device=device, dtype=self.inv_freq.dtype)  # type: ignore
        freqs = einsum("i , j -> i j", seq, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    B, nh, T, hs = x.size()
    x = x.view(B, nh, T, 2, hs // 2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(pos: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    out = (t * pos.cos()) + (rotate_half(t) * pos.sin())
    return out.to(t.dtype)


class Activation(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def output_multiplier(self) -> float:
        raise NotImplementedError

    @classmethod
    def build(cls, config: ModelConfig) -> Activation:
        if config.activation_type == ActivationType.gelu:
            return cast(Activation, GELU(approximate="none"))
        elif config.activation_type == ActivationType.relu:
            return cast(Activation, ReLU(inplace=False))
        elif config.activation_type == ActivationType.swiglu:
            return SwiGLU(config)
        else:
            raise NotImplementedError(f"not sure how to handle activation type '{config.activation_type}'")


class GELU(nn.GELU):
    @property
    def output_multiplier(self) -> float:
        return 1.0


class ReLU(nn.ReLU):
    @property
    def output_multiplier(self) -> float:
        return 1.0


class SwiGLU(Activation):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return F.silu(gate) * x

    @property
    def output_multiplier(self) -> float:
        return 0.5


class OlmoBlock(nn.Module):
    """
    A base class for transformer block implementations.
    """

    def __init__(self, layer_id: int, config: ModelConfig):
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        assert config.d_model % config.n_heads == 0

        # Dropout.
        self.dropout = nn.Dropout(config.residual_dropout)

        # Layer norms.
        self.k_norm: Optional[LayerNormBase] = None
        self.q_norm: Optional[LayerNormBase] = None
        if config.attention_layer_norm:
            self.k_norm = LayerNormBase.build(
                config,
                size=config.d_model // config.n_heads if config.multi_query_attention else None,
                elementwise_affine=True,
            )
            self.q_norm = LayerNormBase.build(config, elementwise_affine=True)

        # Activation function.
        self.act = Activation.build(config)
        assert (self.act.output_multiplier * config.mlp_ratio * config.d_model) % 1 == 0

        # Attention output projection.
        self.attn_out = nn.Linear(
            config.d_model, config.d_model, bias=config.include_bias, device=config.init_device
        )

        # Feed-forward output projection.
        self.ff_out = nn.Linear(
            int(self.act.output_multiplier * config.mlp_ratio * config.d_model),
            config.d_model,
            bias=config.include_bias,
            device=config.init_device,
        )
        self.ff_out._is_residual = True  # type: ignore

        # Rotary embeddings.
        if self.config.rope:
            self.rotary_emb = RotaryEmbedding(config)
            self.register_buffer(
                "pos_emb", self.rotary_emb(config.max_sequence_length, device=config.init_device), persistent=False
            )

    def reset_parameters(self):
        if self.k_norm is not None:
            self.k_norm.reset_parameters()
        if self.q_norm is not None:
            self.q_norm.reset_parameters()
        init_weights(
            self.config,
            self.attn_out,
            d=self.config.d_model,
            layer_id=self.layer_id,
        )
        init_weights(
            self.config,
            self.ff_out,
            d=self.ff_out.in_features,
            layer_id=self.layer_id,
        )

    def get_rotary_embedding(self, seq_len: int, device: Optional[torch.device]) -> torch.Tensor:
        if self.pos_emb is not None and self.pos_emb.shape[-2] >= seq_len:  # type: ignore
            return self.pos_emb[:seq_len]  # type: ignore

        pos_emb = self.rotary_emb(seq_len, device=device)
        self.register_buffer("pos_emb", pos_emb, persistent=False)
        return pos_emb

    def attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        B, T, C = q.size()  # batch size, sequence length, d_model
        dtype = k.dtype

        # Optionally apply layer norm to keys and queries.
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q).to(dtype=dtype)
            k = self.k_norm(k).to(dtype=dtype)

        # Move head forward to be next to the batch dim.
        # shape: (B, nh, T, hs)
        q = q.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)
        if self.config.multi_query_attention:
            # shape: (B, 1, T, hs)
            k = k.view(B, T, 1, C // self.config.n_heads).transpose(1, 2)
            # shape: (B, 1, T, hs)
            v = v.view(B, T, 1, C // self.config.n_heads).transpose(1, 2)
        else:
            # shape: (B, nh, T, hs)
            k = k.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)
            # shape: (B, nh, T, hs)
            v = v.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)

        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)

        if use_cache:
            present = (k, v)
        else:
            present = None

        query_len, key_len = q.shape[-2], k.shape[-2]  # could be different if layer_past not None

        if self.config.rope:
            # Apply rotary embeddings.
            positions = self.get_rotary_embedding(key_len, q.device)
            q = apply_rotary_pos_emb(positions[key_len - query_len : key_len], q)
            k = apply_rotary_pos_emb(positions, k)

        if attention_bias is not None:
            attention_bias = attention_bias[:, :, key_len - query_len : key_len, :key_len]

        # Get the attention scores.
        # shape: (B, nh, T, hs)
        att = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None if attention_bias is None else attention_bias.to(dtype=dtype),
            dropout_p=0.0 if not self.training else self.config.attention_dropout,
            is_causal=attention_bias is None,
        )

        # Re-assemble all head outputs side-by-side.
        att = att.transpose(1, 2).contiguous().view(B, T, C)

        # Apply output projection.
        return self.attn_out(att), present

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        raise NotImplementedError

    @classmethod
    def build(cls, layer_id: int, config: ModelConfig) -> OlmoBlock:
        if config.block_type == BlockType.sequential:
            return OlmoSequentialBlock(layer_id, config)
        elif config.block_type == BlockType.parallel:
            return OlmoParallelBlock(layer_id, config)
        else:
            raise NotImplementedError(f"not sure how to handle block type '{config.block_type}'")


class OlmoSequentialBlock(OlmoBlock):
    """
    This is a typical transformer block where the output is computed as ``MLP(LN(x + Attention(LN(x))))``
    (plus another skip connection).
    """

    def __init__(self, layer_id: int, config: ModelConfig):
        super().__init__(layer_id, config)
        # Layer norms.
        self.attn_norm = LayerNorm.build(config)
        self.ff_norm = LayerNorm.build(config)
        # Attention input projection. Projects x -> (q, k, v)
        if config.multi_query_attention:
            self.fused_dims = (config.d_model, config.d_model // config.n_heads, config.d_model // config.n_heads)
        else:
            self.fused_dims = (config.d_model, config.d_model, config.d_model)
        self.att_proj = nn.Linear(
            config.d_model, sum(self.fused_dims), bias=config.include_bias, device=config.init_device
        )
        # Feed-forward input projection.
        self.ff_proj = nn.Linear(
            config.d_model, config.mlp_ratio * config.d_model, bias=config.include_bias, device=config.init_device
        )

    def reset_parameters(self):
        super().reset_parameters()
        self.attn_norm.reset_parameters()
        self.ff_norm.reset_parameters()
        # NOTE: the standard deviation for these weights does not depend on the layer.
        init_weights(self.config, self.att_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.ff_proj, d=self.config.d_model, layer_id=None)

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Get query, key, value projections.
        # shape:
        #  - for regular attn q, k, v: (batch_size, seq_len, d_model)
        #  - for multi-query attn q: (batch_size, seq_len, d_model)
        #                      k, v: (batch_size, seq_len, d_model // n_heads)
        q, k, v = self.att_proj(self.attn_norm(x)).split(self.fused_dims, dim=-1)

        # Get attention scores.
        att, cache = self.attention(q, k, v, attention_bias, layer_past=layer_past, use_cache=use_cache)

        # Add attention scores.
        # shape: (B, T, C)
        x = x + self.dropout(att)

        # Add feed-forward projection.
        # shape: (batch_size, seq_len, d_model)
        x = x + self.dropout(self.ff_out(self.act(self.ff_proj(self.ff_norm(x)))))

        return x, cache


class OlmoParallelBlock(OlmoBlock):
    """
    This is a transformer block where the output is computed as ``MLP(LN(x)) + Attention(LN(x))``
    as in the PaLM architecture, as opposed to the typical ``MLP(LN(x + Attention(LN(x))))``
    as in :class:`OlmoSequentialBlock` (ignoring some skip connections).

    The decoupling of the MLP and Attention functions allow us to fuse the separate input projections
    into a single linear layer to increase throughput. In this configuration it's also straight-forward
    to fuse the output projections, but we found that didn't help.
    """

    def __init__(self, layer_id: int, config: ModelConfig):
        super().__init__(layer_id, config)
        self.norm = LayerNorm.build(config)
        # Fused attention and feed-forward projection.
        # NOTE: we could also fuse the attention and feed-forward output projections
        # but we found that didn't help, possibly because of the overhead of joining the `att`
        # and `ff` activations together.
        # See https://github.com/allenai/LLM/pull/79 for details.
        if config.multi_query_attention:
            self.fused_dims = (
                config.d_model,
                config.d_model // config.n_heads,
                config.d_model // config.n_heads,
                config.mlp_ratio * config.d_model,
            )
        else:
            self.fused_dims = (config.d_model, config.d_model, config.d_model, config.mlp_ratio * config.d_model)
        self.fused_attn_ff_proj = nn.Linear(
            config.d_model, sum(self.fused_dims), bias=config.include_bias, device=config.init_device
        )

    def reset_parameters(self):
        super().reset_parameters()
        self.norm.reset_parameters()
        # NOTE: the standard deviation for these weights does not depend on the layer.
        init_weights(self.config, self.fused_attn_ff_proj, d=self.config.d_model, layer_id=None)

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # Get query, key, value, and feed-forward projections.
        # shape of q, k, v:
        #  - for regular attn q, k, v: (batch_size, seq_len, d_model)
        #  - for multi-query attn q: (batch_size, seq_len, d_model)
        #                      k, v: (batch_size, seq_len, d_model // n_heads)
        # shape of ff:      (batch_size, seq_len, mlp_ratio x d_model)
        q, k, v, ff = self.fused_attn_ff_proj(self.norm(x)).split(self.fused_dims, dim=-1)

        # Get attention scores.
        # shape: (B, T, C)
        att, cache = self.attention(q, k, v, attention_bias, layer_past=layer_past, use_cache=use_cache)

        # Apply output projections (and activation function) and sum the results.
        # We keep these projections separate because we found that we got better throughput this
        # way compared to fusing them.
        return x + self.dropout(self.ff_out(self.act(ff))) + self.dropout(att), cache


class OlmoOutput(NamedTuple):
    logits: torch.FloatTensor
    """
    A tensor of shape `(batch_size, seq_len, vocab_size)` representing the log probabilities
    for the next token *before* normalization via (log) softmax.
    """

    attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]]
    """
    Attention keys and values from each block.
    """


class OlmoGenerateOutput(NamedTuple):
    token_ids: torch.LongTensor
    """
    The generated token IDs, a tensor of shape `(batch_size, beam_size, max_steps)`.
    These do *not* include the original input IDs.
    """

    scores: torch.FloatTensor
    """
    The scores of the generated sequences, a tensor of shape `(batch_size, beam_size)`.
    """


def causal_attention_bias(config: ModelConfig, device: torch.device) -> torch.FloatTensor:
    size = config.max_sequence_length
    att_bias = torch.triu(
        torch.ones(size, size, device=device, dtype=torch.float),
        diagonal=1,
    )
    att_bias.masked_fill_(att_bias == 1, float("-inf"))
    return att_bias.view(1, 1, size, size)  # type: ignore


def alibi_attention_bias(config: ModelConfig, device: torch.device) -> torch.FloatTensor:
    size = config.max_sequence_length
    alibi_bias = torch.arange(1 - size, 1, dtype=torch.float, device=device).view(1, 1, 1, size)

    # shape: (1, 1, seq_len, seq_len)
    alibi_bias = alibi_bias - torch.arange(1 - size, 1, dtype=torch.float, device=device).view(1, 1, size, 1)
    alibi_bias.abs_().mul_(-1)

    # shape: (n_heads,)
    m = torch.arange(1, config.n_heads + 1, dtype=torch.float, device=device)
    m.mul_(config.alibi_bias_max / config.n_heads)

    # shape: (1, n_heads, seq_len, seq_len)
    return alibi_bias * (1.0 / (2 ** m.view(1, config.n_heads, 1, 1)))  # type: ignore


class Olmo(nn.Module):
    def __init__(self, config: ModelConfig, init_params: bool = True):
        super().__init__()
        self.config = config

        # Validate config.
        if self.config.alibi and self.config.flash_attention:
            raise OlmoConfigurationError("ALiBi is currently not supported with FlashAttention")

        if self.config.alibi and self.config.rope:
            raise OlmoConfigurationError("ALiBi and RoPE are mutually exclusive")

        if self.config.embedding_size is not None and self.config.embedding_size != self.config.vocab_size:
            if self.config.embedding_size < self.config.vocab_size:
                raise OlmoConfigurationError("embedding size should be at least as big as vocab size")
            elif self.config.embedding_size % 128 != 0:
                import warnings

                warnings.warn(
                    "Embedding size is not a multiple of 128! This could hurt throughput performance.", UserWarning
                )

        torch.backends.cuda.enable_flash_sdp(self.config.flash_attention)
        torch.backends.cuda.enable_mem_efficient_sdp(False)  # this is super slow so make sure torch won't use it

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(
                    config.embedding_size or config.vocab_size, config.d_model, device=config.init_device
                ),
                emb_drop=nn.Dropout(config.embedding_dropout),
                blocks=nn.ModuleList([OlmoBlock.build(i, config) for i in range(config.n_layers)]),
                ln_f=LayerNorm.build(config),
            )
        )
        if not (self.config.alibi or self.config.rope):
            self.transformer.update(
                {"wpe": nn.Embedding(config.max_sequence_length, config.d_model, device=config.init_device)}
            )
        # When `init_device="meta"` FSDP will call `reset_parameters()` to initialize weights.
        if init_params and self.config.init_device != "meta":
            self.reset_parameters()
        self.__num_fwd_flops: Optional[int] = None

        # Attention bias cache.
        # We could cache these as buffers, but we've run into various issues doing that with FSDP.
        # In general it appears the way FSDP handles buffers is not well-defined.
        # It doesn't shard them but apparently it does synchronize them across processes, which we want to avoid
        # since (A) it isn't necessary, and (B) we have `-inf` in these biases which might get turned into
        # NaNs when they're synchronized due to casting or some other issue.
        self.__bias_cache: Dict[str, Optional[torch.FloatTensor]] = {
            "causal_attention_bias": None,
            "alibi_attention_bias": None,
        }
        if self.config.alibi:
            # Warm up cache.
            self.causal_attention_bias
            self.alibi_attention_bias

    def reset_parameters(self):
        log.info("Initializing model parameters...")
        # Top-level embeddings / linear layers.
        init_weights(
            self.config,
            self.transformer.wte,  # type: ignore
            std_factor=(0.5 * math.sqrt(self.config.d_model)) if self.config.scale_logits else 1.0,
        )
        if hasattr(self.transformer, "wpe"):
            init_weights(self.config, self.transformer.wpe)  # type: ignore

        # Top-level layer norm.
        self.transformer.ln_f.reset_parameters()  # type: ignore

        # Let the blocks handle themselves.
        for block in self.transformer.blocks:  # type: ignore
            block.reset_parameters()  # type: ignore

    @property
    def device(self) -> torch.device:
        device: torch.device = self.transformer.wte.weight.device  # type: ignore
        if device.type == "meta":
            if self.config.init_device is not None and self.config.init_device != "meta":
                return torch.device(self.config.init_device)
            else:
                return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            return device

    @property
    def causal_attention_bias(self) -> torch.FloatTensor:
        causal_bias = self.__bias_cache["causal_attention_bias"]
        if causal_bias is None:
            causal_bias = causal_attention_bias(self.config, self.device)
            self.__bias_cache["causal_attention_bias"] = causal_bias
        elif causal_bias.device != self.device:  # in case model was moved to different device
            causal_bias = causal_bias.to(device=self.device)
            self.__bias_cache["causal_attention_bias"] = causal_bias  # type: ignore
        return causal_bias  # type: ignore

    @property
    def alibi_attention_bias(self) -> torch.FloatTensor:
        alibi_bias = self.__bias_cache["alibi_attention_bias"]
        if alibi_bias is None:
            alibi_bias = alibi_attention_bias(self.config, self.device)
            self.__bias_cache["alibi_attention_bias"] = alibi_bias
        elif alibi_bias.device != self.device:  # in case model was moved to different device
            alibi_bias = alibi_bias.to(device=self.device)
            self.__bias_cache["alibi_attention_bias"] = alibi_bias  # type: ignore
        return alibi_bias  # type: ignore

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        past_key_values: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        last_logits_only: bool = False,
    ) -> OlmoOutput:
        """
        :param input_ids: A tensor of shape `(batch_size, seq_len)`.
        :param attention_mask: A tensor of shape `(batch_size, seq_len)` that indicates
            which input IDs are masked. A `1` value in the mask means that
            the corresponding input ID should *not* be ignored. A `0` means
            that the corresponding input ID is masked.

            This has the same meaning as the `attention_mask` in HuggingFace's `transformers`
            library.
        :param attention_bias: A tensor of shape `(batch_size, 1, seq_len, seq_len)`,
            `(1, 1, seq_len, seq_len)`, or `(seq_len, seq_len)`. This is used
            to introduce causal or other biases.

            If the tensor is a bool or byte tensor, a `True` or `1` at `attention_bias[:, :, i, j]`
            indicates that the i-th element in the sequence is allowed to attend to the j-th
            element in the sequence.

            If the tensor is a float tensor, it will just be added to the attention
            scores before the softmax.

            The default is causal, which corresponds to a lower-diagonal byte matrix of ones.
        :param past_key_values: Pre-computed keys and values for each attention block.
            Can be used to speed up sequential decoding. The `input_ids` which have
            their past given to this model should not be passed as `input_ids` as they have already been computed.
        :param use_cache: If `True`, return key and value tensors for each block.
        :param last_logits_only: If `True`, only compute the logits for the last token of each sequence.
            This can speed up decoding when you only care about the next token.
        """
        if past_key_values:
            assert len(past_key_values) == self.config.n_layers

        batch_size, seq_len = input_ids.size()
        assert seq_len <= self.config.max_sequence_length, (
            f"Cannot forward input with seq_len={seq_len}, "
            f"this model only supports seq_len<={self.config.max_sequence_length}"
        )

        # Get embeddings of input.
        # shape: (batch_size, seq_len, d_model)
        x = self.transformer.wte(input_ids)  # type: ignore

        if not (self.config.alibi or self.config.rope):
            # Get positional embeddings.
            if past_key_values is None:
                past_length = 0
            else:
                past_length = past_key_values[0][0].size(-2)
            # shape: (1, seq_len)
            pos = torch.arange(
                past_length, past_length + seq_len, dtype=torch.long, device=input_ids.device
            ).unsqueeze(0)
            # shape: (1, seq_len, d_model)
            pos_emb = self.transformer.wpe(pos)  # type: ignore
            x = pos_emb + x

        # Add input + positional embeddings and apply dropout.
        # shape: (batch_size, seq_len, d_model)
        x = self.transformer.emb_drop(x)  # type: ignore

        # Transform the attention mask into what the blocks expect.
        if attention_mask is not None:
            # shape: (batch_size, 1, 1, seq_len)
            attention_mask = attention_mask.to(dtype=x.dtype).view(batch_size, -1)[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * torch.finfo(attention_mask.dtype).min
            attention_mask.masked_fill_(attention_mask == 1.0, float("-inf"))

        # Merge attention mask with attention bias.
        if (
            attention_bias is not None
            or attention_mask is not None
            or self.config.alibi
            # NOTE (epwalsh): we need to initialize the attn bias in order for attn to work properly
            # with key+value cache. Otherwise `F.scaled_dot_product_attention()` doesn't seem to compute
            # scores correctly.
            or past_key_values is not None
        ):
            if attention_bias is None and self.config.alibi:
                attention_bias = self.causal_attention_bias + self.alibi_attention_bias
            elif attention_bias is None:
                attention_bias = self.causal_attention_bias
            elif attention_bias.dtype in (torch.int8, torch.bool):
                attention_bias = attention_bias.to(dtype=x.dtype)
                attention_bias.masked_fill_(attention_bias == 0.0, float("-inf"))

            # Transform to the right shape and data type.
            mask_len = seq_len
            if attention_mask is not None:
                mask_len = attention_mask.shape[-1]
            elif past_key_values is not None:
                mask_len = past_key_values[0][0].shape[-2] + input_ids.shape[-1]
            attention_bias = attention_bias[:, :, :mask_len, :mask_len].to(x.dtype)

            # Add in the masking bias.
            if attention_mask is not None:
                attention_bias = attention_bias + attention_mask

        attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = [] if use_cache else None

        # Apply blocks one-by-one.
        for block, layer_past in zip(
            self.transformer.blocks,  # type: ignore
            past_key_values or [None] * self.config.n_layers,  # type: ignore
        ):
            # shape: (batch_size, seq_len, d_model)
            x, cache = block(x, attention_bias=attention_bias, layer_past=layer_past, use_cache=use_cache)
            if attn_key_values is not None:
                assert cache is not None
                attn_key_values.append(cache)

        if last_logits_only:
            # shape: (batch_size, 1, d_model)
            x = x[:, -1, :].unsqueeze(1)

        # Apply final layer norm.
        # shape: (batch_size, seq_len or 1, d_model)
        x = self.transformer.ln_f(x)  # type: ignore

        # Get logits.
        # shape: (batch_size, seq_len or 1, vocab_size)
        logits = F.linear(x, self.transformer.wte.weight, None)  # type: ignore
        if self.config.scale_logits:
            logits.mul_(1 / math.sqrt(self.config.d_model))

        return OlmoOutput(logits=logits, attn_key_values=attn_key_values)  # type: ignore[arg-type]

    def fsdp_wrap_fn(self, module, recurse: bool = True, nonwrapped_numel: int = 0):
        del nonwrapped_numel
        if recurse:
            return True  # always recurse
        return isinstance(module, OlmoBlock)

    def activation_checkpointing_fn(self, module):
        return isinstance(module, OlmoBlock)

    def num_params(self, include_embedding: bool = True) -> int:
        """
        Get the total number of parameters.
        """
        params = (np for np in self.named_parameters())
        if not include_embedding:
            params = filter(  # type: ignore
                lambda np: ".wte." not in np[0] and ".wpe." not in np[0],
                params,
            )
        return sum(p.numel() for _, p in params)

    @property
    def num_fwd_flops(self):
        if self.__num_fwd_flops:
            return self.__num_fwd_flops
        n_params = self.num_params()
        # the number of parameters is approximately the number of multiply-accumulates (MAC) in the network
        # each MAC has 2 FLOPs - we multiply by 2 ie 2 * n_param
        # this gets us FLOPs / token
        params_flops_per_token = 2 * n_params
        params_flops_per_seq = params_flops_per_token * self.config.max_sequence_length
        # there are 2 FLOPS per mac; there is A=Q*K^T and out=A*V ops (ie mult by 2)
        attn_flops_per_seq = (
            self.config.n_layers * 2 * 2 * (self.config.d_model * (self.config.max_sequence_length**2))
        )
        self.__num_fwd_flops = params_flops_per_seq + attn_flops_per_seq
        return self.__num_fwd_flops

    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        max_steps: int = 10,
        beam_size: int = 1,
        per_node_beam_size: Optional[int] = None,
        sampler: Optional[Sampler] = None,
        min_steps: Optional[int] = None,
        final_sequence_scorer: Optional[FinalSequenceScorer] = None,
        constraints: Optional[List[Constraint]] = None,
    ) -> OlmoGenerateOutput:
        """
        Generate token IDs using beam search.

        Note that by default ``beam_size`` is set to 1, which is greedy decoding.

        :param input_ids: A tensor of shape `(batch_size, seq_len)`.
        :param attention_mask: A optional tensor of shape `(batch_size, seq_len)`, the same
            as for the forward method.
        :param attention_bias: A tensor of shape
            `(batch_size, 1, seq_len + tokens_to_generate, seq_len + tokens_to_generate)`,
            the same as for the forward method except only one shape is excepted here.

        For an explanation of the other arguments, see the :class:`BeamSearch` class.
        """
        beam_search = BeamSearch(
            self.config.eos_token_id,
            max_steps=max_steps,
            beam_size=beam_size,
            per_node_beam_size=per_node_beam_size,
            sampler=sampler,
            min_steps=min_steps,
            final_sequence_scorer=final_sequence_scorer,
            constraints=constraints,
        )

        # Validate inputs.
        batch_size, seq_len = input_ids.shape
        if attention_mask is not None:
            assert attention_mask.shape == (batch_size, seq_len)
        if attention_bias is not None:
            assert len(attention_bias.shape) == 4
            assert attention_bias.shape[:2] == (batch_size, 1)
            assert (
                seq_len + beam_search.max_steps
                <= attention_bias.shape[2]
                == attention_bias.shape[3]
                <= self.config.max_sequence_length
            )

        tokens_generated = 0

        def flatten_past_key_values(
            past_key_values: List[Tuple[torch.Tensor, torch.Tensor]]
        ) -> Dict[str, torch.Tensor]:
            out = {}
            for i, (key, value) in enumerate(past_key_values):
                out[f"past_key_{i}"] = key
                out[f"past_value_{i}"] = value
            return out

        def unflatten_past_key_values(
            past_key_values: Dict[str, torch.Tensor]
        ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
            out = []
            for i in range(self.config.n_layers):
                past_key = past_key_values[f"past_key_{i}"]
                past_value = past_key_values[f"past_value_{i}"]
                out.append((past_key, past_value))
            return out

        def step(
            last_predictions: torch.Tensor, state: dict[str, torch.Tensor]
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            nonlocal tokens_generated

            attention_mask = state.get("attention_mask")
            attention_bias = state.get("attention_bias")

            if tokens_generated > 0:
                past_key_values = unflatten_past_key_values(state)
                input_ids = last_predictions.unsqueeze(1)
                if attention_mask is not None:
                    group_size = input_ids.shape[0]
                    attention_mask = torch.cat((attention_mask, attention_mask.new_ones((group_size, 1))), dim=-1)
            else:
                past_key_values = None
                input_ids = state["input_ids"]

            tokens_generated += 1

            # Run forward pass of model to get logits, then normalize to get log probs.
            output = self(
                input_ids,
                attention_mask=attention_mask,
                attention_bias=attention_bias,
                past_key_values=past_key_values,
                use_cache=True,
                last_logits_only=True,
            )
            log_probs = F.log_softmax(output.logits[:, -1, :], dim=-1)

            # Create new state.
            state = flatten_past_key_values(output.attn_key_values)
            if attention_mask is not None:
                state["attention_mask"] = attention_mask
            if attention_bias is not None:
                state["attention_bias"] = attention_bias

            return log_probs, state

        initial_preds = input_ids.new_zeros((batch_size,))  # This is arbitrary, we won't use this.
        state: dict[str, torch.Tensor] = {"input_ids": input_ids}
        if attention_mask is not None:
            state["attention_mask"] = attention_mask
        if attention_bias is not None:
            state["attention_bias"] = attention_bias
        with torch.no_grad():
            token_ids, scores = beam_search.search(initial_preds, state, step)

        return OlmoGenerateOutput(
            token_ids=token_ids,  # type: ignore[arg-type]
            scores=scores,  # type: ignore[arg-type]
        )

    @classmethod
    def from_checkpoint(cls, checkpoint_dir: PathOrStr, device: str = "cpu") -> Olmo:
        """
        Load an OLMo model from a checkpoint.
        """
        from cached_path import cached_path

        # Load config.
        config_path = cached_path(os.path.join(checkpoint_dir, "config.yaml"))
        model_config = ModelConfig.load(config_path, key="model", validate_paths=False)

        # Initialize model (always on CPU to start with so we don't run out of GPU memory).
        model_config.init_device = "cpu"
        model = Olmo(model_config)
        model.config.init_device = device

        # Load state dict directly to target device.
        state_dict_path = cached_path(os.path.join(checkpoint_dir, "model.pt"))
        state_dict = torch.load(state_dict_path, map_location="cpu")
        model.load_state_dict(model._make_state_dict_compatible(state_dict))

        return model.to(torch.device(device)).eval()

    def _make_state_dict_compatible(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # For backwards compatibility prior to fixing https://github.com/allenai/LLM/issues/222
        prefix = ""
        if next(iter(state_dict.keys())).startswith((fsdp_prefix := "_fsdp_wrapped_module.")):
            prefix = fsdp_prefix
        if self.config.block_type == BlockType.sequential:
            for block_idx in range(self.config.n_layers):
                norm_w_key = f"{prefix}transformer.blocks.{block_idx}.norm.weight"
                norm_b_key = f"{prefix}transformer.blocks.{block_idx}.norm.bias"
                if norm_w_key in state_dict:
                    norm_w = state_dict.pop(norm_w_key)
                    state_dict[f"{prefix}transformer.blocks.{block_idx}.attn_norm.weight"] = norm_w
                    state_dict[f"{prefix}transformer.blocks.{block_idx}.ff_norm.weight"] = norm_w.clone()
                if norm_b_key in state_dict:
                    norm_b = state_dict.pop(norm_b_key)
                    state_dict[f"{prefix}transformer.blocks.{block_idx}.attn_norm.bias"] = norm_b
                    state_dict[f"{prefix}transformer.blocks.{block_idx}.ff_norm.bias"] = norm_b.clone()
        return state_dict
