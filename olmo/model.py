"""
Adapted from
[MosaiclML](https://github.com/mosaicml/examples.git) and
[minGPT](https://github.com/karpathy/minGPT.git)
"""

from __future__ import annotations

import logging
import math
import sys
from abc import abstractmethod
from collections import defaultdict
from functools import partial
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    cast,
)

import torch
import torch.backends.cuda
import torch.nn as nn
import torch.nn.functional as F
from PIL.Image import Image
from torch import einsum

from .aliases import PathOrStr
from .beam_search import BeamSearch, Constraint, FinalSequenceScorer, Sampler
from .config import (
    ActivationCheckpointingStrategy,
    ActivationType,
    BlockType,
    CheckpointType,
    FSDPWrapStrategy,
    LayerNormType,
    ModelConfig,
    VisionBackboneType,
)
from .exceptions import OlmoConfigurationError
from .initialization import ModuleType, init_weights
from .mm_data.image_preprocessing import ClipImageResize, AnyResClipImageResize
from .torch_util import ensure_finite_, get_global_rank
from .clip import create_model as create_clip_model
from .mm_data.image_util import unpad_image

if sys.version_info.minor > 8:
    from collections.abc import MutableMapping
elif sys.version_info.minor == 8:
    from typing import MutableMapping
else:
    raise SystemExit("This script supports Python 3.8 or higher")

__all__ = [
    "LayerNormBase",
    "LayerNorm",
    "RMSLayerNorm",
    "AMDLayerNorm",
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


def activation_checkpoint_function(cfg: ModelConfig):
    preserve_rng_state = (
        (cfg.attention_dropout == 0.0) and (cfg.embedding_dropout == 0.0) and (cfg.residual_dropout == 0.0)
    )
    from torch.utils.checkpoint import checkpoint

    return partial(
        checkpoint,
        preserve_rng_state=preserve_rng_state,
        use_reentrant=False,
    )


class BufferCache(dict, MutableMapping[str, torch.Tensor]):
    """
    Cache for attention biases and other things that would normally be stored as buffers.
    We avoid using buffers because we've run into various issues doing so with FSDP.
    In general it appears the way FSDP handles buffers is not well-defined.
    It doesn't shard them but apparently it does synchronize them across processes, which we want to avoid
    since (A) it isn't necessary, and (B) we sometimes have `-inf` in these biases which might get turned into
    NaNs when they're synchronized due to casting or some other issue.
    """


def _non_meta_init_device(config: ModelConfig) -> torch.device:
    if config.init_device is not None and config.init_device != "meta":
        return torch.device(config.init_device)
    else:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Dropout(nn.Dropout):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.p == 0.0:
            return input
        else:
            return F.dropout(input, self.p, self.training, self.inplace)


class LayerNormBase(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        *,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = True,
        eps: float = 1e-05,
    ):
        super().__init__()
        self.config = config
        self.eps = eps
        self.normalized_shape = (size or config.d_model,)
        if elementwise_affine or (elementwise_affine is None and self.config.layer_norm_with_affine):
            self.weight = nn.Parameter(torch.ones(self.normalized_shape, device=config.init_device))
            use_bias = self.config.bias_for_layer_norm
            if use_bias is None:
                use_bias = self.config.include_bias
            if use_bias:
                self.bias = nn.Parameter(torch.zeros(self.normalized_shape, device=config.init_device))
            else:
                self.register_parameter("bias", None)
        else:
            self.register_parameter("bias", None)
            self.register_parameter("weight", None)

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
            return RMSLayerNorm(config, size=size, **kwargs)
        elif config.layer_norm_type == LayerNormType.amd_compatible:
            return AMDLayerNorm(config, size=size, **kwargs)
        else:
            raise NotImplementedError(f"Unknown LayerNorm type: '{config.layer_norm_type}'")

    def _cast_if_autocast_enabled(self, tensor: torch.Tensor, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        # NOTE: `is_autocast_enabled()` only checks for CUDA autocast, so we use the separate function
        # `is_autocast_cpu_enabled()` for CPU autocast.
        # See https://github.com/pytorch/pytorch/issues/110966.
        if tensor.device.type == "cuda" and torch.is_autocast_enabled():
            return tensor.to(dtype=dtype if dtype is not None else torch.get_autocast_gpu_dtype())
        elif tensor.device.type == "cpu" and torch.is_autocast_cpu_enabled():
            return tensor.to(dtype=dtype if dtype is not None else torch.get_autocast_cpu_dtype())
        else:
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
        eps: float = 1e-05,
    ):
        super().__init__(config, size=size, elementwise_affine=elementwise_affine, eps=eps)
        self.low_precision = low_precision

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


class AMDLayerNorm(LayerNormBase):
    """
    LayerNorm implemented using PyTorch primitives.

    We do this to work around a bug in the PyTorch/ROCm implementation of layer norm that fails with a
    segfault when the bias is not present.
    """

    def __init__(
        self,
        config: ModelConfig,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = None,
        eps: float = 1e-05,
    ):
        super().__init__(config, size=size, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        og_dtype = x.dtype
        x = self._cast_if_autocast_enabled(x, dtype=torch.float32)
        with torch.autocast(enabled=False, device_type=x.device.type):
            var, mean = torch.var_mean(x, dim=-1, correction=0, keepdim=True)
            var.add_(self.eps)
            var.rsqrt_()  # rsqrt should be more stable than 1/sqrt
            x = var * (x - mean)
            if self.weight is not None:
                x.mul_(self.weight)
            if self.bias is not None:
                x.add_(self.bias)
            return x.to(og_dtype)


class RMSLayerNorm(LayerNormBase):
    """
    RMS layer norm, a simplified :class:`LayerNorm` implementation
    """

    def __init__(
        self,
        config: ModelConfig,
        size: Optional[int] = None,
        elementwise_affine: Optional[bool] = None,
        eps: float = 1e-5,
    ):
        super().__init__(config, size=size, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(enabled=False, device_type=x.device.type):
            og_dtype = x.dtype
            x = x.to(torch.float32)
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + self.eps)
            x = x.to(og_dtype)

        if self.weight is not None:
            if self.bias is not None:
                return self.weight * x + self.bias
            else:
                return self.weight * x
        else:
            return x


class RotaryEmbedding(nn.Module):
    """
    [Rotary positional embeddings (RoPE)](https://arxiv.org/abs/2104.09864).
    """

    def __init__(self, config: ModelConfig, cache: BufferCache):
        super().__init__()
        self.config = config
        self.__cache = cache
        # Warm up cache.
        self.get_rotary_embedding(config.max_sequence_length, _non_meta_init_device(config))

    def get_rotary_embedding(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if (
            (pos_sin := self.__cache.get("rope_pos_sin")) is not None
            and (pos_cos := self.__cache.get("rope_pos_cos")) is not None
            and pos_sin.shape[-2] >= seq_len
            and pos_cos.shape[-2] >= seq_len
        ):
            if pos_sin.device != device:
                pos_sin = pos_sin.to(device)
                self.__cache["rope_pos_sin"] = pos_sin
            if pos_cos.device != device:
                pos_cos = pos_cos.to(device)
                self.__cache["rope_pos_cos"] = pos_cos
            return pos_sin[:, :, :seq_len, :], pos_cos[:, :, :seq_len, :]

        with torch.autocast(device.type, enabled=False):
            dim = self.config.d_model // self.config.n_heads
            inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device, dtype=torch.float) / dim))
            seq = torch.arange(seq_len, device=device, dtype=torch.float)
            freqs = einsum("i , j -> i j", seq, inv_freq)
            positions = torch.cat((freqs, freqs), dim=-1)
            pos_sin, pos_cos = positions.sin()[None, None, :, :], positions.cos()[None, None, :, :]
        self.__cache["rope_pos_sin"] = pos_sin
        self.__cache["rope_pos_cos"] = pos_cos
        return pos_sin, pos_cos

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        B, nh, T, hs = x.size()
        x = x.view(B, nh, T, 2, hs // 2)
        x1, x2 = x.unbind(dim=-2)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, pos_sin: torch.Tensor, pos_cos: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return ((t * pos_cos) + (self.rotate_half(t) * pos_sin)).to(t.dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.config.rope_full_precision:
            q_, k_ = q.float(), k.float()
        else:
            q_, k_ = q, k

        with torch.autocast(q.device.type, enabled=False):
            query_len, key_len = q_.shape[-2], k_.shape[-2]  # could be different if layer_past not None
            pos_sin, pos_cos = self.get_rotary_embedding(key_len, q_.device)
            pos_sin = pos_sin.type_as(q_)
            pos_cos = pos_cos.type_as(q_)
            q_ = self.apply_rotary_pos_emb(
                pos_sin[:, :, key_len - query_len : key_len, :],
                pos_cos[:, :, key_len - query_len : key_len, :],
                q_,
            )
            k_ = self.apply_rotary_pos_emb(pos_sin, pos_cos, k_)
        return q_.type_as(q), k_.type_as(k)


class TwoDimSinCosEmbedding(nn.Module):
    """
    [2D sine-consine positional embeddings](https://arxiv.org/abs/2111.06377).
    Assume sqaure grid.
    """

    def __init__(self, config: ModelConfig, cache: BufferCache):
        super().__init__()
        self.config = config
        self.__cache = cache
        # Warm up cache.
        self.get_2d_sincos_pos_embedding(_non_meta_init_device(config))
    
    def get_2d_sincos_pos_embedding(self, device: torch.device, temperature: float = 10000.) -> torch.Tensor:
        """2D Sinusoidal Position Embedding.

        Args:
            emb_dim: int, dimension of the embedding.
            grid_size: int, grid size (H/W)

        Returns:
            position embedding of shape (size*size, emb_dim).
        """
        if (pos_emb := self.__cache.get("sincos_pos")) is not None:
            if pos_emb.device != device:
                pos_emb = pos_emb.to(device)
                self.__cache["sincos_pos"] = pos_emb
            return pos_emb

        emb_dim = self.config.resampler.d_query
        assert emb_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        grid_size = int(math.sqrt(self.config.resampler.n_queries))

        with torch.autocast(device.type, enabled=False):
            grid_w = torch.arange(grid_size, device=device, dtype=torch.float)
            grid_h = torch.arange(grid_size, device=device, dtype=torch.float)
            grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='xy')

            emb_w = self.get_1d_sincos_pos_embed_from_grid(emb_dim // 2, grid_w, device, temperature)
            emb_h = self.get_1d_sincos_pos_embed_from_grid(emb_dim // 2, grid_h, device, temperature)
            pos_emb = torch.cat([emb_w, emb_h], dim=1) # (H*W, D)
        
        self.__cache["sincos_pos"] = pos_emb
        return pos_emb

    def get_1d_sincos_pos_embed_from_grid(self, emb_dim: int, pos: torch.Tensor, device: torch.device, temperature: float = 10000.) -> torch.Tensor:
        """
        (Absolute, additive) 1D sinusoidal positional embeddings used in MoCo v3, MAE
        Args:
            emb_dim (int): output dimension for each position
            pos: a list of positions to be encoded: size (H, W), M = H * W
            out: (M, D)
        """
        assert emb_dim % 2 == 0
        omega = torch.arange(emb_dim // 2, device=device, dtype=torch.float)
        omega /= emb_dim / 2.
        omega = 1. / temperature**omega  # (D/2,)

        out = torch.einsum('m,d->md', pos.flatten(), omega) # (M, D/2), outer product

        emb_sin = torch.sin(out) # (M, D/2)
        emb_cos = torch.cos(out) # (M, D/2)

        emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
        return emb
    
    def interpolate_pos_emb(self, pos_emb: torch.Tensor, target_size: int) -> torch.Tensor:
        # pos_emb: L, C
        # target_size: M
        # return: M, C
        source_grid_size = int(math.sqrt(pos_emb.shape[0]))
        target_grid_size = int(math.sqrt(target_size))
        dtype = pos_emb.dtype

        if source_grid_size != target_grid_size:
            pos_emb = F.interpolate(
                pos_emb.float().reshape(1, source_grid_size, source_grid_size, -1).permute(0, 3, 1, 2),
                size=(target_grid_size, target_grid_size),
                mode="bicubic",
                align_corners=False,
            ).permute(0, 2, 3, 1).flatten(0, 2).to(dtype=dtype)
        return pos_emb
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pos_emb = self.get_2d_sincos_pos_embedding(x.device)
        with torch.autocast(x.device.type, enabled=False):
            pos_emb = self.interpolate_pos_emb(pos_emb, x.shape[1])
            pos_emb = pos_emb.type_as(x)
            x = x + pos_emb
        return x


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
    def build(cls, config: ModelConfig, activation_type: ActivationType=None) -> Activation:
        activation_type = activation_type or config.activation_type
        if activation_type == ActivationType.gelu:
            return cast(Activation, GELU(approximate="none"))
        elif activation_type == ActivationType.relu:
            return cast(Activation, ReLU(inplace=False))
        elif activation_type == ActivationType.swiglu:
            return SwiGLU(config)
        else:
            raise NotImplementedError(f"Unknown activation: '{activation_type}'")


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


def causal_attention_bias(seq_len: int, device: torch.device) -> torch.FloatTensor:
    att_bias = torch.triu(
        torch.ones(seq_len, seq_len, device=device, dtype=torch.float),
        diagonal=1,
    )
    att_bias.masked_fill_(att_bias == 1, torch.finfo(att_bias.dtype).min)
    return att_bias.view(1, 1, seq_len, seq_len)  # type: ignore


def get_causal_attention_bias(cache: BufferCache, seq_len: int, device: torch.device) -> torch.Tensor:
    if (causal_bias := cache.get("causal_attention_bias")) is not None and causal_bias.shape[-1] >= seq_len:
        if causal_bias.device != device:
            causal_bias = causal_bias.to(device)
            cache["causal_attention_bias"] = causal_bias
        return causal_bias
    with torch.autocast(device.type, enabled=False):
        causal_bias = causal_attention_bias(seq_len, device)
    cache["causal_attention_bias"] = causal_bias
    return causal_bias


def alibi_attention_bias(seq_len: int, config: ModelConfig, device: torch.device) -> torch.FloatTensor:
    alibi_bias = torch.arange(1 - seq_len, 1, dtype=torch.float, device=device).view(1, 1, 1, seq_len)

    # shape: (1, 1, seq_len, seq_len)
    alibi_bias = alibi_bias - torch.arange(1 - seq_len, 1, dtype=torch.float, device=device).view(1, 1, seq_len, 1)
    alibi_bias.abs_().mul_(-1)

    # shape: (n_heads,)
    m = torch.arange(1, config.n_heads + 1, dtype=torch.float, device=device)
    m.mul_(config.alibi_bias_max / config.n_heads)

    # shape: (1, n_heads, seq_len, seq_len)
    return alibi_bias * (1.0 / (2 ** m.view(1, config.n_heads, 1, 1)))  # type: ignore


class OlmoBlock(nn.Module):
    """
    A base class for transformer block implementations.
    """

    def __init__(self, layer_id: int, config: ModelConfig, cache: BufferCache):
        super().__init__()
        self.layer_id = layer_id
        self.config = config
        self.hidden_size = (
            config.mlp_hidden_size if config.mlp_hidden_size is not None else config.mlp_ratio * config.d_model
        )
        self.__cache = cache
        assert config.d_model % config.n_heads == 0

        self._activation_checkpoint_fn = None

        # Dropout.
        self.dropout = Dropout(config.residual_dropout)

        # Layer norms.
        self.k_norm: Optional[LayerNormBase] = None
        self.q_norm: Optional[LayerNormBase] = None
        if config.attention_layer_norm:
            self.k_norm = LayerNormBase.build(
                config,
                size=config.d_model // config.n_heads if config.multi_query_attention else None,
                elementwise_affine=config.attention_layer_norm_with_affine,
            )
            self.q_norm = LayerNormBase.build(config, elementwise_affine=config.attention_layer_norm_with_affine)

        # Activation function.
        self.act = Activation.build(config)
        assert (self.act.output_multiplier * self.hidden_size) % 1 == 0

        # Attention output projection.
        self.attn_out = nn.Linear(
            config.d_model, config.d_model, bias=config.include_bias, device=config.init_device
        )

        # Feed-forward output projection.
        self.ff_out = nn.Linear(
            int(self.act.output_multiplier * self.hidden_size),
            config.d_model,
            bias=config.include_bias,
            device=config.init_device,
        )
        self.ff_out._is_residual = True  # type: ignore

        # Rotary embeddings.
        if self.config.rope:
            self.rotary_emb = RotaryEmbedding(config, self.__cache)

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
            type_of_module=ModuleType.out_module,
        )
        init_weights(
            self.config,
            self.ff_out,
            d=self.ff_out.in_features,
            layer_id=self.layer_id,
            type_of_module=ModuleType.out_module,
        )

    def set_activation_checkpointing(self, strategy: Optional[ActivationCheckpointingStrategy]):
        if strategy == ActivationCheckpointingStrategy.fine_grained:
            self._activation_checkpoint_fn = activation_checkpoint_function(self.config)
        else:
            self._activation_checkpoint_fn = None

    @classmethod
    def _cast_attn_bias(cls, bias: torch.Tensor, input_dtype: torch.dtype) -> torch.Tensor:
        target_dtype = input_dtype
        # NOTE: `is_autocast_enabled()` only checks for CUDA autocast, so we use the separate function
        # `is_autocast_cpu_enabled()` for CPU autocast.
        # See https://github.com/pytorch/pytorch/issues/110966.
        if bias.device.type == "cuda" and torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        elif bias.device.type == "cpu" and torch.is_autocast_cpu_enabled():
            target_dtype = torch.get_autocast_cpu_dtype()
        if bias.dtype != target_dtype:
            bias = bias.to(target_dtype)
            ensure_finite_(bias, check_neg_inf=True, check_pos_inf=False)
        return bias

    def _scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
    ) -> torch.Tensor:
        """
        Computes scaled dot product attention on query, key and value tensors, using an optional
        attention mask if passed, and applying dropout if a probability greater than 0.0 is specified.

        This method is based on PyTorch's `scaled_dot_product_attention`.
        """
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )

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

        present = (k, v) if use_cache else None
        query_len, key_len = q.shape[-2], k.shape[-2]  # could be different if layer_past not None

        if self.config.rope:
            # Apply rotary embeddings.
            q, k = self.rotary_emb(q, k)

        if attention_bias is not None:
            # Resize and cast attention bias.
            # The current dtype of the attention bias might not match the dtype that the SDP attn function will
            # run in if AMP is enabled, and this can be a problem if some tokens are masked out due to padding
            # as down-casting the attention bias to the autocast precision will result in -infs, which will
            # cause the SDP attn function to produce NaNs.
            attention_bias = self._cast_attn_bias(
                attention_bias[:, :, key_len - query_len : key_len, :key_len], dtype
            )

        # Get the attention scores.
        # shape: (B, nh, T, hs)
        att = self._scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_bias,
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
        layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        raise NotImplementedError

    @classmethod
    def build(cls, layer_id: int, config: ModelConfig, cache: BufferCache) -> OlmoBlock:
        if config.block_type == BlockType.sequential:
            return OlmoSequentialBlock(layer_id, config, cache)
        elif config.block_type == BlockType.parallel:
            return OlmoParallelBlock(layer_id, config, cache)
        elif config.block_type == BlockType.llama:
            return OlmoLlamaBlock(layer_id, config, cache)
        else:
            raise NotImplementedError(f"Unknown block type: '{config.block_type}'")


class OlmoSequentialBlock(OlmoBlock):
    """
    This is a typical transformer block where the output is computed as ``MLP(LN(x + Attention(LN(x))))``
    (plus another skip connection).
    """

    def __init__(self, layer_id: int, config: ModelConfig, cache: BufferCache):
        super().__init__(layer_id, config, cache)
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
            config.d_model, self.hidden_size, bias=config.include_bias, device=config.init_device
        )

    def reset_parameters(self):
        super().reset_parameters()
        self.attn_norm.reset_parameters()
        self.ff_norm.reset_parameters()
        # NOTE: the standard deviation for these weights does not depend on the layer.
        init_weights(
            self.config, self.att_proj, d=self.config.d_model, layer_id=None, type_of_module=ModuleType.in_module
        )
        init_weights(
            self.config, self.ff_proj, d=self.config.d_model, layer_id=None, type_of_module=ModuleType.in_module
        )

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
        if self._activation_checkpoint_fn is not None:
            q, k, v = self.att_proj(self._activation_checkpoint_fn(self.attn_norm, x)).split(
                self.fused_dims, dim=-1
            )
        else:
            q, k, v = self.att_proj(self.attn_norm(x)).split(self.fused_dims, dim=-1)

        # Get attention scores.
        if self._activation_checkpoint_fn is not None:
            att, cache = self._activation_checkpoint_fn(  # type: ignore
                self.attention, q, k, v, attention_bias, layer_past=layer_past, use_cache=use_cache
            )
        else:
            att, cache = self.attention(q, k, v, attention_bias, layer_past=layer_past, use_cache=use_cache)

        # Add attention scores.
        # shape: (B, T, C)
        x = x + self.dropout(att)

        # Add feed-forward projection.
        # shape: (batch_size, seq_len, d_model)
        og_x = x
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.ff_norm, x)  # type: ignore
        else:
            x = self.ff_norm(x)
        x = self.ff_proj(x)
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.act, x)  # type: ignore
        else:
            x = self.act(x)
        x = self.ff_out(x)
        x = self.dropout(x)
        x = og_x + x

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

    def __init__(self, layer_id: int, config: ModelConfig, cache: BufferCache):
        super().__init__(layer_id, config, cache)
        self.norm = LayerNorm.build(config)
        # Fused attention and feed-forward projection.
        # NOTE: we could also fuse the attention and feed-forward output projections but we
        # found that didn't help, possibly because of the overhead of joining the `att` and
        # `ff` activations together. See https://github.com/allenai/LLM/pull/79 for details.
        if config.multi_query_attention:
            self.fused_dims = (
                config.d_model,
                config.d_model // config.n_heads,
                config.d_model // config.n_heads,
                self.hidden_size,
            )
        else:
            self.fused_dims = (config.d_model, config.d_model, config.d_model, self.hidden_size)
        self.fused_attn_ff_proj = nn.Linear(
            config.d_model, sum(self.fused_dims), bias=config.include_bias, device=config.init_device
        )

    def reset_parameters(self):
        super().reset_parameters()
        self.norm.reset_parameters()
        # NOTE: the standard deviation for these weights does not depend on the layer.
        init_weights(
            self.config,
            self.fused_attn_ff_proj,
            d=self.config.d_model,
            layer_id=None,
            type_of_module=ModuleType.in_module,
        )

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
        # shape of ff:      (batch_size, seq_len, hidden_size)
        if self._activation_checkpoint_fn is not None:
            q, k, v, ff = self.fused_attn_ff_proj(self._activation_checkpoint_fn(self.norm, x)).split(
                self.fused_dims, dim=-1
            )
        else:
            q, k, v, ff = self.fused_attn_ff_proj(self.norm(x)).split(self.fused_dims, dim=-1)

        # Get attention scores.
        # shape: (B, T, C)
        if self._activation_checkpoint_fn is not None:
            att, cache = self._activation_checkpoint_fn(  # type: ignore
                self.attention, q, k, v, attention_bias, layer_past=layer_past, use_cache=use_cache
            )
        else:
            att, cache = self.attention(q, k, v, attention_bias, layer_past=layer_past, use_cache=use_cache)

        # Apply output projections (and activation function) and sum the results.
        # We keep these projections separate because we found that we got better throughput this
        # way compared to fusing them.
        if self._activation_checkpoint_fn is not None:
            return (
                x + self.dropout(self.ff_out(self._activation_checkpoint_fn(self.act, ff))) + self.dropout(att),
                cache,
            )
        else:
            return (
                x + self.dropout(self.ff_out(self.act(ff))) + self.dropout(att),
                cache,
            )


class OlmoLlamaBlock(OlmoBlock):
    """
    This is a transformer block where the output is computed as ``MLP(LN(x + Attention(LN(x))))``
    (plus another skip connection). This block is similar to `OlmoSequentialBlock`
    but some operations have slightly different implementations to imitate the
    behavior of Llama.
    """

    def __init__(self, layer_id: int, config: ModelConfig, cache: BufferCache):
        super().__init__(layer_id, config, cache)
        # Layer norms.
        self.attn_norm = LayerNorm.build(config)
        self.ff_norm = LayerNorm.build(config)
        self.__cache = cache

        # Attention input projection. Projects x -> (q, k, v)
        if config.multi_query_attention:
            q_proj_out_dim = config.d_model
            k_proj_out_dim = config.d_model // config.n_heads
            v_proj_out_dim = config.d_model // config.n_heads
        else:
            q_proj_out_dim = config.d_model
            k_proj_out_dim = config.d_model
            v_proj_out_dim = config.d_model
        self.q_proj = nn.Linear(
            config.d_model, q_proj_out_dim, bias=config.include_bias, device=config.init_device
        )
        self.k_proj = nn.Linear(
            config.d_model, k_proj_out_dim, bias=config.include_bias, device=config.init_device
        )
        self.v_proj = nn.Linear(
            config.d_model, v_proj_out_dim, bias=config.include_bias, device=config.init_device
        )

        # Feed-forward input projection.
        self.ff_proj = nn.Linear(
            config.d_model, self.hidden_size, bias=config.include_bias, device=config.init_device
        )

    def reset_parameters(self):
        super().reset_parameters()
        self.attn_norm.reset_parameters()
        self.ff_norm.reset_parameters()
        # NOTE: the standard deviation for these weights does not depend on the layer.
        init_weights(self.config, self.q_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.k_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.v_proj, d=self.config.d_model, layer_id=None)
        init_weights(self.config, self.ff_proj, d=self.config.d_model, layer_id=None)

    def _scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
    ) -> torch.Tensor:
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))

        if is_causal:
            assert attn_mask is None

            query_len, key_len = q.shape[-2], k.shape[-2]  # could be different if layer_past not None
            attn_bias = get_causal_attention_bias(self.__cache, key_len, q.device)[:, :, :query_len, :key_len]
        elif attn_mask is not None:
            attn_bias = attn_mask.to(q.dtype)
        else:
            attn_bias = torch.zeros_like(attn_weights)

        attn_weights += attn_bias
        attn_weights = nn.functional.softmax(attn_weights, dim=-1).to(q.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=dropout_p)
        return torch.matmul(attn_weights, v)

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
        x_normed = self.attn_norm(x)
        q = self.q_proj(x_normed)
        k = self.k_proj(x_normed)
        v = self.v_proj(x_normed)

        # Get attention scores.
        att, cache = self.attention(q, k, v, attention_bias, layer_past=layer_past, use_cache=use_cache)

        # Add attention scores.
        # shape: (B, T, C)
        x = x + self.dropout(att)

        # Add feed-forward projection.
        # shape: (batch_size, seq_len, d_model)
        og_x = x
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.ff_norm, x)  # type: ignore
        else:
            x = self.ff_norm(x)
        x = self.ff_proj(x)
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.act, x)  # type: ignore
        else:
            x = self.act(x)
        x = self.ff_out(x)
        x = self.dropout(x)
        x = og_x + x

        return x, cache


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


class OlmoBlockGroup(nn.ModuleList):
    def __init__(self, config: ModelConfig, layer_offset: int, modules: Optional[Iterable[nn.Module]] = None):
        super().__init__(modules)
        self.config = config
        self.layer_offset = layer_offset
        self.activation_checkpointing_strategy: Optional[ActivationCheckpointingStrategy] = None
        self._activation_checkpoint_fn = activation_checkpoint_function(self.config)

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.FloatTensor] = None,
        layers_past: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = [] if use_cache else None
        for block_idx, block in enumerate(self):
            layer_past = None if layers_past is None else layers_past[block_idx]
            block_idx += self.layer_offset
            if (
                (self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.whole_layer)
                or (
                    self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_two
                    and block_idx % 2 == 0
                )
                or (
                    self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_three
                    and block_idx % 3 == 0
                )
                or (
                    self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_four
                    and block_idx % 4 == 0
                )
            ):
                # shape: (batch_size, seq_len, d_model)
                x, cache = self._activation_checkpoint_fn(  # type: ignore
                    block, x, attention_bias=attention_bias, layer_past=layer_past, use_cache=use_cache
                )
            else:
                # shape: (batch_size, seq_len, d_model)
                x, cache = block(x, attention_bias=attention_bias, layer_past=layer_past, use_cache=use_cache)
            if attn_key_values is not None:
                assert cache is not None
                attn_key_values.append(cache)
        return x, attn_key_values

    def reset_parameters(self):
        for block in self:
            block.reset_parameters()

    def set_activation_checkpointing(self, strategy: Optional[ActivationCheckpointingStrategy]):
        self.activation_checkpointing_strategy = strategy
        for block in self:
            block.set_activation_checkpointing(strategy)


class Resampler(nn.Module):
    def __init__(self, config: ModelConfig, cache: BufferCache):
        super().__init__()
        self.config = config
        self.__cache = cache
        cfg = self.config.resampler
        assert cfg.d_query % cfg.n_heads == 0

        if config.projector.d_visual != cfg.d_query:
            self.kv_proj = nn.Linear(
                config.projector.d_visual, cfg.d_query, bias=config.include_bias, device=config.init_device
            )
        else:
            self.kv_proj = nn.Identity()

        # Layer norms.
        self.q_norm = LayerNorm.build(config, size=cfg.d_query)
        self.kv_norm = LayerNorm.build(config, size=cfg.d_query)

        # Positional embedding
        self.pos_emb = TwoDimSinCosEmbedding(config, self.__cache)

        # Query embeddings
        self.query = nn.Parameter(
            torch.zeros(cfg.n_queries, cfg.d_query, device=config.init_device)
        )

        # Query/key/value projection.
        self.query_proj = nn.Linear(
            cfg.d_query, cfg.d_query, bias=config.include_bias, device=config.init_device  
        )
        self.key_proj = nn.Linear(
            cfg.d_query, cfg.d_query, bias=config.include_bias, device=config.init_device
        )
        self.value_proj = nn.Linear(
            cfg.d_query, cfg.d_query, bias=config.include_bias, device=config.init_device
        )
        # Attention output projection.
        self.attn_out = nn.Linear(
            cfg.d_query, cfg.d_query, bias=config.include_bias, device=config.init_device
        )
    
    def reset_parameters(self):
        cfg = self.config.resampler
        self.q_norm.reset_parameters()
        self.kv_norm.reset_parameters()
        nn.init.trunc_normal_(self.query, std=0.02)
        if self.config.projector.d_visual != cfg.d_query:
            init_weights(
                self.config, self.kv_proj, d=self.config.projector.d_visual, layer_id=None, type_of_module=ModuleType.in_module
            )
        init_weights(
            self.config, self.query_proj, d=cfg.d_query, layer_id=None, type_of_module=ModuleType.in_module
        )
        init_weights(
            self.config, self.key_proj, d=cfg.d_query, layer_id=None, type_of_module=ModuleType.in_module
        )
        init_weights(
            self.config, self.value_proj, d=cfg.d_query, layer_id=None, type_of_module=ModuleType.in_module
        )
        init_weights(
            self.config, self.attn_out, d=cfg.d_query, layer_id=None, type_of_module=ModuleType.in_module
        )

    def attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        cfg = self.config.resampler
        B, q_len, C = q.size()  # batch size, q_len, d_query
        kv_len = k.shape[1]

        # Move head forward to be next to the batch dim.
        # shape: (B, nh, q_len, hs)
        q = q.view(B, q_len, cfg.n_heads, C // cfg.n_heads).transpose(1, 2)
        # shape: (B, nh, kv_len, hs)
        k = k.view(B, kv_len, cfg.n_heads, C // cfg.n_heads).transpose(1, 2)
        # shape: (B, nh, kv_len, hs)
        v = v.view(B, kv_len, cfg.n_heads, C // cfg.n_heads).transpose(1, 2)

        # Get the attention scores.
        # shape: (B, nh, q_len, hs)
        att = F.scaled_dot_product_attention(q, k, v)

        # Re-assemble all head outputs side-by-side.
        att = att.transpose(1, 2).contiguous().view(B, q_len, C)

        # Apply output projection.
        return self.attn_out(att)
    
    def forward(self, x):
        """Get the reduced number of visual tokens using a single cross-attention layer."""
        # Get query, key value projections.
        # Apply sinusoidal positional embeddings to query/key, do interpolation if needed.
        # shape:
        #  - x -> k/v: (batch_size, num_tokens, d_query)
        #  - q: (batch_size, n_queries, d_query)
        q = self.q_norm(self.query).unsqueeze(0).expand(x.size(0), -1, -1)
        q = self.query_proj(self.pos_emb(q))
        x = self.kv_norm(self.kv_proj(x))
        k = self.key_proj(self.pos_emb(x))
        v = self.value_proj(x)
        # Apply sdpa.
        out = self.attention(q, k, v)
        return out


class Projector(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        cfg = self.config.projector
        self.in_sizes = [cfg.d_visual if config.resampler is None else config.resampler.d_query]
        self.in_sizes += [config.d_model] * (cfg.n_layers - 1)
        self.out_sizes = [config.d_model] * cfg.n_layers
        if cfg.n_layers > 1:
            self.act = Activation.build(config, activation_type=cfg.activation_type)
        ff_layers = [
            nn.Linear(in_size, out_size, bias=config.include_bias, device=config.init_device)
            for in_size, out_size in zip(self.in_sizes, self.out_sizes)
        ]
        self.ff_layers = nn.ModuleList(ff_layers)
    
    def reset_parameters(self):
        for i, layer in enumerate(self.ff_layers):
            init_weights(
                self.config, layer, d=self.in_sizes[i], layer_id=None, type_of_module=ModuleType.in_module
            )
    
    def forward(self, x):
        # Project visual features into language embedding space using an MLP.
        cfg = self.config.projector
        for layer in self.ff_layers:
            x = layer(x)
            if cfg.n_layers > 1:
                x = self.act(x)
        return x


class OlmoVisionBackbone(nn.Module):
    def __init__(self, config: ModelConfig, cache: BufferCache):
        super().__init__()
        self.config = config
        self.__cache = cache
        v_cfg = self.config.vision_backbone

        self.image_newline = None
        if v_cfg.anyres:
            self.image_newline = nn.Parameter(torch.zeros(config.d_model, device=config.init_device))

        self.resampler = None
        if config.resampler is not None:
            assert v_cfg.anyres
            self.resampler = Resampler(config, self.__cache)
        
        self.projector = Projector(config)

    @classmethod
    def build(cls, config: ModelConfig, cache: BufferCache) -> OlmoVisionBackbone:
        v_cfg = config.vision_backbone
        assert v_cfg is not None
        return OlmoPretrainedVisionBackbone(config, cache)

    def get_image_preprocessor(self):
        raise NotImplementedError()

    def reset_parameters(self):
        if self.resampler is not None:
            self.resampler.reset_parameters()
        self.projector.reset_parameters()
        if self.image_newline is not None:
            embed_std = 1 / math.sqrt(self.config.d_model)
            nn.init.normal_(self.image_newline, std=embed_std)


class OlmoPretrainedVisionBackbone(OlmoVisionBackbone):
    def __init__(self, config: ModelConfig, cache: BufferCache):
        super().__init__(config, cache)
        v_cfg = self.config.vision_backbone
        image_size = (v_cfg.image_width, v_cfg.image_height)
        patch_size = (v_cfg.patch_width, v_cfg.patch_height)
        resample_tokens = config.resampler.n_queries if config.resampler is not None else None
        if v_cfg.anyres:
            self.preprocessor = AnyResClipImageResize(
                image_size, patch_size, v_cfg.possible_resolutions, resample_tokens
            )
        else:
            self.preprocessor = ClipImageResize(image_size, patch_size, v_cfg.pad_image)
        
        self.image_token_sizer = self.preprocessor.image_token_sizer()
        
        if config.init_device == "meta" and config.low_cpu_fsdp:
            """
            for FSDP, we can save cpu memory by loading pretrained model on rank0 only.
            this avoids cpu oom when loading large models.
            reference: https://github.com/facebookresearch/llama-recipes/blob/main/src/llama_recipes/finetuning.py
            """
            rank = get_global_rank()
            if rank == 0:
                self.vision_tower = create_clip_model(v_cfg.name, v_cfg.pretrained, cache_dir=config.cache_dir)
            else:
                with torch.device("meta"):
                    self.vision_tower = create_clip_model(v_cfg.name, device='meta')
        else:
            device = config.init_device if config.init_device != "meta" else "cpu"
            self.vision_tower = create_clip_model(v_cfg.name, v_cfg.pretrained, device=device, cache_dir=config.cache_dir)

        if v_cfg.frozen:
            for param in self.vision_tower.parameters():
                param.requires_grad = False
        
    def reset_parameters(self):
        super().reset_parameters()
    
    def get_image_preprocessor(self):
        return self.preprocessor
    
    def encode_images(self, image_patches: torch.Tensor) -> torch.Tensor:
        v_cfg = self.config.vision_backbone
        # Output all hidden states
        # n_layers x (batch_num_patches, 1+n_tokens, d_visual)
        image_features = self.vision_tower(image_patches)
        # Features from the selected layer
        image_features = image_features[v_cfg.select_layer][:, 1:] # remove the [CLS] token
        # Reduce the number of tokens per patch, if needed
        if self.resampler is not None:
            image_features = self.resampler(image_features)
        # Project vision features into language embedding space using an MLP
        # (batch_num_patches, n_tokens, d_model)
        image_features = self.projector(image_features)
        return image_features

    def merge_patches(self, patch_features: torch.Tensor, image_size: List[int]) -> torch.Tensor:
        grid_width, grid_height = self.image_token_sizer.get_grid_shape(*[int(x) for x in image_size])
        # downsampled_feature: features for the downsampled image
        downsampled_feature, patch_features = patch_features[0], patch_features[1:]
        height = width = int(math.sqrt(self.image_token_sizer.n_tokens))
        assert height * width == downsampled_feature.shape[0]
        patch_features = patch_features.view(
            grid_height, grid_width, height, width, -1
        )
        patch_features = patch_features.permute(4, 0, 2, 1, 3).contiguous()
        patch_features = patch_features.flatten(1, 2).flatten(2, 3) # (d_model, padded_height, padded_width)
        if self.resampler is None:
            # Remove zero paddings
            patch_features = unpad_image(patch_features, image_size) # (d_model, H, W)
        # Add a new_line embedding after each row
        new_patch_features = torch.cat(
            [
                patch_features,
                self.image_newline[:, None, None].expand(*patch_features.shape[:-1], 1).to(patch_features.device)
            ],
            dim=-1
        )
        # (n_tokens, d_model)
        new_patch_features = new_patch_features.flatten(1, 2).transpose(0, 1)
        new_patch_features = torch.cat((downsampled_feature, new_patch_features), dim=0)

        return new_patch_features
            
    
    def forward(
        self, image_patches: torch.Tensor, num_patches_per_image: torch.Tensor,
        image_sizes: torch.Tensor = None,
    ) -> torch.Tensor:
        # image_patches: (batch_size, num_patches, 3, height, width)
        # num_patches_per_image: (batch_size, n_images)
        # image_sizes: (batch_size, num_images, 2)
        v_cfg = self.config.vision_backbone
        # image_features: (batch_num_patches, n_tokens, d_model)
        batch_size, num_patches = image_patches.shape[:2]
        image_features = self.encode_images(image_patches.flatten(0, 1))
        # Remove paddings
        image_features = image_features.view(batch_size, num_patches, *image_features.shape[1:])
        n_patches_seq = num_patches_per_image.sum(dim=1)
        max_patches = n_patches_seq.max().item()
        mask = torch.arange(max_patches, device=n_patches_seq.device).expand(batch_size, max_patches) < n_patches_seq.unsqueeze(1)
        # (batch_num_patches, n_tokens, d_model)
        image_features = image_features[mask]
        if v_cfg.anyres:
            # TODO Need to implement faster logic for anyres
            # Each image might have different resolution, so consisting of different number of patches

            # Group patches by image
            # Each element in the tuple is a tensor of shape (n_patches, n_tokens, d_model)
            # where n_patches is the number of patches in the sequence
            image_features = torch.split(
                image_features, n_patches_seq.tolist(), dim=0,
            )
            # Each elment in the tuple is a tensor of shape (n_images, 2)
            # where n_images is the number of images in the sequence
            image_sizes = image_sizes[num_patches_per_image > 0]
            image_sizes = torch.split(
                image_sizes, (num_patches_per_image > 0).sum(dim=1).tolist(), dim=0
            )
            # Merge patches
            new_image_features = []
            masked_num_patches_per_iamge = num_patches_per_image[n_patches_seq > 0]
            for sequence_patch_features, num_patches, sequence_image_size in zip(image_features, masked_num_patches_per_iamge, image_sizes):
                # num_patches: (n_images,)
                # each element in the sequence_patch_features: patch features for each image
                # (n_patches_for_the_image, n_tokens, d_model)
                sequence_patch_features = torch.split(
                    sequence_patch_features, num_patches.tolist(), dim=0
                )
                for patch_features, image_size in zip(sequence_patch_features, sequence_image_size):
                    patch_features = self.merge_patches(patch_features, image_size.tolist())
                    new_image_features.append(patch_features)
            image_features = torch.cat(new_image_features, dim=0)
        else:
            image_features = image_features.flatten(0, 1)
        
        # (batch_n_tokens, d_model)
        return image_features


class Olmo(nn.Module):
    def __init__(self, config: ModelConfig, init_params: bool = True):
        super().__init__()
        self.config = config
        self.__cache = BufferCache()

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

        self.activation_checkpointing_strategy: Optional[ActivationCheckpointingStrategy] = None
        self._activation_checkpoint_fn: Callable = activation_checkpoint_function(self.config)

        if not (
            0 < self.config.block_group_size <= self.config.n_layers
            and self.config.n_layers % self.config.block_group_size == 0
        ):
            raise OlmoConfigurationError("n layers must be divisible by block group size")

        torch.backends.cuda.enable_flash_sdp(self.config.flash_attention)
        torch.backends.cuda.enable_mem_efficient_sdp(False)  # this is super slow so make sure torch won't use it

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(
                    config.embedding_size or config.vocab_size, config.d_model, device=config.init_device
                ),
                emb_drop=Dropout(config.embedding_dropout),
                ln_f=LayerNorm.build(config),
            )
        )

        blocks = [OlmoBlock.build(i, config, self.__cache) for i in range(config.n_layers)]
        if self.config.block_group_size > 1:
            block_groups = [
                OlmoBlockGroup(config, i, blocks[i : i + config.block_group_size])
                for i in range(0, config.n_layers, config.block_group_size)
            ]
            self.transformer.update({"block_groups": nn.ModuleList(block_groups)})
        else:
            self.transformer.update({"blocks": nn.ModuleList(blocks)})

        if not (self.config.alibi or self.config.rope):
            self.transformer.update(
                {"wpe": nn.Embedding(config.max_sequence_length, config.d_model, device=config.init_device)}
            )

        if not config.weight_tying:
            self.transformer.update(
                {
                    "ff_out": nn.Linear(
                        config.d_model,
                        config.embedding_size or config.vocab_size,
                        bias=config.include_bias,
                        device=config.init_device,
                    )
                }
            )

        self.vision_backbone: Optional[OlmoVisionBackbone] = None
        if config.vision_backbone is not None:
            self.vision_backbone = OlmoVisionBackbone.build(config, self.__cache)

        # When `init_device="meta"` FSDP will call `reset_parameters()` to initialize weights.
        if init_params and self.config.init_device != "meta":
            self.reset_parameters()
        self.__num_fwd_flops: Optional[int] = None

        # Warm up cache.
        if self.config.alibi:
            get_causal_attention_bias(self.__cache, config.max_sequence_length, _non_meta_init_device(config))
            self.get_alibi_attention_bias(config.max_sequence_length, _non_meta_init_device(config))
        
        # Freeze the model if needed
        if self.config.llm_frozen:
            for param in self.transformer.parameters():
                param.requires_grad = False
        
        if self.config.llm_load_path:
            from .util import resource_path
            from pathlib import Path
            state_dict_path = resource_path(
                Path(self.config.llm_load_path).parent, Path(self.config.llm_load_path).name,
                local_cache=self.config.cache_dir,
            )
            assert state_dict_path.is_file(), f"Model file {str(state_dict_path)} not found"
            if config.init_device == "meta" and config.low_cpu_fsdp:
                rank = get_global_rank()
                if rank == 0:
                    state_dict = torch.load(state_dict_path, map_location="cpu")
                    self.transformer.to_empty(device="cpu")
                    self.transformer.load_state_dict(state_dict)
            else:
                state_dict = torch.load(state_dict_path, map_location="cpu")
                self.transformer.to_empty(device="cpu")
                self.transformer.load_state_dict(state_dict)

    def set_activation_checkpointing(self, strategy: Optional[ActivationCheckpointingStrategy]):
        self.activation_checkpointing_strategy = strategy
        if self.config.block_group_size != 1:
            for block_group in self.transformer.block_groups:
                block_group.set_activation_checkpointing(strategy)
        else:
            for block in self.transformer.blocks:
                block.set_activation_checkpointing(strategy)

    @property
    def device(self) -> torch.device:
        device: torch.device = self.transformer.wte.weight.device  # type: ignore
        if device.type == "meta":
            return _non_meta_init_device(self.config)
        else:
            return device

    def reset_parameters(self):
        log.info("Initializing model parameters...")
        # Top-level embeddings / linear layers.
        # Vision backbone.
        if self.vision_backbone is not None:
            self.vision_backbone.reset_parameters()
        
        if self.config.llm_load_path:
            log.info("LLM Transformer loaded from a checkpoint, skipping parameter initialization")
            return

        init_weights(
            self.config,
            self.transformer.wte,  # type: ignore
            std_factor=(0.5 * math.sqrt(self.config.d_model)) if self.config.scale_logits else 1.0,
            type_of_module=ModuleType.emb,
        )
        if hasattr(self.transformer, "wpe"):
            init_weights(self.config, self.transformer.wpe, type_of_module=ModuleType.emb)  # type: ignore

        # Top-level layer norm.
        self.transformer.ln_f.reset_parameters()  # type: ignore

        # Output weights.
        if hasattr(self.transformer, "ff_out"):
            init_weights(self.config, self.transformer.ff_out, type_of_module=ModuleType.final_out)  # type: ignore

        # Let the blocks handle themselves.
        if self.config.block_group_size == 1:
            for block in self.transformer.blocks:
                block.reset_parameters()
        else:
            for block_group in self.transformer.block_groups:
                block_group.reset_parameters()

    def get_alibi_attention_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if (alibi_bias := self.__cache.get("alibi_attention_bias")) is not None and alibi_bias.shape[
            -1
        ] >= seq_len:
            if alibi_bias.device != device:
                alibi_bias = alibi_bias.to(device)
                self.__cache["alibi_attention_bias"] = alibi_bias
            return alibi_bias
        with torch.autocast(device.type, enabled=False):
            alibi_bias = alibi_attention_bias(seq_len, self.config, device)
        self.__cache["alibi_attention_bias"] = alibi_bias
        return alibi_bias

    def forward(
        self,
        input_ids: torch.LongTensor,
        input_embeddings: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        past_key_values: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
        image_patches: Optional[torch.Tensor] = None,
        image_offsets: Optional[torch.Tensor] = None,
        num_patches_per_image: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        last_logits_only: bool = False,
    ) -> OlmoOutput:
        """
        :param input_ids: A tensor of shape `(batch_size, seq_len)`.
        :param input_embeddings: A tensor of shape `(batch_size, seq_len, d_model)` with input
            embeddings. When provided, it is treated as the output of the input embedding layer.
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
        :param image_patches: For multi-modal models, image patch inputs of shape
            `(batch_size, num_patches, 3, height, width)`.
        :param image_offsets: For mulit-modal models, specifies where in the input IDs the embedded image
            patches should go. Shape `(batch_size, n_tokens)`.
            n_tokens is the (maximum) number of image tokens in each sequence
        :param num_patches_per_image: For mulit-modal models, specifies the number of patches in each image
            in the batch. Shape `(batch_size, n_images)`.
            n_images is the (maximum) number of images in each sequence
        :param image_sizes: For mulit-modal models, specifies the size of each image
            in the batch. Shape `(batch_size, num_images, 2)`.
            batch_num_images is the number of images in the batch
        :param use_cache: If `True`, return key and value tensors for each block.
        :param last_logits_only: If `True`, only compute the logits for the last token of each sequence.
            This can speed up decoding when you only care about the next token.
        """
        if past_key_values:
            assert len(past_key_values) == self.config.n_layers

        batch_size, seq_len = input_ids.size() if input_embeddings is None else input_embeddings.size()[:2]
        if past_key_values is None:
            past_length = 0
        else:
            past_length = past_key_values[0][0].size(-2)

        img_emb: Optional[torch.Tensor] = None
        if image_patches is not None:
            # Get image patch embeddings.
            assert self.vision_backbone is not None
            assert image_offsets is not None
            # shape: (batch_n_tokens, d_model)
            img_emb = self.vision_backbone(image_patches, num_patches_per_image, image_sizes)

        # Get embeddings of input.
        # shape: (batch_size, seq_len, d_model)
        x = self.transformer.wte(input_ids) if input_embeddings is None else input_embeddings  # type: ignore

        if img_emb is not None:
            # Inject image patch embeddings into input embeddings.
            assert image_offsets is not None
            image_offsets_mask = image_offsets >= 0
            batch_idx = torch.arange(0, batch_size, device=x.device).repeat_interleave(
                image_offsets_mask.sum(dim=-1)
            )
            x.index_put_((batch_idx, image_offsets[image_offsets_mask]), img_emb)

        if not (self.config.alibi or self.config.rope):
            # Get positional embeddings.
            # shape: (1, seq_len)
            pos = torch.arange(past_length, past_length + seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
            # shape: (1, seq_len, d_model)
            pos_emb = self.transformer.wpe(pos)  # type: ignore
            x = pos_emb + x

        # Add input + positional embeddings and apply dropout.
        # shape: (batch_size, seq_len, d_model)
        x = self.transformer.emb_drop(x)  # type: ignore

        # Transform the attention mask into what the blocks expect.
        if attention_mask is not None:
            # shape: (batch_size, 1, 1, seq_len)
            attention_mask = attention_mask.to(dtype=torch.float).view(batch_size, -1)[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * torch.finfo(attention_mask.dtype).min

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
                attention_bias = get_causal_attention_bias(
                    self.__cache, past_length + seq_len, x.device
                ) + self.get_alibi_attention_bias(past_length + seq_len, x.device)
            elif attention_bias is None:
                attention_bias = get_causal_attention_bias(self.__cache, past_length + seq_len, x.device)
            elif attention_bias.dtype in (torch.int8, torch.bool):
                attention_bias = attention_bias.to(dtype=torch.float)
                attention_bias.masked_fill_(attention_bias == 0.0, torch.finfo(attention_bias.dtype).min)

            # Transform to the right shape and data type.
            mask_len = seq_len
            if attention_mask is not None:
                mask_len = attention_mask.shape[-1]
            elif past_key_values is not None:
                mask_len = past_key_values[0][0].shape[-2] + seq_len
            attention_bias = attention_bias[:, :, :mask_len, :mask_len].to(dtype=torch.float)

            # Add in the masking bias.
            if attention_mask is not None:
                attention_bias = attention_bias + attention_mask
                # Might get -infs after adding attention mask, since dtype.min + dtype.min = -inf.
                # `F.scaled_dot_product_attention()` doesn't handle -inf like you'd expect, instead
                # it can produce NaNs.
                ensure_finite_(attention_bias, check_neg_inf=True, check_pos_inf=False)

        attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = [] if use_cache else None

        # Apply blocks one-by-one.
        if self.config.block_group_size == 1:
            for block_idx, block in enumerate(self.transformer.blocks):
                layer_past = None if past_key_values is None else past_key_values[block_idx]
                if (
                    (self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.whole_layer)
                    or (
                        self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_two
                        and block_idx % 2 == 0
                    )
                    or (
                        self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_three
                        and block_idx % 3 == 0
                    )
                    or (
                        self.activation_checkpointing_strategy == ActivationCheckpointingStrategy.one_in_four
                        and block_idx % 4 == 0
                    )
                ):
                    # shape: (batch_size, seq_len, d_model)
                    x, cache = self._activation_checkpoint_fn(
                        block, x, attention_bias=attention_bias, layer_past=layer_past, use_cache=use_cache
                    )
                else:
                    # shape: (batch_size, seq_len, d_model)
                    x, cache = block(x, attention_bias=attention_bias, layer_past=layer_past, use_cache=use_cache)
                if attn_key_values is not None:
                    assert cache is not None
                    attn_key_values.append(cache)
        else:
            for group_idx, block_group in enumerate(self.transformer.block_groups):
                layers_past = (
                    None
                    if past_key_values is None
                    else past_key_values[
                        group_idx * self.config.block_group_size : (group_idx + 1) * self.config.block_group_size
                    ]
                )
                x, cache = block_group(
                    x, attention_bias=attention_bias, layers_past=layers_past, use_cache=use_cache
                )
                if attn_key_values is not None:
                    assert cache is not None
                    attn_key_values.extend(cache)

        if last_logits_only:
            # shape: (batch_size, 1, d_model)
            x = x[:, -1, :].unsqueeze(1)

        # Apply final layer norm.
        # shape: (batch_size, seq_len or 1, d_model)
        x = self.transformer.ln_f(x)  # type: ignore

        # Get logits.
        # shape: (batch_size, seq_len or 1, vocab_size)
        if self.config.weight_tying:
            logits = F.linear(x, self.transformer.wte.weight, None)  # type: ignore
        else:
            logits = self.transformer.ff_out(x)  # type: ignore
        if self.config.scale_logits:
            logits.mul_(1 / math.sqrt(self.config.d_model))

        return OlmoOutput(logits=logits, attn_key_values=attn_key_values)  # type: ignore[arg-type]

    def get_fsdp_wrap_policy(self, wrap_strategy: Optional[FSDPWrapStrategy] = None):
        if wrap_strategy is None:
            return None
        elif wrap_strategy == FSDPWrapStrategy.by_block:

            def fsdp_wrap_fn(module, recurse: bool = True, nonwrapped_numel: int = 0):
                del nonwrapped_numel
                if recurse:
                    return True  # always recurse for simplicity
                return isinstance(module, (OlmoVisionBackbone, OlmoBlock))

            return fsdp_wrap_fn
        elif wrap_strategy == FSDPWrapStrategy.by_block_and_size:

            def fsdp_wrap_fn(module, recurse: bool = True, nonwrapped_numel: int = 0):
                del nonwrapped_numel
                if recurse:
                    return True  # always recurse for simplicity
                return not isinstance(module, nn.modules.linear.NonDynamicallyQuantizableLinear) and \
                    isinstance(module, (OlmoBlock, nn.Linear, nn.Embedding))

            return fsdp_wrap_fn
        elif wrap_strategy == FSDPWrapStrategy.by_block_group:
            if self.config.block_group_size <= 1:
                raise OlmoConfigurationError(
                    "'by_block_group' FSDP wrapping strategy requires block group size greater than 1"
                )

            def fsdp_wrap_fn(module, recurse: bool = True, nonwrapped_numel: int = 0):
                del nonwrapped_numel
                if recurse:
                    return True  # always recurse for simplicity
                return isinstance(module, (OlmoVisionBackbone, OlmoBlockGroup))

            return fsdp_wrap_fn
        elif wrap_strategy == FSDPWrapStrategy.by_block_group_and_size:
            if self.config.block_group_size <= 1:
                raise OlmoConfigurationError(
                    "'by_block_group_and_size' FSDP wrapping strategy requires block group size greater than 1"
                )

            def fsdp_wrap_fn(module, recurse: bool = True, nonwrapped_numel: int = 0):
                del nonwrapped_numel
                if recurse:
                    return True  # always recurse for simplicity
                return not isinstance(module, nn.modules.linear.NonDynamicallyQuantizableLinear) and \
                    isinstance(module, (OlmoBlockGroup, nn.Linear, nn.Embedding))

            return fsdp_wrap_fn
        elif wrap_strategy == FSDPWrapStrategy.size_based:
            from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

            return partial(size_based_auto_wrap_policy, force_leaf_modules={OlmoVisionBackbone})
        elif wrap_strategy in {
            FSDPWrapStrategy.one_in_two,
            FSDPWrapStrategy.one_in_three,
            FSDPWrapStrategy.one_in_four,
            FSDPWrapStrategy.one_in_five,
        }:
            c = {
                FSDPWrapStrategy.one_in_two: 2,
                FSDPWrapStrategy.one_in_three: 3,
                FSDPWrapStrategy.one_in_four: 4,
                FSDPWrapStrategy.one_in_five: 5,
            }[wrap_strategy]

            def fsdp_wrap_fn(module, recurse: bool = True, nonwrapped_numel: int = 0):
                del nonwrapped_numel
                if recurse:
                    return True  # always recurse for simplicity
                return isinstance(module, (OlmoVisionBackbone, OlmoBlock)) and module.layer_id % c == 0

            return fsdp_wrap_fn
        else:
            raise NotImplementedError(wrap_strategy)

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

        For an explanation of the other arguments, see :class:`BeamSearch`.
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
            past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
        ) -> Dict[str, torch.Tensor]:
            out = {}
            for i, (key, value) in enumerate(past_key_values):
                out[f"past_key_{i}"] = key
                out[f"past_value_{i}"] = value
            return out

        def unflatten_past_key_values(
            past_key_values: Dict[str, torch.Tensor],
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
    def from_checkpoint(
        cls, checkpoint_dir: PathOrStr, device: str = "cpu", checkpoint_type: Optional[CheckpointType] = None
    ) -> Olmo:
        """
        Load an OLMo model from a checkpoint.
        """
        from .util import resource_path

        # Guess checkpoint type.
        if checkpoint_type is None:
            try:
                if resource_path(checkpoint_dir, "model.pt").is_file():
                    checkpoint_type = CheckpointType.unsharded
                else:
                    checkpoint_type = CheckpointType.sharded
            except FileNotFoundError:
                checkpoint_type = CheckpointType.sharded

        # Load config.
        config_path = resource_path(checkpoint_dir, "config.yaml")
        model_config = ModelConfig.load(config_path, key="model", validate_paths=False)

        if checkpoint_type == CheckpointType.unsharded:
            # Initialize model (always on CPU to start with so we don't run out of GPU memory).
            model_config.init_device = "cpu"
            model = Olmo(model_config)

            # Load state dict directly to target device.
            state_dict_path = resource_path(checkpoint_dir, "model.pt")
            state_dict = torch.load(state_dict_path, map_location="cpu")
            model.load_state_dict(model._make_state_dict_compatible(state_dict)[0])
            model = model.to(torch.device(device))
        else:
            from .checkpoint import load_model_state

            # Initialize model on target device. In this case the state dict is loaded in-place
            # so it's not necessary to start on CPU if the target device is a GPU.
            model_config.init_device = device
            model = Olmo(model_config)

            # Load state dict in place.
            load_model_state(checkpoint_dir, model)

        return model.eval()

    def _make_state_dict_compatible(
        self, state_dict: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Set[str]]]:
        """
        Handles some cases where the state dict is valid yet may need to be transformed in order to
        be loaded.

        This modifies the state dict in-place and also returns it, along with a mapping of original key
        names to new key names in cases where the keys were simply renamed. That mapping can be used
        to make a corresponding optimizer state dict compatible as well.
        """
        import re
        from fnmatch import fnmatch

        new_keys_to_og_keys: Dict[str, str] = {}

        # Remove "_fsdp_wrapped_module." prefix from all keys. We don't want this prefix when the model is
        # not wrapped in FSDP. And when the model is wrapped in FSDP, loading this state dict will still work
        # fine without the prefixes. This also simplifies the other steps below.
        for key in list(state_dict.keys()):
            state_dict[(new_key := key.replace("_fsdp_wrapped_module.", ""))] = state_dict.pop(key)
            new_keys_to_og_keys[new_key] = key

        # For backwards compatibility prior to fixing https://github.com/allenai/LLM/issues/222
        if self.config.block_type == BlockType.sequential:
            for key in list(state_dict.keys()):
                if fnmatch(key, "transformer.*.norm.weight"):
                    tensor = state_dict.pop(key)
                    state_dict[(new_key := key.replace("norm.weight", "attn_norm.weight"))] = tensor
                    new_keys_to_og_keys[new_key] = new_keys_to_og_keys[key]
                    state_dict[(new_key := key.replace("norm.weight", "ff_norm.weight"))] = tensor.clone()
                    new_keys_to_og_keys[new_key] = new_keys_to_og_keys[key]
                    del new_keys_to_og_keys[key]
                elif fnmatch(key, "transformer.*.norm.bias"):
                    tensor = state_dict.pop(key)
                    state_dict[(new_key := key.replace("norm.bias", "attn_norm.bias"))] = tensor
                    new_keys_to_og_keys[new_key] = new_keys_to_og_keys[key]
                    state_dict[(new_key := key.replace("norm.bias", "ff_norm.bias"))] = tensor.clone()
                    new_keys_to_og_keys[new_key] = new_keys_to_og_keys[key]
                    del new_keys_to_og_keys[key]

        # For loading a state dict that was saved with a different `block_group_size`.
        if "transformer.block_groups.0.0.attn_out.weight" in state_dict.keys():
            state_dict_block_group_size = len(
                [k for k in state_dict.keys() if fnmatch(k, "transformer.block_groups.0.*.attn_out.weight")]
            )
        else:
            state_dict_block_group_size = 1
        if self.config.block_group_size != state_dict_block_group_size:
            log.info(
                f"Regrouping state dict blocks from group size {state_dict_block_group_size} to "
                f"group size {self.config.block_group_size}"
            )
            # For simplicity we're first going to flatten out the block groups in the state dict (if necessary)
            # and then (re-)group them into the right block sizes.
            if state_dict_block_group_size > 1:
                for key in list(state_dict.keys()):
                    if (m := re.match(r"transformer.block_groups\.(\d+)\.(\d+)\..*", key)) is not None:
                        group_idx, group_block_idx = int(m.group(1)), int(m.group(2))
                        block_idx = (group_idx * state_dict_block_group_size) + group_block_idx
                        state_dict[
                            (
                                new_key := key.replace(
                                    f"block_groups.{group_idx}.{group_block_idx}.", f"blocks.{block_idx}."
                                )
                            )
                        ] = state_dict.pop(key)
                        new_keys_to_og_keys[new_key] = new_keys_to_og_keys.pop(key)

            if self.config.block_group_size > 1:
                # Group the state dict blocks into the right block size.
                for key in list(state_dict.keys()):
                    if (m := re.match(r"transformer.blocks\.(\d+)\..*", key)) is not None:
                        block_idx = int(m.group(1))
                        group_idx, group_block_idx = (
                            block_idx // self.config.block_group_size,
                            block_idx % self.config.block_group_size,
                        )
                        state_dict[
                            (
                                new_key := key.replace(
                                    f"blocks.{block_idx}.", f"block_groups.{group_idx}.{group_block_idx}."
                                )
                            )
                        ] = state_dict.pop(key)
                        new_keys_to_og_keys[new_key] = new_keys_to_og_keys.pop(key)

        og_keys_to_new: Dict[str, Set[str]] = defaultdict(set)
        for new_key, og_key in new_keys_to_og_keys.items():
            og_keys_to_new[og_key].add(new_key)

        return state_dict, og_keys_to_new
