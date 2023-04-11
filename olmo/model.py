"""
Adapted from
[MosaiclML](https://github.com/mosaicml/examples.git) and
[minGPT](https://github.com/karpathy/minGPT.git)
"""

from __future__ import annotations

import math
from abc import abstractmethod
from typing import List, NamedTuple, Optional, cast

import torch
import torch.backends.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

from .beam_search import BeamSearch, Constraint, FinalSequenceScorer, Sampler
from .config import ActivationType, BlockType, LayerNormType, ModelConfig
from .exceptions import OlmoConfigurationError

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


class LayerNormBase(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def build(cls, config: ModelConfig) -> LayerNormBase:
        if config.layer_norm_type == LayerNormType.default:
            return LayerNorm(config, low_precision=False)
        elif config.layer_norm_type == LayerNormType.low_precision:
            return LayerNorm(config, low_precision=True)
        elif config.layer_norm_type == LayerNormType.rms:
            return RMSLayerNorm(config, low_precision=False)
        elif config.layer_norm_type == LayerNormType.low_precision_rms:
            return RMSLayerNorm(config, low_precision=True)
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


class LayerNorm(LayerNormBase):
    """
    The default :class:`LayerNorm` implementation which can optionally run in low precision.
    """

    def __init__(self, config: ModelConfig, low_precision: bool = False):
        super().__init__(config)
        self.normalized_shape = (config.d_model,)
        self.eps = 1e-05
        self.weight = nn.Parameter(torch.ones(self.normalized_shape, device=config.init_device))
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape, device=config.init_device))
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
                return F.layer_norm(downcast_x, self.normalized_shape, downcast_weight, downcast_bias, self.eps)
        else:
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)


class RMSLayerNorm(LayerNorm):
    """
    RMS layer norm, a simplified :class:`LayerNorm` implementation that can optionally run
    in low-precision.
    """

    def __init__(self, config: ModelConfig, low_precision: bool = False):
        super().__init__(config)
        self.eps = 1e-08
        self.weight = nn.Parameter(torch.ones(self.config.d_model))
        if self.config.include_bias:
            self.bias = nn.Parameter(torch.zeros(self.config.d_model))
        else:
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

    def rms_norm(self, x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
        norm_x = x.norm(2, dim=-1, keepdim=True)

        rms_x = norm_x * self.config.d_model ** (-1.0 / 2)
        x_normed = x / (rms_x + self.eps)

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

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        assert config.d_model % config.n_heads == 0

        # Dropout.
        self.dropout = nn.Dropout(config.residual_dropout)

        # Layer norms.
        self.norm = LayerNorm.build(config)
        self.k_norm: Optional[LayerNormBase] = None
        self.q_norm: Optional[LayerNormBase] = None
        if config.attention_layer_norm:
            self.k_norm = LayerNormBase.build(config)
            self.q_norm = LayerNormBase.build(config)

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

    def get_rotary_embedding(self, seq_len: int, device: Optional[torch.device]) -> torch.Tensor:
        if self.pos_emb is not None and self.pos_emb.shape[-2] >= seq_len:  # type: ignore
            return self.pos_emb[:seq_len]  # type: ignore

        pos_emb = self.rotary_emb(seq_len, device=device)
        self.register_buffer("pos_emb", pos_emb, persistent=False)
        return pos_emb

    def attention(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, attention_bias: Optional[torch.FloatTensor] = None
    ) -> torch.Tensor:
        assert not q.isnan().any()
        assert not k.isnan().any()
        assert not v.isnan().any()

        B, T, C = q.size()  # batch size, sequence length, d_model
        dtype = k.dtype

        assert not q.isnan().any()
        assert not k.isnan().any()
        assert not v.isnan().any()

        # Optionally apply layer norm to keys and queries.
        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q).to(dtype=dtype)
            k = self.k_norm(k).to(dtype=dtype)

        assert not q.isnan().any()
        assert not k.isnan().any()
        assert not v.isnan().any()

        # Move head forward to be next to the batch dim.
        # shape (all): (B, nh, T, hs)
        q = q.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)
        k = k.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)
        v = v.view(B, T, self.config.n_heads, C // self.config.n_heads).transpose(1, 2)

        assert not q.isnan().any()
        assert not k.isnan().any()
        assert not v.isnan().any()

        if self.config.rope:
            # Apply rotary embeddings.
            positions = self.get_rotary_embedding(T, q.device)
            q, k = map(lambda t: apply_rotary_pos_emb(positions, t), (q, k))

        assert not q.isnan().any()
        assert not k.isnan().any()
        assert not v.isnan().any()
        if attention_bias is not None:
            assert not attention_bias.isnan().any()
            assert not attention_bias.to(dtype=dtype).isnan().any()

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

        assert not att.isnan().any()

        # Re-assemble all head outputs side-by-side.
        att = att.transpose(1, 2).contiguous().view(B, T, C)

        # Apply output projection.
        return self.attn_out(att)

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def build(cls, config: ModelConfig) -> OlmoBlock:
        if config.block_type == BlockType.sequential:
            return OlmoSequentialBlock(config)
        elif config.block_type == BlockType.parallel:
            return OlmoParallelBlock(config)
        else:
            raise NotImplementedError(f"not sure how to handle block type '{config.block_type}'")


class OlmoSequentialBlock(OlmoBlock):
    """
    This is a typical transformer block where the output is computed as ``MLP(LN(x + Attention(LN(x))))``
    (plus another skip connection).
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        # Attention input projection. Projects x -> (q, k, v)
        self.att_proj = nn.Linear(
            config.d_model, 3 * config.d_model, bias=config.include_bias, device=config.init_device
        )
        self.att_proj._fused = (0, (self.config.d_model, 2 * self.config.d_model))  # type: ignore
        # Feed-forward input projection.
        self.ff_proj = nn.Linear(
            config.d_model, config.mlp_ratio * config.d_model, bias=config.include_bias, device=config.init_device
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        # Get query, key, value projections.
        # shape (all): (batch_size, seq_len, d_model)
        q, k, v = self.att_proj(self.norm(x)).split(self.config.d_model, dim=2)

        # Add attention scores.
        # shape: (B, T, C)
        x = x + self.dropout(self.attention(q, k, v, attention_bias))

        # Add feed-forward projection.
        # shape: (batch_size, seq_len, d_model)
        x = x + self.dropout(self.ff_out(self.act(self.ff_proj(self.norm(x)))))

        return x


class OlmoParallelBlock(OlmoBlock):
    """
    This is a transformer block where the output is computed as ``MLP(LN(x)) + Attention(LN(x))``
    as in the PaLM architecture, as opposed to the typical ``MLP(LN(x + Attention(LN(x))))``
    as in :class:`OlmoSequentialBlock` (ignoring some skip connections).

    The decoupling of the MLP and Attention functions allow us to fuse the separate input projections
    into a single linear layer to increase throughput. In this configuration it's also straight-forward
    to fuse the output projections, but we found that didn't help.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        # Fused attention and feed-forward projection.
        # NOTE: we could also fuse the attention and feed-forward output projections
        # but we found that didn't help, possibly because of the overhead of joining the `att`
        # and `ff` activations together.
        # See https://github.com/allenai/LLM/pull/79 for details.
        self.fused_dims = (config.d_model, config.d_model, config.d_model, config.mlp_ratio * config.d_model)
        self.fused_attn_ff_proj = nn.Linear(
            config.d_model, sum(self.fused_dims), bias=config.include_bias, device=config.init_device
        )
        self.fused_attn_ff_proj._fused = (0, self.fused_dims)  # type: ignore

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        # Get query, key, value, and feed-forward projections.
        # shape of q, k, v: (batch_size, seq_len, d_model)
        # shape of ff:      (batch_size, seq_len, mlp_ratio x d_model)
        q, k, v, ff = self.fused_attn_ff_proj(self.norm(x)).split(self.fused_dims, dim=-1)

        # Get attention scores.
        # shape: (B, T, C)
        att = self.attention(q, k, v, attention_bias)

        # Apply output projections (and activation function) and sum the results.
        # We keep these projections separate because we found that we got better throughput this
        # way compared to fusing them.
        return x + self.dropout(self.ff_out(self.act(ff))) + self.dropout(att)


class OlmoOutput(NamedTuple):
    logits: torch.FloatTensor
    """
    A tensor of shape `(batch_size, seq_len, vocab_size)` representing the log probabilities
    for the next token *before* normalization via (log) softmax.
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
        torch.backends.cuda.enable_mem_efficient_sdp(self.config.memory_efficient_attention)

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(
                    config.embedding_size or config.vocab_size, config.d_model, device=config.init_device
                ),
                emb_drop=nn.Dropout(config.embedding_dropout),
                blocks=nn.ModuleList([OlmoBlock.build(config) for _ in range(config.n_layers)]),
                ln_f=LayerNorm.build(config),
            )
        )
        if not (self.config.alibi or self.config.rope):
            self.transformer.update(
                {"wpe": nn.Embedding(config.max_sequence_length, config.d_model, device=config.init_device)}
            )
        if init_params and self.config.init_device != "meta":
            self.apply(self.param_init_fn)
        self.__num_fwd_flops = None

        # Initialize attention bias buffers up front since calling `register_buffer`
        # while compiling will cause a break in the graph.
        if self.config.alibi:
            self.causal_attention_bias
            self.alibi_attention_bias

    @property
    def buffer_dtype(self) -> torch.dtype:
        """
        For some reason when we use :func:`torch.compile()` and AMP, we have to create the
        attention bias buffers with the right data type.
        """
        if self.config.precision == "amp_bf16":
            return torch.bfloat16
        elif self.config.precision == "amp_fp16":
            return torch.float16
        else:
            return torch.float

    @property
    def causal_attention_bias(self) -> torch.FloatTensor:
        if not hasattr(self, "_causal_attention_bias"):
            att_bias = torch.triu(
                torch.ones(
                    self.config.max_sequence_length,
                    self.config.max_sequence_length,
                    device=self.config.device,
                    dtype=torch.float,
                ),
                diagonal=1,
            )
            att_bias = att_bias.to(self.buffer_dtype)
            att_bias.masked_fill_(att_bias == 1, float("-inf"))
            self.register_buffer(
                "_causal_attention_bias",
                att_bias.view(
                    1, 1, self.config.max_sequence_length, self.config.max_sequence_length
                ),
                persistent=False,
            )
        assert not self._causal_attention_bias.isnan().any()
        return self._causal_attention_bias  # type: ignore[return-type]

    @property
    def alibi_attention_bias(self) -> torch.FloatTensor:
        if not hasattr(self, "_alibi_attention_bias"):
            # shape: (1, 1, 1, seq_len)
            alibi_bias = torch.arange(
                1 - self.config.max_sequence_length, 1, dtype=torch.float, device=self.config.device
            ).view(1, 1, 1, self.config.max_sequence_length)

            # shape: (1, 1, seq_len, seq_len)
            alibi_bias = alibi_bias - torch.arange(
                1 - self.config.max_sequence_length, 1, dtype=torch.float, device=self.config.device
            ).view(1, 1, self.config.max_sequence_length, 1)
            alibi_bias.abs_().mul_(-1)

            # shape: (n_heads,)
            m = torch.arange(1, self.config.n_heads + 1, dtype=torch.float, device=self.config.device)
            m.mul_(self.config.alibi_bias_max / self.config.n_heads)

            # shape: (1, n_heads, seq_len, seq_len)
            alibi_bias = alibi_bias * (1.0 / (2 ** m.view(1, self.config.n_heads, 1, 1)))
            self.register_buffer("_alibi_attention_bias", alibi_bias.to(dtype=self.buffer_dtype), persistent=False)
        return self._alibi_attention_bias  # type: ignore[return-type]

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
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
        """
        batch_size, seq_len = input_ids.size()
        assert seq_len <= self.config.max_sequence_length, (
            f"Cannot forward input with seq_len={seq_len}, "
            f"this model only supports seq_len<={self.config.max_sequence_length}"
        )

        assert attention_mask is None or not attention_mask.isnan().any()
        assert attention_bias is None or not attention_bias.isnan().any()

        # Get embeddings of input.
        # shape: (batch_size, seq_len, d_model)
        x = self.transformer.wte(input_ids)  # type: ignore

        if not (self.config.alibi or self.config.rope):
            # Get positional embeddings.
            # shape: (1, seq_len)
            pos = torch.arange(0, seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)
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
            assert not attention_mask.isnan().any()

        # Merge attention mask with attention bias.
        if attention_bias is not None or attention_mask is not None or self.config.alibi:
            if attention_bias is None:
                # Default to causal attention bias.
                attention_bias = self.causal_attention_bias
                assert not attention_bias.isnan().any()
            elif attention_bias.dtype in (torch.int8, torch.bool):
                attention_bias = attention_bias.to(dtype=x.dtype)
                attention_bias.masked_fill_(attention_bias == 0.0, float("-inf"))
                assert not attention_bias.isnan().any()

            attention_bias = attention_bias[:, :, :seq_len, :seq_len]

            # Add in the masking bias.
            if attention_mask is not None:
                assert not attention_mask.isnan().any()
                attention_bias = attention_bias + attention_mask
                assert not attention_bias.isnan().any()

            if self.config.alibi:
                # Add in ALiBi attention bias.
                attention_bias = attention_bias + self.alibi_attention_bias[:, :, :seq_len, :seq_len].to(x.dtype)
                assert not attention_bias.isnan().any()

        assert not attention_bias.isnan().any()

        # Apply blocks one-by-one.
        for block in self.transformer.blocks:  # type: ignore
            # shape: (batch_size, seq_len, d_model)
            x = block(x, attention_bias=attention_bias)

        # Apply final layer norm.
        # shape: (batch_size, seq_len, d_model)
        x = self.transformer.ln_f(x)  # type: ignore

        # Get logits.
        # shape: (batch_size, seq_len, vocab_size)
        logits = F.linear(x, self.transformer.wte.weight, None)  # type: ignore

        return OlmoOutput(logits=logits)  # type: ignore[arg-type]

    def fsdp_wrap_fn(self, module):
        return isinstance(module, OlmoBlock)

    def activation_checkpointing_fn(self, module):
        return isinstance(module, OlmoBlock)

    def param_init_fn(self, module):
        from functools import partial

        init_fn = partial(nn.init.normal_, mean=0.0, std=self.config.init_std)

        def fused_init_fn(module):
            # Parameter initialization is often based on the parameters shape.
            # If a layer is fused, initialization should be based on the shapes
            # of the original tensor instead of the shape of the fused tensor.
            # Layers which are fused should have the _fused attribute defined.
            # The first element of _fused is the dimension along which the tensor is fused.
            # This is followed by an iterable of split indices.
            _fused = getattr(module, "_fused", None)
            if _fused is None:
                raise RuntimeError("Internal logic error")

            dim, splits = _fused
            splits = (0, *splits, module.weight.size(dim))
            for s, e in zip(splits[:-1], splits[1:]):
                slice_indices = [slice(None)] * module.weight.ndim
                slice_indices[dim] = slice(s, e)
                init_fn(module.weight[slice_indices])

        # Linear
        if isinstance(module, nn.Linear):
            if hasattr(module, "_fused"):
                fused_init_fn(module)
            else:
                init_fn(module.weight)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

            if getattr(module, "_is_residual", False):
                with torch.no_grad():
                    module.weight.div_(math.sqrt(2 * self.config.n_layers))

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        # Embedding
        if isinstance(module, nn.Embedding):
            init_fn(module.weight)

        # LayerNorm
        if isinstance(module, (nn.LayerNorm, LayerNorm, RMSLayerNorm)):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

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

        def step(
            last_predictions: torch.Tensor, state: dict[str, torch.Tensor]
        ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
            nonlocal tokens_generated

            input_ids = state["input_ids"]
            attention_mask = state.get("attention_mask")
            attention_bias = state.get("attention_bias")
            group_size = input_ids.shape[0]

            if tokens_generated > 0:
                input_ids = torch.cat((input_ids, last_predictions.unsqueeze(1)), dim=-1)
                if attention_mask is not None:
                    attention_mask = torch.cat((attention_mask, attention_mask.new_ones((group_size, 1))), dim=-1)

            tokens_generated += 1

            # Run forward pass of model to get logits, then normalize to get log probs.
            output = self(input_ids, attention_mask=attention_mask, attention_bias=attention_bias)
            log_probs = F.log_softmax(output.logits[:, -1, :], dim=-1)

            # Create new state.
            state = {"input_ids": input_ids}
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
        token_ids, scores = beam_search.search(initial_preds, state, step)

        return OlmoGenerateOutput(
            token_ids=token_ids,  # type: ignore[arg-type]
            scores=scores,  # type: ignore[arg-type]
        )
