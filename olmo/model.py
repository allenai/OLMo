"""
Adapted from
[MosaiclML](https://github.com/mosaicml/examples.git) and
[minGPT](https://github.com/karpathy/minGPT.git)
"""

from __future__ import annotations

import logging
import math
from abc import abstractmethod
from typing import List, NamedTuple, Optional, Sequence, Tuple

import torch
import torch.backends.cuda
import torch.nn as nn
import torch.nn.functional as F
from .config import LayerNormType, ModelConfig
from .exceptions import OlmoConfigurationError
from .initialization import init_weights

__all__ = [
    "LayerNormBase",
    "LayerNorm",
    "Olmo",
    "OlmoOutput",
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
    def build(cls, config: ModelConfig, size: Optional[int] = None) -> LayerNormBase:
        if config.layer_norm_type == LayerNormType.default:
            return LayerNorm(config, size=size, low_precision=False)
        elif config.layer_norm_type == LayerNormType.low_precision:
            return LayerNorm(config, size=size, low_precision=True)
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
        torch.nn.init.ones_(self.weight)  # type: ignore
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)  # type: ignore


class LayerNorm(LayerNormBase):
    """
    The default :class:`LayerNorm` implementation which can optionally run in low precision.
    """

    def __init__(self, config: ModelConfig, size: Optional[int] = None, low_precision: bool = False):
        super().__init__(config)
        self.normalized_shape = (size or config.d_model,)
        self.eps = 1e-05
        self.weight = nn.Parameter(torch.ones(self.normalized_shape, device=config.init_device))
        if self.config.include_bias:
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape, device=config.init_device))
        else:
            self.register_parameter("bias", None)
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
                ln_f=LayerNorm.build(config),
            )
        )
        # When `init_device="meta"` FSDP will call `reset_parameters()` to initialize weights.
        if init_params and self.config.init_device != "meta":
            self.reset_parameters()
        self.__num_fwd_flops: Optional[int] = None

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

        # Add input + positional embeddings and apply dropout.
        # shape: (batch_size, seq_len, d_model)
        x = self.transformer.emb_drop(x)  # type: ignore

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

        return OlmoOutput(logits=logits, attn_key_values=None)  # type: ignore[arg-type]

    def fsdp_wrap_fn(self, module, recurse: bool = True, nonwrapped_numel: int = 0):
        del nonwrapped_numel
        if recurse:
            return True  # always recurse
        return False

    def activation_checkpointing_fn(self, module):
        return False

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
