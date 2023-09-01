"""
Adapted from
[MosaiclML](https://github.com/mosaicml/examples.git) and
[minGPT](https://github.com/karpathy/minGPT.git)
"""

from __future__ import annotations

import logging
import math
from typing import List, NamedTuple, Optional, Sequence, Tuple

import torch
import torch.backends.cuda
import torch.nn as nn
import torch.nn.functional as F
from .initialization import init_weights

__all__ = [
    "LayerNorm",
    "Olmo",
    "OlmoOutput",
]


log = logging.getLogger(__name__)


class LayerNorm(nn.Module):
    """
    The default :class:`LayerNorm` implementation which can optionally run in low precision.
    """

    def __init__(self, low_precision: bool = False):
        super().__init__()
        self.normalized_shape = (4096,)
        self.eps = 1e-05
        self.weight = nn.Parameter(torch.ones(self.normalized_shape, device="meta"))
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
    def __init__(self):
        super().__init__()

        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)  # this is super slow so make sure torch won't use it

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(
                    50304, 4096, device="meta"
                ),
                emb_drop=nn.Dropout(0.0),
                ln_f=LayerNorm(low_precision=False),
            )
        )
        # FSDP will call `reset_parameters()` to initialize weights.
        #self.reset_parameters()
        self.__num_fwd_flops: Optional[int] = None

    def reset_parameters(self):
        log.info("Initializing model parameters...")
        # Top-level embeddings / linear layers.
        init_weights(
            self.transformer.wte,  # type: ignore
            std_factor=1.0,
        )
        if hasattr(self.transformer, "wpe"):
            init_weights(self.transformer.wpe)  # type: ignore

        # Top-level layer norm.
        self.transformer.ln_f.reset_parameters()  # type: ignore

    @property
    def device(self) -> torch.device:
        device: torch.device = self.transformer.wte.weight.device  # type: ignore
        if device.type == "meta":
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
        batch_size, seq_len = input_ids.size()

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
        return 1
