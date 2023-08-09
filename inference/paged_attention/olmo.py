from typing import Dict, List, Optional, Tuple
import math
import torch
from torch import nn
import torch.nn.functional as F

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.attention import PagedAttentionWithALiBi
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.weight_utils import (hf_model_weights_iterator,
                                              load_tensor_parallel_weights)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from vllm.model_executor.parallel_utils.tensor_parallel import (
    VocabParallelEmbedding, ColumnParallelLinear, RowParallelLinear)
from vllm.sequence import SequenceOutputs

from olmo.model import LayerNormBase, LayerNorm, RMSLayerNorm, RotaryEmbedding, OlmoBlock, OlmoParallelBlock, OlmoOutput, Activation, apply_rotary_pos_emb, causal_attention_bias
from olmo.config import ActivationType, BlockType, LayerNormType, ModelConfig
from olmo.exceptions import OlmoConfigurationError

KVCache = Tuple[torch.Tensor, torch.Tensor]


def _get_alibi_slopes(
    total_num_heads: int,
    alibi_bias_max: int,
) -> torch.Tensor:
    next_power_of_2 = 2**math.ceil(math.log2(total_num_heads))
    m = torch.arange(1, next_power_of_2 + 1, dtype=torch.float32)
    m = m.mul(alibi_bias_max / next_power_of_2)
    slopes = 1.0 / torch.pow(2, m)
    if next_power_of_2 != total_num_heads:
        slopes = torch.concat([slopes[1::2], slopes[::2]])[:total_num_heads]
    return slopes


class PagedAttentionOlmoSequentialBlock(OlmoBlock):
    """
    This is a typical transformer block where the output is computed as ``MLP(LN(x + Attention(LN(x))))``
    (plus another skip connection).
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
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
        self.att_proj._fused = (0, self.fused_dims)  # type: ignore
        # Feed-forward input projection.
        self.ff_proj = nn.Linear(
            config.d_model, config.mlp_ratio * config.d_model, bias=config.include_bias, device=config.init_device
        )

        # TODO: confirm
        slopes = _get_alibi_slopes(config.n_heads, config.alibi_bias_max)
        self.paged_attn = PagedAttentionWithALiBi(config.n_heads, config.d_model // config.n_heads, scale=1.0,
                                                  slopes=slopes)

    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
    ) -> torch.Tensor:
        # Get query, key, value projections.
        # shape:
        #  - for regular attn q, k, v: (batch_size, seq_len, d_model)
        #  - for multi-query attn q: (batch_size, seq_len, d_model)
        #                      k, v: (batch_size, seq_len, d_model // n_heads)
        q, k, v = self.att_proj(self.attn_norm(x)).split(self.fused_dims, dim=-1)

        # Get attention scores.
        att = self.paged_attn(q, k, v, kv_cache[0], kv_cache[1], input_metadata, cache_event)

        # Add attention scores.
        # shape: (B, T, C)
        x = x + self.dropout(att)

        # Add feed-forward projection.
        # shape: (batch_size, seq_len, d_model)
        x = x + self.dropout(self.ff_out(self.act(self.ff_proj(self.ff_norm(x)))))

        return x


class OlmoModel(nn.Module):
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
                blocks=nn.ModuleList([PagedAttentionOlmoSequentialBlock(config) for _ in range(config.n_layers)]),
                ln_f=LayerNorm.build(config),
            )
        )

        if init_params and self.config.init_device != "meta":
            self.apply(self.param_init_fn)
        self.__num_fwd_flops: Optional[int] = None

    def param_init_fn(self, module):
        # TODO: minimize code repetition
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

    def _make_state_dict_compatible(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # For backwards compatibility prior to fixing https://github.com/allenai/LLM/issues/222
        if self.config.block_type == BlockType.sequential:
            for block_idx in range(self.config.n_layers):
                norm_w_key = f"transformer.blocks.{block_idx}.norm.weight"
                norm_b_key = f"transformer.blocks.{block_idx}.norm.bias"
                if norm_w_key in state_dict:
                    norm_w = state_dict.pop(norm_w_key)
                    state_dict[f"transformer.blocks.{block_idx}.attn_norm.weight"] = norm_w
                    state_dict[f"transformer.blocks.{block_idx}.ff_norm.weight"] = norm_w.clone()
                if norm_b_key in state_dict:
                    norm_b = state_dict.pop(norm_b_key)
                    state_dict[f"transformer.blocks.{block_idx}.attn_norm.bias"] = norm_b
                    state_dict[f"transformer.blocks.{block_idx}.ff_norm.bias"] = norm_b.clone()
        return state_dict

    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> torch.Tensor:

        batch_size, seq_len = input_ids.size()
        assert seq_len <= self.config.max_sequence_length, (
            f"Cannot forward input with seq_len={seq_len}, "
            f"this model only supports seq_len<={self.config.max_sequence_length}"
        )

        # Get embeddings of input.
        # shape: (batch_size, seq_len, d_model)
        x = self.transformer.wte(input_ids)  # type: ignore

        # Add input + positional embeddings and apply dropout.
        # shape: (batch_size, seq_len, d_model)
        x = self.transformer.emb_drop(x)  # type: ignore

        # Apply blocks one-by-one.

        for i, block in enumerate(self.transformer.blocks):
            if cache_events is None:
                cache_event = None
            else:
                cache_event = cache_events[i]

            # shape: (batch_size, seq_len, d_model)
            x = block(x, positions, kv_caches[i], input_metadata, cache_event)

        # if last_logits_only:
        #     # shape: (batch_size, 1, d_model)
        #     x = x[:, -1, :].unsqueeze(1)

        # Apply final layer norm.
        # shape: (batch_size, seq_len or 1, d_model)
        x = self.transformer.ln_f(x)  # type: ignore

        return x

        # # Get logits.
        # # shape: (batch_size, seq_len or 1, vocab_size)
        # logits = F.linear(x, self.transformer.wte.weight, None)  # type: ignore
        #
        # return OlmoOutput(logits=logits, attn_key_values=attn_key_values)  # type: ignore[arg-type]


class OlmoModelForCausalLM(nn.Module):

    def __init__(self, config):
        super().__init__()
        device = config.init_device
        config.init_device = "cpu"
        self.model = OlmoModel(config)
        config.init_device = device
        self.sampler = Sampler(config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> Dict[int, SequenceOutputs]:
        hidden_states = self.model(input_ids, positions, kv_caches,
                                   input_metadata, cache_events)
        next_tokens = self.sampler(self.model.transformer.wte.weight, hidden_states,
                                   input_metadata)
        return next_tokens

    _column_parallel_weights = []
    _row_parallel_weights = []

    def load_weights(self, model_name_or_path: str, cache_dir: Optional[str] = None, use_np_cache: bool = False):

        # Reference: Olmo.from_checkpoint()

        from cached_path import cached_path
        import os

        # Load config.
        config_path = cached_path(os.path.join(model_name_or_path, "config.yaml"))
        model_config = ModelConfig.load(config_path, key="model", validate_paths=False)

        # Initialize model (always on CPU to start with so we don't run out of GPU memory).
        # model_config.init_device = "cpu"
        # model = Olmo(model_config)
        # model.config.init_device = device

        # Load state dict directly to target device.
        state_dict_path = cached_path(os.path.join(model_name_or_path, "model.pt"))
        state_dict = torch.load(state_dict_path, map_location="cpu")
        self.model.load_state_dict(self.model._make_state_dict_compatible(state_dict))

