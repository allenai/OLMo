from typing import Dict, List, Optional, Tuple

import torch
from torch import nn

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.attention import PagedAttentionWithRoPE
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.weight_utils import (hf_model_weights_iterator,
                                              load_tensor_parallel_weights)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from vllm.model_executor.parallel_utils.tensor_parallel import (
    VocabParallelEmbedding, ColumnParallelLinear, RowParallelLinear)
from vllm.sequence import SequenceOutputs

from olmo.model import LayerNormBase, LayerNorm, RotaryEmbedding, OlmoParallelBlock, OlmoOutput
from olmo.config import ActivationType, BlockType, LayerNormType, ModelConfig

KVCache = Tuple[torch.Tensor, torch.Tensor]


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
        self.k_norm: Optional[LayerNormBase] = None
        self.q_norm: Optional[LayerNormBase] = None
        if config.attention_layer_norm:
            self.k_norm = LayerNormBase.build(
                config, size=config.d_model // config.n_heads if config.multi_query_attention else None
            )
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

        return OlmoOutput(logits=logits, attn_key_values=attn_key_values)  # type: ignore[arg-type]
