from typing import Dict, List, Optional, Tuple
import math
import torch
from torch import nn
import torch.nn.functional as F

from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.attention import PagedAttentionWithALiBi, PagedAttention
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.weight_utils import (hf_model_weights_iterator,
                                              load_tensor_parallel_weights)
from vllm.model_executor.parallel_utils.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size)
from vllm.model_executor.parallel_utils.tensor_parallel import (
    VocabParallelEmbedding, ColumnParallelLinear, RowParallelLinear)
from vllm.sequence import SequenceOutputs
from vllm.model_executor.layers.sampler import _get_output_tokens, _get_penalties, _get_temperatures, _apply_penalties, _get_top_p_top_k, _apply_top_p_top_k, _prune_hidden_states, _SAMPLING_EPS, _sample 

from olmo.model import LayerNormBase, LayerNorm, RMSLayerNorm, RotaryEmbedding, OlmoBlock, OlmoParallelBlock, OlmoOutput, Activation, apply_rotary_pos_emb, causal_attention_bias, alibi_attention_bias
from olmo.config import ActivationType, BlockType, LayerNormType, ModelConfig
from olmo.exceptions import OlmoConfigurationError

KVCache = Tuple[torch.Tensor, torch.Tensor]
import pdb

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

class OlmoSampler(nn.Module):
    """Samples the next tokens from the model's outputs.

    This layer does the following:
    3. Apply presence and frequency penalties.
    4. Apply temperature scaling.
    5. Apply top-p and top-k truncation.
    6. Sample the next tokens.
    Here, each sequence group within the batch can have different sampling
    parameters (e.g., sampling method, temperature, top-p, top-k, etc.).
    """

    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.vocab_size = vocab_size

    def forward(
        self,
        logits: torch.Tensor,
        input_metadata: InputMetadata,
        embedding_bias: Optional[torch.Tensor] = None,
    ) -> Dict[int, SequenceOutputs]:
        # Get the hidden states that we use for sampling.
        #hidden_states = _prune_hidden_states(hidden_states, input_metadata)
        logits = _prune_hidden_states(logits, input_metadata)

        # Get the logits for the next tokens.
        #logits = torch.matmul(hidden_states, embedding.t())
        #if embedding_bias is not None:
        #    logits += embedding_bias
        #logits = gather_from_tensor_model_parallel_region(logits)
        # Remove paddings in vocab (if any).
        logits = logits[:, :self.vocab_size]

        # Apply presence and frequency penalties.
        output_tokens = _get_output_tokens(input_metadata)
        assert len(output_tokens) == logits.shape[0]
        presence_penalties, frequency_penalties = _get_penalties(
            input_metadata)
        assert len(presence_penalties) == logits.shape[0]
        assert len(frequency_penalties) == logits.shape[0]
        logits = _apply_penalties(logits, output_tokens, presence_penalties,
                                  frequency_penalties, self.vocab_size)

        # Apply temperature scaling.
        temperatures = _get_temperatures(input_metadata)
        assert len(temperatures) == logits.shape[0]
        if any(t != 1.0 for t in temperatures):
            t = torch.tensor(temperatures,
                             dtype=logits.dtype,
                             device=logits.device)
            # Use in-place division to avoid creating a new tensor.
            logits.div_(t.unsqueeze(dim=1))

        # We use float32 for probabilities and log probabilities.
        # Compute the probabilities.
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        # Compute the log probabilities (before applying top-p and top-k).
        logprobs = torch.log(probs)

        # Apply top-p and top-k truncation.
        top_ps, top_ks = _get_top_p_top_k(input_metadata, self.vocab_size)
        assert len(top_ps) == len(top_ks) == probs.shape[0]
        do_top_p = any(p < 1.0 - _SAMPLING_EPS for p in top_ps)
        do_top_k = any(k != self.vocab_size for k in top_ks)
        if do_top_p or do_top_k:
            probs = _apply_top_p_top_k(probs, top_ps, top_ks)

        # Sample the next tokens.
        return _sample(probs, logprobs, input_metadata)

use_parallel = True

class PagedAttentionOlmoSequentialBlock(OlmoBlock):
    """
    This is a typical transformer block where the output is computed as ``MLP(LN(x + Attention(LN(x))))``
    (plus another skip connection).
    """

    def __init__(self, layer_id: int, config: ModelConfig):
        super().__init__(layer_id, config)

        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.n_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.head_dim = config.d_model // config.n_heads
        assert self.head_dim * self.total_num_heads == config.d_model

        # Layer norms.
        self.k_norm: Optional[LayerNormBase] = None
        self.q_norm: Optional[LayerNormBase] = None
        if config.attention_layer_norm:
            self.k_norm = LayerNormBase.build(
                config,
                size=self.head_dim if config.multi_query_attention else None,
                elementwise_affine=True,
            )
            self.q_norm = LayerNormBase.build(config, size=config.d_model, elementwise_affine=True)

        self.q_norm = None
        self.k_norm = None

        self.attn_norm = LayerNorm.build(config)
        self.ff_norm = LayerNorm.build(config)
        # Attention input projection. Projects x -> (q, k, v)
        if config.multi_query_attention:
            self.fused_dims = (config.d_model, config.d_model // config.n_heads, config.d_model // config.n_heads)
        else:
            self.fused_dims = (config.d_model, config.d_model, config.d_model)

        #self.att_proj = nn.Linear(
        #    config.d_model, sum(self.fused_dims), bias=config.include_bias, device=config.init_device
        #)

        if config.multi_query_attention:
            if not use_parallel:
                self.q_att_proj = nn.Linear(
                    config.d_model, config.d_model, bias=config.include_bias, device=config.init_device
                )
            else:
                self.q_att_proj = ColumnParallelLinear(
                    config.d_model,
                    config.d_model,
                    bias=config.include_bias,
                    perform_initialization=False,
                    gather_output=True,
                )

            self.kv_att_proj = nn.Linear(
                config.d_model, 2 * self.head_dim, bias=config.include_bias, device=config.init_device
            )
        else:
            self.fused_dims = (config.d_model, config.d_model, config.d_model)
            self.att_proj = ColumnParallelLinear(
                config.d_model,
                sum(self.fused_dims),
                bias=config.include_bias,
                gather_output=True,
                perform_initialization=False,
            )

            self.att_proj._fused = (0, self.fused_dims)  # type: ignore

        
        # Feed-forward input projection.

        if not use_parallel:
            self.ff_proj = nn.Linear(
                config.d_model, config.mlp_ratio * config.d_model, bias=config.include_bias, device=config.init_device
            )

        else:
            self.ff_proj = ColumnParallelLinear(
                config.d_model,
                config.mlp_ratio * config.d_model,
                bias=config.include_bias,
                gather_output=False,
                perform_initialization=False,
            )

        head_dim = config.d_model // config.n_heads
        scaling = head_dim**-0.5

        if config.multi_query_attention:
            num_kv_heads = 1
        else:
            num_kv_heads = config.n_heads

        if self.config.alibi:
            slopes = _get_alibi_slopes(config.n_heads, config.alibi_bias_max)
            tp_rank = get_tensor_model_parallel_rank()
            head_start = tp_rank * self.num_heads
            head_end = (tp_rank + 1) * self.num_heads
            slopes = slopes[head_start:head_end].tolist()
            # We use config.n_heads instead of 1 here because we have moved the kv repeat part into
            # our forward method rather than relying on PagedAttentionWithALiBi.
            self.paged_attn = PagedAttentionWithALiBi(self.num_heads, self.head_dim, scale=scaling,
                                                      slopes=slopes, num_kv_heads=self.num_heads)
        else:
            self.paged_attn = PagedAttention(self.num_heads, self.head_dim, scale=scaling, num_kv_heads=num_kv_heads)

        # Attention output projection.
        if not use_parallel:
            self.attn_out = nn.Linear(
                config.d_model, config.d_model, bias=config.include_bias, device=config.init_device
            )
        else:
            self.attn_out = RowParallelLinear(
                config.d_model,
                config.d_model,
                bias=config.include_bias,
                input_is_parallel=True,
                perform_initialization=False,
                reduce_results=True,
                skip_bias_add=True,
            )

        # Feed-forward output projection.
        if not use_parallel:
            self.ff_out = nn.Linear(
                int(self.act.output_multiplier * config.mlp_ratio * config.d_model),
                config.d_model,
                bias=config.include_bias,
                device=config.init_device,
            )
            self.ff_out._is_residual = True  # type: ignore

        else:
            self.ff_out = RowParallelLinear(
                int(self.act.output_multiplier * config.mlp_ratio * config.d_model),
                config.d_model,
                bias=config.include_bias,
                input_is_parallel=True,
                perform_initialization=False,
                #reduce_results=True,
            )

            self.ff_out._is_residual = True  # type: ignore


    def forward(
        self,
        x: torch.Tensor,
        positions: torch.Tensor,
        kv_cache: KVCache,
        input_metadata: InputMetadata,
        cache_event: Optional[torch.cuda.Event],
        debug: bool = False,
    ) -> torch.Tensor:
        # Get query, key, value projections.
        # shape:
        #  - for regular attn q, k, v: (batch_size, seq_len, d_model)
        #  - for multi-query attn q: (batch_size, seq_len, d_model)
        #                      k, v: (batch_size, seq_len, d_model // n_heads)
        #q, k, v = self.att_proj(self.attn_norm(x)).split(self.fused_dims, dim=-1)

        t = self.attn_norm(x)
        if not use_parallel:
            q = self.q_att_proj(t)
        else:
            q = self.q_att_proj(t)[0]
        k, v = self.kv_att_proj(t).split((self.head_dim, self.head_dim), dim=-1)
        k_cache, v_cache = kv_cache
  
        dtype = k.dtype

        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q).to(dtype=dtype)
            k = self.k_norm(k).to(dtype=dtype)

        if self.config.multi_query_attention:
            num_queries_per_kv = self.config.n_heads
            k = k.repeat(1, self.config.n_heads)
            v = v.repeat(1, self.config.n_heads)
        
        # Get attention scores.
        att = self.paged_attn(q, k, v, k_cache, v_cache, input_metadata, cache_event) #, debug=debug)
        if use_parallel:
            att = self.attn_out(att)[0]
        else:
            att = self.attn_out(att)

        # Add attention scores.
        # shape: (B, T, C)

        tp_rank = get_tensor_model_parallel_rank()
        print(tp_rank, x.shape, att.shape)
        x = x + self.dropout(att)

        # Add feed-forward projection.
        # shape: (batch_size, seq_len, d_model)
        if use_parallel:
            x = x + self.dropout(self.ff_out(self.act(self.ff_proj(self.ff_norm(x))[0]))[0])
        else:
            x = x + self.dropout(self.ff_out(self.act(self.ff_proj(self.ff_norm(x)))))
        return x


class OlmoModel(nn.Module):
    def __init__(self, config: ModelConfig, init_params: bool = True):
        super().__init__()
        self.config = config

        torch.backends.cuda.enable_flash_sdp(self.config.flash_attention)
        torch.backends.cuda.enable_mem_efficient_sdp(False)  # this is super slow so make sure torch won't use it

        self.transformer = nn.ModuleDict(
            dict(
                #wte=nn.Embedding(
                #    config.embedding_size or config.vocab_size, config.d_model, device=config.init_device
                #),
                wte=VocabParallelEmbedding(
                    config.embedding_size or config.vocab_size, config.d_model, perform_initialization=False
                ),
                emb_drop=nn.Dropout(config.embedding_dropout),
                blocks=nn.ModuleList([PagedAttentionOlmoSequentialBlock(i, config) for i in range(config.n_layers)]),
                ln_f=LayerNorm.build(config),
            )
        )

        if not (self.config.alibi or self.config.rope):
            self.transformer.update(
                {"wpe": nn.Embedding(config.max_sequence_length, config.d_model, device=config.init_device)}
            )

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

        seq_len = input_ids.shape[-1]
        # Get embeddings of input.
        # shape: (batch_size, seq_len, d_model)
        x = self.transformer.wte(input_ids)  # type: ignore

        if not (self.config.alibi or self.config.rope):
            # Get positional embeddings.
            if kv_caches[0][0] is None:
                past_length = 0
            else:
                past_length = kv_caches[0][0].size(-2)
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

        # Apply blocks one-by-one.

        for i, block in enumerate(self.transformer.blocks):
            if cache_events is None:
                cache_event = None
            else:
                cache_event = cache_events[i]

            # shape: (batch_size, seq_len, d_model)
            x = block(x, positions, kv_caches[i], input_metadata, cache_event) #, debug=(i==0))

        #if last_logits_only:
        #    # shape: (batch_size, 1, d_model)
        #    x = x[:, -1, :].unsqueeze(1)

        # Apply final layer norm.
        # shape: (batch_size, seq_len or 1, d_model)
        x = self.transformer.ln_f(x)  # type: ignore

        #return x

        # # Get logits.
        # # shape: (batch_size, seq_len or 1, vocab_size)
        logits = F.linear(x, self.transformer.wte.weight, None)  # type: ignore
        return logits
        #
        # return OlmoOutput(logits=logits, attn_key_values=attn_key_values)  # type: ignore[arg-type]


class OlmoModelForCausalLM(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        device = config.init_device
        config.init_device = "cpu"
        self.model = OlmoModel(config)
        #self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        config.init_device = device
        self.sampler = OlmoSampler(config.vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
        cache_events: Optional[List[torch.cuda.Event]],
    ) -> Dict[int, SequenceOutputs]:
        #hidden_states = self.model(input_ids, positions, kv_caches,
        #                           input_metadata, cache_events)
        #next_tokens = self.sampler(self.model.transformer.wte.weight, hidden_states,
        #                           input_metadata)
        logits = self.model(input_ids, positions, kv_caches, input_metadata, cache_events)
        next_tokens = self.sampler(logits, input_metadata)
        #next_tokens = self.sampler(self.lm_head.weight, hidden_states, input_metadata)
        return next_tokens

    #_column_parallel_weights = [] #["wte.weight"]
    #_row_parallel_weights = []

    _column_parallel_weights = ["wte.weight", "ff_proj.weight", "ff_proj.bias"] #, "att_proj.weight", "att_proj.bias"] #["wte.weight"]
    _row_parallel_weights = ["ff_out.weight", "attn_out.weight"] #, "ff_out.bias", "attn_out.bias"]

    def load_weights(self, model_name_or_path: str, cache_dir: Optional[str] = None, use_np_cache: bool = False):

        # Reference: Olmo.from_checkpoint()

        from cached_path import cached_path
        import os

        # Load config.
        # config_path = cached_path(os.path.join(model_name_or_path, "config.yaml"))
        # model_config = ModelConfig.load(config_path, key="model", validate_paths=False)

        # Initialize model (always on CPU to start with so we don't run out of GPU memory).
        # model_config.init_device = "cpu"
        # model = Olmo(model_config)
        # model.config.init_device = device

        # Load state dict directly to target device.
        state_dict_path = cached_path(os.path.join(model_name_or_path, "model.pt"))
        state_dict = torch.load(state_dict_path, map_location="cpu")
        state_dict = self.model._make_state_dict_compatible(state_dict)

        #for i in range(model_config.n_layers):
        #    state_dict[f"transformer.blocks.{i}.att_proj.weight"] = reshape_mqa_weight(
        #        state_dict[f"transformer.blocks.{i}.att_proj.weight"], model_config.d_model, model_config.n_heads
        #    )

        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()

        hidden_size = self.config.hidden_size
        total_num_heads = self.config.num_attention_heads
        num_heads = total_num_heads // tp_size
        head_size = hidden_size // total_num_heads
        head_start = tp_rank * num_heads
        head_end = (tp_rank + 1) * num_heads
        kv_head_start = 0
        kv_head_end = 1

        #empty_state_dict = self.model.state_dict()
        for name, param in self.model.state_dict().items():

            if "att_proj" in name:
                config = self.model.config
                if config.multi_query_attention:
                    fused_dims = (config.d_model, (config.d_model // config.n_heads) + (config.d_model // config.n_heads))
                else:
                    fused_dims = (config.d_model, config.d_model + config.d_model)

                if "q_att_proj" in name:
                    state_dict_param_name = name.replace("q_att_proj", "att_proj")
                    loaded_weight = state_dict[state_dict_param_name]
                    loaded_weight, _ = loaded_weight.split(fused_dims, dim=0)
                    loaded_weight = loaded_weight[head_size*head_start:head_size*head_end]

                elif "kv_att_proj" in name:
                    state_dict_param_name = name.replace("kv_att_proj", "att_proj")
                    loaded_weight = state_dict[state_dict_param_name]
                    _, loaded_weight = loaded_weight.split(fused_dims, dim=0)       

            else:
                loaded_weight = state_dict[name]

            if "attn_out" in name:
                print(tp_rank, "attn_out", param.shape, loaded_weight.shape)
            load_tensor_parallel_weights(param, loaded_weight, name,
                                         self._column_parallel_weights,
                                         self._row_parallel_weights, tp_rank)
        # self.model.load_state_dict(state_dict)


def reshape_mqa_weight(tensor, d, n):
    assert tensor.shape[0] == d + d//n + d//n
    q_part = tensor[:d]
    k_part = tensor[d:d+(d//n)]
    v_part = tensor[d+(d//n):]
    k_part = k_part.repeat(n, 1)
    v_part = v_part.repeat(n, 1)
    repeated_tensor = torch.cat((q_part, k_part, v_part), dim=0)  # Concatenate tensors

    return repeated_tensor
