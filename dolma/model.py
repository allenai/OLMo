"""
Adapted from
[MosaiclML](https://github.com/mosaicml/examples.git) and
[minGPT](https://github.com/karpathy/minGPT.git)
"""

import math
from typing import NamedTuple, Optional, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig

__all__ = ["SelfAttention", "GPTMLP", "GPTBlock", "DolmaGPT"]


class SelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.n_heads = config.n_heads
        self.d_model = config.d_model
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, device=config.init_device)
        # output projection
        self.c_proj = nn.Linear(config.d_model, config.d_model, device=config.init_device)
        # regularization
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.residual_dropout)
        # optional layer norm for keys and queries.
        self.k_ln: Optional[nn.LayerNorm] = None
        self.q_ln: Optional[nn.LayerNorm] = None
        if config.attention_layer_norm:
            self.k_ln = nn.LayerNorm(self.d_model, device=config.init_device)
            self.q_ln = nn.LayerNorm(self.d_model, device=config.init_device)

    def forward(
        self,
        x: torch.FloatTensor,
        attention_bias: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        :param x: A tensor of shape `(batch_size, seq_len, d_model)`.
        :param attention_bias: A tensor of shape `(batch_size, n_heads, seq_len, seq_len)`
            or an equivalently broadcastable shape. This is used to introduce causal or other biases
            and it is simply added to the attention scores before the softmax.
        """
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (d_model)

        # Calculate query, key, values for all heads in batch.
        # shape (all): (B, T, C)
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)

        # Optionally apply layer norm to keys and queries.
        if self.k_ln is not None and self.q_ln is not None:
            k = self.k_ln(k)
            q = self.q_ln(q)

        # Move head forward to be next to the batch dim.
        # shape (all): (B, nh, T, hs)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)

        # Self-attention: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # Apply bias.
        if attention_bias is not None:
            att = att + attention_bias[:, :, :T, :T]

        # Apply softmax and dropout.
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        # Get head outputs.
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # Re-assemble all head outputs side by side.
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Apply output projection.
        y = self.resid_dropout(self.c_proj(y))

        return y


class GPTMLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, config.mlp_ratio * config.d_model, device=config.init_device)
        self.act = nn.GELU(approximate="none")
        self.c_proj = nn.Linear(config.mlp_ratio * config.d_model, config.d_model, device=config.init_device)
        self.c_proj._is_residual = True  # type: ignore
        self.dropout = nn.Dropout(config.residual_dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.act(self.c_fc(x))))


class GPTBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model, device=config.init_device)
        self.attn = SelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model, device=config.init_device)
        self.mlp = GPTMLP(config)

    def forward(
        self,
        x: torch.Tensor,
        attention_bias: Optional[torch.FloatTensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x), attention_bias=attention_bias)
        x = x + self.mlp(self.ln_2(x))
        return x


class DolmaGPTOutput(NamedTuple):
    logits: torch.FloatTensor
    """
    A tensor of shape `(batch_size, seq_len, vocab_size)` representing the log probabilities
    for the next token *before* normalization via (log) softmax.
    """


class DolmaGPT(nn.Module):
    def __init__(self, config: ModelConfig, init_params: bool = True):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.d_model, device=config.init_device),
                emb_drop=nn.Dropout(config.embedding_dropout),
                blocks=nn.ModuleList([GPTBlock(config) for _ in range(config.n_layers)]),
                ln_f=nn.LayerNorm(config.d_model, device=config.init_device),
            )
        )
        if not self.config.alibi:
            self.transformer.update(
                {"wpe": nn.Embedding(config.max_sequence_length, config.d_model, device=config.init_device)}
            )
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False, device=config.init_device)
        if init_params and self.config.init_device != "meta":
            self.apply(self.param_init_fn)
        self.__num_fwd_flops = None

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
            att_bias.masked_fill_(att_bias == 1, float("-inf"))
            self.register_buffer(
                "_causal_attention_bias",
                att_bias.view(1, 1, self.config.max_sequence_length, self.config.max_sequence_length),
            )
        return cast(torch.FloatTensor, self._causal_attention_bias)

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
            self.register_buffer("_alibi_attention_bias", alibi_bias)
        return cast(torch.FloatTensor, self._alibi_attention_bias)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
    ) -> DolmaGPTOutput:
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

        # Get embeddings of input.
        # shape: (batch_size, seq_len, d_model)
        x = self.transformer.wte(input_ids)  # type: ignore

        if not self.config.alibi:
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
            attention_mask = attention_mask.to(dtype=torch.float).view(batch_size, -1)[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * torch.finfo(attention_mask.dtype).min
            attention_mask.masked_fill_(attention_mask == 1.0, float("-inf"))

        # Default to causal attention bias.
        attention_bias = cast(
            torch.Tensor, attention_bias if attention_bias is not None else self.causal_attention_bias
        )
        if attention_bias.dtype in (torch.int8, torch.bool):
            attention_bias = attention_bias.to(dtype=torch.float)
            attention_bias.masked_fill_(attention_bias == 0.0, float("-inf"))

        attention_bias = attention_bias[:, :, :seq_len, :seq_len]

        # Add in the masking bias.
        if attention_mask is not None:
            attention_bias = attention_bias + attention_mask

        if self.config.alibi:
            # Add in ALiBi attention bias.
            attention_bias = attention_bias + self.alibi_attention_bias[:, :, :seq_len, :seq_len]

        # Apply blocks one-by-one.
        for block in self.transformer.blocks:  # type: ignore
            # shape: (batch_size, seq_len, d_model)
            x = block(x, attention_bias=attention_bias)

        # Apply final layer norm.
        # shape: (batch_size, seq_len, d_model)
        x = self.transformer.ln_f(x)  # type: ignore

        # Get logits.
        # shape: (batch_size, seq_len, vocab_size)
        logits = self.lm_head(x)  # type: ignore

        return DolmaGPTOutput(logits=cast(torch.FloatTensor, logits))

    def configure_optimizer(
        self,
        learning_rate: Optional[float] = None,
        weight_decay: float = 0.01,
        **kwargs,
    ) -> torch.optim.AdamW:
        """
        Get a suitable AdamW optimizer for training/fine-tuning.

        :param learning_rate: The learning rate. If not specified, a default learning
            rate will calculated according to the equation from the Scaling Laws paper
            `0.003239 - 0.0001395 * math.log(N)`,
            where `N` is the number of trainable parameters excluding embeddings.
        :param weight_decay: The weight decay coefficient. This does not apply to
            biases and layernorm/embedding weights, which will have a weight decay
            coefficient of 0.
        :param kwargs: Other keyword arguments passed to torch's `AdamW` optimizer.
        """
        # Separate out all parameters to those that will and won't experience regularizing weight decay.
        decay = set()
        no_decay = set()
        all_params = {}
        num_trainable_non_embedding_weights = 0
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                # NOTE: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times, but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if not p.requires_grad:
                    continue

                fpn = f"{mn}.{pn}" if mn else pn
                all_params[fpn] = p

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

                if fpn not in {"transformer.wte.weight", "transformer.wpe.weight"}:
                    num_trainable_non_embedding_weights += p.numel()

        # Validate that we've considered every parameter
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"parameters {inter_params} made it into both decay/no_decay sets!"
        assert (
            len(all_params.keys() - union_params) == 0
        ), f"parameters {all_params.keys() - union_params} were not separated into either decay/no_decay set!"

        # Create the pytorch optimizer groups.
        optim_groups = [
            {"params": [all_params[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [all_params[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        if learning_rate is None:
            learning_rate = 0.003239 - 0.0001395 * math.log(num_trainable_non_embedding_weights)

        return torch.optim.AdamW(optim_groups, lr=learning_rate, **kwargs)

    def fsdp_wrap_fn(self, module):
        return isinstance(module, GPTBlock)

    def activation_checkpointing_fn(self, module):
        return isinstance(module, GPTBlock)

    def param_init_fn(self, module):
        from functools import partial

        init_fn = partial(torch.nn.init.normal_, mean=0.0, std=self.config.init_std)

        # Linear
        if isinstance(module, nn.Linear):
            init_fn(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

            if getattr(module, "_is_residual", False):
                module.weight.data.normal_(
                    mean=0.0, std=(self.config.init_std / math.sqrt(2 * self.config.n_layers))
                )

        # Embedding
        if isinstance(module, nn.Embedding):
            init_fn(module.weight)

        # LayerNorm
        if isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def num_params(self, include_embedding: bool = True) -> int:
        """
        Get the total number of parameters.
        """
        params = (np for np in self.named_parameters())
        if not include_embedding:
            params = filter(lambda np: ".wte." not in np[0] and ".wpe." not in np[0], params)
        return sum(p.numel() for _, p in params)

    @property
    def num_fwd_flops(self):
        if self.__num_fwd_flops:
            return self.__num_fwd_flops
        n_params = sum(p.numel() for p in self.parameters())
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
