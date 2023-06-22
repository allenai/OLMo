from typing import Optional, Union, Sequence, Tuple
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.auto import AutoModelForCausalLM
import os

from olmo.model import Olmo
from olmo.config import ModelConfig
from .configuration_olmo import OLMoConfig

import torch

#
# # XPretrainedModel + XModel
# class OlmoModel(PreTrainedModel):
#     config_class = OLMoConfig
#     base_model_prefix = "model"
#     supports_gradient_checkpointing = False  # TODO: confirm
#     _skip_keys_device_placement = ["past_key_values"]  # TODO: confirm
#
#     def _init_weights(self, module):
#         # underlying model should take care of it with `init_params`.
#         pass
#
#     def __init__(self, config: OLMoConfig):
#         super().__init__(config)
#
#         model_config = ModelConfig(**config.asdict())
#         self.model = Olmo(model_config, init_params=False)  # TODO: pass as input?
#
#     def forward(
#         self,
#         input_ids: torch.LongTensor,
#         attention_mask: Optional[torch.Tensor] = None,
#         attention_bias: Optional[torch.Tensor] = None,
#         past_key_values: Optional[Sequence[Tuple[torch.Tensor, torch.Tensor]]] = None,
#         use_cache: bool = False,
#         last_logits_only: bool = False,
#     ) -> BaseModelOutputWithPast:
#         """
#         :param input_ids: A tensor of shape `(batch_size, seq_len)`.
#         :param attention_mask: A tensor of shape `(batch_size, seq_len)` that indicates
#             which input IDs are masked. A `1` value in the mask means that
#             the corresponding input ID should *not* be ignored. A `0` means
#             that the corresponding input ID is masked.
#
#             This has the same meaning as the `attention_mask` in HuggingFace's `transformers`
#             library.
#         :param attention_bias: A tensor of shape `(batch_size, 1, seq_len, seq_len)`,
#             `(1, 1, seq_len, seq_len)`, or `(seq_len, seq_len)`. This is used
#             to introduce causal or other biases.
#
#             If the tensor is a bool or byte tensor, a `True` or `1` at `attention_bias[:, :, i, j]`
#             indicates that the i-th element in the sequence is allowed to attend to the j-th
#             element in the sequence.
#
#             If the tensor is a float tensor, it will just be added to the attention
#             scores before the softmax.
#
#             The default is causal, which corresponds to a lower-diagonal byte matrix of ones.
#         :param past_key_values: Pre-computed keys and values for each attention block.
#             Can be used to speed up sequential decoding. The `input_ids` which have
#             their past given to this model should not be passed as `input_ids` as they have already been computed.
#         :param use_cache: If `True`, return key and value tensors for each block.
#         :param last_logits_only: If `True`, only compute the logits for the last token of each sequence.
#             This can speed up decoding when you only care about the next token.
#         """
#         if past_key_values:
#             assert len(past_key_values) == self.config.n_layers
#
#         batch_size, seq_len = input_ids.size()
#         assert seq_len <= self.config.max_sequence_length, (
#             f"Cannot forward input with seq_len={seq_len}, "
#             f"this model only supports seq_len<={self.config.max_sequence_length}"
#         )
#
#         # Get embeddings of input.
#         # shape: (batch_size, seq_len, d_model)
#         x = self.transformer.wte(input_ids)  # type: ignore
#
#         if not (self.config.alibi or self.config.rope):
#             # Get positional embeddings.
#             if past_key_values is None:
#                 past_length = 0
#             else:
#                 past_length = past_key_values[0][0].size(-2)
#             # shape: (1, seq_len)
#             pos = torch.arange(
#                 past_length, past_length + seq_len, dtype=torch.long, device=input_ids.device
#             ).unsqueeze(0)
#             # shape: (1, seq_len, d_model)
#             pos_emb = self.transformer.wpe(pos)  # type: ignore
#             x = pos_emb + x
#
#         # Add input + positional embeddings and apply dropout.
#         # shape: (batch_size, seq_len, d_model)
#         x = self.transformer.emb_drop(x)  # type: ignore
#
#         # Transform the attention mask into what the blocks expect.
#         if attention_mask is not None:
#             # shape: (batch_size, 1, 1, seq_len)
#             attention_mask = attention_mask.to(dtype=x.dtype).view(batch_size, -1)[:, None, None, :]
#             attention_mask = (1.0 - attention_mask) * torch.finfo(attention_mask.dtype).min
#             attention_mask.masked_fill_(attention_mask == 1.0, float("-inf"))
#
#         # Merge attention mask with attention bias.
#         if (
#             attention_bias is not None
#             or attention_mask is not None
#             or self.config.alibi
#             # NOTE (epwalsh): we need to initialize the attn bias in order for attn to work properly
#             # with key+value cache. Otherwise `F.scaled_dot_product_attention()` doesn't seem to compute
#             # scores correctly.
#             or past_key_values is not None
#         ):
#             if attention_bias is None and self.config.alibi:
#                 attention_bias = self.causal_attention_bias + self.alibi_attention_bias
#             elif attention_bias is None:
#                 attention_bias = self.causal_attention_bias
#             elif attention_bias.dtype in (torch.int8, torch.bool):
#                 attention_bias = attention_bias.to(dtype=x.dtype)
#                 attention_bias.masked_fill_(attention_bias == 0.0, float("-inf"))
#
#             # Transform to the right shape and data type.
#             mask_len = seq_len
#             if attention_mask is not None:
#                 mask_len = attention_mask.shape[-1]
#             elif past_key_values is not None:
#                 mask_len = past_key_values[0][0].shape[-2] + input_ids.shape[-1]
#             attention_bias = attention_bias[:, :, :mask_len, :mask_len].to(x.dtype)
#
#             # Add in the masking bias.
#             if attention_mask is not None:
#                 attention_bias = attention_bias + attention_mask
#
#         attn_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = [] if use_cache else None
#
#         # Apply blocks one-by-one.
#         for block, layer_past in zip(
#             self.transformer.blocks,  # type: ignore
#             past_key_values or [None] * self.config.n_layers,  # type: ignore
#         ):
#             # shape: (batch_size, seq_len, d_model)
#             x, cache = block(x, attention_bias=attention_bias, layer_past=layer_past, use_cache=use_cache)
#             if attn_key_values is not None:
#                 assert cache is not None
#                 attn_key_values.append(cache)
#
#         if last_logits_only:
#             # shape: (batch_size, 1, d_model)
#             x = x[:, -1, :].unsqueeze(1)
#
#         # Apply final layer norm.
#         # shape: (batch_size, seq_len or 1, d_model)
#         x = self.transformer.ln_f(x)  # type: ignore
#
#         # Get logits.
#         # shape: (batch_size, seq_len or 1, vocab_size)
#         logits = F.linear(x, self.transformer.wte.weight, None)  # type: ignore
#
#         # return OlmoOutput(logits=logits, attn_key_values=attn_key_values)  # type: ignore[arg-type]
#         return BaseModelOutputWithPast(
#             last_hidden_state=
#         )
#
# class OLMoPretrainedModel(PreTrainedModel):
#     config_class = OLMoConfig
#
#     def __init__(self, config: OLMoConfig):
#         super().__init__(config)
#
#         model_config = ModelConfig(**config.asdict())
#         self.model = Olmo(model_config, init_params=False)  # TODO: pass as input?
#
#     def _init_weights(self, module):
#         # Underlying model should take care of this.
#         pass
#
#     def forward(self, *args, **kwargs):
#         return self.model.forward(*args, **kwargs)
#
#     @classmethod
#     def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
#         model = Olmo.from_checkpoint(pretrained_model_name_or_path, *model_args, **kwargs)
#         return cls(model.config, model)


class OLMoForCausalLM(PreTrainedModel):
    """
    Extremely barebones HF model wrapper.
    """
    def __init__(self, config: OLMoConfig, model: Optional[Olmo] = None):
        super().__init__(config)

        if not model:
            model_config = ModelConfig(**config.asdict())
            self.model = Olmo(model_config, init_params=True)
        else:
            self.model = model

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    # def generate(self, *args, **kwargs):
    #     return self.model.generate(*args, **kwargs)

    def generate(
        self,
        input_ids,
        max_length,
        max_new_tokens=None,
        eos_token_id=None,
        do_sample=False,
        pad_token_id=None,
        **kwargs,
    ):
        max_steps = max_new_tokens or max_length - input_ids.shape[1]  # max new tokens
        with torch.no_grad():
            res = self.model.generate(input_ids, max_steps=max_steps, eos_token_id=eos_token_id, beam_size=1, **kwargs)
        # Add back input_ids to top beam output since this is what's expected
        return torch.cat((input_ids, res.token_ids[:, 0]), dim=1)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs):
        model = Olmo.from_checkpoint(pretrained_model_name_or_path)
        return cls(model.config, model)