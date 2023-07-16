import os

# import warnings
from typing import List, Optional, Tuple, Union

import torch
from transformers import PreTrainedModel

# from transformers.generation.utils import (  # BaseStreamer,
#     GenerateOutput,
#     LogitsProcessorList,
#     StoppingCriteriaList,
# )
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.auto import AutoModelForCausalLM

from olmo.config import ModelConfig
from olmo.model import Olmo

from .configuration_olmo import OLMoConfig

# from typing import Callable, Sequence


def create_model_config_from_pretrained_config(config: OLMoConfig):
    """
    Utility function
    """
    model_config = ModelConfig(
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        mlp_ratio=config.mlp_ratio,
        activation_type=config.activation_type,
        block_type=config.block_type,
        alibi=config.alibi,
        alibi_bias_max=config.alibi_bias_max,
        rope=config.rope,
        flash_attention=config.flash_attention,
        attention_dropout=config.attention_dropout,
        attention_layer_norm=config.attention_layer_norm,
        multi_query_attention=config.multi_query_attention,
        residual_dropout=config.residual_dropout,
        embedding_dropout=config.embedding_dropout,
        layer_norm_type=config.layer_norm_type,
        max_sequence_length=config.max_sequence_length,
        include_bias=config.include_bias,
        vocab_size=config.vocab_size,
        embedding_size=config.embedding_size,
        eos_token_id=config.eos_token_id,
        pad_token_id=config.pad_token_id,
        init_device=config.init_device,
        init_std=config.init_std,
        precision=config.precision,
    )
    return model_config


class OLMoForCausalLM(PreTrainedModel):
    """
    Extremely barebones HF model wrapper.
    """

    config_class = OLMoConfig

    def __init__(self, config: OLMoConfig, model: Optional[Olmo] = None):
        super().__init__(config)

        if not model:
            model_config = create_model_config_from_pretrained_config(config)
            self.model = Olmo(model_config, init_params=True)
        else:
            self.model = model

    # def forward(self, *args, **kwargs):
    #     # use_cache = self.config.use_cache or kwargs.pop("use_cache", False)
    #     kwargs["use_cache"] = kwargs.pop("use_cache", self.config.use_cache)
    #     return self.model.forward(*args, **kwargs)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if use_cache is None:
            use_cache = self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        logits = outputs.logits

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = torch.nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.embedding_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.attn_key_values,
        )

    def can_generate(self) -> bool:
        return True

    # Note (akshitab): This model does not use OLMo's generate() function as it does not support all the
    # bells and whistles that HF's generation-compatible models do, such as `StoppingCriteria` or top-p sampling, etc.
    # Instead, the model sets `can_generate` to True, and relies on HF's default `.generate()`, and implements
    # supporting functions like `prepare_inputs_for_generation()`. This allows us to use HF's various generation
    # options.

    # def generate(
    #     self,
    #     input_ids: Optional[torch.Tensor] = None,
    #     max_length: int = 20,
    #     max_new_tokens: Optional[int] = None,
    #     logits_processor: Optional[LogitsProcessorList] = None,
    #     stopping_criteria: Optional[StoppingCriteriaList] = None,
    #     prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
    #     synced_gpus: Optional[bool] = None,
    #     assistant_model: Optional["PreTrainedModel"] = None,
    #     streamer: Optional["BaseStreamer"] = None,
    #     **kwargs,
    # ) -> Union[GenerateOutput, torch.LongTensor]:
    #
    #     assert input_ids is not None
    #
    #     # TODO: use stopping_criteria, since it's being used by instruct-eval
    #     if stopping_criteria is not None:
    #         warnings.warn(
    #             "OLMo's generate() function does not currently support `stopping_criteria`. "
    #             "This will likely result in worse performance on tasks."
    #         )
    #
    #     max_steps = max_new_tokens or max_length - input_ids.shape[1]
    #     result = self.model.generate(
    #         input_ids,
    #         max_steps=max_steps,
    #         beam_size=1,
    #         **kwargs,
    #     )
    #
    #     return torch.cat((input_ids, result.token_ids[:, 0]), dim=1)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs
    ):
        assert pretrained_model_name_or_path is not None
        if kwargs.get("device_map", "auto") == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        else:
            device = "cpu"
        model = Olmo.from_checkpoint(pretrained_model_name_or_path, device=device)
        try:
            config = OLMoConfig.from_pretrained(pretrained_model_name_or_path)
        except FileNotFoundError:
            config = OLMoConfig(use_cache=True, **model.config.asdict())
        return cls(config, model)

    def prepare_inputs_for_generation(
        self, input_ids: torch.LongTensor, past_key_values: Optional[List[Tuple]] = None, **kwargs
    ):
        if past_key_values:
            # This is because we want the model to only process the last generated token.
            input_ids = input_ids[:, -1:]
        model_inputs = {"input_ids": input_ids, "past_key_values": past_key_values}

        model_inputs.update(kwargs)
        model_inputs["use_cache"] = kwargs.pop("use_cache", self.config.use_cache)
        return model_inputs

    # TODO: these are required to make the implementation complete.
    # def resize_position_embeddings(self, new_num_position_embeddings: int):
    #     pass
    #
    # def get_position_embeddings(self) -> Union[nn.Embedding, Tuple[nn.Embedding]]:
    #     pass
    #
    # def _reorder_cache(self, past_key_values, beam_idx):
    #     pass


# Register the model so that it is available for transformer pipelines, auto-loading, etc.
AutoModelForCausalLM.register(OLMoConfig, OLMoForCausalLM)
