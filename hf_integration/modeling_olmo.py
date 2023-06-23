import os
from typing import Optional, Union

import torch
from transformers import PreTrainedModel
from transformers.models.auto import AutoModelForCausalLM

from olmo.config import ModelConfig
from olmo.model import Olmo

from .configuration_olmo import OLMoConfig


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

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def generate(self, input_ids, *args, **kwargs):
        with torch.no_grad():
            res = self.model.generate(input_ids, **kwargs)
        # Add back input_ids to top beam output since this is what's expected for AutoModelForCausalLM
        return torch.cat((input_ids, res.token_ids[:, 0]), dim=1)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: Optional[Union[str, os.PathLike]], *model_args, **kwargs
    ):
        assert pretrained_model_name_or_path is not None
        model = Olmo.from_checkpoint(pretrained_model_name_or_path)
        config = OLMoConfig(**model.config.asdict())
        return cls(config, model)

    # TODO: these 4 are required to make the implementation complete.
    # def resize_position_embeddings(self, new_num_position_embeddings: int):
    #     pass
    #
    # def get_position_embeddings(self) -> Union[nn.Embedding, Tuple[nn.Embedding]]:
    #     pass
    #
    # def prepare_inputs_for_generation(self, *args, **kwargs):
    #     pass
    #
    # def _reorder_cache(self, past_key_values, beam_idx):
    #     pass


# Register the model so that it is available for transformer pipelines, auto-loading, etc.
AutoModelForCausalLM.register(OLMoConfig, OLMoForCausalLM)
