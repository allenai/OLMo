"""
OLMo configuration
"""

from transformers import AutoConfig, PretrainedConfig
from transformers.utils import logging

from olmo.config import ModelConfig
from olmo.exceptions import OLMoConfigurationError

logger = logging.get_logger(__name__)


class OLMoConfig(PretrainedConfig):
    model_type = "hf_olmo"
    keys_to_ignore_at_inference = ["past_key_values"]  # TODO: confirm

    def __init__(self, use_cache: bool = False, **kwargs):
        model_config = ModelConfig()
        all_kwargs = model_config.asdict()
        all_kwargs.update(kwargs)
        all_kwargs.update({"use_cache": use_cache})
        all_kwargs.update(
            {"architectures": all_kwargs.get("architectures", ["OLMoForCausalLM"]) or ["OLMoForCausalLM"]}
        )
        super().__init__(**all_kwargs)

    @property
    def num_attention_heads(self):
        return self.n_heads

    @property
    def num_hidden_layers(self):
        return self.n_layers

    @property
    def hidden_size(self):
        return self.d_model

    @property
    def effective_n_kv_heads(self) -> int:
        if self.n_kv_heads is None:
            if self.multi_query_attention is True:
                return 1
            else:
                return self.n_heads
        else:
            if self.multi_query_attention is None:
                return self.n_kv_heads
            if self.multi_query_attention:
                n_kv_heads_should_be = 1
            else:
                n_kv_heads_should_be = self.n_heads
            if self.n_kv_heads == n_kv_heads_should_be:
                return n_kv_heads_should_be
            else:
                raise OLMoConfigurationError(
                    "You can't set `multi_query_attention` and `n_kv_heads` at the same time."
                )


# Register the config class so that it is available for transformer pipelines, auto-loading etc.
# OLMo is integrated directly in transformers from v4.40.0 onwards, but the version in transformers
# may not support the newest architectures we create.
AutoConfig.register("hf_olmo", OLMoConfig)
