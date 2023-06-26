"""
OLMo configuration
"""

from transformers import AutoConfig, PretrainedConfig
from transformers.utils import logging

from olmo.config import ModelConfig

logger = logging.get_logger(__name__)


class OLMoConfig(PretrainedConfig):
    model_type = "olmo"
    keys_to_ignore_at_inference = ["past_key_values"]  # TODO: confirm

    def __init__(self, use_cache: bool = False, **kwargs):
        model_config = ModelConfig()
        all_kwargs = model_config.asdict()
        all_kwargs.update(kwargs)
        all_kwargs.update({"use_cache": use_cache})
        super().__init__(**all_kwargs)


# Register the config class so that it is available for transformer pipelines, auto-loading etc.
AutoConfig.register("olmo", OLMoConfig)
