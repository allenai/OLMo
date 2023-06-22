"""
OLMo configuration
"""

from transformers import AutoConfig, PretrainedConfig
from transformers.utils import logging

from olmo.config import ModelConfig

logger = logging.get_logger(__name__)


class OLMoConfig(PretrainedConfig):  # trying to keep it as simple as possible.
    model_type = "olmo"
    keys_to_ignore_at_inference = ["past_key_values"]  # TODO: confirm

    def __init__(self, **kwargs):
        # TODO: pass it as arg?
        model_config = ModelConfig()
        all_kwargs = model_config.asdict()
        all_kwargs.update(kwargs)
        super().__init__(**all_kwargs)

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

    # def to_dict(self) -> str:
    #     the_dict = PretrainedConfig.to_dict(self)
    #     the_dict.update(ModelConfig.asdict(self))
    #     return the_dict


# Register
AutoConfig.register("olmo", OLMoConfig)
