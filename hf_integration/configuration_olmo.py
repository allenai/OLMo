"""
OLMo configuration
"""

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from olmo.config import ModelConfig

logger = logging.get_logger(__name__)

OLMO_PRETRAINED_CONFIG_ARCHIVE_MAP = {}


class OLMoConfig(PretrainedConfig, ModelConfig):  # trying to keep it as simple as possible.

    model_type = "olmo"
    keys_to_ignore_at_inference = ["past_key_values"]   # TODO: confirm

    def __init__(self, **kwargs):
        # TODO: confirm name mapping.
        super().__init__(**kwargs)

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

    def to_dict(self) -> str:
        the_dict = PretrainedConfig.to_dict(self)
        the_dict.update(ModelConfig.asdict(self))
        return the_dict
