"""
OLMo configuration
"""

from transformers import AutoConfig, PretrainedConfig
from transformers.utils import logging

from olmo.config import ModelConfig

logger = logging.get_logger(__name__)


class InfgramConfig(PretrainedConfig):
    def __init__(
        self,
        model_type, # dummy field
        index_dir,
        min_cnt=2,
        support=20,
        cpp_log_path='/tmp/cpp_engine.log',
        mode='prod',
        sharded=False, # sharded index is not supported
        prefetch=False, # prefetch is not supported
        method_train=2,
        method_eval=2,
    ):
        self.index_dir = index_dir
        self.min_cnt = min_cnt
        self.support = support
        self.cpp_log_path = cpp_log_path
        self.mode = mode
        self.sharded = sharded
        self.prefetch = prefetch
        self.method_train = method_train
        self.method_eval = method_eval


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
        if 'infgram' in all_kwargs:
            all_kwargs['infgram'] = InfgramConfig(**all_kwargs['infgram'])
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


# Register the config class so that it is available for transformer pipelines, auto-loading etc.
# OLMo is integrated directly in transformers from v4.40.0 onwards, but the version in transformers
# may not support the newest architectures we create.
AutoConfig.register("hf_olmo", OLMoConfig)
