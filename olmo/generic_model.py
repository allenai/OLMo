from __future__ import annotations

import logging
import math
import sys
from abc import abstractmethod
from collections import defaultdict
from functools import partial
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    cast,
)

import torch
import torch.backends.cuda
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

from .aliases import PathOrStr
from .config import (
    ActivationCheckpointingStrategy,
    ActivationType,
    BlockType,
    CheckpointType,
    FSDPWrapStrategy,
    LayerNormType,
    ModelConfig,
)
from .exceptions import OLMoConfigurationError
from .initialization import ModuleType, init_weights
from .torch_util import ensure_finite_
from .model import _non_meta_init_device, OLMoOutput

from mamba_ssm import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig

if sys.version_info.minor > 8:
    from collections.abc import MutableMapping
elif sys.version_info.minor == 8:
    from typing import MutableMapping
else:
    raise SystemExit("This script supports Python 3.8 or higher")


class GenericOLMoModel(nn.Module):
    """
    OLMo like model class and OLMo config to initialize generic models

    TODO: add generate, from_checkpoint, _make_state_dict_compatible?
    """
    def __init__(self, config: ModelConfig, init_params: bool = True):
        super().__init__()
        self.config = config

    @abstractmethod
    def adapt_olmo_config(self, olmo_config: ModelConfig):
        pass

    @abstractmethod
    def set_activation_checkpointing(self, strategy: Optional[ActivationCheckpointingStrategy]):
        pass

    @property
    def device(self) -> torch.device:
        device: torch.device = self.transformer.wte.weight.device  # type: ignore
        if device.type == "meta":
            return _non_meta_init_device(self.config)
        else:
            return device

    @abstractmethod
    def reset_parameters(self):
        pass

    @abstractmethod
    def forward(self, input_ids: torch.LongTensor) -> OLMoOutput:
        """
        :param input_ids: A tensor of shape `(batch_size, seq_len)`.
        """
        pass

    @abstractmethod
    def get_fsdp_wrap_policy(self, wrap_strategy: Optional[FSDPWrapStrategy] = None):
        pass

    @abstractmethod
    def num_params(self, include_embedding: bool = True) -> int:
        pass

    @classmethod
    def build(cls, config: ModelConfig, size: Optional[int] = None, **kwargs) -> GenericOLMoModel:
        if config.model.model_name == 'mamba':
            return Mamba(config, **kwargs)
        else:
            raise NotImplementedError(f"Unknown model: '{config.model.model_name}'")


class Mamba(GenericOLMoModel):
    def __init__(self, config: ModelConfig, init_params: bool = True):
        super().__init__(config, init_params)

        dtype = None
        if config.precision == 'amp_bf16' or not config.precision:
            dtype = torch.bfloat16
        elif config.precision == 'amp_fp16':
            dtype = torch.float16
        elif config.precision == 'fp32':
            dtype = torch.float32

        self.model = MambaLMHeadModel(
            config=self.adapt_olmo_config(config),
            initializer_cfg={
                'initializer_range': config.model.init_std,
                'rescale_prenorm_residual': config.model.rescale_prenorm_residual,
            },   # params to _init_weights function
            device=config.model.init_device,
            dtype=dtype,
        )

    def adapt_olmo_config(self, olmo_config: ModelConfig) -> MambaConfig:
        mamba_config = MambaConfig()

        # patch config
        mamba_config.d_model = olmo_config.model.d_model
        mamba_config.n_layer = olmo_config.model.n_layers
        mamba_config.vocab_size = olmo_config.model.vocab_size
        mamba_config.ssm_cfg = {}

        # ssm_cfg in mamba layer
        mamba_config.ssm_cfg["d_state"] = olmo_config.model.d_state
        mamba_config.ssm_cfg["d_conv"] = olmo_config.model.d_conv
        mamba_config.ssm_cfg["expand"] = olmo_config.model.mlp_ratio

        # ssm ops config
        mamba_config.ssm_cfg["dt_rank"] = olmo_config.model.time_step_rank
        mamba_config.ssm_cfg["dt_min"] = olmo_config.model.time_step_min
        mamba_config.ssm_cfg["dt_max"] = olmo_config.model.time_step_max
        mamba_config.ssm_cfg["dt_init"] = olmo_config.model.time_step_init_scheme
        mamba_config.ssm_cfg["dt_scale"] = olmo_config.model.time_step_scale
        mamba_config.ssm_cfg["dt_init_floor"] = olmo_config.model.time_step_floor

        mamba_config.ssm_cfg["conv_bias"] = olmo_config.model.conv_bias
        mamba_config.ssm_cfg["bias"] = olmo_config.model.include_bias
        mamba_config.ssm_cfg["use_fast_path"] = olmo_config.model.use_fast_path

        mamba_config.rms_norm = True if olmo_config.model.layer_norm_type == LayerNormType.rms else False
        mamba_config.residual_in_fp32 = True
        mamba_config.fused_add_norm = True
        mamba_config.pad_vocab_size_multiple = 128 if olmo_config.model.embedding_size > olmo_config.model.vocab_size else 0
        mamba_config.tie_embeddings = olmo_config.model.weight_tying

        return mamba_config

    def reset_parameters(self):
        """
        Mamba has its own init weights method which is called in __init__
        """
        return

    def forward(self, input_ids: torch.LongTensor) -> OLMoOutput:
        """
        :param input_ids: A tensor of shape `(batch_size, seq_len)`.
        """
        return OLMoOutput(logits=self.model(input_ids))

    def get_fsdp_wrap_policy(self, wrap_strategy: Optional[FSDPWrapStrategy] = None):
        if wrap_strategy is None:
            return None

    def num_params(self, include_embedding: bool = True) -> int:
        params = (np for np in self.model.named_parameters())
        if not include_embedding:
            params = filter(lambda np: ".embeddings." not in np[0], params)

        return sum(p.numel() for _, p in params)

    def set_activation_checkpointing(self, strategy: Optional[ActivationCheckpointingStrategy]):
        self._activation_checkpoint_fn = None
