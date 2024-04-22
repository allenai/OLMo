from __future__ import annotations

import sys
from abc import abstractmethod
from typing import Optional

import torch
import torch.backends.cuda
import torch.nn as nn
import torch.nn.functional as F

from .config import (
    ActivationCheckpointingStrategy,
    FSDPWrapStrategy,
    LayerNormType,
    ModelConfig,
)
from .model import _non_meta_init_device, OLMoOutput, OLMoSequentialBlock

from mamba_ssm import MambaLMHeadModel
from mamba_ssm.modules.mamba_simple import Block
from mamba_ssm.models.config_mamba import MambaConfig


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
        if config.model_name == 'mamba':
            return Mamba(config, **kwargs)
        elif config.model_name == 'mlp_mamba':
            return MLPMamba(config, **kwargs)
        elif config.model_name == 'zamba':
            return Zamba(config, **kwargs)
        else:
            raise NotImplementedError(f"Unknown model: '{config.model_name}'")


class Mamba(GenericOLMoModel):
    def __init__(self, config: ModelConfig, init_params: bool = True, precision: str = 'fp32'):
        super().__init__(config, init_params)

        # main training script sends precision at bf16
        dtype = None
        if precision == 'amp_bf16':
            dtype = torch.bfloat16
        elif precision == 'amp_fp16':
            dtype = torch.float16
        elif precision == 'fp32':
            dtype = torch.float32

        self.model = MambaLMHeadModel(
            config=self.adapt_olmo_config(config),
            initializer_cfg={
                'initializer_range': config.init_std,
                'rescale_prenorm_residual': config.rescale_prenorm_residual,
            },   # params to _init_weights function
            device=config.init_device,
            dtype=torch.float32,
        )

    def adapt_olmo_config(self, olmo_config: ModelConfig) -> MambaConfig:
        mamba_config = MambaConfig()

        # patch config
        mamba_config.d_model = olmo_config.d_model
        mamba_config.n_layer = olmo_config.n_layers
        mamba_config.vocab_size = olmo_config.vocab_size
        mamba_config.ssm_cfg = {}

        # ssm_cfg in mamba layer
        mamba_config.ssm_cfg["d_state"] = olmo_config.d_state
        mamba_config.ssm_cfg["d_conv"] = olmo_config.d_conv
        mamba_config.ssm_cfg["expand"] = olmo_config.expand

        # ssm ops config
        mamba_config.ssm_cfg["dt_rank"] = olmo_config.time_step_rank
        mamba_config.ssm_cfg["dt_min"] = olmo_config.time_step_min
        mamba_config.ssm_cfg["dt_max"] = olmo_config.time_step_max
        mamba_config.ssm_cfg["dt_init"] = olmo_config.time_step_init_scheme
        mamba_config.ssm_cfg["dt_scale"] = olmo_config.time_step_scale
        mamba_config.ssm_cfg["dt_init_floor"] = olmo_config.time_step_floor

        mamba_config.ssm_cfg["conv_bias"] = olmo_config.conv_bias
        mamba_config.ssm_cfg["bias"] = olmo_config.include_bias
        mamba_config.ssm_cfg["use_fast_path"] = olmo_config.use_fast_path

        mamba_config.rms_norm = True if olmo_config.layer_norm_type == LayerNormType.rms else False
        mamba_config.residual_in_fp32 = olmo_config.residual_in_fp32
        mamba_config.fused_add_norm = olmo_config.fused_add_norm
        mamba_config.pad_vocab_size_multiple = 128 if olmo_config.embedding_size > olmo_config.vocab_size else 0
        mamba_config.tie_embeddings = olmo_config.weight_tying

        return mamba_config

    def reset_parameters(self):
        """
        Mamba has its own init weights method which is called in __init__
        """
        return

    def forward(self, input_ids: torch.LongTensor, **kwargs) -> OLMoOutput:
        """
        :param input_ids: A tensor of shape `(batch_size, seq_len)`.
        """
        return OLMoOutput(
            logits=self.model(input_ids).logits,
            attn_key_values=None,
            hidden_states=None,
        )

    def get_fsdp_wrap_policy(self, wrap_strategy: Optional[FSDPWrapStrategy] = None):
        if wrap_strategy is None:
            return None
        elif wrap_strategy == FSDPWrapStrategy.by_block:
            def fsdp_wrap_fn(module, recurse: bool = True, nonwrapped_numel: int = 0):
                del nonwrapped_numel
                wrap = isinstance(module, Block)
                if recurse:
                    return True
                else:
                    return wrap

            return fsdp_wrap_fn
        else:
            raise NotImplementedError(wrap_strategy)

    def num_params(self, include_embedding: bool = True) -> int:
        params = (np for np in self.model.named_parameters())
        if not include_embedding:
            params = filter(lambda np: ".embedding." not in np[0], params)

        return sum(p.numel() for _, p in params)

    def set_activation_checkpointing(self, strategy: Optional[ActivationCheckpointingStrategy]):
        self._activation_checkpoint_fn = None


class MLPMambaBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__(config)


class MLPMamba(GenericOLMoModel):
    def __init__(self, config: ModelConfig, init_params: bool = True, precision: str = 'fp32'):
        super().__init__(config, init_params)

    def adapt_olmo_config(self, olmo_config: ModelConfig) -> MambaConfig:
        mamba_config = MambaConfig()

    def set_activation_checkpointing(self, strategy: Optional[ActivationCheckpointingStrategy]):
        self._activation_checkpoint_fn = None

    def reset_parameters(self):
        """
        Mamba has its own init weights method which is called in __init__
        """
        return

    def forward(self, input_ids: torch.LongTensor, **kwargs) -> OLMoOutput:
        """
        :param input_ids: A tensor of shape `(batch_size, seq_len)`.
        """
        logits = self.model(input_ids).logits

        return OLMoOutput(
            logits=logits,
            attn_key_values=None,
            hidden_states=None,
        )

    def get_fsdp_wrap_policy(self, wrap_strategy: Optional[FSDPWrapStrategy] = None):
        if wrap_strategy is None:
            return None
        elif wrap_strategy == FSDPWrapStrategy.by_block:
            def fsdp_wrap_fn(module, recurse: bool = True, nonwrapped_numel: int = 0):
                del nonwrapped_numel
                wrap = isinstance(module, MLPMambaBlock)
                if recurse:
                    return True
                else:
                    return wrap

            return fsdp_wrap_fn
        else:
            raise NotImplementedError(wrap_strategy)

    def num_params(self, include_embedding: bool = True) -> int:
        params = (np for np in self.model.named_parameters())
        if not include_embedding:
            params = filter(lambda np: ".embedding." not in np[0], params)

        return sum(p.numel() for _, p in params)


class Zamba(Mamba):
    def __init__(self, config: ModelConfig, init_params: bool = True, precision: str = 'fp32'):
        super().__init__(config, init_params, precision)

        # delete 4 mamba layers to substitute in 1 OLMoSequentialBlock
        del self.model.backbone.layers[-4:]

        self.attention_block = OLMoSequentialBlock(
            layer_id=config.n_layers - 4,
            config=config,
            cache=None,
        )

    def forward(self, input_ids: torch.LongTensor, **kwargs) -> OLMoOutput:
        """
        Interleave single attention block between equally spaced out mamba-blocks
        """
        hidden_states = self.model.embedding(input_ids)

        # 1st pass through attention block
        hidden_states, _ = self.attention_block(hidden_states)
        residual = None

        # pass through half of the mamba layers
        for layer in self.model.backbone.layers[:len(self.model.backbone.layers) // 2]:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=kwargs.get("inference_params")
            )

        # 2nd pass through attention block
        # attention block has its own norm at the beginning of the block
        # => only add residual and hidden
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states, _ = self.attention_block(hidden_states)
        residual = None

        # pass through the remaining mamba layers
        for layer in self.model.backbone.layers[len(self.model.backbone.layers) // 2]:
            hidden_states, residual = layer(
                hidden_states, residual, inference_params=kwargs.get("inference_params")
            )

        # 3rd pass through attention block
        # attention block has its own norm at the beginning of the block
        # => only add residual and hidden
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states, _ = self.attention_block(hidden_states)
        hidden_states = self.model.backbone.norm_f(hidden_states)

        return OLMoOutput(
            logits=self.model.lm_head(hidden_states),
            attn_key_values=None,
            hidden_states=None,
        )

    def get_fsdp_wrap_policy(self, wrap_strategy: Optional[FSDPWrapStrategy] = None):
        if wrap_strategy is None:
            return None

    def reset_parameters(self):
        """
        Mamba has its own init weights method which is called in __init__
        """
        # NOTE: the standard deviation for these weights does not depend on the layer.
        self.attention_block.reset_parameters()
