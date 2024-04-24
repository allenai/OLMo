from __future__ import annotations

import sys
import logging
from abc import abstractmethod
from typing import Optional
from functools import partial

import torch
import torch.backends.cuda
import torch.nn as nn
import torch.nn.functional as F

from .initialization import init_weights, ModuleType
from .config import (
    ActivationCheckpointingStrategy,
    FSDPWrapStrategy,
    LayerNormType,
    ModelConfig,
)
from .model import (
    _non_meta_init_device,
    activation_checkpoint_function,
    _non_meta_init_device,
    OLMoOutput,
    OLMoSequentialBlock,
    Activation,
    Dropout,
    LayerNorm,
)

from mamba_ssm import MambaLMHeadModel
from mamba_ssm.modules.mamba_simple import Block, Mamba
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import _init_weights


log = logging.getLogger(__name__)


class GenericOLMoModel(nn.Module):
    """
    OLMo like model class and OLMo config to initialize generic models

    TODO: add generate, from_checkpoint, _make_state_dict_compatible?
    """
    def __init__(self, config: ModelConfig, init_params: bool = True):
        """
        For each class inheriting this base class, add: device property and 

        if init_params and self.config.init_device != 'meta':
            self.reset_parameters()

        as the last piece of code in init
        """
        super().__init__()
        self.config = config

    @abstractmethod
    def set_activation_checkpointing(self, strategy: Optional[ActivationCheckpointingStrategy]):
        pass

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
            return OGMamba(config, **kwargs)
        elif config.model_name == 'mlp_mamba':
            return MLPMamba(config, **kwargs)
        elif config.model_name == 'zamba':
            return Zamba(config, **kwargs)
        else:
            raise NotImplementedError(f"Unknown model: '{config.model_name}'")


class OGMamba(GenericOLMoModel):
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
                'initializer_range': config.mamba_initializer_range,
                'rescale_prenorm_residual': config.rescale_prenorm_residual,
            },   # params to _init_weights function
            device=config.init_device,
            dtype=torch.float32,
        )

        # When `init_device="meta"` FSDP will call `reset_parameters()` to initialize weights.
        if init_params and self.config.init_device != "meta":
            self.reset_parameters()

    @property
    def device(self) -> torch.device:
        device: torch.device = self.model.backbone.embedding.weight.device  # type: ignore
        if device.type == "meta":
            return _non_meta_init_device(self.config)
        else:
            return device

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
            params = filter(lambda np: "embedding" not in np[0], params)

        return sum(p.numel() for _, p in params)

    def set_activation_checkpointing(self, strategy: Optional[ActivationCheckpointingStrategy]):
        self._activation_checkpoint_fn = None


class MLPMambaBlock(nn.Module):
    def __init__(self, layer_id: int, config: ModelConfig):
        super().__init__()

        # block setup
        self.layer_id = layer_id
        self.config = config
        self._activation_checkpoint_fn = None

        # Dropout.
        self.dropout = Dropout(config.residual_dropout)

        # Layer Norms
        self.temporal_mix_norm = LayerNorm.build(config)
        self.ff_norm = LayerNorm.build(config)

        # mamba block
        # fp32 is used in original mamba implementation
        self.mamba_block = Mamba(
            d_model=config.d_model,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand=config.expand,
            dt_rank=config.time_step_rank,
            dt_min=config.time_step_min,
            dt_max=config.time_step_max,
            dt_init=config.time_step_init_scheme,
            dt_scale=config.time_step_scale,
            dt_init_floor=config.time_step_floor,
            conv_bias=config.conv_bias,
            bias=config.include_bias,
            use_fast_path=config.use_fast_path,
            layer_idx=layer_id,
            device=config.init_device,
            dtype=torch.float32,
        )

        # Gated MLP block
        self.hidden_size = (
            config.mlp_hidden_size if config.mlp_hidden_size is not None else config.mlp_ratio * config.d_model
        )

        self.act = Activation.build(config)
        assert (self.act.output_multiplier * self.hidden_size) % 1 == 0

        self.ff_proj = nn.Linear(
            config.d_model, self.hidden_size, bias=config.include_bias, device=config.init_device
        )

        self.ff_out = nn.Linear(
            int(self.act.output_multiplier * self.hidden_size),
            config.d_model,
            bias=config.include_bias,
            device=config.init_device,
        )
        self.ff_out._is_residual = True  # type: ignore

    def reset_parameters(self):
        self.temporal_mix_norm.reset_parameters()
        self.ff_norm.reset_parameters()

        self.mamba_block.apply(
            partial(
                _init_weights,
                n_layer=self.config.n_layers,
                initializer_range=self.config.mamba_initializer_range,
                rescale_prenorm_residual=self.config.rescale_prenorm_residual,
            )
        )

        init_weights(
            self.config,
            self.ff_proj,
            d=self.config.d_model,
            layer_id=None,
            type_of_module=ModuleType.in_module,
        )

        init_weights(
            self.config,
            self.ff_out,
            d=self.ff_out.in_features,
            layer_id=self.layer_id,
            type_of_module=ModuleType.out_module,
        )

    def set_activation_checkpointing(self, strategy: Optional[ActivationCheckpointingStrategy]):
        if strategy == ActivationCheckpointingStrategy.fine_grained:
            self._activation_checkpoint_fn = activation_checkpoint_function(self.config)
        else:
            self._activation_checkpoint_fn = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # sequence mixing block
        if self._activation_checkpoint_fn is not None:
            out = self._activation_checkpoint_fn(self.mamba_block, self.temporal_mix_norm(x))
        else:
            out = self.mamba_block(self.temporal_mix_norm(x))

        x = x + self.dropout(out)

        # pass through MLP block
        # shape: (batch_size, seq_len, d_model)
        og_x = x
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.ff_norm, x)  # type: ignore
        else:
            x = self.ff_norm(x)

        x = self.ff_proj(x)
        if self._activation_checkpoint_fn is not None:
            x = self._activation_checkpoint_fn(self.act, x)  # type: ignore
        else:
            x = self.act(x)

        x = self.ff_out(x)
        x = og_x + self.dropout(x)

        return x


class MLPMamba(GenericOLMoModel):
    def __init__(self, config: ModelConfig, init_params: bool = True, precision: str = 'fp32'):
        super().__init__(config, init_params)

        # Validate config.
        if self.config.embedding_size is not None and self.config.embedding_size != self.config.vocab_size:
            if self.config.embedding_size < self.config.vocab_size:
                raise OLMoConfigurationError("embedding size should be at least as big as vocab size")
            elif self.config.embedding_size % 128 != 0:
                import warnings

                warnings.warn(
                    "Embedding size is not a multiple of 128! This could hurt throughput performance.", UserWarning
                )

        self.activation_checkpointing_strategy: Optional[ActivationCheckpointingStrategy] = None
        self._activation_checkpoint_fn: Callable = activation_checkpoint_function(self.config)

        self.model = nn.ModuleDict(
            dict(
                embedding=nn.Embedding(
                    config.embedding_size or config.vocab_size, config.d_model, device=config.init_device
                ),
                emb_drop=Dropout(config.embedding_dropout),
                ln_f=LayerNorm.build(config),
            )
        )

        blocks = [MLPMambaBlock(i, config) for i in range(config.n_layers)]
        self.model.update({"blocks": nn.ModuleList(blocks)})

        if not config.weight_tying:
            self.model.update(
                {
                    "ff_out": nn.Linear(
                        config.d_model,
                        config.embedding_size or config.vocab_size,
                        bias=config.include_bias,
                        device=config.init_device,
                    )
                }
            )

        # When `init_device="meta"` FSDP will call `reset_parameters()` to initialize weights.
        if init_params and self.config.init_device != 'meta':
            self.reset_parameters()

    @property
    def device(self) -> torch.device:
        device: torch.device = self.model.embedding.weight.device  # type: ignore
        if device.type == "meta":
            return _non_meta_init_device(self.config)
        else:
            return device

    def set_activation_checkpointing(self, strategy: Optional[ActivationCheckpointingStrategy]):
        # TODO: self._activation_checkpoint_fn is not set at a model level?
        self.activation_checkpointing_strategy = strategy
        for block in self.model.blocks:
            block.set_activation_checkpointing(strategy)

    def reset_parameters(self):
        log.info("Initializing model parameters...")
        # Top-level embeddings / linear layers
        init_weights(
            self.config,
            self.model.embedding,  # type: ignore
            std_factor=(0.5 * math.sqrt(self.config.d_model)) if self.config.scale_logits else 1.0,
            type_of_module=ModuleType.emb,
        )

        # Top-level layer norm
        self.model.ln_f.reset_parameters()  # type: ignore

        # Output weights
        if hasattr(self.model, "ff_out"):
            init_weights(self.config, self.model.ff_out, type_of_module=ModuleType.final_out)  # type: ignore

        # MLPMamba blocks init
        for block in self.model.blocks:
            block.reset_parameters()

    def forward(self, input_ids: torch.LongTensor, **kwargs) -> OLMoOutput:
        """
        :param input_ids: A tensor of shape `(batch_size, seq_len)`.
        """
        batch_size, seq_len = input_ids.size()

        # shape: (batch_size, seq_len, d_model)
        x = self.model.embedding(input_ids)
        x = self.model.emb_drop(x)  # type: ignore

        for block_idx, block in enumerate(self.model.blocks):
            x = block(x)

        x = self.model.ln_f(x)  # type: ignore
        if self.config.weight_tying:
            logits = F.linear(x, self.model.embedding.weight, None)  # type: ignore
        else:
            logits = self.model.ff_out(x)  # type: ignore

        return OLMoOutput(logits=logits, attn_key_values=None, hidden_states=None)

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
            params = filter(lambda np: "embedding" not in np[0], params)

        return sum(p.numel() for _, p in params)


class Zamba(OGMamba):
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
