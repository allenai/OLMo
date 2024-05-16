import math
from typing import Optional, Union

import torch
import torch.nn as nn

from .config import InitFnType, ModelConfig
from .util import StrEnum

__all__ = ["init_weights", "ModuleType"]


class ModuleType(StrEnum):
    in_module = "in"
    out_module = "out"
    emb = "emb"
    final_out = "final_out"


def init_weights(
    config: ModelConfig,
    module: Union[nn.Linear, nn.Embedding],
    d: Optional[int] = None,
    layer_id: Optional[int] = None,
    std_factor: float = 1.0,
    type_of_module: Optional[ModuleType] = None,
) -> None:
    """
    Initialize weights of a linear or embedding module.

    :param config: The model config.
    :param module: The linear or embedding submodule to initialize.
    :param d: The effective input dimensionality of the weights. This could be smaller than the actual dimensions
        for fused layers.
    :param layer_id: When set, the standard deviation for the "mitchell" method will be adjusted by
        ``1 / sqrt(2 * (layer_id + 1))``.
    """
    d = d if d is not None else config.d_model
    if config.init_fn == InitFnType.normal:
        std = config.init_std * std_factor
        if config.init_cutoff_factor is not None:
            cutoff_value = config.init_cutoff_factor * std
            nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-cutoff_value, b=cutoff_value)
        else:
            nn.init.normal_(module.weight, mean=0.0, std=std)
    elif config.init_fn == InitFnType.mitchell:
        std = std_factor / math.sqrt(d)
        if layer_id is not None:
            std = std / math.sqrt(2 * (layer_id + 1))
        nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
    elif config.init_fn == InitFnType.kaiming_normal:
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
    elif config.init_fn == InitFnType.fan_in:
        std = std_factor / math.sqrt(d)
        nn.init.normal_(module.weight, mean=0.0, std=std)
    elif config.init_fn == InitFnType.full_megatron:
        if type_of_module is None:
            raise RuntimeError(f"When using the {InitFnType.full_megatron} init, every module must have a type.")

        cutoff_factor = config.init_cutoff_factor
        if cutoff_factor is None:
            cutoff_factor = 3

        if type_of_module == ModuleType.in_module:
            # for att_proj (same as QKV), ff_proj
            std = config.init_std
        elif type_of_module == ModuleType.out_module:
            # for attn_out, ff_out
            std = config.init_std / math.sqrt(2.0 * config.n_layers)
        elif type_of_module == ModuleType.emb:
            # positional embeddings (wpe)
            # token embeddings (wte)
            std = config.init_std * math.sqrt(config.d_model)
        elif type_of_module == ModuleType.final_out:
            # final output (ff_out)
            std = config.d_model**-0.5
        else:
            raise RuntimeError(f"Unknown module type '{type_of_module}'")
        nn.init.trunc_normal_(
            module.weight,
            mean=0.0,
            std=std,
            a=-cutoff_factor * std,
            b=cutoff_factor * std,
        )
    else:
        raise NotImplementedError(config.init_fn)

    if isinstance(module, nn.Linear):
        if module.bias is not None:
            nn.init.zeros_(module.bias)

        if config.init_fn == InitFnType.normal and getattr(module, "_is_residual", False):
            with torch.no_grad():
                module.weight.div_(math.sqrt(2 * config.n_layers))
