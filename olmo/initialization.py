import math
from typing import Optional, Union

import torch
import torch.nn as nn

from .config import ModelConfig, WeightsInitFnType


def linear_init_fn(
    config: ModelConfig, module: Union[nn.Linear, nn.Embedding], std: Optional[float] = None
) -> None:
    init_cfg = config.weights_init_fn
    if init_cfg.name == WeightsInitFnType.normal:
        nn.init.normal_(module.weight, mean=0.0, std=init_cfg.std)
    elif init_cfg.name == WeightsInitFnType.mitchell:
        std = std if std is not None else 1.0 / math.sqrt(config.d_model)
        nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
    else:
        raise NotImplementedError(init_cfg.name)

    if isinstance(module, nn.Linear):
        if module.bias is not None:
            nn.init.zeros_(module.bias)

        if init_cfg.name == WeightsInitFnType.normal and getattr(module, "_is_residual", False):
            with torch.no_grad():
                module.weight.div_(math.sqrt(2 * config.n_layers))
