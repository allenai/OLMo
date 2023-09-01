import math
from typing import Optional, Union

import torch
import torch.nn as nn

from .config import InitFnType, ModelConfig

__all__ = ["init_weights"]


def init_weights(
    module: Union[nn.Linear, nn.Embedding],
    d: Optional[int] = None,
    layer_id: Optional[int] = None,
    std_factor: float = 1.0,
) -> None:
    """
    Initialize weights of a linear or embedding module.

    :param module: The linear or embedding submodule to initialize.
    :param d: The effective dimensionality of the weights. This could be smaller than the actual dimensions
        for fused layers.
    :param layer_id: When set, the standard deviation for the "mitchell" method will be adjusted by
        ``1 / sqrt(2 * (layer_id + 1))``.
    """
    d = d if d is not None else 4096
    nn.init.normal_(module.weight, mean=0.0, std=0.02 * std_factor)

    if isinstance(module, nn.Linear):
        if module.bias is not None:
            nn.init.zeros_(module.bias)
