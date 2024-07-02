from typing import Optional, Union

import torch.nn as nn
from mup import normal_ as mup_normal
from mup import trunc_normal_ as mup_trunc_normal

__all__ = ["init_normal"]


def init_normal(
    module: Union[nn.Linear, nn.Embedding],
    std: float,
    init_cutoff_factor: Optional[float] = None,
    use_mup: bool = False,
):
    # weights
    if init_cutoff_factor is not None:
        cutoff_value = init_cutoff_factor * std
        trunc_normal = mup_trunc_normal if use_mup else nn.init.trunc_normal_
        trunc_normal(module.weight, mean=0.0, std=std, a=-cutoff_value, b=cutoff_value)
    else:
        normal = mup_normal if use_mup else nn.init.normal_
        normal(module.weight, mean=0.0, std=std)

    # biases
    if isinstance(module, nn.Linear) and module.bias is not None:
        nn.init.zeros_(module.bias)
