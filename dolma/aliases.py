from os import PathLike
from typing import Optional, TypedDict, Union

import torch

__all__ = ["PathOrStr", "BatchDict"]


PathOrStr = Union[str, PathLike]


class BatchDict(TypedDict):
    input_ids: torch.LongTensor
    attention_mask: Optional[torch.Tensor]
    attention_bias: Optional[torch.Tensor]
