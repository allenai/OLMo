from dataclasses import dataclass
from typing import List, Union, cast

import torch
import torch.nn.functional as F

from ..aliases import BatchDict
from ..config import Config
from ..util import StrEnum

__all__ = ["PaddingDirection", "DataCollator"]


class PaddingDirection(StrEnum):
    right = "right"
    left = "left"


@dataclass
class DataCollator:
    config: Config
    pad_direction: PaddingDirection = PaddingDirection.left

    def __call__(self, items: Union[List[BatchDict], List[torch.Tensor]]) -> BatchDict:
        assert items
        max_len = max((len(x["input_ids"] if isinstance(x, dict) else x) for x in items))
        all_input_ids = []
        all_attention_mask = []
        all_attention_bias = []
        for x in items:
            input_ids = x["input_ids"] if isinstance(x, dict) else x
            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.tensor(input_ids)

            pad_shape = (
                (max_len - len(input_ids), 0)
                if self.pad_direction == PaddingDirection.left
                else (0, max_len - len(input_ids))
            )

            # Pad input IDs.
            all_input_ids.append(
                F.pad(
                    input_ids.to(dtype=torch.long, device=self.config.device),
                    pad_shape,
                    value=self.config.pad_token_id,
                )
            )

            # Pad attention mask.
            attention_mask = x.get("attention_mask") if isinstance(x, dict) else None
            if attention_mask is not None:
                if not isinstance(attention_mask, torch.Tensor):
                    attention_mask = torch.tensor(attention_mask)
                all_attention_mask.append(
                    F.pad(
                        attention_mask.to(dtype=torch.float, device=self.config.device),
                        pad_shape,
                        value=0.0,
                    )
                )

            # Pad attention bias.
            attention_bias = x.get("attention_bias") if isinstance(x, dict) else None
            if attention_bias is not None:
                if not isinstance(attention_bias, torch.Tensor):
                    attention_bias = torch.tensor(attention_bias)
                # Reshape to `(1, seq_len, seq_len)`
                while len(attention_bias.shape) < 3:
                    attention_bias = attention_bias.unsqueeze(0)
                pad_value = False if attention_bias.dtype == torch.bool else float("-inf")
                all_attention_bias.append(
                    F.pad(
                        attention_bias.to(device=self.config.device),
                        pad_shape + pad_shape,
                        value=pad_value,
                    )
                )

        return {
            "input_ids": cast(torch.LongTensor, torch.stack(all_input_ids)),
            "attention_mask": None if not all_attention_mask else torch.stack(all_attention_mask),
            "attention_bias": None if not all_attention_bias else torch.stack(all_attention_bias),
        }
