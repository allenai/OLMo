from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import torch.nn.functional as F

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

    def __call__(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        assert items
        max_len = max((len(x["input_ids"]) for x in items))
        all_input_ids = []
        all_attention_mask = []
        for x in items:
            pad_shape = (
                (max_len - len(x["input_ids"]), 0)
                if self.pad_direction == PaddingDirection.left
                else (0, max_len - len(x["input_ids"]))
            )
            all_input_ids.append(
                F.pad(
                    torch.tensor(x["input_ids"], dtype=torch.long, device=self.config.device),
                    pad_shape,
                    value=self.config.pad_token_id,
                )
            )
            if "attention_mask" in x:
                all_attention_mask.append(
                    F.pad(
                        torch.tensor(
                            [1.0] * len(x["attention_mask"]), dtype=torch.float, device=self.config.device
                        ),
                        pad_shape,
                        value=0.0,
                    )
                )

        out = {"input_ids": torch.stack(all_input_ids)}
        if all_attention_mask:
            out["attention_mask"] = torch.stack(all_attention_mask)
        return out
