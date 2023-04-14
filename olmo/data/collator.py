from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

import torch
import torch.nn.functional as F

from ..aliases import BatchDict
from ..config import PaddingDirection, TrainConfig

__all__ = ["DataCollator"]


@dataclass
class DataCollator:
    config: TrainConfig

    _alibi_causal_attention_bias: Optional[torch.Tensor] = None

    @property
    def pad_direction(self) -> PaddingDirection:
        return self.config.data.pad_direction

    @property
    def pad_token_id(self) -> int:
        return self.config.model.pad_token_id

    @property
    def alibi_causal_attention_bias(self) -> torch.FloatTensor:
        if self._alibi_causal_attention_bias is None:
            from ..model import alibi_attention_bias, causal_attention_bias

            self._alibi_causal_attention_bias = causal_attention_bias(
                self.config.model, device="cpu"
            ) + alibi_attention_bias(self.config.model, device="cpu")
        return self._alibi_causal_attention_bias  # type: ignore

    @classmethod
    def from_train_config(cls, config: TrainConfig) -> DataCollator:
        return cls(config=config)

    def __call__(self, items: Union[List[BatchDict], List[torch.Tensor]]) -> BatchDict:
        assert items
        max_len = max((len(x["input_ids"] if isinstance(x, dict) else x) for x in items))
        batch_size = len(items)

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
                    input_ids.to(dtype=torch.long),
                    pad_shape,
                    value=self.pad_token_id,
                )
            )

            # Pad attention mask.
            attention_mask = x.get("attention_mask") if isinstance(x, dict) else None
            if attention_mask is not None:
                if not isinstance(attention_mask, torch.Tensor):
                    attention_mask = torch.tensor(attention_mask)
                all_attention_mask.append(
                    F.pad(
                        attention_mask.to(dtype=torch.float),
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
                        attention_bias,
                        pad_shape + pad_shape,
                        value=pad_value,
                    )
                )

        out = {"input_ids": torch.stack(all_input_ids)}

        if all_attention_mask:
            out["attention_mask"] = torch.stack(all_attention_mask)

        if all_attention_bias:
            out["attention_bias"] = torch.stack(all_attention_bias)
        elif self.config.model.alibi:
            out["attention_bias"] = self.alibi_causal_attention_bias.expand(batch_size, -1, -1, -1).clone()

        return out  # type: ignore
