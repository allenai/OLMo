import math
from typing import Any, Dict, Union

import torch
from torchmetrics.aggregation import BaseAggregator


class EntropyMetric(BaseAggregator):
    def __init__(
        self,
        base: int = 2,  # Does anyone ever use anything but 2 here?
        nan_strategy: Union[str, float] = "warn",
        **kwargs: Dict[str, Any],
    ):
        super().__init__("sum", [], nan_strategy, **kwargs)
        self.base = base
        self.add_state("loglikelihood", default=torch.tensor(0.0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("characters", default=torch.tensor(0, dtype=torch.int), dist_reduce_fx="sum")

    def update(
        self, loglikelihood: Union[float, torch.Tensor], characters: Union[int, torch.Tensor]
    ) -> None:  # type: ignore
        loglikelihood = self._cast_and_nan_check_input(loglikelihood)
        if not isinstance(characters, torch.Tensor):
            characters = torch.tensor(characters)
        self.loglikelihood += loglikelihood.sum()
        self.characters += characters.sum()

    def compute(self) -> torch.Tensor:
        return -(self.loglikelihood / self.characters) / math.log(self.base)
