from typing import Any, Dict, Union

import torch
from torchmetrics.aggregation import BaseAggregator


class PerplexityMetric(BaseAggregator):
    def __init__(
        self,
        nan_strategy: Union[str, float] = "warn",
        **kwargs: Dict[str, Any],
    ):
        super().__init__("sum", [], nan_strategy, **kwargs)
        self.add_state("loglikelihood", default=torch.tensor(0.0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("num_tokens", default=torch.tensor(0, dtype=torch.int), dist_reduce_fx="sum")

    def update(
        self, loglikelihood: Union[float, torch.Tensor], num_tokens: Union[int, torch.Tensor]
    ) -> None:  # type: ignore
        loglikelihood = self._cast_and_nan_check_input(loglikelihood)
        if not isinstance(num_tokens, torch.Tensor):
            num_tokens = torch.tensor(num_tokens)
        self.loglikelihood += loglikelihood.sum()
        self.num_tokens += num_tokens.sum()

    def compute(self) -> torch.Tensor:
        return torch.exp(-self.loglikelihood / self.num_tokens)
