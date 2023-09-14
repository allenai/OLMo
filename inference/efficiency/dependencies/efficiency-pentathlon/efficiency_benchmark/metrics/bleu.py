from typing import Any, Dict, List, Union

import torch
from sacrebleu.metrics import BLEU
from torchmetrics.aggregation import BaseAggregator


class BLEUMetric(BaseAggregator):
    def __init__(
        self,
        base: int = 2,  # Does anyone ever use anything but 2 here?
        nan_strategy: Union[str, float] = "warn",
        **kwargs: Dict[str, Any],
    ):
        super().__init__("sum", [], nan_strategy, **kwargs)
        self.base = base
        self._outputs: List[str] = []
        self._targets: List[str] = []
        self._bleu_metric = BLEU()

    def update(self, output: str, target: str) -> None:  # type: ignore
        self._outputs.append(output)
        self._targets.append(target)

    def compute(self) -> torch.Tensor:
        bleu = self._bleu_metric.corpus_score(self._outputs, [self._targets])
        return torch.Tensor([bleu.score])
