from dataclasses import dataclass
from typing import Any, Dict, Iterator

import torch
from torch.utils.data import DataLoader
from torchmetrics import Metric

from ..config import EvaluatorConfig, EvaluatorType

__all__ = ["Evaluator"]


@dataclass
class Evaluator:
    cfg: EvaluatorConfig
    eval_loader: DataLoader
    eval_batches: Iterator[Dict[str, Any]]
    eval_metric: Metric

    def reset_metrics(self) -> None:
        self.eval_metric.reset()

    def compute_metrics(self) -> Dict[str, float]:
        metric_val = self.eval_metric.compute()
        if self.cfg.type == EvaluatorType.downstream:
            # Metric is ICLMetric
            return {
                f"eval/downstream/{self.cfg.label}_{self.eval_metric.metric_type}": metric_val.item(),
            }
        elif self.cfg.type == EvaluatorType.lm:
            # Metric is cross entropy loss
            loss = metric_val
            return {
                f"eval/{self.cfg.label}/CrossEntropyLoss": loss.item(),
                f"eval/{self.cfg.label}/Perplexity": torch.exp(loss).item(),
            }
        else:
            raise ValueError(f"Unexpected evaluator type '{self.cfg.type}'")

    def update_metrics(
        self,
        batch: Dict[str, Any],
        loss: torch.Tensor,
        logits: torch.Tensor,
    ) -> None:
        if self.cfg.type == EvaluatorType.downstream:
            # Metric is ICLMetric
            self.eval_metric.update(batch, logits)  # type: ignore
        elif self.cfg.type == EvaluatorType.lm:
            # Metric is cross entropy loss
            self.eval_metric.update(loss)
        else:
            raise ValueError(f"Unexpected evaluator type '{self.cfg.type}'")
