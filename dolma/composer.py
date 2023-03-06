from typing import Dict, Optional

import torch
import torch.nn.functional as F
from composer.models import ComposerModel
from torchmetrics import Metric

from .aliases import BatchDict
from .config import ModelConfig
from .model import DolmaGPT, DolmaGPTOutput


class ComposerDolmaGPT(ComposerModel):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.model = DolmaGPT(config)

        from composer.metrics.nlp import LanguageCrossEntropy, Perplexity

        self.train_metrics = {
            "LanguageCrossEntropy": LanguageCrossEntropy(config.vocab_size),
            "Perplexity": Perplexity(),
        }
        self.eval_metrics = {
            "LanguageCrossEntropy": LanguageCrossEntropy(config.vocab_size),
            "Perplexity": Perplexity(),
        }

    def get_labels(self, batch: BatchDict) -> torch.Tensor:
        # Labels are just input IDs shifted to the left (first item is ignored).
        labels, attention_mask = batch["input_ids"], batch.get("attention_mask")
        if attention_mask is not None:
            labels = labels.masked_fill(attention_mask == 0.0, -100)
        return labels[..., 1:].contiguous()

    def forward(self, batch: BatchDict) -> DolmaGPTOutput:
        return self.model(**batch)

    def loss(self, outputs: DolmaGPTOutput, batch: BatchDict) -> torch.Tensor:
        labels = self.get_labels(batch)
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        return F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1), ignore_index=-100)

    def eval_forward(self, batch: BatchDict, outputs: Optional[DolmaGPTOutput] = None) -> DolmaGPTOutput:
        return outputs if outputs is not None else self.forward(batch)

    def get_metrics(self, is_train: bool = False) -> Dict[str, "Metric"]:
        return self.train_metrics if is_train else self.eval_metrics

    def update_metric(self, batch: BatchDict, outputs: DolmaGPTOutput, metric: "Metric") -> None:
        labels = self.get_labels(batch)
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        metric.update(shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1))
