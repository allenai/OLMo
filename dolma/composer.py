import logging
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from composer.loggers import ConsoleLogger
from composer.loggers.logger import format_log_data_value
from composer.models import ComposerModel
from composer.utils import dist
from torchmetrics import Metric

from .aliases import BatchDict
from .config import ModelConfig, SchedulerConfig, SchedulerType
from .model import DolmaGPT, DolmaGPTOutput

log = logging.getLogger(__name__)

__all__ = ["ComposerDolmaGPT", "DolmaConsoleLogger", "build_scheduler", "build_algorithm"]


class ComposerDolmaGPT(ComposerModel):
    def __init__(self, config: ModelConfig):
        super().__init__()
        model = DolmaGPT(config)
        if config.compile:
            log.info("compiling model...")
            self.model = torch.compile(model)
        else:
            self.model = model

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

    @property
    def num_fwd_flops(self):
        return self.model.num_fwd_flops

    def flops_per_batch(self, batch: BatchDict):
        # Note: this computation does not take into account padding, and assumes
        # that the dataset has been constructed without padding. Additionally, we
        # assume the backward pass is approximately 2x the forward pass
        return self.num_fwd_flops * 3 * batch["input_ids"].shape[0]


class DolmaConsoleLogger(ConsoleLogger):
    def _log_hparams_to_console(self):
        if dist.get_local_rank() == 0:
            log_str = "Config:"
            for name, value in self.hparams.items():
                value_str = format_log_data_value(value)
                log_str += f"\n\t {name}: {value_str}"
            self._log_to_console(log_str)

    def _log_to_console(self, log_str: str):
        log.info(log_str)


def build_scheduler(cfg: SchedulerConfig):
    from composer.optim.scheduler import (
        ConstantWithWarmupScheduler,
        CosineAnnealingWithWarmupScheduler,
        LinearWithWarmupScheduler,
    )

    if cfg.name == SchedulerType.constant_with_warmup:
        return ConstantWithWarmupScheduler(t_warmup=cfg.t_warmup)
    elif cfg.name == SchedulerType.cosine_with_warmup:
        return CosineAnnealingWithWarmupScheduler(t_warmup=cfg.t_warmup, alpha_f=cfg.alpha_f)
    elif cfg.name == SchedulerType.linear_decay_with_warmup:
        return LinearWithWarmupScheduler(t_warmup=cfg.t_warmup, alpha_f=cfg.alpha_f)
    else:
        raise NotImplementedError(f"Not sure how to build scheduler '{cfg.name}'")


def build_algorithm(name: str, kwargs: Dict[str, Any]):
    from composer import algorithms

    if name == "gradient_clipping":
        return algorithms.GradientClipping(**kwargs)
    elif name == "fused_layernorm":
        return algorithms.FusedLayerNorm(**kwargs)
    elif name == "gated_linear_units":
        return algorithms.GatedLinearUnits(**kwargs)
    elif name == "low_precision_layernorm":
        return algorithms.LowPrecisionLayerNorm(**kwargs)
    else:
        raise NotImplementedError(f"Not sure how to build algorithm '{name}'")
