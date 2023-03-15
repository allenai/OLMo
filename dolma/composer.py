import warnings
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from composer.core import State
from composer.loggers import ConsoleLogger
from composer.loggers.logger import format_log_data_value
from composer.models import ComposerModel
from composer.utils import dist
from torchmetrics import Metric

from .aliases import BatchDict
from .config import ModelConfig, SchedulerConfig, SchedulerType
from .model import DolmaGPT, DolmaGPTOutput
from .util import echo

__all__ = ["ComposerDolmaGPT", "DolmaConsoleLogger", "build_scheduler", "build_algorithm"]


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

    @property
    def num_fwd_flops(self):
        return self.model.num_fwd_flops

    def flops_per_batch(self, batch: BatchDict):
        # Note: this computation does not take into account padding, and assumes
        # that the dataset has been constructed without padding. Additionally, we
        # assume the backward pass is approximately 2x the forward pass
        return self.num_fwd_flops * 3 * batch["input_ids"].shape[0]


GPU_AVAILABLE_FLOPS = {
    # source: https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet
    # nvidia publishes spec sheet with a 2x sparsity factor
    "h100-sxm": {
        "fp64": 67e12,
        "fp32": 67e12,
        "tf32": 989e12 / 2,
        "fp16": 1.979e15 / 2,
        "amp_fp16": 1.979e15 / 2,
        "bf16": 1.979e15 / 2,
        "amp_bf16": 1.979e15 / 2,
        "fp8": 3.958e15 / 2,
        "amp_fp8": 3.958e15 / 2,
        "int8": 3.958e15 / 2,
    },
    "h100-pcie": {
        "fp64": 51e12,
        "fp32": 51e12,
        "tf32": 756e12 / 2,
        "fp16": 1.513e15 / 2,
        "amp_fp16": 1.513e15 / 2,
        "bf16": 1.513e15 / 2,
        "amp_bf16": 1.513e15 / 2,
        "fp8": 3.026e15 / 2,
        "amp_fp8": 3.026e15 / 2,
        "int8": 3.026e15 / 2,
    },
    # source: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-us-nvidia-1758950-r4-web.pdf
    # sxm and pcie have same flop counts
    "a100": {
        "fp64": 19.5e12,
        "fp32": 19.5e12,
        "tf32": 156e12,
        "fp16": 312e12,
        "amp_fp16": 312e12,
        "bf16": 312e12,
        "amp_bf16": 312e12,
    },
    # source: https://images.nvidia.com/content/technologies/volta/pdf/volta-v100-datasheet-update-us-1165301-r5.pdf
    "v100-sxm": {
        "fp64": 7.8e12,
        "fp32": 15.7e12,
        "fp16": 125e12,
        "amp_fp16": 125e12,
    },
    "v100-pcie": {
        "fp64": 7e12,
        "fp32": 14e12,
        "fp16": 112e12,
        "amp_fp16": 112e12,
    },
    "v100s-pcie": {
        "fp64": 8.2e12,
        "fp32": 16.4e12,
        "fp16": 130e12,
        "amp_fp16": 130e12,
    },
    # source: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet-951643.pdf
    # sxm and pcie have same flop counts
    "t4": {
        "fp32": 8.1e12,
        "fp16": 65e12,
        "amp_fp16": 65e12,
        "int8": 130e12,
        "int4": 260e12,
    },
}


def get_gpu_flops_available(state: State):
    gpu_flops_available = None

    # Return 0 if no CUDA device (e.g., when running with CPU only)
    if not torch.cuda.is_available():
        return 0

    # torch.cuda.get_device_name() ex output: 'NVIDIA A100-SXM4-40GB'
    dev_name = torch.cuda.get_device_name().lower()
    if "h100-sxm" in dev_name:
        dev_name = "h100-sxm"
    elif "h100-pcie" in dev_name:
        dev_name = "h100-pcie"
    elif "a100" in dev_name:
        dev_name = "a100"
    elif "v100-sxm" in dev_name:
        dev_name = "v100-sxm"
    elif "v100-pcie" in dev_name:
        dev_name = "v100-pcie"
    elif "t4" in dev_name:
        dev_name = "t4"
    else:
        dev_name = None

    if dev_name:
        try:
            gpu_flops_available = int(GPU_AVAILABLE_FLOPS[dev_name][state.precision.value])
        except KeyError:
            gpu_flops_available = None

    if gpu_flops_available is None:
        warnings.warn(
            f"gpu_flop count not found for {dev_name=} with precision: {state.precision.value}; "
            f"MFU cannot be calculated and reported. gpu_flops_available can be manually "
            f"overridden by setting gpu_flops_available in SpeedMonitorMFU()"
        )
        # Setting to 0 will disable MFU computation and prevent
        # the speed monitor from running this helper every batch
        gpu_flops_available = 0

    return gpu_flops_available


class DolmaConsoleLogger(ConsoleLogger):
    def _log_hparams_to_console(self):
        if dist.get_local_rank() == 0:
            log_str = "Config:"
            for name, value in self.hparams.items():
                value_str = format_log_data_value(value)
                log_str += f"\n\t {name}: {value_str}"
            self._log_to_console(log_str)

    def _log_to_console(self, log_str: str):
        echo.info(log_str)


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
