import warnings
from collections import deque
from typing import Any, Deque, Dict, Optional, Union

import torch
import torch.nn.functional as F
from composer.core import Callback, State
from composer.loggers import ConsoleLogger, Logger
from composer.loggers.logger import format_log_data_value
from composer.models import ComposerModel
from composer.utils import dist
from torch.utils.data import DataLoader
from torchmetrics import Metric

from .aliases import BatchDict
from .config import ModelConfig
from .model import DolmaGPT, DolmaGPTOutput
from .util import echo

__all__ = ["ComposerDolmaGPT", "SpeedMonitorMFU", "DolmaConsoleLogger"]


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


class SpeedMonitorMFU(Callback):
    """Logs the training throughput and MFU."""

    def __init__(self, window_size: int = 100, gpu_flops_available: Optional[Union[float, int]] = None):
        # Track the batch num samples and wct to compute throughput over a window of batches
        self.history_samples: Deque[float] = deque(maxlen=window_size + 1)
        self.history_wct: Deque[float] = deque(maxlen=window_size + 1)

        self.set_gpu_flops_available = False
        self.gpu_flops_available = gpu_flops_available

        # Keep track of time spent evaluating
        self.total_eval_wct = 0.0

    def state_dict(self) -> Dict[str, Any]:
        return {
            "total_eval_wct": self.total_eval_wct,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.total_eval_wct = state["total_eval_wct"]

    def init(self, state: State, logger: Logger) -> None:
        del logger  # unused
        # Get available GPU FLOPS
        if self.gpu_flops_available is None:
            self.gpu_flops_available = get_gpu_flops_available(state)

    def batch_end(self, state: State, logger: Logger):
        # Add the new element
        self.history_wct.append(state.timestamp.total_wct.total_seconds())
        self.history_samples.append(int(state.timestamp.sample))

        # Log the throughput
        if len(self.history_wct) == self.history_wct.maxlen:
            world_size = dist.get_world_size()
            elapsed_batches = len(self.history_samples) - 1
            elapsed_samples = self.history_samples[-1] - self.history_samples[0]
            elapsed_wct = self.history_wct[-1] - self.history_wct[0]
            batches_per_sec = elapsed_batches / elapsed_wct
            samples_per_sec = elapsed_samples / elapsed_wct
            dev_batches_per_sec = batches_per_sec / world_size
            dev_samples_per_sec = samples_per_sec / world_size
            logger.log_metrics({"throughput/batches_per_sec": batches_per_sec})
            logger.log_metrics({"throughput/samples_per_sec": samples_per_sec})
            logger.log_metrics({"throughput/device/batches_per_sec": dev_batches_per_sec})
            logger.log_metrics({"throughput/device/samples_per_sec": dev_samples_per_sec})

            if isinstance(state.dataloader, DataLoader) and hasattr(state.dataloader.dataset, "chunk_size"):
                max_seq_len = state.dataloader.dataset.chunk_size  # type: ignore
                # only applicable to seq data / models
                logger.log_metrics({"throughput/tokens_per_sec": samples_per_sec * max_seq_len})
                logger.log_metrics({"throughput/device/tokens_per_sec": dev_samples_per_sec * max_seq_len})

            composer_model = state.model
            if not isinstance(composer_model, ComposerModel):
                composer_model = composer_model.module  # Handles DDP case until Composer fixes this
            if hasattr(composer_model, "num_fwd_flops"):
                assert isinstance(composer_model.num_fwd_flops, (int, float))
                # Assume that training flops is 3x fwd flops (1 fwd, 2 bkw)
                flops_per_sec = 3 * composer_model.num_fwd_flops * samples_per_sec
                logger.log_metrics({"throughput/flops_per_sec": flops_per_sec})
                dev_flops_per_sec = flops_per_sec / world_size
                logger.log_metrics({"throughput/device/flops_per_sec": dev_flops_per_sec})
                if self.gpu_flops_available:
                    mfu = dev_flops_per_sec / self.gpu_flops_available
                    logger.log_metrics({"throughput/device/mfu": mfu})

        # Log the time
        # `state.timestamp` excludes any time spent in evaluation
        logger.log_metrics(
            {
                "wall_clock/train": state.timestamp.total_wct.total_seconds(),
                "wall_clock/val": self.total_eval_wct,
                "wall_clock/total": state.timestamp.total_wct.total_seconds() + self.total_eval_wct,
            }
        )

    def eval_end(self, state: State, logger: Logger):
        del logger  # unused
        self.total_eval_wct += state.eval_timestamp.total_wct.total_seconds()


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
