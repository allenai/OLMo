import logging
import math
import os
import shutil
import warnings
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Set, Tuple, TypedDict, Union

import torch
import torch.distributed.checkpoint as checkpoint
import torch.nn as nn
import torch.nn.functional as F
from composer.callbacks import CheckpointSaver
from composer.core import Event, State, Time
from composer.loggers import ConsoleLogger, Logger
from composer.loggers.logger import format_log_data_value
from composer.models import ComposerModel
from composer.trainer import Trainer
from composer.utils import dist, reproducibility
from torch.distributed.checkpoint.optimizer import load_sharded_optimizer_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader
from torchmetrics import Metric

from .aliases import BatchDict
from .config import (
    ModelConfig,
    OptimizerType,
    SchedulerConfig,
    SchedulerType,
    TrainConfig,
)
from .data import DataCollator, MemMapDataset
from .exceptions import OlmoConfigurationError
from .model import LayerNormBase, Olmo
from .optim import DecoupledLionW

log = logging.getLogger(__name__)

__all__ = [
    "TrainBatchPerplexity",
    "ComposerOlmoLM",
    "OlmoConsoleLogger",
    "OlmoCheckpointer",
    "build_dataloader",
    "build_optimizer",
    "build_scheduler",
    "build_algorithm",
]


class TrainBatchOutput(TypedDict, total=True):
    logits: torch.Tensor
    """
    The (shifted) logits.
    """

    labels: torch.Tensor
    """
    The (shifted) label token IDs.
    """

    loss: torch.Tensor
    """
    The cross-entropy loss.
    """


class TrainBatchPerplexity(Metric):
    """
    A metric for tracking training perplexity on a per-batch basis.
    We use this as a training metric instead of composer's built-in
    :class:`LanguageCrossEntropy` to avoid recomputing the loss.
    """

    def __init__(self) -> None:
        super().__init__(sync_on_compute=False)
        self.loss: Optional[torch.Tensor]

    def update(self, loss: torch.Tensor):
        self.loss = loss

    def compute(self) -> torch.Tensor:
        assert self.loss is not None
        return torch.exp(self.loss)


class ComposerOlmoLM(ComposerModel):
    def __init__(self, model_or_config: Union[Olmo, ModelConfig]):
        super().__init__()
        self.model = Olmo(model_or_config) if isinstance(model_or_config, ModelConfig) else model_or_config
        self.config = self.model.config
        self.num_fwd_flops = self.model.num_fwd_flops

        from composer.metrics.nlp import LanguageCrossEntropy, LanguagePerplexity

        self.train_metrics: Dict[str, Metric] = {
            "Perplexity": TrainBatchPerplexity(),
        }
        self.eval_metrics: Dict[str, Metric] = {
            "Perplexity": LanguagePerplexity(),
            "CrossEntropy": LanguageCrossEntropy(),
        }

    def get_labels(self, batch: BatchDict) -> torch.Tensor:
        # Labels are just input IDs shifted to the left (first item is ignored).
        labels, attention_mask = batch["input_ids"], batch.get("attention_mask")
        if attention_mask is not None:
            labels = labels.masked_fill(attention_mask == 0.0, -100)
        return labels[..., 1:].contiguous()

    def forward(self, batch: BatchDict) -> TrainBatchOutput:
        logits = self.model(**batch).logits[..., :-1, :].contiguous()
        labels = self.get_labels(batch)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        return {"logits": logits, "labels": labels, "loss": loss}

    def loss(self, outputs: TrainBatchOutput, batch: BatchDict) -> torch.Tensor:
        del batch
        return outputs["loss"]

    def eval_forward(self, batch: BatchDict, outputs: Optional[TrainBatchOutput] = None) -> TrainBatchOutput:
        return outputs if outputs is not None else self.forward(batch)

    def get_metrics(self, is_train: bool = False) -> Dict[str, Metric]:
        return self.train_metrics if is_train else self.eval_metrics

    def update_metric(self, batch: BatchDict, outputs: TrainBatchOutput, metric: Metric) -> None:
        del batch
        if isinstance(metric, TrainBatchPerplexity):
            metric.update(outputs["loss"].detach())
        else:
            logits, labels = outputs["logits"], outputs["labels"]
            metric.update(logits.view(-1, logits.size(-1)), labels.view(-1))

    def flops_per_batch(self, batch: BatchDict):
        # Note: this computation does not take into account padding, and assumes
        # that the dataset has been constructed without padding. Additionally, we
        # assume the backward pass is approximately 2x the forward pass
        return self.num_fwd_flops * 3 * batch["input_ids"].shape[0]


class OlmoConsoleLogger(ConsoleLogger):
    metrics_to_log: Set[str] = {"loss/train/total", "trainer/global_step", "metrics/*"}

    def log_metrics(self, metrics: dict[str, float], step: Optional[int] = None) -> None:
        del step
        # Lazy logging of metrics.
        # Stores all metrics logged until they are cleared with a log_to_console call
        self.logged_metrics.update(
            {k: v for k, v in metrics.items() if any(fnmatch(k, pattern) for pattern in self.metrics_to_log)}
        )

    def _log_hparams_to_console(self):
        if dist.get_local_rank() == 0:
            log_str = "Config:"
            for name, value in self.hparams.items():
                value_str = format_log_data_value(value)
                log_str += f"\n\t {name}: {value_str}"
            self._log_to_console(log_str)

    def _log_to_console(self, log_str: str):
        log.info(log_str)


class OlmoCheckpointer(CheckpointSaver):
    """
    We override the default `CheckpointSaver` to make use of torch's new `distributed.checkpoint` functions,
    and because the current checkpointing mechanism in composer doesn't work with Torch 2 + FSDP.
    """

    def __init__(
        self,
        *,
        folder: str = "{run_name}/checkpoints",
        save_interval: Union[Time, str, int, Callable[[State, Event], bool]] = "1ep",
        overwrite: bool = False,
        num_checkpoints_to_keep: int = -1,
    ):
        super().__init__(
            folder=folder,
            filename="ep{epoch}-ba{batch}",
            remote_file_name=None,
            latest_filename="latest",
            latest_remote_file_name=None,
            save_interval=save_interval,
            overwrite=overwrite,
            num_checkpoints_to_keep=num_checkpoints_to_keep,
        )

    @property
    def is_rank0(self) -> bool:
        return not dist.is_initialized() or dist.get_global_rank() == 0

    def _save_checkpoint(self, state: State, logger: Logger):
        del logger
        self.last_checkpoint_batch = state.timestamp.batch

        dirname = Path(self.filename.format(state))
        dirname.mkdir(parents=True, exist_ok=True)

        # Save state dict.
        state_dict = self.get_state_dict(state)
        state_dict["optimizers"] = state_dict["state"].pop("optimizers")  # move optimizer state to top level
        checkpoint.save_state_dict(state_dict, checkpoint.FileSystemWriter(dirname))

        if dist.is_initialized():
            dist.barrier()

        if self.latest_filename is not None and self.is_rank0:
            symlink = self.latest_filename.format(state)
            try:
                os.remove(symlink)
            except FileNotFoundError:
                pass
            os.symlink(dirname.name, symlink)

        self.saved_checkpoints.append(str(dirname))

        if self.num_checkpoints_to_keep >= 0 and self.is_rank0:
            self._rotate_checkpoints()

    def _rotate_checkpoints(self):
        while len(self.saved_checkpoints) > self.num_checkpoints_to_keep:
            checkpoint = self.saved_checkpoints.pop(0)
            shutil.rmtree(checkpoint)

    def restore_checkpoint(self, load_path: str, trainer: Trainer):
        """
        This is a function we added to reproduce the behavior of passing ``load_path``
        to the :class:`Trainer`.
        """
        state = trainer.state

        # `torch.distributed.checkpoint` modifies `state_dict` in-place.
        state_dict = self.get_state_dict(state)
        del state_dict["optimizers"]  # have to load the optimizer separately
        checkpoint.load_state_dict(state_dict, checkpoint.FileSystemReader(load_path))

        # still need to call `load_state_dict` though.
        state.load_state_dict(state_dict["state"], trainer.logger)

        # Load optimizer state.
        optim_state = load_sharded_optimizer_state_dict(
            model_state_dict=state_dict["state"]["model"],
            optimizer_key="optimizers",
            storage_reader=checkpoint.FileSystemReader(load_path),
        )

        # NOTE: careful, the order of these arguments has changed since the 2.0 release.
        flattened_osd = FSDP.optim_state_dict_to_load(optim_state["optimizers"], state.model, state.optimizers[0])
        state.load_state_dict(flattened_osd, trainer.logger)

        if dist.is_initialized():
            dist.barrier()

        # See the `Trainer.__init__`.
        trainer._rng_state = state_dict["rng"]
        reproducibility.seed_all(state.seed)


def build_dataloader(config: TrainConfig, batch_size: int) -> DataLoader:
    from composer.utils.dist import get_sampler

    collator = DataCollator.from_train_config(config)
    dataset = MemMapDataset.from_train_config(config)
    sampler = get_sampler(dataset, shuffle=True, drop_last=config.data.drop_last)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=config.data.num_workers,
        sampler=sampler,
        pin_memory=config.data.pin_memory,
        prefetch_factor=config.data.prefetch_factor,
        persistent_workers=config.data.persistent_workers,
        timeout=config.data.timeout,
    )


def build_optimizer(
    model,
    name: OptimizerType = OptimizerType.decoupled_lionw,
    learning_rate: Optional[float] = None,
    weight_decay: float = 0.0,
    betas: Tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
) -> torch.optim.Optimizer:
    """
    Get a suitable optimizer for training/fine-tuning.

    :param learning_rate: The learning rate. If not specified, a default learning
        rate will calculated according to the equation from the Scaling Laws paper
        `0.003239 - 0.0001395 * math.log(N)`,
        where `N` is the number of trainable parameters excluding embeddings.
    :param weight_decay: The weight decay coefficient. This does not apply to
        biases and layernorm/embedding weights, which will have a weight decay
        coefficient of 0.
    :param kwargs: Other keyword arguments passed to the optimizer.
    """
    # Separate out all parameters to those that will and won't experience regularizing weight decay.
    decay = set()
    no_decay = set()
    all_params = {}
    num_trainable_non_embedding_weights = 0
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            # NOTE: because named_modules and named_parameters are recursive
            # we will see the same tensors p many many times, but doing it this way
            # allows us to know which parent module any tensor p belongs to...
            if not p.requires_grad:
                continue

            fpn = f"{mn}.{pn}" if mn else pn
            all_params[fpn] = p

            if pn.endswith("bias"):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, nn.Linear):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith("weight") and isinstance(m, (LayerNormBase, nn.LayerNorm, nn.Embedding)):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)

            if fpn not in {"transformer.wte.weight", "transformer.wpe.weight"}:
                num_trainable_non_embedding_weights += p.numel()

    # Validate that we've considered every parameter
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, f"parameters {inter_params} made it into both decay/no_decay sets!"
    assert (
        len(all_params.keys() - union_params) == 0
    ), f"parameters {all_params.keys() - union_params} were not separated into either decay/no_decay set!"

    # Create the pytorch optimizer groups.
    optim_groups = [
        {"params": [all_params[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [all_params[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]

    if learning_rate is None:
        learning_rate = 0.003239 - 0.0001395 * math.log(num_trainable_non_embedding_weights)

    if name == OptimizerType.decoupled_lionw:
        return DecoupledLionW(optim_groups, lr=learning_rate, betas=betas)
    elif name == OptimizerType.decoupled_adamw:
        from composer.optim import DecoupledAdamW

        return DecoupledAdamW(optim_groups, lr=learning_rate, betas=betas, eps=eps)
    elif name == OptimizerType.adamw:
        return torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, eps=eps)
    else:
        raise NotImplementedError(f"Not sure how to build optimizer '{name}'")


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
    else:
        raise NotImplementedError(f"Not sure how to build algorithm '{name}'")


def calculate_batch_size_info(
    global_batch_size: int, device_microbatch_size: Union[int, str]
) -> Tuple[int, Union[str, int], Union[str, int]]:
    if global_batch_size % dist.get_world_size() != 0:
        raise OlmoConfigurationError(
            f"Global batch size {global_batch_size} is not divisible by {dist.get_world_size()} "
            "as a result, the batch size would be truncated, please adjust `global_batch_size` "
            f"to be divisible by world size, {dist.get_world_size()}."
        )
    device_batch_size = global_batch_size // dist.get_world_size()
    if device_microbatch_size == "auto":
        device_grad_accum = "auto"
    elif isinstance(device_microbatch_size, int):
        if device_microbatch_size > device_batch_size:
            warnings.warn(
                f"device_microbatch_size > device_batch_size, "
                f"will be reduced from {device_microbatch_size} -> {device_batch_size}.",
                UserWarning,
            )
            device_microbatch_size = device_batch_size
        device_grad_accum = math.ceil(device_batch_size / device_microbatch_size)
    else:
        raise OlmoConfigurationError(f"Not sure how to parse {device_microbatch_size=}")

    return device_batch_size, device_microbatch_size, device_grad_accum


# Coming soon: this conversion math will be done inside Composer Trainer
def update_batch_size_info(cfg: TrainConfig):
    device_train_batch_size, device_train_microbatch_size, device_train_grad_accum = calculate_batch_size_info(
        cfg.global_train_batch_size, cfg.device_train_microbatch_size
    )
    cfg.n_gpus = dist.get_world_size()
    cfg.device_train_batch_size = device_train_batch_size
    cfg.device_train_microbatch_size = device_train_microbatch_size
    cfg.device_train_grad_accum = device_train_grad_accum
    # Safely set `device_eval_batch_size` if not provided by user
    if cfg.device_eval_batch_size is None:
        if cfg.device_train_microbatch_size == "auto":
            cfg.device_eval_batch_size = 1  # TODO debug auto eval microbatching
        elif isinstance(cfg.device_train_microbatch_size, int):
            cfg.device_eval_batch_size = cfg.device_train_microbatch_size
        else:
            raise OlmoConfigurationError(
                f"Not sure how to parse device_train_microbatch_size={cfg.device_train_microbatch_size}"
            )
    return cfg
