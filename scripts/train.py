"""Run this script with 'torchrun'."""

import logging
import os
import random
import shutil
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from functools import partial
from itertools import islice
from pathlib import Path
from typing import Any, Deque, Dict, Generator, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import wandb
from packaging import version
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
from torch.distributed.fsdp.api import (
    FullOptimStateDictConfig,
    ShardedOptimStateDictConfig,
    ShardedStateDictConfig,
)
from torch.utils.data import DataLoader, DistributedSampler
from torchmetrics import MeanMetric

from olmo.aliases import BatchDict
from olmo.config import (
    DataConfig,
    EvaluatorConfig,
    ModelConfig,
    OptimizerType,
    SchedulerType,
    SpeedMonitorConfig,
    TrainConfig,
)
from olmo.data import DataCollator, MemMapDataset
from olmo.exceptions import OlmoCliError, OlmoConfigurationError
from olmo.model import Olmo
from olmo.optim import DecoupledLionW
from olmo.util import (
    clean_opt,
    global_rank,
    local_rank,
    log_extra_field,
    move_to_device,
    prepare_cli_environment,
    seed_all,
)

log = logging.getLogger("train")


@dataclass
class SpeedMonitor:
    cfg: SpeedMonitorConfig
    start_times: Deque[float] = field(default_factory=lambda: deque([]))
    num_tokens: Deque[int] = field(default_factory=lambda: deque([]))

    def batch_start(self, num_tokens: int) -> None:
        if len(self.start_times) >= self.cfg.window_size:
            self.start_times.popleft()
            self.num_tokens.popleft()
        self.start_times.append(time.monotonic())
        self.num_tokens.append(num_tokens)

    def reset(self) -> None:
        self.start_times.clear()
        self.num_tokens.clear()

    def check(self) -> Dict[str, float]:
        total_seconds = time.monotonic() - self.start_times[0]
        total_tokens = sum(self.num_tokens)
        total_batches = len(self.start_times)
        return {
            "throughput/device/tokens_per_second": total_tokens / total_seconds,
            "throughput/device/batches_per_second": total_batches / total_seconds,
        }


@dataclass
class LRMonitor:
    optim: torch.optim.Optimizer

    def check(self) -> Dict[str, float]:
        lrs = [group["lr"] for group in self.optim.param_groups]
        return {f"optim/learning_rate_group{idx}": lr for idx, lr in enumerate(lrs)}


@dataclass
class Evaluator:
    cfg: EvaluatorConfig
    eval_loader: DataLoader
    eval_batches: Iterator[Tuple[int, BatchDict]]
    eval_loss_metric: MeanMetric

    def reset_metrics(self) -> None:
        self.eval_loss_metric.reset()

    def compute_metrics(self) -> Dict[str, float]:
        loss = self.eval_loss_metric.compute()
        return {
            f"eval/{self.cfg.label}/CrossEntropyLoss": loss.item(),
            f"eval/{self.cfg.label}/Perplexity": torch.exp(loss).item(),
        }

    def update_metrics(self, loss: torch.Tensor) -> None:
        self.eval_loss_metric.update(loss)


@dataclass
class Trainer:
    cfg: TrainConfig
    model: Olmo
    fsdp_model: FSDP
    optim: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    train_loader: DataLoader
    training_batches: Iterator[Tuple[int, Tuple[int, BatchDict]]]
    device: torch.device
    train_loss_metric: MeanMetric
    evaluators: List[Evaluator]
    global_step: int = 0
    global_data_step: int = 0
    checkpoints: List[Path] = field(default_factory=list)
    unsharded_checkpoints: List[Path] = field(default_factory=list)

    def state_dict(self) -> Dict[str, Any]:
        state_dict = self.non_tensor_state_dict()
        state_dict["model"] = self.fsdp_model.state_dict()
        state_dict["optim"] = FSDP.optim_state_dict(self.fsdp_model, self.optim)
        return state_dict

    def non_tensor_state_dict(self) -> Dict[str, Any]:
        return {
            "scheduler": self.scheduler.state_dict(),
            "global_step": self.global_step,  # move forward one batch
            "global_data_step": self.global_data_step,  # move forward one batch
            "checkpoints": self.checkpoints,
            "unsharded_checkpoints": self.unsharded_checkpoints,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.random.get_rng_state(),
                "cuda": torch.cuda.get_rng_state(),
            },
        }

    def save_sharded_checkpoint(self) -> Path:
        checkpoint_dir = Path(self.cfg.save_folder) / f"step{self.global_step}"

        try:
            next(checkpoint_dir.glob("*"))
            if cfg.save_overwrite:
                if global_rank() == 0:
                    shutil.rmtree(checkpoint_dir)
            else:
                raise OlmoConfigurationError(
                    f"Checkpoint for step {self.global_step} already exists, use --save-overwrite to overwrite it"
                )
        except StopIteration:
            pass

        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        dist.barrier()

        # Write the checkpoint.
        with FSDP.state_dict_type(
            self.fsdp_model,
            state_dict_type=StateDictType.SHARDED_STATE_DICT,
            state_dict_config=ShardedStateDictConfig(offload_to_cpu=True),
            optim_state_dict_config=ShardedOptimStateDictConfig(offload_to_cpu=True),
        ):
            # NOTE: Alternatively we could use the checkpointing method in this test
            # https://github.com/pytorch/pytorch/blob/main/test/distributed/checkpoint/test_fsdp_optim_state.py
            # but we've had issues with that on AMD GPUs. See
            # https://github.com/pytorch/pytorch/issues/100041
            #  checkpoint.save_state_dict(self.state_dict(), checkpoint.FileSystemWriter(checkpoint_dir))
            torch.save(self.state_dict(), checkpoint_dir / f"rank{global_rank()}.pt")

        # Link to 'latest'.
        if global_rank() == 0:
            latest_path = Path(self.cfg.save_folder) / "latest"
            latest_path.unlink(missing_ok=True)
            latest_path.symlink_to(checkpoint_dir.name, target_is_directory=True)

        self.checkpoints.append(checkpoint_dir)

        # Remove old checkpoints.
        if self.cfg.save_num_checkpoints_to_keep > 0:
            while len(self.checkpoints) > self.cfg.save_num_checkpoints_to_keep:
                oldest_checkpoint = self.checkpoints.pop(0)
                if global_rank() == 0 and oldest_checkpoint.is_dir():
                    shutil.rmtree(oldest_checkpoint, ignore_errors=True)

        dist.barrier()

        return checkpoint_dir

    def restore_sharded_checkpoint(self, load_path: Path):
        # Zero-gradients to avoid gathering them.
        self.optim.zero_grad(set_to_none=True)

        with FSDP.state_dict_type(
            self.fsdp_model,
            state_dict_type=StateDictType.SHARDED_STATE_DICT,
            state_dict_config=ShardedStateDictConfig(offload_to_cpu=True),
            optim_state_dict_config=ShardedOptimStateDictConfig(offload_to_cpu=True),
        ):
            # NOTE: Alternatively we could use the checkpointing method in this test
            # https://github.com/pytorch/pytorch/blob/main/test/distributed/checkpoint/test_fsdp_optim_state.py
            # but we've had issues with that on AMD GPUs. See
            # https://github.com/pytorch/pytorch/issues/100041
            # But basically it would look like this.
            # Load the serialized state dict in place.
            #  state_dict = self.state_dict()
            #  del state_dict["optim"]  # Can't load optimizer together with the model
            #  checkpoint.load_state_dict(state_dict, checkpoint.FileSystemReader(load_path))
            #  self.fsdp_model.load_state_dict(state_dict["model"])
            # Load other state...
            # Load optim state.
            #  optim_state = load_sharded_optimizer_state_dict(
            #      model_state_dict=state_dict["model"],
            #      optimizer_key="optim",
            #      storage_reader=checkpoint.FileSystemReader(load_path),
            #  )
            #  flattened_osd = FSDP.optim_state_dict_to_load(optim_state["optim"], self.fsdp_model, self.optim)
            #  self.optim.load_state_dict(flattened_osd)

            # Deserialize state dictionary.
            state_dict = torch.load(load_path / f"rank{global_rank()}.pt")

            # Load state.
            self.fsdp_model.load_state_dict(state_dict["model"])
            self.global_step = state_dict["global_step"]
            self.global_data_step = state_dict["global_data_step"]
            self.checkpoints = [
                path
                for path in state_dict["checkpoints"]
                if path.is_dir() and path.resolve().parent == Path(self.cfg.save_folder)
            ]
            self.unsharded_checkpoints = [
                path
                for path in state_dict["unsharded_checkpoints"]
                if path.is_dir() and path.resolve().parent == Path(self.cfg.save_folder)
            ]
            self.scheduler.load_state_dict(state_dict["scheduler"])
            # NOTE: careful, the order of these arguments has changed since the 2.0 release.
            if version.parse(torch.__version__) < version.parse("2.1.0"):
                #  flattened_osd = FSDP.optim_state_dict_to_load(optim_state["optim"], self.fsdp_model, self.optim)  # type: ignore
                flattened_osd = FSDP.optim_state_dict_to_load(state_dict["optim"], self.fsdp_model, self.optim)  # type: ignore
            else:
                #  flattened_osd = FSDP.optim_state_dict_to_load(self.fsdp_model, self.optim, optim_state["optim"])  # type: ignore
                flattened_osd = FSDP.optim_state_dict_to_load(self.fsdp_model, self.optim, state_dict["optim"])  # type: ignore
            self.optim.load_state_dict(flattened_osd)

            rng_state = state_dict.pop("rng")
            del state_dict

        dist.barrier()

        if not self.cfg.restore_dataloader:
            self.global_data_step = 0
        elif self.cfg.fast_forward_batches:
            self.global_data_step += self.cfg.fast_forward_batches

        # Fast-forward data loader.
        if not self.cfg.dry_run:
            self.fast_forward_batches()
            dist.barrier()

        # Set rng state.
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(rng_state["torch"])
        torch.cuda.set_rng_state(rng_state["cuda"])

    def save_unsharded_checkpoint(self) -> Path:
        # Zero-gradients to avoid gathering them.
        self.optim.zero_grad(set_to_none=True)

        checkpoint_dir = Path(self.cfg.save_folder) / f"step{self.global_step}-unsharded"

        try:
            next(checkpoint_dir.glob("*"))
            if cfg.save_overwrite:
                if global_rank() == 0:
                    shutil.rmtree(checkpoint_dir)
            else:
                raise OlmoConfigurationError(
                    f"Unsharded checkpoint for step {self.global_step} already exists, use --save-overwrite to overwrite it"
                )
        except StopIteration:
            pass

        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        dist.barrier()

        # Write the checkpoint.
        with FSDP.state_dict_type(
            self.fsdp_model,
            state_dict_type=StateDictType.FULL_STATE_DICT,
            state_dict_config=FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            optim_state_dict_config=FullOptimStateDictConfig(rank0_only=True, offload_to_cpu=True),
        ):
            # We'll write the model and optimizer state dicts individually to reduce (CPU) memory consumption.
            # First the model state.
            model_state_dict = self.fsdp_model.state_dict()
            if global_rank() == 0:
                torch.save(model_state_dict, checkpoint_dir / "model.pt")
            del model_state_dict

            # Then the optimizer state.
            optim_state_dict = FSDP.optim_state_dict(self.fsdp_model, self.optim)
            if global_rank() == 0:
                torch.save(optim_state_dict, checkpoint_dir / "optim.pt")
            del optim_state_dict

            # Then everything else.
            other_state_dict = self.non_tensor_state_dict()
            if global_rank() == 0:
                torch.save(other_state_dict, checkpoint_dir / "other.pt")

        # Link to 'latest'.
        if global_rank() == 0:
            latest_path = Path(self.cfg.save_folder) / "latest-unsharded"
            latest_path.unlink(missing_ok=True)
            latest_path.symlink_to(checkpoint_dir.name, target_is_directory=True)

        self.unsharded_checkpoints.append(checkpoint_dir)

        # Remove old checkpoints.
        if self.cfg.save_num_unsharded_checkpoints_to_keep > 0:
            while len(self.unsharded_checkpoints) > self.cfg.save_num_unsharded_checkpoints_to_keep:
                oldest_checkpoint = self.unsharded_checkpoints.pop(0)
                if global_rank() == 0 and oldest_checkpoint.is_dir():
                    shutil.rmtree(oldest_checkpoint, ignore_errors=True)

        dist.barrier()

        return checkpoint_dir

    def restore_unsharded_checkpoint(self, load_path: Path):
        # Zero-gradients to avoid gathering them.
        self.optim.zero_grad(set_to_none=True)

        with FSDP.state_dict_type(
            self.fsdp_model,
            state_dict_type=StateDictType.FULL_STATE_DICT,
            state_dict_config=FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            optim_state_dict_config=FullOptimStateDictConfig(rank0_only=True, offload_to_cpu=True),
        ):
            # Load model state.
            self.fsdp_model.load_state_dict(torch.load(load_path / "model.pt"))

            # Load optimizer state.
            optim_state_dict = torch.load(load_path / "optim.pt")
            # NOTE: careful, the order of these arguments has changed since the 2.0 release.
            if version.parse(torch.__version__) < version.parse("2.1.0"):
                #  flattened_osd = FSDP.optim_state_dict_to_load(optim_state["optim"], self.fsdp_model, self.optim)  # type: ignore
                flattened_osd = FSDP.optim_state_dict_to_load(optim_state_dict, self.fsdp_model, self.optim)  # type: ignore
            else:
                #  flattened_osd = FSDP.optim_state_dict_to_load(self.fsdp_model, self.optim, optim_state["optim"])  # type: ignore
                flattened_osd = FSDP.optim_state_dict_to_load(self.fsdp_model, self.optim, optim_state_dict)  # type: ignore
            del optim_state_dict
            self.optim.load_state_dict(flattened_osd)
            del flattened_osd

            # Load other state.
            other_state_dict = torch.load(load_path / "other.pt")
            self.global_step = other_state_dict["global_step"]
            self.global_data_step = other_state_dict["global_data_step"]
            self.checkpoints = [
                path
                for path in other_state_dict["checkpoints"]
                if path.is_dir() and path.resolve().parent == Path(self.cfg.save_folder)
            ]
            self.unsharded_checkpoints = [
                path
                for path in other_state_dict["unsharded_checkpoints"]
                if path.is_dir() and path.resolve().parent == Path(self.cfg.save_folder)
            ]
            self.scheduler.load_state_dict(other_state_dict["scheduler"])

        dist.barrier()

        if not self.cfg.restore_dataloader:
            self.global_data_step = 0
        elif self.cfg.fast_forward_batches:
            self.global_data_step += self.cfg.fast_forward_batches

        # Fast-forward data loader.
        if not self.cfg.dry_run:
            self.fast_forward_batches()
            dist.barrier()

    def fast_forward_batches(self):
        if self.global_data_step > 0:
            if self.global_data_step > self.global_step:
                log.info(
                    f"Fast-forwarding data loader to {self.global_step}+{self.global_data_step-self.global_step}..."
                )
            else:
                log.info(f"Fast-forwarding data loader to {self.global_data_step}...")
            for step, _ in self.training_batches:
                if step + 1 >= self.global_data_step:
                    log.info(f"Fast-forwarded to {self.global_data_step}")
                    break
                elif step + 1 % 1000 == 0:
                    log.info(f"Fast-forwarding... {step + 1}/{self.global_data_step}")

    def restore_checkpoint(self, load_path: Path):
        if load_path.name.endswith("-unsharded"):
            self.restore_unsharded_checkpoint(load_path)
        else:
            self.restore_sharded_checkpoint(load_path)

    def get_labels(self, batch: BatchDict) -> torch.Tensor:
        # Labels are just input IDs shifted to the left (first item is ignored).
        labels, attention_mask = batch["input_ids"], batch.get("attention_mask")
        if attention_mask is not None:
            labels = labels.masked_fill(attention_mask == 0.0, -100)
        return labels[..., 1:].contiguous()

    def model_forward(self, batch: BatchDict) -> torch.Tensor:
        with torch.autocast("cuda", enabled=True, dtype=self.cfg.autocast_precision):
            logits = self.fsdp_model(**batch).logits[..., :-1, :].contiguous()
            labels = self.get_labels(batch)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        return loss

    def train_batch(self, batch: BatchDict) -> torch.Tensor:
        # Split into micro-batches.
        micro_batches = self.split_batch(batch)

        # In case this helps with memory utilization.
        del batch

        batch_loss = torch.tensor(0.0, device=self.device)
        for micro_batch in micro_batches:
            # Run forward pass.
            loss = self.model_forward(micro_batch) / len(micro_batches)

            # In case this helps with memory utilization.
            del micro_batch

            # Check for nan loss.
            if torch.isnan(loss):
                raise ValueError("nan loss encountered")

            # Run backward pass.
            loss.backward()

            # Update overall batch loss.
            batch_loss += loss.detach()

        return batch_loss

    def train_step(self, batch: BatchDict) -> Dict[str, float]:
        # Zero-gradients.
        self.optim.zero_grad(set_to_none=True)

        # Reset metric.
        self.train_loss_metric.reset()

        # Move tensors to the right device.
        batch = move_to_device(batch, self.device)

        # Run forward-backward pass.
        batch_loss = self.train_batch(batch)

        # Clip gradient norms.
        grad_norm: Optional[float] = None
        if self.cfg.max_grad_norm is not None:
            grad_norm = self.fsdp_model.clip_grad_norm_(self.cfg.max_grad_norm).item()

        # Optimizer step.
        self.optim.step()
        self.scheduler.step()

        # Reduce loss across ranks.
        self.train_loss_metric.update(batch_loss)
        batch_loss = self.train_loss_metric.compute()

        metrics = {"train/CrossEntropyLoss": batch_loss.item(), "train/Perplexity": torch.exp(batch_loss).item()}
        if grad_norm is not None:
            metrics["optim/grad_norm"] = grad_norm
        return metrics

    def eval_batch(self, batch: BatchDict) -> torch.Tensor:
        return self.model_forward(batch)

    def eval_step(self, batch: BatchDict, evaluator: Evaluator) -> Dict[str, float]:
        # Move tensors to the right device.
        batch = move_to_device(batch, self.device)

        # Run forward pass.
        with torch.no_grad():  # NOTE: 'torch.inference_mode()' doesn't work with 'torch.compile()'.
            loss = self.eval_batch(batch)

        # Update metrics.
        evaluator.update_metrics(loss)

        return evaluator.compute_metrics()

    def split_batch(self, batch: BatchDict) -> List[BatchDict]:
        batch_size = batch["input_ids"].shape[0]
        if batch_size <= self.cfg.device_train_microbatch_size:
            return [batch]
        else:
            micro_batches = {}
            for key, tensor in batch.items():
                micro_batches[key] = tensor.split(self.cfg.device_train_microbatch_size, dim=0)  # type: ignore
            return [
                {key: tensor[i] for key, tensor in micro_batches.items()}  # type: ignore
                for i in range(len(micro_batches["input_ids"]))
            ]

    def fit(self):
        # Set model to 'train' mode.
        self.fsdp_model.train()

        # Initialize monitors.
        speed_monitor = SpeedMonitor(self.cfg.speed_monitor)
        lr_monitor = LRMonitor(self.optim)

        # Train.
        first_batch: bool = True
        for step, (epoch, batch) in self.training_batches:
            self.global_step += 1
            self.global_data_step = step + 1

            # We start monitoring speed after the first batch since the first
            # batch might be an outlier due to compiling and other initialization overhead.
            if not first_batch:
                num_tokens = batch["input_ids"].shape[0] * batch["input_ids"].shape[1]
                speed_monitor.batch_start(num_tokens)

            # Run train step on batch.
            metrics = self.train_step(batch)

            # Get speed metrics.
            if not first_batch:
                metrics.update(speed_monitor.check())

            # Log metrics to console.
            if self.global_step % self.cfg.console_log_interval == 0:
                log.info(
                    f"[epoch={epoch}, step={self.global_step}/{self.cfg.max_duration}]\n"
                    + "\n".join([f"    {name}={value:.4f}" for name, value in metrics.items()])
                )

            # Maybe save checkpoint.
            if self.global_step % self.cfg.save_interval == 0:
                log.info("Saving checkpoint...")
                checkpoint_path = self.save_sharded_checkpoint()
                log.info(f"Checkpoint saved to {checkpoint_path}")

                # Reset speed monitor so that we don't count the time taken to save checkpoints.
                speed_monitor.reset()

            # Maybe save unsharded model-only checkpoint.
            if (
                self.cfg.save_interval_unsharded is not None
                and self.global_step % self.cfg.save_interval_unsharded == 0
            ):
                log.info("Saving unsharded checkpoint...")
                checkpoint_path = self.save_unsharded_checkpoint()
                log.info(f"Unsharded checkpoint saved to {checkpoint_path}")

                # Reset speed monitor so that we don't count the time taken to save checkpoints.
                speed_monitor.reset()

            # Maybe run evaluations.
            if self.global_step % self.cfg.eval_interval == 0:
                # Zero gradients and set model to 'eval' mode.
                self.optim.zero_grad(set_to_none=True)
                self.fsdp_model.eval()

                for evaluator in self.evaluators:
                    log.info(f"Running evaluation for '{evaluator.cfg.label}'...")

                    # Reset metrics.
                    evaluator.reset_metrics()

                    # Check how many batches to evaluate on.
                    num_eval_batches = evaluator.cfg.subset_num_batches
                    if num_eval_batches <= 0:
                        num_eval_batches = len(evaluator.eval_loader)

                    # Run model over batches.
                    for eval_step, (_, eval_batch) in enumerate(islice(evaluator.eval_batches, num_eval_batches)):
                        eval_metrics = self.eval_step(eval_batch, evaluator)

                        # Log to console.
                        if (eval_step + 1) % self.cfg.console_log_interval == 0:
                            log.info(
                                f"[eval_step={eval_step + 1}/{num_eval_batches}]\n"
                                + "\n".join([f"    {name}={value:.4f}" for name, value in eval_metrics.items()])
                            )

                    # Get final metrics.
                    metrics.update(evaluator.compute_metrics())

                # Reset speed monitor so that we don't count the time taken to run evaluations.
                speed_monitor.reset()

                # Reset model to 'train' mode.
                self.fsdp_model.train()

            # Log metrics to W&B.
            if wandb.run is not None:
                metrics.update(lr_monitor.check())
                wandb.log(metrics, step=self.global_step)

            first_batch = False

        # Save final unsharded model-only checkpoint.
        log.info("Saving final unsharded model checkpoint...")
        checkpoint_path = self.save_unsharded_checkpoint()
        log.info(f"Unsharded checkpoint saved to {checkpoint_path}")

    def close(self) -> None:
        if wandb.run is not None:
            wandb.finish()


def main(cfg: TrainConfig) -> None:
    # Ensure run name set.
    if cfg.run_name is None:
        cfg.run_name = os.environ.get("COMPOSER_RUN_NAME", "train-llm")
    log_extra_field("run_name", cfg.run_name)

    # Initialize process group and set device.
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(f"cuda:{local_rank()}")

    # Fill some configuration options.
    cfg.model.precision = cfg.precision
    cfg.device_train_batch_size = cfg.global_train_batch_size // dist.get_world_size()
    assert cfg.device_train_batch_size is not None  # for mypy
    cfg.device_train_grad_accum = cfg.device_train_batch_size // cfg.device_train_microbatch_size

    # Display and save configuration.
    if global_rank() == 0:
        log.info("Configuration:")
        log.info(cfg)
        if not cfg.dry_run and (cfg.load_path is None or Path(cfg.load_path).parent != Path(cfg.save_folder)):
            # Save config.
            save_path = Path(cfg.save_folder) / "config.yaml"
            if save_path.is_file() and not cfg.save_overwrite:
                raise OlmoConfigurationError(f"{save_path} already exists, use --save_overwrite to overwrite")
            else:
                log.info(f"Saving config to {save_path}")
                save_path.parent.mkdir(exist_ok=True, parents=True)
                cfg.save(save_path)
            del save_path

    # Set seed.
    seed_all(cfg.seed)

    # Maybe start W&B run.
    if cfg.wandb is not None and (global_rank() == 0 or not cfg.wandb.rank_zero_only):
        wandb_dir = Path(cfg.save_folder) / "wandb"
        wandb_dir.mkdir(parents=True, exist_ok=True)
        wandb.init(
            dir=wandb_dir,
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            group=cfg.wandb.group,
            name=cfg.wandb.name,
            tags=cfg.wandb.tags,
            config=cfg.asdict(exclude=["wandb"]),
        )

    dist.barrier()

    # Initialize the model.
    log.info("Initializing model...")
    olmo_model = Olmo(cfg.model)
    if global_rank() == 0:
        log.info(f"Total number of parameters: {olmo_model.num_params():,d}")
        log.info(
            f"Number of non-embedding parameters: {olmo_model.num_params(include_embedding=False):,d}",
        )

    # Wrap the model in FSDP.
    fsdp_model = FSDP(
        olmo_model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(  # equivalent to MosaicML's "PURE"
            param_dtype=cfg.autocast_precision,
            reduce_dtype=cfg.autocast_precision,
            buffer_dtype=cfg.autocast_precision,
        ),
        auto_wrap_policy=olmo_model.fsdp_wrap_fn,
        use_orig_params=cfg.fsdp.use_orig_params,  # needed for compile
        limit_all_gathers=True,
        device_id=local_rank(),
    )

    if cfg.activation_checkpointing:
        # verify we have FSDP activation support ready by importing:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointImpl,
            apply_activation_checkpointing,
            checkpoint_wrapper,
        )

        non_reentrant_wrapper = partial(
            checkpoint_wrapper,
            offload_to_cpu=False,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        apply_activation_checkpointing(
            fsdp_model,
            checkpoint_wrapper_fn=non_reentrant_wrapper,  # type: ignore
            check_fn=olmo_model.activation_checkpointing_fn,  # type: ignore
        )

    # Construct optimizer and learning rate scheduler.
    optim = build_optimizer(cfg, fsdp_model)
    scheduler = build_scheduler(cfg, optim)

    # Construct data loader.
    train_loader = build_dataloader(cfg.data, cfg.model, cfg.device_train_batch_size)
    training_batches = enumerate(islice(cycle_through_epochs(train_loader), cfg.max_duration))

    # Construct evaluators.
    evaluators = []
    for eval_cfg in cfg.evaluators:
        eval_loader = build_dataloader(eval_cfg.data, cfg.model, eval_cfg.device_eval_batch_size)
        evaluator = Evaluator(
            cfg=eval_cfg,
            eval_loader=eval_loader,
            eval_batches=cycle_through_epochs(eval_loader),
            eval_loss_metric=MeanMetric(nan_strategy="error").to(torch.device(cfg.device)),
        )
        evaluators.append(evaluator)

    # Consolidate components into `Trainer` object.
    trainer = Trainer(
        cfg=cfg,
        model=olmo_model,
        fsdp_model=fsdp_model,
        optim=optim,
        scheduler=scheduler,
        train_loader=train_loader,
        training_batches=training_batches,
        device=torch.device(cfg.device),
        train_loss_metric=MeanMetric(nan_strategy="error").to(torch.device(cfg.device)),
        evaluators=evaluators,
    )

    if not cfg.dry_run and cfg.load_path is None:
        # We save a checkpoint up-front to make sure this won't fail (due to disk space or whatever).
        log.info("Saving pre-train checkpoint...")
        checkpoint_path = trainer.save_sharded_checkpoint()
        log.info(f"Checkpoint saved to {checkpoint_path}")

        # And they we verify that we can load it.
        log.info("Attempting to load pre-train checkpoint...")
        trainer.restore_sharded_checkpoint(checkpoint_path)
        log.info("Checkpoint successfully loaded")

    if cfg.load_path is not None:
        log.info(f"Loading checkpoint from {cfg.load_path}...")
        trainer.restore_checkpoint(Path(cfg.load_path))
        log.info("Checkpoint successfully loaded")

    if cfg.force_save_unsharded:
        log.info(f"Saving unsharded checkpoint...")
        checkpoint_path = trainer.save_unsharded_checkpoint()
        log.info(f"Unsharded checkpoint saved to {checkpoint_path}")

    if cfg.compile is not None:
        # NOTE: trying to compile the whole train step results in a compile-time error from within
        # the optimizer. We should investigate this further at some point.
        #  trainer.train_step = torch.compile(trainer.train_step, **cfg.compile.asdict())
        trainer.train_batch = torch.compile(trainer.train_batch, **cfg.compile.asdict())  # type: ignore
        trainer.eval_batch = torch.compile(trainer.eval_batch, **cfg.compile.asdict())  # type: ignore
        # Alternatively, could just do this:
        #  trainer.fsdp_model = torch.compile(trainer.fsdp_model, **cfg.compile.asdict())

    if not cfg.dry_run:
        log.info("Starting training...")
        trainer.fit()
    else:
        log.info("Dry run complete")

    trainer.close()


def build_dataloader(
    data_config: DataConfig, model_config: ModelConfig, batch_size: int, shuffle: bool = True
) -> DataLoader:
    collator = DataCollator(pad_direction=data_config.pad_direction, pad_token_id=model_config.pad_token_id)
    dataset = MemMapDataset(*data_config.paths, chunk_size=model_config.max_sequence_length)
    sampler = DistributedSampler(
        dataset,
        drop_last=True,
        shuffle=shuffle,
        num_replicas=dist.get_world_size(),
        rank=global_rank(),
        seed=cfg.seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=cfg.data.num_workers,
        sampler=sampler,
        pin_memory=cfg.data.pin_memory,
        prefetch_factor=cfg.data.prefetch_factor,
        persistent_workers=cfg.data.persistent_workers,
        timeout=cfg.data.timeout,
    )


def build_optimizer(cfg: TrainConfig, model: nn.Module) -> torch.optim.Optimizer:
    if cfg.optimizer.name == OptimizerType.decoupled_lionw:
        return DecoupledLionW(
            model.parameters(),
            lr=cfg.optimizer.learning_rate,
            betas=cfg.optimizer.betas,
            weight_decay=cfg.optimizer.weight_decay,
        )
    else:
        raise NotImplementedError


def build_scheduler(cfg: TrainConfig, optim: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
    if cfg.scheduler.name == SchedulerType.cosine_with_warmup:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optim, start_factor=cfg.scheduler.alpha_f, end_factor=1.0, total_iters=cfg.scheduler.t_warmup
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim,
            cfg.max_duration - cfg.scheduler.t_warmup,
            eta_min=cfg.optimizer.learning_rate * cfg.scheduler.alpha_f,
        )
        return torch.optim.lr_scheduler.SequentialLR(optim, [warmup, cosine], [cfg.scheduler.t_warmup])
    else:
        raise NotImplementedError


def cycle_through_epochs(dataloader: DataLoader) -> Generator[Tuple[int, BatchDict], None, None]:
    epoch = 0
    while True:
        if isinstance(dataloader.sampler, DistributedSampler):
            dataloader.sampler.set_epoch(epoch)
        for batch in dataloader:
            yield epoch, batch
        epoch += 1


if __name__ == "__main__":
    prepare_cli_environment()

    try:
        yaml_path, args_list = sys.argv[1], sys.argv[2:]
    except IndexError:
        raise OlmoCliError(f"Usage: {sys.argv[0]} [CONFIG_PATH] [OPTIONS]")

    cfg = TrainConfig.load(yaml_path, [clean_opt(s) for s in args_list])
    main(cfg)
