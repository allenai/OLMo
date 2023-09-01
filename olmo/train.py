from __future__ import annotations

import logging
import math
import os
import random
import shutil
import time
from collections import deque
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, TextIO, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from packaging import version
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.api import (
    FullOptimStateDictConfig,
    ShardedOptimStateDictConfig,
    ShardedStateDictConfig,
)
from torch.utils.data import DataLoader
from torchmetrics import MeanMetric

from .aliases import PathOrStr
from .config import CheckpointType, SpeedMonitorConfig, TrainConfig
from .data import IterableDataset
from .eval import Evaluator
from .exceptions import OlmoConfigurationError
from .model import Olmo
from .optim import Optimizer, Scheduler
from .util import (
    barrier,
    fsdp_clip_grads_and_get_norms,
    get_fs_local_rank,
    get_global_rank,
    get_world_size,
    move_to_device,
    peak_gpu_memory,
    resource_path,
    syncronize_flag,
    upload,
    wait_on,
)

__all__ = ["SpeedMonitor", "LRMonitor", "Trainer"]

log = logging.getLogger(__name__)


@dataclass
class SpeedMonitor:
    cfg: SpeedMonitorConfig
    start_times: Deque[float] = field(default_factory=lambda: deque([]))
    global_total_tokens: int = 0
    device_interval_tokens: Deque[int] = field(default_factory=lambda: deque([]))

    def batch_start(self, global_total_tokens: int, device_batch_num_tokens: int, record: bool = True) -> None:
        self.global_total_tokens = global_total_tokens
        if record:
            if len(self.start_times) >= self.cfg.window_size:
                self.start_times.popleft()
                self.device_interval_tokens.popleft()
            self.start_times.append(time.monotonic())
            self.device_interval_tokens.append(device_batch_num_tokens)

    def reset(self) -> None:
        self.start_times.clear()
        self.device_interval_tokens.clear()

    def check(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {"throughput/total_tokens": self.global_total_tokens}
        if self.start_times:
            interval_seconds = time.monotonic() - self.start_times[0]
            interval_batches = len(self.start_times)
            interval_tokens = sum(self.device_interval_tokens)
            metrics["throughput/device/tokens_per_second"] = interval_tokens / interval_seconds
            metrics["throughput/device/batches_per_second"] = interval_batches / interval_seconds
        return metrics


@dataclass
class LRMonitor:
    optim: torch.optim.Optimizer

    def check(self) -> Dict[str, float]:
        lrs = [group["lr"] for group in self.optim.param_groups]
        return {f"optim/learning_rate_group{idx}": lr for idx, lr in enumerate(lrs)}


@dataclass
class Trainer:
    cfg: TrainConfig
    model: Olmo
    fsdp_model: FSDP
    optim: Optimizer
    scheduler: Scheduler
    device: torch.device
    ce_train_loss_metric: MeanMetric
    z_train_loss_metric: Optional[MeanMetric] = None
    global_step: int = 0
    global_data_step: int = 0
    """This is now redundant since adding 'global_train_examples_seen'."""
    global_train_examples_seen: int = 0
    """Tracks the global number of training examples seen for the purpose of restoring the dataset
    position on restarts."""
    global_train_tokens_seen: int = 0
    """Tracks the global total number of tokens trained on."""
    checkpoints: List[Path] = field(default_factory=list)
    unsharded_checkpoints: List[Path] = field(default_factory=list)
    min_train_loss: float = float("inf")
    indices_file: Optional[TextIO] = None
    _start_time: float = 0.0

    def state_dict(self) -> Dict[str, Any]:
        state_dict = self.non_tensor_state_dict()
        state_dict["model"] = self.fsdp_model.state_dict()
        state_dict["optim"] = FSDP.optim_state_dict(self.fsdp_model, self.optim)
        return state_dict

    def non_tensor_state_dict(self) -> Dict[str, Any]:
        return {
            "global_step": self.global_step,
            "global_data_step": self.global_data_step,
            "global_train_examples_seen": self.global_train_examples_seen,
            "global_train_tokens_seen": self.global_train_tokens_seen,
            "checkpoints": self.checkpoints,
            "unsharded_checkpoints": self.unsharded_checkpoints,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.random.get_rng_state(),
                "cuda": torch.cuda.get_rng_state(),
            },
        }

    def load_non_tensor_state_dict(self, state_dict: Dict[str, Any]) -> None:
        # Checkpoint paths.
        self.checkpoints = [
            path
            for path in state_dict["checkpoints"]
            if path.is_dir() and path.resolve().parent == Path(self.cfg.save_folder).resolve()
        ]
        self.unsharded_checkpoints = [
            path
            for path in state_dict["unsharded_checkpoints"]
            if path.is_dir() and path.resolve().parent == Path(self.cfg.save_folder).resolve()
        ]

        # Dataset / dataloader position.
        self.global_step = state_dict["global_step"]
        self.global_data_step = state_dict["global_data_step"]
        self.global_train_examples_seen = state_dict.get(  # newer addition
            "global_train_examples_seen", self.global_data_step * self.cfg.global_train_batch_size
        )
        self.global_train_tokens_seen = state_dict.get(  # newer addition
            "global_train_tokens_seen",
            self.global_data_step * self.cfg.global_train_batch_size * self.cfg.model.max_sequence_length,
        )
        if not self.cfg.restore_dataloader:
            self.global_data_step = 0
            self.global_train_examples_seen = 0
            self.global_train_tokens_seen = 0
        elif self.cfg.fast_forward_batches:
            self.global_data_step += self.cfg.fast_forward_batches
            # Technically we don't "see" these batches that we fast-forward through, but we use
            # this variable to update the position of the dataset so we need to include them here.
            self.global_train_examples_seen += self.cfg.fast_forward_batches * self.cfg.global_train_batch_size
            # NOTE: on the other hand we don't add anything to 'self.global_train_tokens_seen' here because
            # that variable is meant to track the actual number of tokens trained on.

        # Reset learning rate and weight decay to the values from the config, not the checkpoint.
        new_learning_rate = self.scheduler.get_lr(
            self.cfg.optimizer.learning_rate, self.global_step, self.cfg.max_duration
        )
        for group in self.optim.param_groups:
            group["lr"] = new_learning_rate
            group["initial_lr"] = self.cfg.optimizer.learning_rate
            if "weight_decay" in group and group["weight_decay"] > 0.0:
                group["weight_decay"] = self.cfg.optimizer.weight_decay

        # RNG states.
        if "rng" in state_dict:
            rng_state = state_dict["rng"]
            self.restore_rng_state(rng_state)

    def restore_rng_state(self, rng_state: Dict[str, Any]) -> None:
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(rng_state["torch"])
        torch.cuda.set_rng_state(rng_state["cuda"])

    def save_sharded_checkpoint(self) -> Path:
        # Zero-gradients to avoid gathering them.
        self.optim.zero_grad(set_to_none=True)

        checkpoint_dir = Path(self.cfg.save_folder) / f"step{self.global_step}"
        checkpoint_dir_tmp = Path(self.cfg.save_folder) / f"step{self.global_step}-tmp"

        try:
            next(checkpoint_dir.glob("*"))
            if self.cfg.save_overwrite:
                if get_fs_local_rank() == 0:
                    shutil.rmtree(checkpoint_dir, ignore_errors=True)
            else:
                raise OlmoConfigurationError(
                    f"Checkpoint for step {self.global_step} already exists, use --save-overwrite to overwrite it"
                )
        except StopIteration:
            pass

        if get_fs_local_rank() == 0:
            checkpoint_dir_tmp.mkdir(parents=True, exist_ok=True)

        self.checkpoints.append(checkpoint_dir)
        barrier()

        # Flush data indices file.
        if self.indices_file is not None:
            self.indices_file.flush()

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
            torch.save(self.state_dict(), checkpoint_dir_tmp / f"rank{get_global_rank()}.pt")
            # Save config too.
            if get_global_rank() == 0:
                self.cfg.save(checkpoint_dir_tmp / "config.yaml")
            barrier()

        if get_fs_local_rank() == 0:
            # Replace temp directory with target checkpoint directory.
            checkpoint_dir_tmp.replace(checkpoint_dir)

            # Link to 'latest'.
            latest_path = Path(self.cfg.save_folder) / "latest"
            latest_path.unlink(missing_ok=True)
            latest_path.symlink_to(checkpoint_dir.name, target_is_directory=True)

        # In the cases where we're using a shared NFS drive between ranks to save checkpoints,
        # replacing the temp directory with the final directory from rank 0 might not be immediately
        # realized in the file systems of the other ranks.
        # So we wait here across all ranks until that final checkpoint directory is visible.
        wait_on(lambda: checkpoint_dir.exists(), "Waiting for checkpoint directory", timeout=10.0)

        # Remove old checkpoints.
        if self.cfg.save_num_checkpoints_to_keep > 0:
            while len(self.checkpoints) > self.cfg.save_num_checkpoints_to_keep:
                self.remove_sharded_checkpoint(0)

        barrier()

        # Upload checkpoint to bucket.
        if self.cfg.remote_save_folder is not None:
            files_to_upload = [f"rank{get_global_rank()}.pt"]
            if get_global_rank() == 0:
                files_to_upload.append("config.yaml")
            for fname in files_to_upload:
                source = checkpoint_dir / fname
                target = f"{self.cfg.remote_save_folder}/{checkpoint_dir.name}/{fname}"
                log.info(f"Uploading {source} to {target}...")
                upload(source, target, save_overwrite=self.cfg.save_overwrite)
            barrier()

        return checkpoint_dir

    def remove_sharded_checkpoint(self, idx: int = 0):
        oldest_checkpoint = self.checkpoints.pop(idx)
        barrier()
        if get_fs_local_rank() == 0 and oldest_checkpoint.is_dir():
            shutil.rmtree(oldest_checkpoint, ignore_errors=True)
            latest_path = Path(self.cfg.save_folder) / "latest"
            if latest_path.resolve() == oldest_checkpoint.resolve():
                latest_path.unlink()
        barrier()

    def restore_sharded_checkpoint(self, load_path: PathOrStr):
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
            state_dict = torch.load(resource_path(load_path, f"rank{get_global_rank()}.pt"))

            # Load model and optimizer state.
            log.info("Loading model state...")
            self.fsdp_model.load_state_dict(state_dict["model"])
            log.info("Loading optimizer state...")
            # NOTE: careful, the order of these arguments has changed since the 2.0 release.
            if version.parse(torch.__version__) < version.parse("2.1.0"):
                #  flattened_osd = FSDP.optim_state_dict_to_load(optim_state["optim"], self.fsdp_model, self.optim)  # type: ignore
                flattened_osd = FSDP.optim_state_dict_to_load(state_dict["optim"], self.fsdp_model, self.optim)  # type: ignore
            else:
                #  flattened_osd = FSDP.optim_state_dict_to_load(self.fsdp_model, self.optim, optim_state["optim"])  # type: ignore
                flattened_osd = FSDP.optim_state_dict_to_load(self.fsdp_model, self.optim, state_dict["optim"])  # type: ignore
            self.optim.load_state_dict(flattened_osd)

            # Load non-tensor state.
            self.load_non_tensor_state_dict(state_dict)

            del state_dict, flattened_osd

        barrier()

    def save_unsharded_checkpoint(self) -> Path:
        # Zero-gradients to avoid gathering them.
        self.optim.zero_grad(set_to_none=True)

        checkpoint_dir = Path(self.cfg.save_folder) / f"step{self.global_step}-unsharded"
        checkpoint_dir_tmp = Path(self.cfg.save_folder) / f"step{self.global_step}-unsharded-tmp"

        try:
            next(checkpoint_dir.glob("*"))
            if self.cfg.save_overwrite:
                if get_global_rank() == 0:
                    shutil.rmtree(checkpoint_dir)
            else:
                raise OlmoConfigurationError(
                    f"Unsharded checkpoint for step {self.global_step} already exists, use --save-overwrite to overwrite it"
                )
        except StopIteration:
            pass

        if get_global_rank() == 0:
            checkpoint_dir_tmp.mkdir(parents=True, exist_ok=True)

        self.unsharded_checkpoints.append(checkpoint_dir)
        barrier()

        # Flush data indices file.
        if self.indices_file is not None:
            self.indices_file.flush()

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
            if get_global_rank() == 0:
                torch.save(model_state_dict, checkpoint_dir_tmp / "model.pt")
            del model_state_dict

            # Then the optimizer state.
            optim_state_dict = FSDP.optim_state_dict(self.fsdp_model, self.optim)
            if get_global_rank() == 0:
                torch.save(optim_state_dict, checkpoint_dir_tmp / "optim.pt")
            del optim_state_dict

            # Then everything else.
            other_state_dict = self.non_tensor_state_dict()
            if get_global_rank() == 0:
                torch.save(other_state_dict, checkpoint_dir_tmp / "other.pt")
                self.cfg.save(checkpoint_dir_tmp / "config.yaml")
            barrier()

        if get_global_rank() == 0:
            # Replace temp directory with target checkpoint directory.
            checkpoint_dir_tmp.replace(checkpoint_dir)

            # Link to 'latest'.
            latest_path = Path(self.cfg.save_folder) / "latest-unsharded"
            latest_path.unlink(missing_ok=True)
            latest_path.symlink_to(checkpoint_dir.name, target_is_directory=True)

        # In the cases where we're using a shared NFS drive between ranks to save checkpoints,
        # replacing the temp directory with the final directory from rank 0 might not be immediately
        # realized in the file systems of the other ranks.
        # So we wait here across all ranks until that final checkpoint directory is visible.
        wait_on(lambda: checkpoint_dir.exists(), "Waiting for checkpoint directory", timeout=10.0)

        # Remove old checkpoints.
        if self.cfg.save_num_unsharded_checkpoints_to_keep > 0:
            while len(self.unsharded_checkpoints) > self.cfg.save_num_unsharded_checkpoints_to_keep:
                self.remove_unsharded_checkpoint(0)

        barrier()

        # Upload checkpoint to bucket.
        if self.cfg.remote_save_folder is not None:
            if get_global_rank() == 0:
                for fname in ["config.yaml", "model.pt", "optim.pt", "other.pt"]:
                    source = checkpoint_dir / fname
                    target = f"{self.cfg.remote_save_folder}/{checkpoint_dir.name}/{fname}"
                    log.info(f"Uploading {source} to {target}...")
                    upload(source, target, save_overwrite=self.cfg.save_overwrite)
            barrier()

        return checkpoint_dir

    def remove_unsharded_checkpoint(self, idx: int = 0):
        barrier()
        oldest_checkpoint = self.unsharded_checkpoints.pop(idx)
        if get_global_rank() == 0 and oldest_checkpoint.is_dir():
            shutil.rmtree(oldest_checkpoint, ignore_errors=True)
            latest_path = Path(self.cfg.save_folder) / "latest-unsharded"
            if latest_path.resolve() == oldest_checkpoint.resolve():
                latest_path.unlink()
        barrier()

    def restore_unsharded_checkpoint(self, load_path: PathOrStr):
        # Zero-gradients to avoid gathering them.
        self.optim.zero_grad(set_to_none=True)

        with FSDP.state_dict_type(
            self.fsdp_model,
            state_dict_type=StateDictType.FULL_STATE_DICT,
            state_dict_config=FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            optim_state_dict_config=FullOptimStateDictConfig(rank0_only=True, offload_to_cpu=True),
        ):
            # Load model state.
            log.info("Loading model state...")
            self.fsdp_model.load_state_dict(
                self.model._make_state_dict_compatible(torch.load(resource_path(load_path, "model.pt")))
            )

            # Load optimizer state.
            log.info("Loading optimizer state...")
            optim_state_dict = torch.load(resource_path(load_path, "optim.pt"))
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
            other_state_dict = torch.load(resource_path(load_path, "other.pt"))
            self.load_non_tensor_state_dict(other_state_dict)

        barrier()

    def save_checkpoint(self, checkpoint_type: CheckpointType = CheckpointType.sharded) -> Path:
        if checkpoint_type == CheckpointType.sharded:
            return self.save_sharded_checkpoint()
        elif checkpoint_type == CheckpointType.unsharded:
            return self.save_unsharded_checkpoint()
        else:
            raise NotImplementedError(checkpoint_type)

    def restore_checkpoint(self, load_path: PathOrStr, checkpoint_type: Optional[CheckpointType] = None):
        if checkpoint_type == CheckpointType.unsharded or (
            checkpoint_type is None and str(load_path).endswith("-unsharded")
        ):
            self.restore_unsharded_checkpoint(load_path)
        elif checkpoint_type == CheckpointType.sharded or checkpoint_type is None:
            self.restore_sharded_checkpoint(load_path)
        elif checkpoint_type is not None:
            raise NotImplementedError(checkpoint_type)

    def remove_checkpoint(self, idx: int = 0, checkpoint_type: CheckpointType = CheckpointType.sharded):
        if checkpoint_type == CheckpointType.sharded:
            self.remove_sharded_checkpoint(idx=idx)
        elif checkpoint_type == CheckpointType.unsharded:
            self.remove_unsharded_checkpoint(idx=idx)
        else:
            raise NotImplementedError(checkpoint_type)

    def get_labels(self, batch: Dict[str, Any]) -> torch.Tensor:
        # Labels are just input IDs shifted to the left (first item is ignored).
        labels, attention_mask = batch["input_ids"], batch.get("attention_mask")
        if attention_mask is not None:
            labels = labels.masked_fill(attention_mask == 0.0, -100)
        return labels[..., 1:].contiguous()

    def model_forward(
        self, batch: Dict[str, Any], loss_reduction: str = "mean"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # shape: (batch_size, seq_len, vocab_size)
        logits = self.fsdp_model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            attention_bias=batch.get("attention_bias"),
        ).logits
        logits_for_loss = logits[..., :-1, :].contiguous()
        # shape: (batch_size * seq_len, vocab_size)
        logits_for_loss = logits_for_loss.view(-1, logits_for_loss.size(-1))
        # shape: (batch_size, seq_len)
        labels = self.get_labels(batch)
        # shape: (batch_size * seq_len,)
        labels = labels.view(-1)
        ce_loss = F.cross_entropy(logits_for_loss, labels, ignore_index=-100, reduction=loss_reduction)
        if loss_reduction == "none":
            # Reshape (batch_size * seq_len,) -> (batch_size, seq_len)
            ce_loss = ce_loss.view(batch["input_ids"].shape[0], -1)
        return ce_loss, logits

    def train_batch(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Split into micro-batches.
        micro_batches = self.split_batch(batch)

        # In case this helps with memory utilization.
        del batch

        ce_batch_loss = torch.tensor(0.0, device=self.device)
        z_batch_loss = None if not self.cfg.softmax_auxiliary_loss else torch.tensor(0.0, device=self.device)
        for micro_batch in micro_batches:
            with torch.autocast("cuda", enabled=True, dtype=self.cfg.autocast_precision):
                # Run forward pass.
                ce_loss, logits = self.model_forward(micro_batch)
                ce_loss = ce_loss / len(micro_batches)

                # In case this helps with memory utilization.
                del micro_batch

                # Update overall CE batch loss.
                ce_batch_loss += ce_loss.detach()

                # Get loss to optimize for.
                if self.cfg.softmax_auxiliary_loss:
                    z_squared = logits.logsumexp(-1).pow(2).mean()
                    z_loss = 1e-4 * z_squared / len(micro_batches)
                    loss = ce_loss + z_loss

                    # Update overall Z batch loss.
                    z_batch_loss += z_loss.detach()
                else:
                    loss = ce_loss

                del logits

            # Check for nan.
            if torch.isnan(loss):
                raise ValueError("nan loss encountered")

            # Run backward pass.
            log.info("Before backward")
            loss.backward()
            log.info("After backward")

        return ce_batch_loss, z_batch_loss

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        # Write data-indices to file.
        log.info("Writing indices to file")
        if self.indices_file is not None and "index" in batch:
            indices = "\t".join(str(int(i)) for i in batch["index"])
            self.indices_file.write(f"{self.global_step}\t{indices}\n")

        # Zero-gradients.
        log.info("Zeroing gradients")
        self.optim.zero_grad(set_to_none=True)

        # Reset metrics.
        log.info("Resetting metrics")
        self.ce_train_loss_metric.reset()
        if self.z_train_loss_metric is not None:
            self.z_train_loss_metric.reset()

        # Move tensors to the right device.
        log.info("Moving tensors to device")
        batch = move_to_device(batch, self.device)

        # Run forward-backward pass.
        log.info("Running forward-backward pass.")
        ce_batch_loss, z_batch_loss = self.train_batch(batch)

        # Maybe collect pre-step optimizer-specific metrics.
        if self.should_log_optim_metrics_this_step():
            optim_metrics = self.optim.get_pre_step_metrics(self.fsdp_model)
            for key, value in optim_metrics.items():
                metrics[f"optim/{key}"] = value.item()

        # Clip gradient norms.
        grad_norm: Optional[float] = None
        param_norm: Optional[float] = None
        if self.cfg.max_grad_norm is not None:
            norms = fsdp_clip_grads_and_get_norms(self.fsdp_model, self.cfg.max_grad_norm)
            grad_norm = norms.grad_norm.item()
            if norms.param_norm is not None:
                param_norm = norms.param_norm.item()

        # Adjust the learning rate.
        for group in self.optim.param_groups:
            group["lr"] = self.scheduler.get_lr(
                self.cfg.optimizer.learning_rate, self.global_step, self.cfg.max_duration
            )

        # Optimizer step.
        self.optim.step()

        # Reduce loss metrics across ranks.
        self.ce_train_loss_metric.update(ce_batch_loss)
        ce_batch_loss = self.ce_train_loss_metric.compute()
        metrics["train/CrossEntropyLoss"] = ce_batch_loss.item()
        metrics["train/Perplexity"] = torch.exp(ce_batch_loss).item()
        if z_batch_loss is not None and self.z_train_loss_metric is not None:
            self.z_train_loss_metric.update(z_batch_loss)
            z_batch_loss = self.z_train_loss_metric.compute()
            metrics["train/ZLoss"] = z_batch_loss.item()

        if grad_norm is not None:
            metrics["optim/grad_norm"] = grad_norm
        if param_norm is not None:
            metrics["optim/param_norm"] = param_norm

        # Maybe collect post-step optimizer-specific metrics.
        if self.should_log_optim_metrics_this_step():
            optim_metrics = self.optim.get_post_step_metrics(self.fsdp_model)
            for key, value in optim_metrics.items():
                metrics[f"optim/{key}"] = value.item()

        # Update min train loss and see if we should stop early.
        self.min_train_loss = min(self.min_train_loss, ce_batch_loss.item())  # type: ignore
        if (
            self.cfg.early_stopping_factor is not None
            and self.global_step > self.cfg.scheduler.t_warmup
            and ce_batch_loss.item() > self.cfg.early_stopping_factor * self.min_train_loss
        ):
            raise ValueError("Stopping early because train loss has increased substantially")

        return metrics

    def eval_batch(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast("cuda", enabled=True, dtype=self.cfg.autocast_precision):
            ce_loss, logits = self.model_forward(batch, loss_reduction="none")
        return ce_loss.mean(dim=-1), logits

    def eval_step(self, batch: Dict[str, Any], evaluator: Evaluator) -> None:
        # Move tensors to the right device.
        batch = move_to_device(batch, self.device)

        # Run forward pass.
        with torch.no_grad():  # NOTE: 'torch.inference_mode()' doesn't work with 'torch.compile()'.
            ce_loss, logits = self.eval_batch(batch)

        # Update metrics.
        evaluator.update_metrics(
            batch, ce_loss, logits
        )  # batch includes all keys that the downstream evaluation needs

        barrier()

    def split_batch(self, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        microbatch_size = self.cfg.device_train_microbatch_size
        batch_size = batch["input_ids"].shape[0]
        if batch_size <= microbatch_size:
            return [batch]
        else:
            micro_batches = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    micro_batches[key] = value.split(microbatch_size, dim=0)
                elif isinstance(value, list):
                    micro_batches[key] = [
                        value[microbatch_size * i : microbatch_size * i + microbatch_size]
                        for i in range(math.ceil(batch_size / microbatch_size))
                    ]
                else:
                    raise ValueError(f"unexpected item in batch: '{key}={value}'")
            return [
                {key: value[i] for key, value in micro_batches.items()}  # type: ignore
                for i in range(len(micro_batches["input_ids"]))
            ]

    def system_metrics(self) -> Dict[str, float]:
        metrics = {}
        peak_gpu_mb = peak_gpu_memory()
        if peak_gpu_mb is not None:
            metrics["System/Peak GPU Memory (MB)"] = peak_gpu_mb
        return metrics

    def log_metrics_to_console(self, prefix: str, metrics: Dict[str, float]):
        def format_float(value: float) -> str:
            if value < 0.0001:
                return str(value)  # scientific notation
            elif value > 1000:
                return f"{int(value):,d}"
            elif value > 100:
                return f"{value:.1f}"
            elif value > 10:
                return f"{value:.2f}"
            elif value > 1:
                return f"{value:.3f}"
            else:
                return f"{value:.4f}"

        log.info(
            f"{prefix}\n"
            + "\n".join(
                [
                    f"    {name}={format_float(value)}"
                    for name, value in metrics.items()
                    if not name.startswith("optim/")  # there's too many optimizer metrics
                ]
            )
        )

    def should_log_optim_metrics_this_step(self) -> bool:
        return False

    def should_log_this_step(self) -> bool:
        if self.global_step % self.cfg.console_log_interval == 0:
            return True
        else:
            return False

    def eval(self) -> Dict[str, Any]:
        # Zero gradients and set model to 'eval' mode.
        self.optim.zero_grad(set_to_none=True)
        self.fsdp_model.eval()

        eval_metrics = {}
        return eval_metrics

    def check_if_cancelled(self) -> bool:
        # First check if we've reached the training time limit.
        run_canceled = syncronize_flag(
            self.cfg.time_limit is not None and time.time() - self._start_time >= self.cfg.time_limit,
            self.device,
        )
        if run_canceled:
            log.warning("Run canceled due to time limit")
            return run_canceled

        run_canceled = syncronize_flag(False, self.device)

        return run_canceled

    def fit(self):
        self._start_time = time.time()

        if self.cfg.load_path is not None and self.global_step > 0 and self.cfg.eval_on_load:
            eval_metrics = self.eval()

        # Set model to 'train' mode.
        self.fsdp_model.train()

        # Initialize monitors.
        assert self.cfg.device_train_batch_size is not None
        speed_monitor = SpeedMonitor(self.cfg.speed_monitor)
        lr_monitor = LRMonitor(self.optim)

        # Log system metrics at the start of training.
        sys_metrics = self.system_metrics()
        if sys_metrics:
            self.log_metrics_to_console("Pre-train system metrics", sys_metrics)

        # Train.
        first_batch: bool = True
        canceled: bool = False

        batches = [
            {
                "input_ids": torch.tensor([[  570, 12287,    13,  3395,   625,  2590,    15,   329, 30942,   273,
                                              5448, 22007,   949,   253,  7423,    13,   285,   187,  6972, 20625,
                                             32372,   949,   253,  9129,  2419,    15,   380, 36555,   327,   253,
                                              2829,   187,    71,   663, 40617,    15,  1244,  1335,   253,   767,
                                              2206, 15368,    14,  9458,    13, 39712,   272,  5412,  3564,    13,
                                               347,   187,   783, 39709, 44210,  4817,   689,   253,  7887,    13,
                                               285,   323,   253,  1390,   673,   187,  8732, 12469,  2205,  2210,
                                              1728,   281,  4600, 17421,   521,  5098,  1128, 45217,  1417,  1024,
                                               273,   987,   390,   187,  7041,    13,   390,  5958,   390,  2614,
                                                15,   187,   187,    11,   475,   475,   475,   475,   187,   187],
                                            [ 1439,  1469,   281,  1390,   562,   436,  2137,  2391,   309,  1849,
                                               644,  2820,   281,  2028,   368,   849,  1199,   187,  2520,   294,
                                              1763,  1596,   318,   273,  4266,   275,   368,   556,  5486,   275,
                                               253,  1390,  1643,  1107,  1051,   187,  1915,  8140, 19605,   359,
                                               403,  1051,  1095,  8140, 12401,    15,  7088,    14,  1615,    13,
                                             11761,  5006,    13,   285,  2656,   187,  1257,   342,   368,    15,
                                              4392,  8506,   947,   399, 28077,    58,    15,   535, 50268, 46928,
                                               187,   187,  3172,    35, 18004,  2637,  4915,   427,  9981,   187,
                                               187,  8096,   590,  4395,  3579,   327,   253, 12595,  1919,   344,
                                              1119,   247, 32217,   762,   271,  5637,   187,  3243,    15,   754],
                                            [  352,  4895,   281,   697, 15178,    13,   533,  3517,  2210,   896,
                                               187,  3113,  2067,  2571,    15,  2053,    13,  7824,  3790,   253,
                                             28335,    81, 33787,    13,  6730,   352,   187,  6672,  2920,    13,
                                               347,   858,   253,   806,    15,  9067,   597,   512,  2427,    28,
                                              3517,   597,  4895,   342,   187, 23350,   625,   273,   616, 24835,
                                                13,   840,   597,  4447,   247,  1048,  5084,    13,  1077,   187,
                                              3022,   247, 20762,    13,   285, 22944,   779,   281,   253,  5024,
                                               273,   253,  1854,    15,   380, 15178,  1146,   275,   187,   783,
                                             17514,    14,  3026,    84,   597,   574,   281,  3785,   779,   689,
                                               253, 28390,  8704,   253,   187,    72,  6702,    13,   533,  2378],
                                            [ 1659,   835,   352,   369,   187,  7053,  2326,    15,   380,  3884,
                                               273,   253,  1854,   369,  3164,   432,   411,    15,   407,   322,
                                                15,   281,   187,    38,    15,   407,   427,    15,   380, 31710,
                                               369,  9090,  1469,   432,   253, 19969,    13,   697,  1854,   187,
                                             11849,   271,  6907,   342,   253, 26961,  7844,   273,   670,  3925,
                                              3272,    15,   187,   187,    89, 12211,    15, 17842,  4314,   285,
                                              3578,  2909,   846,   253, 31852,   273,   253,   187,  3899, 22683,
                                                13,   627,   369,   247, 11216,   285,  5536,  1304,    13, 11704,
                                               407,   247,  1077,   187,  6209,   917, 21152,    28,   352,   858,
                                               417,  1199, 28788,  2057, 24511,   390,   253,  1304,   187,  1171],
                                            [ 1608,   482,   327,   368,  1476,   187,   187,    52,   798,   272,
                                             11306,   347,   344, 16535,  1066,   344, 19336,   521,  2159, 14228,
                                              1411,   247,   187, 10228,    13,   285,  2206,  1066,   327,   521,
                                             25296,  2822,   281, 10867, 11304,    15,   187,   187,     3,  4497,
                                               937,   344, 30807,   275, 13775,    13,   346, 30875,   352,   310,
                                               417,  7154,   449,   187,   187,     3,  3650,  7154,     2,  6049,
                                             48570,   369,  4704,   281,  2740,   619,  3802,   483,   323,   479,
                                                15,  2652,  1137,  1476,   187,   187,     3,  1989,   752,   858,
                                               368,   439,   710,   323,  1476,   187,   187,     3,  1276,   323,
                                               315,     3, 13892, 14050, 10867, 11304,    15,   346,    42,   369],
                                            [ 5313,   326,  2168,   187, 20774,   447,   352,    15,  1916,   346,
                                             12467,     3,   253,  4450,   310,   281,   452,   253,  7437,   594,
                                              4845,   187,  3529,   247,  1270,  1142,   273,   253, 19325,   403,
                                             26814,    15,   187,   187,    36, 32671,  4827,   443,  2300, 14609,
                                             18295,  5035,   346, 18237,     3,  1996,   327,    15,   187,   187,
                                                36, 32671,  4827, 35645,  8100, 18295,  5035,   346, 18237,     3,
                                              1996,   327,    15,   187,   187,  2573, 30741, 18295,    34,  5112,
                                              3159,    13,  4495,   346, 33729,   937,  4536,   908,   407,  4383,
                                               187, 34782,   327, 29992,    15,   187,   187,  2573,  6766, 18295,
                                                34,  5112,  3159,  4495,   346, 48781,   937,   285,  4536,   908],
                                            [  187,  2773,   253, 28245,   373,   285, 29555,   262, 22348,   373,
                                               497,   767,  1027, 16308,   273,   187,   264,   692,   265,    13,
                                               369,  1620, 17801,    13,   253,  3438,  1146,  4270,   275,   253,
                                              5281,   273,   247,   187,  6017,   280,  1426,   282,    28,   253,
                                               643,  3839, 32679,    13,   594,   347,   281,  1056,   253,  1072,
                                              4677,   347,   187,   338,   767, 28245,   373,   943,   320,  7416,
                                              2366,    15,  9110,   253,  1072,  1659,   310,  2223,   187,  8890,
                                               407,   841,  4454,   275,  2067,  4477,    15,  1583,  1646,    13,
                                              1512,    13,   281,   452,   644,   187, 38061,   323,  3240,  1027,
                                              7637,    27,   253, 28245,   373,   323,  3924,  7120,    13,   253],
                                            [   13,  1014,  2167,   344,  6057,   326,   253, 43412,   556,   816,
                                             35517,   521,   187, 13875,    27,   187,   187,     3,   688,   253,
                                              1083,  1411,   368,    13,  2305,    15,  3619, 20111,    13,   275,
                                               634,  5928,   309,   452,  6507,   314,   187,  3354,   728,   253,
                                              4179,    15,   496,   253,  1083,  1411, 18682,   659,  7352,  1634,
                                               386,   309, 13292,   512,   187, 36871,   273,    13,   285, 10343,
                                               281,  3890,   271,  4743,  1919,   309,   452,   574,   271,   187,
                                             10468, 32268,   273,  2819,   715,    13,   253,  5989,   273,   253,
                                              5575, 34306,   449,   187,   187,  3996, 20111,   665,   574,  6225,
                                              1309,   436,  2427,   689,   285,  2335,   247,  7319,   387,   253]], dtype=torch.int64, device="cpu"),
                "index": torch.tensor([24604650, 44278198, 10199766, 29398698, 30323890, 26381103,  6706633, 44120804], dtype=torch.int64, device="cpu")
            }
        ]

        for batch in batches:  #self.train_loader:
            log.info("Got a batch")
            torch.set_printoptions(profile="full")
            for key, value in batch.items():
                log.info(f"{key}: dtype: {repr(value.dtype)} device: {repr(value.device)}")

            # Bookkeeping.
            # NOTE: To track the global batch size / number of tokens per batch we make the assumption that all
            # batches see the same number of tokens, which should be the case for language model pre-training
            # (at least when drop_last=True).
            # Alternatively we'd have to use a distributed all reduce over seq_len here, but I don't want that overhead.
            # So for now I'm putting these assertions here so if the assumption is violated it will fail loudly.
            batch_size, seq_len = batch["input_ids"].shape
            assert seq_len == self.cfg.model.max_sequence_length
            assert batch_size == self.cfg.device_train_batch_size
            global_batch_size = batch_size * get_world_size()  # assumes batch size equal across ranks
            self.global_step += 1
            self.global_data_step += 1
            self.global_train_examples_seen += global_batch_size
            self.global_train_tokens_seen += global_batch_size * seq_len
            speed_monitor.batch_start(
                self.global_train_tokens_seen,
                batch_size * seq_len,  # num tokens in batch for this device
                # We start monitoring speed after the first batch since the first
                # batch might be an outlier due to compiling and other initialization overhead.
                record=not first_batch,
            )

            # Run train step on batch.
            log.info("About to start training one step")
            metrics = self.train_step(batch)
            log.info("Trained one step")

            # Maybe collect other metrics.
            if self.should_log_this_step():
                # Speed metrics.
                metrics.update(speed_monitor.check())
                # System metrics.
                metrics.update(self.system_metrics())
                # Learning rate metrics.
                metrics.update(lr_monitor.check())

            # Log metrics to console.
            if self.global_step % self.cfg.console_log_interval == 0:
                self.log_metrics_to_console(f"[step={self.global_step}/{self.cfg.max_duration}]", metrics)

            # Check if run should be canceled.
            if self.global_step % self.cfg.canceled_check_interval == 0:
                canceled = self.check_if_cancelled()

            # Maybe save sharded checkpoint.
            if canceled or (
                self.global_step % self.cfg.save_interval == 0 and self.cfg.save_num_checkpoints_to_keep != 0
            ):
                log.info("Saving checkpoint...")
                checkpoint_path = self.save_sharded_checkpoint()
                log.info(f"Checkpoint saved to {checkpoint_path}")

                # Reset speed monitor so that we don't count the time taken to save checkpoints.
                speed_monitor.reset()

            # Maybe save unsharded checkpoint.
            if (
                not canceled  # we already save a sharded checkpoint when canceled
                and self.cfg.save_interval_unsharded is not None
                and self.global_step % self.cfg.save_interval_unsharded == 0
                and self.cfg.save_num_unsharded_checkpoints_to_keep != 0
            ):
                log.info("Saving unsharded checkpoint...")
                checkpoint_path = self.save_unsharded_checkpoint()
                log.info(f"Unsharded checkpoint saved to {checkpoint_path}")

                # Reset speed monitor so that we don't count the time taken to save checkpoints.
                speed_monitor.reset()

            # Maybe run evaluations.
            if not canceled and self.global_step % self.cfg.eval_interval == 0:
                eval_metrics = self.eval()

                # Reset speed monitor so that we don't count the time taken to run evaluations.
                speed_monitor.reset()

                # Reset model to 'train' mode.
                self.fsdp_model.train()

            # End of batch.
            first_batch = False

            if canceled:
                break
        else:
            log.info("Training loop complete")

        # Save final unsharded model-only checkpoint.
        if not canceled and self.cfg.save_interval_unsharded is not None:
            log.info("Saving final unsharded model checkpoint...")
            checkpoint_path = self.save_unsharded_checkpoint()
            log.info(f"Unsharded checkpoint saved to {checkpoint_path}")

    def close(self, exit_code: int = 0) -> None:
        if self.indices_file is not None:
            self.indices_file.flush()
            self.indices_file.close()

    def __enter__(self) -> Trainer:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        del exc_val, exc_tb
        self.close(0 if exc_type is None else 1)
