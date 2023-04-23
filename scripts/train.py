"""Run this script with 'torchrun'."""

import logging
import os
import random
import shutil
import sys
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as checkpoint
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.checkpoint.optimizer import load_sharded_optimizer_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType
from torch.utils.data import DataLoader, DistributedSampler
from torchmetrics import MeanMetric

from olmo import Olmo, TrainConfig
from olmo.aliases import BatchDict
from olmo.config import OptimizerType, SchedulerType
from olmo.data import DataCollator, MemMapDataset
from olmo.exceptions import OlmoCliError, OlmoConfigurationError
from olmo.optim import DecoupledLionW
from olmo.util import (
    clean_opt,
    log_extra_field,
    move_to_device,
    prepare_cli_environment,
    seed_all,
)

log = logging.getLogger(__name__)


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
    global_step: int = 0

    def state_dict(self) -> Dict[str, Any]:
        return {
            "model": self.fsdp_model.state_dict(),
            "optim": FSDP.optim_state_dict(self.fsdp_model, self.optim),
            "scheduler": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.random.get_rng_state(),
                "cuda": torch.cuda.get_rng_state(),
            },
        }

    def save_checkpoint(self) -> Path:
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
        with FSDP.state_dict_type(self.fsdp_model, state_dict_type=StateDictType.SHARDED_STATE_DICT):
            checkpoint.save_state_dict(self.state_dict(), checkpoint.FileSystemWriter(checkpoint_dir))

        return checkpoint_dir

    def restore_checkpoint(self, load_path: Path):
        # The only way I figured out how to do this was by reading the unit tests here
        # https://github.com/pytorch/pytorch/blob/main/test/distributed/checkpoint/test_fsdp_optim_state.py

        with FSDP.state_dict_type(self.fsdp_model, state_dict_type=StateDictType.SHARDED_STATE_DICT):
            # Load the serialized state dict in place.
            state_dict = self.state_dict()
            del state_dict["optim"]  # Can't load optimizer together with the model
            checkpoint.load_state_dict(state_dict, checkpoint.FileSystemReader(load_path))

            # Load state (other than optimizer).
            self.fsdp_model.load_state_dict(state_dict["model"])
            self.global_step = state_dict["global_step"]
            self.scheduler.load_state_dict(state_dict["scheduler"])
            rng_state = state_dict.pop("rng")

            # Load optim state.
            optim_state = load_sharded_optimizer_state_dict(
                model_state_dict=state_dict["model"],
                optimizer_key="optim",
                storage_reader=checkpoint.FileSystemReader(load_path),
            )
            # NOTE: careful, the order of these arguments has changed since the 2.0 release. Cool!
            # flattened_osd = FSDP.optim_state_dict_to_load(model, optim, optim_state["optim"])  # post 2.0
            flattened_osd = FSDP.optim_state_dict_to_load(optim_state["optim"], self.fsdp_model, self.optim)
            self.optim.load_state_dict(flattened_osd)

        dist.barrier()

        # Fast-forward dataloader.
        if self.global_step > 0:
            log.info(f"Fast-forwarding data loader to {self.global_step}...")
            for step, (current_epoch, batch) in self.training_batches:
                del batch, current_epoch
                if step >= self.global_step - 1:
                    break
                elif step % 1000 == 0:
                    log.info(f"Fast-forwarding... {step}/{self.global_step}")

        dist.barrier()

        # Set rng state.
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(rng_state["torch"])
        torch.cuda.set_rng_state(rng_state["cuda"])

    def get_labels(self, batch: BatchDict) -> torch.Tensor:
        # Labels are just input IDs shifted to the left (first item is ignored).
        labels, attention_mask = batch["input_ids"], batch.get("attention_mask")
        if attention_mask is not None:
            labels = labels.masked_fill(attention_mask == 0.0, -100)
        return labels[..., 1:].contiguous()

    def train_step(self, batch: BatchDict) -> Tuple[float, float]:
        # Zero-gradients.
        self.optim.zero_grad(set_to_none=True)

        # Move tensors to the right device.
        batch = move_to_device(batch, self.device)

        # Split into micro-batches.
        micro_batches = self.split_batch(batch)

        # In case this helps with memory utilization.
        del batch

        # Reset metric.
        self.train_loss_metric.reset()

        batch_loss = torch.tensor(0.0, device=self.device)
        for micro_batch in micro_batches:
            # Run forward pass.
            with torch.autocast("cuda", enabled=True, dtype=self.cfg.autocast_precision):
                logits = self.fsdp_model(**micro_batch).logits[..., :-1, :].contiguous()
                labels = self.get_labels(micro_batch)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
                loss = loss / len(micro_batches)

            # In case this helps with memory utilization.
            del micro_batch

            # Check for nan loss.
            if torch.isnan(loss):
                raise ValueError("nan loss encountered")

            # Run backward pass.
            loss.backward()

            # Update overall batch loss.
            batch_loss += loss.detach()

        # Clip gradient norms.
        if self.cfg.max_grad_norm is not None:
            self.fsdp_model.clip_grad_norm_(self.cfg.max_grad_norm)

        # Optimizer step.
        self.optim.step()
        self.scheduler.step()

        # Reduce loss across ranks.
        self.train_loss_metric.update(batch_loss)
        batch_loss = self.train_loss_metric.compute()

        # Return loss and perplexity.
        return batch_loss.item(), torch.exp(batch_loss).item()

    def split_batch(self, batch: BatchDict) -> List[BatchDict]:
        batch_size = batch["input_ids"].shape[0]
        if batch_size <= self.cfg.device_train_microbatch_size:
            return [batch]
        else:
            micro_batches = {}
            for key, tensor in batch.items():
                micro_batches[key] = tensor.split(self.cfg.device_train_microbatch_size, dim=0)  # type: ignore
            return [  # type: ignore
                {key: tensor[i] for key, tensor in micro_batches.items()}
                for i in range(len(micro_batches["input_ids"]))
            ]

    def fit(self):
        self.fsdp_model.train()
        for step, (epoch, batch) in self.training_batches:
            loss, ppl = self.train_step(batch)

            # Log to console.
            if step % self.cfg.console_log_interval == 0:
                log.info(
                    f"[epoch={epoch}, step={step}/{self.cfg.max_duration}]\n"
                    f"    CrossEntropyLoss={loss:.4f}\n"
                    f"    Perplexity={ppl:.4f}"
                )


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
        use_orig_params=True,  # needed for compile
        limit_all_gathers=True,
        device_id=local_rank(),
    )

    # Construct optimizer and learning rate scheduler.
    optim = build_optimizer(cfg, fsdp_model)
    scheduler = build_scheduler(cfg, optim)

    # Construct data loader.
    train_loader = build_dataloader(cfg, cfg.device_train_batch_size)
    training_batches = enumerate(islice(cycle_through_epochs(train_loader), cfg.max_duration))

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
    )

    if not cfg.dry_run and cfg.load_path is None:
        # We save a checkpoint up-front to make sure this won't fail (due to disk space or whatever)
        log.info("Saving pre-train checkpoint...")
        checkpoint_path = trainer.save_checkpoint()
        log.info(f"Checkpoint saved to {checkpoint_path}")

    if cfg.load_path is not None:
        log.info(f"Loading checkpoint from {cfg.load_path}...")
        trainer.restore_checkpoint(Path(cfg.load_path))

    if cfg.compile is not None:
        trainer.train_step = torch.compile(trainer.train_step, **cfg.compile.asdict())

    if not cfg.dry_run:
        log.info("Starting training...")
        trainer.fit()
    else:
        log.info("Dry run complete")


def global_rank() -> int:
    return dist.get_rank()


def local_rank() -> int:
    return int(os.environ["LOCAL_RANK"])


def build_dataloader(cfg: TrainConfig, batch_size: int, shuffle: bool = True) -> DataLoader:
    collator = DataCollator.from_train_config(cfg)
    dataset = MemMapDataset.from_train_config(cfg)
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
