"""Run this script with 'torchrun'."""

import gzip
import logging
import os
import sys
from functools import partial
from pathlib import Path
from typing import Optional, TextIO

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torchmetrics import MeanMetric

from olmo.config import CheckpointType, TrainConfig
from olmo.data import build_train_dataloader
from olmo.eval import build_evaluators
from olmo.exceptions import OlmoCliError, OlmoConfigurationError
from olmo.model import Olmo
from olmo.optim import build_optimizer, build_scheduler
from olmo.train import Trainer
from olmo.util import (
    barrier,
    clean_opt,
    get_global_rank,
    get_local_rank,
    get_world_size,
    log_extra_field,
    peak_gpu_memory,
    prepare_cli_environment,
    seed_all,
)

log = logging.getLogger("train")


def main(cfg: TrainConfig) -> None:
    # Ensure run name set.
    if cfg.run_name is None:
        cfg.run_name = os.environ.get("COMPOSER_RUN_NAME", "train-llm")
    log_extra_field("run_name", cfg.run_name)

    # Initialize process group and set device.
    dist.init_process_group(backend="nccl")
    barrier()
    torch.cuda.set_device(f"cuda:{get_local_rank()}")
    device = torch.device("cuda")

    # Fill some configuration options.
    cfg.model.precision = cfg.precision
    cfg.device_train_batch_size = cfg.global_train_batch_size // get_world_size()
    assert cfg.device_train_batch_size is not None  # for mypy
    cfg.device_train_grad_accum = cfg.device_train_batch_size // cfg.device_train_microbatch_size

    barrier()

    # Set seed.
    seed_all(cfg.seed)

    barrier()

    # Initialize the model.
    log.info("Building model...")
    olmo_model = Olmo(cfg.model)
    log.info(f"Total number of parameters: {olmo_model.num_params():,d}")
    log.info(f"Number of non-embedding parameters: {olmo_model.num_params(include_embedding=False):,d}")
    log.info(f"Peak GPU Memory (MB) before FSDP: {int(peak_gpu_memory() or 0)}")

    # Wrap the model in FSDP.
    log.info("Wrapping model with FDSP...")
    fsdp_model = FSDP(
        olmo_model,
        sharding_strategy=cfg.fsdp.sharding_strategy,
        mixed_precision=MixedPrecision(  # equivalent to MosaicML's "PURE"
            param_dtype=cfg.autocast_precision,
            reduce_dtype=cfg.autocast_precision,
            buffer_dtype=cfg.autocast_precision,
        ),
        auto_wrap_policy=olmo_model.fsdp_wrap_fn,
        use_orig_params=cfg.fsdp.use_orig_params,  # needed for compile and some of our optimizer/parameter metrics
        limit_all_gathers=True,
        device_id=get_local_rank(),
    )

    log.info(f"Peak GPU Memory (MB) after FSDP: {int(peak_gpu_memory() or 0)}")

    assert not cfg.activation_checkpointing

    log.info("Model:")
    log.info(fsdp_model)

    # Construct optimizer and learning rate scheduler.
    optim = build_optimizer(cfg, fsdp_model)
    scheduler = build_scheduler(cfg)

    # Consolidate components into `Trainer` object.
    with Trainer(
        cfg=cfg,
        model=olmo_model,
        fsdp_model=fsdp_model,
        optim=optim,
        scheduler=scheduler,
        device=device,
        ce_train_loss_metric=MeanMetric(nan_strategy="error").to(device),
        z_train_loss_metric=None
        if not cfg.softmax_auxiliary_loss
        else MeanMetric(nan_strategy="error").to(device),
        indices_file=None,
    ) as trainer:
        log.info("Starting training...")
        trainer.fit()
        log.info("Training complete")


if __name__ == "__main__":
    prepare_cli_environment()

    try:
        yaml_path, args_list = sys.argv[1], sys.argv[2:]
    except IndexError:
        raise OlmoCliError(f"Usage: {sys.argv[0]} [CONFIG_PATH] [OPTIONS]")

    cfg = TrainConfig.load(yaml_path, [clean_opt(s) for s in args_list])
    main(cfg)
