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
import wandb
from packaging import version
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

from olmo.config import CheckpointType, FSDPWrapStrategy, TrainConfig
from olmo.data import build_train_dataloader
from olmo.eval import build_evaluators
from olmo.exceptions import OlmoCliError, OlmoConfigurationError
from olmo.model import Olmo
from olmo.optim import build_optimizer, build_scheduler, BoltOnWarmupScheduler
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

    # Sanity check
    if cfg.reset_optimizer_state and cfg.load_path is None:
        log.warning(
            "You want to reset the optimizer state, but we're not loading from the checkpoint. The"
            "setting has no effect."
        )

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

    # Display and save configuration.
    if get_global_rank() == 0:
        if cfg.data.paths is not None and len(cfg.data.paths) < 50:
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

    barrier()

    # Maybe start W&B run.
    if cfg.wandb is not None and (get_global_rank() == 0 or not cfg.wandb.rank_zero_only):
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

    barrier()

    # Set seed.
    seed_all(cfg.seed)

    # Construct data loader.
    train_loader = build_train_dataloader(cfg)

    # Construct evaluators.
    evaluators = build_evaluators(cfg, device)
    barrier()

    # Initialize the model.
    log.info("Building model...")
    olmo_model = Olmo(cfg.model)
    log.info(f"Total number of parameters: {olmo_model.num_params():,d}")
    log.info(f"Number of non-embedding parameters: {olmo_model.num_params(include_embedding=False):,d}")
    log.info(f"Peak GPU Memory (MB) before FSDP: {int(peak_gpu_memory() or 0)}")

    # Wrap the model in FSDP.
    log.info("Wrapping model with FDSP...")
    wrap_policy = None
    if cfg.fsdp.wrapping_strategy == FSDPWrapStrategy.by_block:
        wrap_policy = olmo_model.fsdp_wrap_fn
    elif cfg.fsdp.wrapping_strategy == FSDPWrapStrategy.size_based:
        wrap_policy = size_based_auto_wrap_policy

    if version.parse(torch.__version__) >= version.parse("2.1.0"):
        # This prevents any parameters from being initialized twice
        def dummy_init_fn(module: torch.nn.Module) -> None:
            module.to_empty(device=get_local_rank())

        param_init_fn = dummy_init_fn
    else:
        param_init_fn = None

    fsdp_model = FSDP(
        olmo_model,
        sharding_strategy=cfg.fsdp.sharding_strategy,
        mixed_precision=cfg.fsdp_precision,
        auto_wrap_policy=wrap_policy,
        use_orig_params=cfg.fsdp.use_orig_params,  # needed for compile and some of our optimizer/parameter metrics
        limit_all_gathers=True,
        device_id=get_local_rank(),
        param_init_fn=param_init_fn,
    )
    # when param_init_fn is None, FSDP will call reset_parameters() automatically
    if param_init_fn is not None:
        olmo_model.reset_parameters()

    log.info(f"Peak GPU Memory (MB) after FSDP: {int(peak_gpu_memory() or 0)}")

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

    log.info("Model:")
    log.info(fsdp_model)

    # Construct optimizer and learning rate scheduler.
    optim = build_optimizer(cfg, fsdp_model)
    scheduler = build_scheduler(cfg)

    # Data indices file.
    indices_file: Optional[TextIO] = None
    if cfg.save_data_indices:
        indices_file_path = Path(cfg.save_folder) / f"data-indices/rank{get_global_rank()}.tsv.gz"
        if indices_file_path.exists() and not cfg.save_overwrite:
            raise OlmoConfigurationError(f"{indices_file_path} already exists, use --save_overwrite to overwrite")
        indices_file_path.parent.mkdir(exist_ok=True, parents=True)
        indices_file = gzip.open(indices_file_path, "wt")

    # Consolidate components into `Trainer` object.
    with Trainer(
        cfg=cfg,
        model=olmo_model,
        fsdp_model=fsdp_model,
        optim=optim,
        scheduler=scheduler,
        train_loader=train_loader,
        device=device,
        evaluators=evaluators,
        indices_file=indices_file,
    ) as trainer:
        if not cfg.dry_run and cfg.load_path is None:
            checkpoint_type = (
                CheckpointType.sharded if cfg.save_num_checkpoints_to_keep != 0 else CheckpointType.unsharded
            )

            # We save a checkpoint up-front to make sure this won't fail (due to disk space or whatever).
            log.info("Saving pre-train checkpoint...")
            checkpoint_path, local_checkpoint_cache = trainer.save_checkpoint(checkpoint_type=checkpoint_type)
            log.info(f"Checkpoint saved to {checkpoint_path}")

            # And they we verify that we can load it.
            log.info("Attempting to load pre-train checkpoint...")
            trainer.restore_checkpoint(
                checkpoint_path, checkpoint_type=checkpoint_type, local_cache=local_checkpoint_cache
            )
            log.info("Checkpoint successfully loaded")

            # NOTE: https://github.com/allenai/LLM/issues/233
            #  log.info("Removing pre-train checkpoint...")
            #  trainer.remove_checkpoint(checkpoint_type=checkpoint_type)
            #  log.info("Successfully removed checkpoint")

        if cfg.load_path is not None:
            log.info(f"Loading checkpoint from {cfg.load_path}...")
            trainer.restore_checkpoint(cfg.load_path, load_optimizer_state=not cfg.reset_optimizer_state)
            log.info("Checkpoint successfully loaded")

            # If we have to, set a new scheduler:
            if cfg.reset_optimizer_state:
                trainer.scheduler = BoltOnWarmupScheduler(
                    trainer.scheduler, trainer.global_step, trainer.global_step + cfg.scheduler.t_warmup
                )

        if cfg.force_save_unsharded:
            log.info("Saving unsharded checkpoint...")
            checkpoint_path, _ = trainer.save_unsharded_checkpoint()
            log.info(f"Unsharded checkpoint saved to {checkpoint_path}")

        if cfg.compile is not None:
            # TODO (epwalsh): trying to compile the whole train step results in a compile-time error from within
            # the optimizer. We should investigate this further at some point.
            #  trainer.train_step = torch.compile(trainer.train_step, **cfg.compile.asdict())
            trainer.train_batch = torch.compile(trainer.train_batch, **cfg.compile.asdict())  # type: ignore
            # TODO (epwalsh): compiling the `eval_batch()` method is a little sketchy since the inputs will look
            # different for different eval tasks. That might be okay, but it might not be.
            #  trainer.eval_batch = torch.compile(trainer.eval_batch, **cfg.compile.asdict())  # type: ignore
            # Alternatively, could just do this:
            #  trainer.fsdp_model = torch.compile(trainer.fsdp_model, **cfg.compile.asdict())

        if not cfg.dry_run:
            log.info("Starting training...")
            trainer.fit()
            log.info("Training complete")
        else:
            log.info("Dry run complete")


if __name__ == "__main__":
    prepare_cli_environment()

    try:
        yaml_path, args_list = sys.argv[1], sys.argv[2:]
    except IndexError:
        raise OlmoCliError(f"Usage: {sys.argv[0]} [CONFIG_PATH] [OPTIONS]")

    cfg = TrainConfig.load(yaml_path, [clean_opt(s) for s in args_list])
    main(cfg)
