"""Run this script with 'torchrun'."""

import logging
import sys
from datetime import timedelta
from pathlib import Path
from typing import Optional, TextIO
import glob

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb
from packaging import version
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.nn.parallel import DistributedDataParallel as DDP

from olmo.config import (
    DDPGradSyncMode,
    DistributedStrategy,
    TrainConfig,
)
from olmo.eval import build_evaluators
from olmo.exceptions import OLMoCliError, OLMoConfigurationError
from olmo.model import OLMo
from olmo.optim import build_optimizer, build_scheduler
from olmo.torch_util import (
    barrier,
    get_default_device,
    get_global_rank,
    get_local_rank,
    get_local_world_size,
    get_world_size,
    peak_gpu_memory,
    seed_all,
)
from olmo.train import TrainerForEval
from olmo.util import (
    add_cached_path_clients,
    clean_opt,
    log_extra_field,
    prepare_cli_environment,
)

log = logging.getLogger("train")


def main(cfg: TrainConfig) -> None:
    # Ensure run name set.
    if cfg.run_name is None:
        raise OLMoConfigurationError("--run_name is required")
    log_extra_field("run_name", cfg.run_name)

    # Sanity check
    if (cfg.reset_optimizer_state or cfg.reset_trainer_state) and cfg.load_path is None:
        log.warning(
            "You want to reset the optimizer or trainer state, but we're not loading from the checkpoint. The"
            "setting has no effect."
        )

    barrier()

    device = torch.device("cuda")

    # Fill some configuration options.
    cfg.model.precision = cfg.precision
    cfg.device_train_batch_size = cfg.global_train_batch_size // get_world_size()
    assert cfg.device_train_batch_size is not None  # for mypy
    cfg.device_train_grad_accum = cfg.device_train_batch_size // cfg.device_train_microbatch_size
    if cfg.optimizer.no_decay_norm_and_bias is not None:
        log.warning(
            "You set the deprecated config option `no_decay_norm_and_bias`. For compatibility, this"
            "setting will take precedence over all other weight decay configurations. Please change"
            "your config to use `decay_norm_and_bias` and `decay_embeddings` instead."
        )
        cfg.optimizer.decay_norm_and_bias = not cfg.optimizer.no_decay_norm_and_bias
        cfg.optimizer.decay_embeddings = not cfg.optimizer.no_decay_norm_and_bias
        cfg.optimizer.no_decay_norm_and_bias = None  # So nobody uses this by accident.

    # Display and save configuration.
    if get_global_rank() == 0:
        if cfg.data.paths is not None and len(cfg.data.paths) < 50:
            log.info("Configuration:")
            log.info(cfg)
        if not cfg.dry_run and (cfg.load_path is None or Path(cfg.load_path).parent != Path(cfg.save_folder)):
            # Save config.
            save_path = Path(cfg.save_folder) / "config.yaml"
            if save_path.is_file() and not cfg.save_overwrite:
                raise OLMoConfigurationError(f"{save_path} already exists, use --save_overwrite to overwrite")
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

    # Construct evaluators.
    evaluators = build_evaluators(cfg, device)
    barrier()

    if cfg.load_path is None:
        raise OLMoConfigurationError("To run eval you must provide a load_path")
    if 'step' in cfg.load_path.split('/')[-1]:
        load_paths = [cfg.load_path]
    else:
        # This globbing does not work with remote paths.
        load_paths = list(glob.glob(f"{cfg.load_path}/step*"))
        load_paths = [x for x in load_paths if x[-1].isdigit()]
        # load_paths = [x for x in load_paths if int(x.split('/')[-1].split('step')[-1]) >= 712000] # TODO: delete this
        load_paths = list(sorted(load_paths, key=lambda x: int(x.split('/')[-1].split('step')[-1])))

    for load_path in load_paths:
        step = int(load_path.split('/')[-1].split('step')[-1])

        # Initialize the model.
        log.info("Building model...")
        olmo_model = OLMo(cfg.model)
        log.info(f"Total number of parameters: {olmo_model.num_params():,d}")
        log.info(f"Number of non-embedding parameters: {olmo_model.num_params(include_embedding=False):,d}")
        log.info(f"Peak GPU Memory (MB) before {cfg.distributed_strategy}: {int(peak_gpu_memory() or 0)}")

        if cfg.distributed_strategy == DistributedStrategy.ddp:
            log.info("Wrapping model with DDP...")
            assert cfg.ddp is not None, "DistributedStrategy ddp needs cfg.ddp to be set!"

            if cfg.model.init_device != "cuda":
                raise OLMoConfigurationError("DDP does not work with init_device set to anything other than `cuda`.")

            if cfg.ddp.find_unused_params is True and cfg.ddp.grad_sync_mode != DDPGradSyncMode.micro_batch:
                raise OLMoConfigurationError(
                    "`find_unused_params` is set to True. DDP needs to synchronize gradients for every micro-batch to avoid errors. Set `grad_sync_mode` to `micro_batch`."
                )

            param_init_fn = None

            # move to cuda before calling ddp
            dist_model = DDP(olmo_model.to(device), find_unused_parameters=cfg.ddp.find_unused_params)
        elif cfg.distributed_strategy == DistributedStrategy.fsdp:
            # Wrap the model in FSDP.
            log.info("Wrapping model with FSDP...")
            assert cfg.fsdp is not None, "DistributedStrategy fsdp needs cfg.fsdp to be set!"
            wrap_policy = olmo_model.get_fsdp_wrap_policy(cfg.fsdp.wrapping_strategy)

            if version.parse(torch.__version__) >= version.parse("2.1.0"):
                # This prevents any parameters from being initialized twice
                def dummy_init_fn(module: torch.nn.Module) -> None:
                    module.to_empty(device=get_default_device())

                param_init_fn = dummy_init_fn
            else:
                param_init_fn = None

            # Set up device mesh for hybrid sharding in order to specify which nodes are assoicated to a given model replica
            device_mesh = None
            hybrid_sharding_fsdp_kwargs = {}
            if cfg.fsdp.sharding_strategy in (ShardingStrategy.HYBRID_SHARD, ShardingStrategy._HYBRID_SHARD_ZERO2):
                if version.parse(torch.__version__) < version.parse("2.2.0"):
                    # Device mesh was not added to PyTorch until v2.2.0
                    raise OLMoConfigurationError(
                        "OLMo training does not correctly support hybrid sharding before torch 2.2.0"
                    )

                from torch.distributed.device_mesh import init_device_mesh

                num_model_replicas = cfg.fsdp.hybrid_sharding_num_model_replicas or (
                    get_world_size() // get_local_world_size()
                )

                if num_model_replicas <= 0:
                    raise OLMoConfigurationError("fsdp.hybrid_sharding_num_model_replicas must be a positive integer")

                if get_world_size() % num_model_replicas != 0:
                    raise OLMoConfigurationError("fsdp.hybrid_sharding_num_model_replicas must divide world size")

                device_mesh = init_device_mesh("cuda", (num_model_replicas, get_world_size() // num_model_replicas))
                hybrid_sharding_fsdp_kwargs["device_mesh"] = device_mesh

            dist_model = FSDP(
                olmo_model,
                sharding_strategy=cfg.fsdp.sharding_strategy,
                mixed_precision=cfg.fsdp_precision,
                auto_wrap_policy=wrap_policy,
                use_orig_params=cfg.fsdp.use_orig_params,  # needed for compile and some of our optimizer/parameter metrics
                limit_all_gathers=True,
                device_id=get_local_rank(),
                param_init_fn=param_init_fn,
                **hybrid_sharding_fsdp_kwargs,
            )
        elif cfg.distributed_strategy is None:
            raise NotImplementedError("Single accelerator training not implemented yet!")

        # when param_init_fn is None, FSDP will call reset_parameters() automatically
        if param_init_fn is not None or cfg.distributed_strategy == DistributedStrategy.ddp:
            olmo_model.reset_parameters()

        log.info(f"Peak GPU Memory (MB) after {cfg.distributed_strategy}: {int(peak_gpu_memory() or 0)}")
        log.info("Model:")
        log.info(dist_model)

        # Construct optimizer and learning rate scheduler.
        optim = build_optimizer(cfg, dist_model)
        scheduler = build_scheduler(cfg)

        # Consolidate components into `Trainer` object.
        with TrainerForEval(
            cfg=cfg,
            epoch=cfg.epoch,
            model=olmo_model,
            dist_model=dist_model,
            device=device,
            evaluators=evaluators,
            optim=optim,
            scheduler=scheduler,
        ) as trainer:

            log.info(f"Loading checkpoint from {load_path}...")
            trainer.restore_checkpoint(
                load_path,
                load_optimizer_state=False,
                load_trainer_state=False,
                sharded_checkpointer=cfg.load_path_sharded_checkpointer,
            )
            log.info("Checkpoint successfully loaded")

            log.info("Starting evaluating...")
            eval_metrics = trainer.eval()
            if wandb.run is not None:
                wandb.log(eval_metrics, step=step)
            log.info("Evaluating complete")


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError as e:
        print(f"failed to set multiprocessing start method: {e}")
    log.info(f"Multiprocessing start method set to '{mp.get_start_method()}'")

    # Set CUDA device.
    torch.cuda.set_device(f"cuda:{get_local_rank()}")

    # Initialize process group.
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=30))
    log.info("Process group initialized")

    prepare_cli_environment()
    log.info("CLI environment prepared")

    add_cached_path_clients()

    try:
        yaml_path, args_list = sys.argv[1], sys.argv[2:]
    except IndexError:
        raise OLMoCliError(f"Usage: {sys.argv[0]} [CONFIG_PATH] [OPTIONS]")

    cfg = TrainConfig.load(yaml_path, [clean_opt(s) for s in args_list])
    main(cfg)
