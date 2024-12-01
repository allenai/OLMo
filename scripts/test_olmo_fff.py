"""Run this script with 'torchrun'."""

import glob
import logging
import sys
from pathlib import Path

import torch

import wandb
from olmo.config import TrainConfig
from olmo.eval import build_evaluators
from olmo.exceptions import OLMoCliError, OLMoConfigurationError
from olmo.model import OLMo
from olmo.optim import build_optimizer, build_scheduler
from olmo.torch_util import (
    get_global_rank,
    get_world_size,
    peak_gpu_memory,
    seed_all,
    get_local_rank
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

    # torch.cuda.set_device(f"cuda:{get_local_rank()}")
    # device = f"cuda:{get_local_rank()}"

    torch.set_default_device("cpu")
    device = "cpu"

    # Fill some configuration options.
    cfg.model.precision = cfg.precision
    cfg.device_train_batch_size = cfg.global_train_batch_size // get_world_size()
    assert cfg.device_train_batch_size is not None  # for mypy

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

    # # Maybe start W&B run.
    # if cfg.wandb is not None and (get_global_rank() == 0 or not cfg.wandb.rank_zero_only):
    #     wandb_dir = Path(cfg.save_folder) / "wandb"
    #     wandb_dir.mkdir(parents=True, exist_ok=True)
    #     wandb.init(
    #         dir=wandb_dir,
    #         project=cfg.wandb.project,
    #         entity=cfg.wandb.entity,
    #         group=cfg.wandb.group,
    #         name=cfg.wandb.name,
    #         tags=cfg.wandb.tags,
    #         config=cfg.asdict(exclude=["wandb"]),
    #     )

    # Set seed.
    seed_all(cfg.seed)

    # Construct evaluators.
    evaluators = build_evaluators(cfg, device)

    if cfg.load_path is None:
        raise OLMoConfigurationError("To run eval you must provide a load_path")
    elif "://" in cfg.load_path:
        raise OLMoConfigurationError(
            "Eval does not support remote paths. Please specify a local path or WEKA mounted path."
        )
    if "step" in cfg.load_path.split("/")[-1]:
        load_paths = [cfg.load_path]
    else:
        # This globbing only works with local paths
        load_paths = list(glob.glob(f"{cfg.load_path}/step*"))
        load_paths = [x for x in load_paths if x[-1].isdigit()]
        load_paths = list(
            sorted(load_paths, key=lambda x: int(x.split("/")[-1].replace("-unsharded", "").split("step")[-1]))
        )

    for load_path in load_paths:
        step = int(load_path.split("/")[-1].replace("-unsharded", "").split("step")[-1])

        # We are using a single accelerator!
        cfg.distributed_strategy = None
        cfg.model.init_device = device

        # Initialize the model.
        log.info("Building model...")
        olmo_model = OLMo(cfg.model)
        log.info(f"Total number of parameters: {olmo_model.num_params():,d}")
        log.info(f"Number of non-embedding parameters: {olmo_model.num_params(include_embedding=False):,d}")
        log.info(f"Peak GPU Memory (MB) before {cfg.distributed_strategy}: {int(peak_gpu_memory() or 0)}")

        dist_model = olmo_model.to(device)

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
    prepare_cli_environment()
    log.info("CLI environment prepared")

    add_cached_path_clients()

    try:
        yaml_path, args_list = sys.argv[1], sys.argv[2:]
    except IndexError:
        raise OLMoCliError(f"Usage: {sys.argv[0]} [CONFIG_PATH] [OPTIONS]")

    cfg = TrainConfig.load(yaml_path, [clean_opt(s) for s in args_list])

    main(cfg)

