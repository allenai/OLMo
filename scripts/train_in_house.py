import logging
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy

from olmo import Olmo, TrainConfig
from olmo.exceptions import OlmoCliError, OlmoConfigurationError
from olmo.optim import DecoupledLionW
from olmo.util import clean_opt, log_extra_field, prepare_cli_environment, seed_all

log = logging.getLogger(__name__)


def main(cfg: TrainConfig) -> None:
    # Ensure run name set.
    if cfg.run_name is None:
        cfg.run_name = os.environ.get("COMPOSER_RUN_NAME", "train-llm")
    log_extra_field("run_name", cfg.run_name)

    cfg.model.precision = cfg.precision

    # Initialize process group and set device.
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(f"cuda:{local_rank}")

    # Display and save configuration.
    if dist.get_rank() == 0:
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

    # Initialize the model.
    log.info("Initializing model...")
    olmo_model = Olmo(cfg.model)
    if dist.get_rank() == 0:
        log.info(f"Total number of parameters: {olmo_model.num_params():,d}")
        log.info(
            f"Number of non-embedding parameters: {olmo_model.num_params(include_embedding=False):,d}",
        )

    # Wrap the model in FSDP.
    fsdp_model = FSDP(
        olmo_model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=cfg.autocast_precision,
            reduce_dtype=cfg.autocast_precision,
            buffer_dtype=cfg.autocast_precision,
        ),
        auto_wrap_policy=olmo_model.fsdp_wrap_fn,
        param_init_fn=olmo_model.param_init_fn,
        use_orig_params=True,  # needed for compile
        limit_all_gathers=True,
        device_id=local_rank,
    )

    # Construct optimizer.
    assert cfg.optimizer.learning_rate is not None
    optim = DecoupledLionW(
        fsdp_model.parameters(),
        lr=cfg.optimizer.learning_rate,
        betas=cfg.optimizer.betas,
        weight_decay=cfg.optimizer.weight_decay,
    )

    # TODO: learning rate scheduler

    if not cfg.dry_run and cfg.load_path is None:
        # We save a checkpoint up-front to make sure this won't fail (due to disk space or whatever)
        log.info("Saving pre-train checkpoint...")
        raise NotImplementedError

    if cfg.load_path is not None:
        log.info(f"Loading checkpoint from {cfg.load_path}...")
        raise NotImplementedError

    # TODO: compile forward/backward/step


if __name__ == "__main__":
    prepare_cli_environment()

    try:
        yaml_path, args_list = sys.argv[1], sys.argv[2:]
    except IndexError:
        raise OlmoCliError(f"Usage: {sys.argv[0]} [CONFIG_PATH] [OPTIONS]")

    cfg = TrainConfig.load(yaml_path, [clean_opt(s) for s in args_list])
    main(cfg)
