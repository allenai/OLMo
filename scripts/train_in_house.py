import logging
import os
import shutil
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as checkpoint
from torch.distributed.checkpoint.optimizer import load_sharded_optimizer_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType

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
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(f"cuda:{local_rank()}")

    # Display and save configuration.
    if rank() == 0:
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
    if rank() == 0:
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
        use_orig_params=True,  # needed for compile
        limit_all_gathers=True,
        device_id=local_rank(),
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

    # TODO: data loader

    if not cfg.dry_run and cfg.load_path is None:
        # We save a checkpoint up-front to make sure this won't fail (due to disk space or whatever)
        log.info("Saving pre-train checkpoint...")
        checkpoint_path = save_checkpoint(0, cfg, fsdp_model, optim)
        log.info(f"Checkpoint saved to {checkpoint_path}")

    if cfg.load_path is not None:
        log.info(f"Loading checkpoint from {cfg.load_path}...")
        restore_checkpoint(Path(cfg.load_path), cfg, fsdp_model, optim)

    # TODO: compile forward/backward/step


def rank() -> int:
    return dist.get_rank()


def local_rank() -> int:
    return int(os.environ["LOCAL_RANK"])


def save_checkpoint(step: int, cfg: TrainConfig, model: FSDP, optim: torch.optim.Optimizer) -> Path:
    checkpoint_dir = Path(cfg.save_folder) / f"step{step}"

    try:
        next(checkpoint_dir.glob("*"))
        if cfg.save_overwrite:
            if rank() == 0:
                shutil.rmtree(checkpoint_dir)
        else:
            raise OlmoConfigurationError(
                f"Checkpoint for step {step} already exists, use --save-overwrite to overwrite it"
            )
    except StopIteration:
        pass

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()

    # Write the checkpoint.
    with FSDP.state_dict_type(model, state_dict_type=StateDictType.SHARDED_STATE_DICT):
        # TODO: save rng states
        checkpoint.save_state_dict(
            {"model": model.state_dict(), "optim": FSDP.optim_state_dict(model, optim)},
            checkpoint.FileSystemWriter(checkpoint_dir),
        )

    return checkpoint_dir


def restore_checkpoint(load_path: Path, cfg: TrainConfig, model: FSDP, optim: torch.optim.Optimizer):
    del cfg

    # The only way I figured out how to do this was by reading the unit tests here
    # https://github.com/pytorch/pytorch/blob/main/test/distributed/checkpoint/test_fsdp_optim_state.py

    with FSDP.state_dict_type(model, state_dict_type=StateDictType.SHARDED_STATE_DICT):
        # Load the serialized state dict in place.
        state_dict = {
            "model": model.state_dict(),
            # Can't load optimizer together with the model
        }
        checkpoint.load_state_dict(state_dict, checkpoint.FileSystemReader(load_path))

        # Load into the model.
        model.load_state_dict(state_dict["model"])

        # Load optim state.
        optim_state = load_sharded_optimizer_state_dict(
            model_state_dict=state_dict["model"],
            optimizer_key="optim",
            storage_reader=checkpoint.FileSystemReader(load_path),
        )
        # NOTE: careful, the order of these arguments has changed since the 2.0 release. Cool!
        # flattened_osd = FSDP.optim_state_dict_to_load(model, optim, optim_state["optim"])  # post 2.0
        flattened_osd = FSDP.optim_state_dict_to_load(optim_state["optim"], model, optim)
        optim.load_state_dict(flattened_osd)

    dist.barrier()

    # TODO: restore rng states


if __name__ == "__main__":
    prepare_cli_environment()

    try:
        yaml_path, args_list = sys.argv[1], sys.argv[2:]
    except IndexError:
        raise OlmoCliError(f"Usage: {sys.argv[0]} [CONFIG_PATH] [OPTIONS]")

    cfg = TrainConfig.load(yaml_path, [clean_opt(s) for s in args_list])
    main(cfg)
