import logging
import sys

import torch
import torch.distributed as dist

from olmo.config import TrainConfig
from olmo.data import build_train_dataloader
from olmo.exceptions import OlmoCliError
from olmo.util import (
    barrier,
    clean_opt,
    get_local_rank,
    get_world_size,
    move_to_device,
    prepare_cli_environment,
)

log = logging.getLogger(__name__)


def main(cfg: TrainConfig) -> None:
    dist.init_process_group(backend="nccl")
    dist.barrier()
    log.info("Distributed process group initialized")

    torch.cuda.set_device(f"cuda:{get_local_rank()}")
    barrier()

    # Fill some configuration options.
    cfg.model.precision = cfg.precision
    cfg.device_train_batch_size = cfg.global_train_batch_size // get_world_size()
    assert cfg.device_train_batch_size is not None  # for mypy
    cfg.device_train_grad_accum = cfg.device_train_batch_size // cfg.device_train_microbatch_size

    log.info("Building data loader...")
    data_loader = build_train_dataloader(cfg)
    for i, batch in enumerate(data_loader):
        batch = move_to_device(batch, torch.device("cuda"))
        log.info(f"Batch {i}, size: ({batch['input_ids'].shape[0]}, {batch['input_ids'].shape[1]})")
        barrier()
        if i > 9:
            break

    log.info("Done!")


if __name__ == "__main__":
    prepare_cli_environment()
    try:
        yaml_path, args_list = sys.argv[1], sys.argv[2:]
    except IndexError:
        raise OlmoCliError(f"Usage: {sys.argv[0]} [CONFIG_PATH] [OPTIONS]")

    cfg = TrainConfig.load(yaml_path, [clean_opt(s) for s in args_list])
    main(cfg)
