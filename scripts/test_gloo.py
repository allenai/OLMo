"""Run this script with 'torchrun'."""

import gzip
import logging
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Optional, TextIO

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb
from packaging import version
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.nn.parallel import DistributedDataParallel as DDP

from olmo.config import (
    CheckpointType,
    DDPGradSyncMode,
    DistributedStrategy,
    TrainConfig,
)
from olmo.data import build_train_dataloader
from olmo.eval import build_evaluators
from olmo.exceptions import OLMoCliError, OLMoConfigurationError
from olmo.model import OLMo
from olmo.optim import BoltOnWarmupScheduler, build_optimizer, build_scheduler
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
from olmo.train import Trainer
from olmo.util import (
    add_cached_path_clients,
    clean_opt,
    find_latest_checkpoint,
    log_extra_field,
    prepare_cli_environment,
)

log = logging.getLogger("train")


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError as e:
        print(f"failed to set multiprocessing start method: {e}")
    log.info(f"Multiprocessing start method set to '{mp.get_start_method()}'")

    # Set CUDA device.
    torch.cuda.set_device(f"cuda:{get_local_rank()}")

    # Initialize process group.
    device_as_string = f"cuda:{get_local_rank()}"
    torch.cuda.set_device(
        device_as_string
    )  # Set this early to prevent GPU 0 from picking up a bunch of tensors it shouldn't have.
    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl", timeout=timedelta(minutes=1440), device_id=torch.device(device_as_string)
    )
    log.info("Process group initialized")

    # rank = get_global_rank()
    # input = torch.arange(4) + rank * 4
    # input = list(input.chunk(4))
    # output = list(torch.empty([4], dtype=torch.int64).chunk(4))
    # dist.all_to_all(output, input)
    # print(f"Rank {get_global_rank()}, output: {output}")

    rank = get_global_rank()
    input = torch.arange(16) + rank * 16
    output = torch.empty([16], dtype=torch.int64)
    dist.all_to_all_single(output, input)
    print(f"Rank {get_global_rank()}, output: {output}")
