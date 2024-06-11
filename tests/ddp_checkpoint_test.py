# 2 tests: ddp checkpointing local, ddp checkpointing s3
# in both cases, init model, optim state, trainer state, do 2 steps, dump checkpoint, init new model and optim state, check they are different, load the checkpoint, make sure the loaded checkpoint into new model and optim is same as model at step 2
import os
from pathlib import Path
from typing import Dict, Tuple

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from olmo.checkpoint import FullCheckpointer
from olmo.config import OptimizerConfig, OptimizerType, TrainConfig
from olmo.optim import Optimizer, build_optimizer


def _init_model_and_optim(config: TrainConfig) -> Tuple[DDP, Optimizer]:
    model = DDP(torch.nn.Linear(4, 4).cuda(dist.get_rank()), find_unused_parameters=False)
    optim = build_optimizer(config, model)

    # update model
    model(torch.rand(4, 4)).sum().backward()
    optim.step()

    return model, optim


def _run_local_unsharded_checkpointer(rank: int, world_size: int, tmp_path: Path):
    # Set up world pg
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)

    # Initialize the process group
    dist.init_process_group("cpu:gloo,cuda:nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # init train config
    train_config = TrainConfig(
        optimizer=OptimizerConfig(name=OptimizerType.adamw, learning_rate=0.1, weight_decay=0.1)
    )

    # init model and optim
    model_1, optim_1 = _init_model_and_optim(train_config)

    # checkpoint
    checkpointer = FullCheckpointer(train_config)
    checkpoint_dir = tmp_path / "checkpoint"
    checkpointer.save_checkpoint(checkpoint_dir, model_1, optim_1, {})

    # init new model and optim state
    # assert the model weights, optim states are different
    model_2, optim_2 = _init_model_and_optim(train_config)

    # load from checkpoint into new model
    checkpointer.restore_checkpoint(checkpoint_dir, model_2, optim_2)

    # assert loaded model and optim state are same

    # Shut down world pg
    dist.destroy_process_group()


@pytest.mark.gpu
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires 2 or more CUDA devices")
def test_local_sharded_checkpointer(tmp_path: Path):
    world_size = torch.cuda.device_count()
    mp.spawn(
        _run_local_unsharded_checkpointer,
        args=(world_size, tmp_path),
        nprocs=world_size,
        join=True,
    )
