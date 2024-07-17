import os
from pathlib import Path
from typing import Dict, Tuple

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from olmo.checkpoint import FullCheckpointer, LocalShardedCheckpointer
from olmo.config import OptimizerConfig, OptimizerType, TrainConfig
from olmo.optim import Optimizer, build_optimizer


def opt_at(opt, idx, key):
    return list(opt.state.values())[idx][key]


def _init_model_and_optim(config: TrainConfig) -> Tuple[FSDP, Optimizer]:
    model = FSDP(torch.nn.Linear(4, 4).cuda(dist.get_rank()), use_orig_params=True)
    optim = build_optimizer(config, model)
    model(torch.rand(4, 4).cuda(dist.get_rank())).sum().backward()
    optim.step()
    return model, optim


def _run_local_sharded_checkpointer_test(rank: int, world_size: int, tmp_path: Path):
    # Set up world pg
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)

    # Initialize the process group
    dist.init_process_group("cpu:gloo,cuda:nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    # Initialize model, optimizer, and checkpointer.
    train_config = TrainConfig(
        optimizer=OptimizerConfig(name=OptimizerType.adamw, learning_rate=0.1, weight_decay=0.1)
    )
    fsdp_model1, optim1 = _init_model_and_optim(train_config)
    checkpointer = LocalShardedCheckpointer(train_config)
    checkpoint_dir = tmp_path / "checkpoint"

    # Save checkpoint.
    checkpointer.save_checkpoint(checkpoint_dir, fsdp_model1, optim1, {})

    # Create a 2nd model and optimizer and load the checkpoint.
    fsdp_model2, optim2 = _init_model_and_optim(train_config)
    checkpointer.restore_checkpoint(checkpoint_dir, fsdp_model2, optim2)

    # Validate parameters and optimizer state are the same now.
    with FSDP.summon_full_params(fsdp_model1), FSDP.summon_full_params(fsdp_model2):
        torch.testing.assert_close(fsdp_model1.weight, fsdp_model2.weight)
        torch.testing.assert_close(fsdp_model1.bias, fsdp_model2.bias)
    torch.testing.assert_close(opt_at(optim1, 0, "exp_avg"), opt_at(optim2, 0, "exp_avg"))
    torch.testing.assert_close(opt_at(optim1, 0, "exp_avg_sq"), opt_at(optim2, 0, "exp_avg_sq"))

    # Now save a full unsharded checkpoint.
    full_checkpointer = FullCheckpointer(train_config)
    checkpoint_dir_full = tmp_path / "checkpoint-unsharded"
    full_checkpointer.save_checkpoint(checkpoint_dir_full, fsdp_model1, optim1, {})

    # Load the full unsharded checkpoint and unshard the local sharded checkpoint.
    full_model_state_dict, full_optim_state_dict = full_checkpointer.load_checkpoint(
        checkpoint_dir_full, device=torch.device("cuda")
    )
    unsharded_model_state_dict, unsharded_optim_state_dict, _ = checkpointer.unshard_checkpoint(
        checkpoint_dir, device=torch.device("cuda")
    )
    assert full_optim_state_dict is not None
    assert unsharded_optim_state_dict is not None

    # Validate the unsharded model state.
    assert unsharded_model_state_dict.keys() == full_model_state_dict.keys()
    for key in unsharded_model_state_dict.keys():
        unsharded_weight = unsharded_model_state_dict[key]
        original_weight = full_model_state_dict[key]
        torch.testing.assert_close(unsharded_weight, original_weight)

    fqn_to_id: Dict[str, int] = {}
    for group in unsharded_optim_state_dict["param_groups"]:
        for fqn, id in zip(group["param_names"], group["params"]):
            fqn = fqn.replace("_fsdp_wrapped_module.", "")
            fqn_to_id[fqn] = id

    # Validate the unsharded optim param groups.
    assert len(unsharded_optim_state_dict["param_groups"]) == len(full_optim_state_dict["param_groups"])
    for unsharded_group, full_group in zip(
        unsharded_optim_state_dict["param_groups"], full_optim_state_dict["param_groups"]
    ):
        assert unsharded_group.keys() == full_group.keys()
        for key in unsharded_group.keys():
            if key == "param_names":
                assert unsharded_group[key] == [n.replace("_fsdp_wrapped_module.", "") for n in full_group[key]]
            elif key == "params" and isinstance(full_group[key][0], str):
                # These are FQNs instead of IDs.
                assert unsharded_group[key] == [fqn_to_id[fqn] for fqn in full_group[key]]
            else:
                assert unsharded_group[key] == full_group[key], key

    # Validate the unsharded optim state tensors.
    if isinstance(next(iter(full_optim_state_dict["state"].keys())), str):
        full_optim_state_dict["state"] = {fqn_to_id[fqn]: s for fqn, s in full_optim_state_dict["state"].items()}
    assert unsharded_optim_state_dict["state"].keys() == full_optim_state_dict["state"].keys()
    for id in unsharded_optim_state_dict["state"].keys():
        unsharded_state, full_state = unsharded_optim_state_dict["state"][id], full_optim_state_dict["state"][id]
        assert unsharded_state.keys() == full_state.keys()
        for key in unsharded_state.keys():
            torch.testing.assert_close(unsharded_state[key], full_state[key])

    # Shut down world pg
    dist.destroy_process_group()


@pytest.mark.gpu
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires 2 or more CUDA devices")
def test_local_sharded_checkpointer(tmp_path: Path):
    world_size = torch.cuda.device_count()
    mp.spawn(
        _run_local_sharded_checkpointer_test,
        args=(world_size, tmp_path),
        nprocs=world_size,
        join=True,
    )
