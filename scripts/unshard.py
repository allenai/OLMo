import logging
import shutil
import sys
from pathlib import Path
from typing import Any, Union, List, Dict, Tuple, cast
from functools import reduce

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor
from torch.distributed import _remote_device
from torch.distributed._shard.metadata import ShardMetadata
from torch.distributed._shard.sharded_tensor import ShardedTensor

logger = logging.getLogger(__name__)


def gather(shards: List[ShardedTensor]) -> Tensor:
    world_size = len(shards)
    shard0_md = shards[0].metadata()
    # Make sure all shards agree on the metadata
    assert all(shard.metadata() == shard0_md for shard in shards)
    # Make sure the nth shard expects to be the nth shard.
    assert all(shard_md.placement.rank() == rank for rank, shard_md in enumerate(shard0_md.shards_metadata))

    def shard_size(shard_md):
        return reduce((lambda x, y: x * y), shard_md.shard_sizes)  # type: ignore[attr-defined]
    rank_sizes = [0 for _ in range(world_size)]
    max_rank_size = 0
    shard_placement: Dict[ShardMetadata, Tuple[int, int]] = {}
    for shard_md in shard0_md.shards_metadata:
        shard_rank = cast(_remote_device, shard_md.placement).rank()
        assert shard_rank is not None

        shard_placement[shard_md] = (shard_rank, rank_sizes[shard_rank])
        rank_sizes[shard_rank] += shard_size(shard_md)
        max_rank_size = max(max_rank_size, rank_sizes[shard_rank])

    gather_list: List[Tensor] = [torch.empty((max_rank_size,)) for _ in range(world_size)]

    datas = []
    with torch.no_grad():
        for shard in shards:
            data = torch.empty(max_rank_size)

            for local_shard in shard.local_shards():
                src = local_shard.tensor.flatten()
                shard_offset = shard_placement[local_shard.metadata][1]
                data[shard_offset: shard_offset + src.numel()].copy_(src)

            datas.append(data)

    # torch.gather in a nutshell
    for rank, data in enumerate(datas):
        gather_list[rank].copy_(data)

    full_size = shard0_md.size
    out = torch.empty(*full_size, dtype=shard0_md.tensor_properties.dtype)
    dims = len(full_size)
    for shard_md in shard0_md.shards_metadata:
        rank, rank_offset = shard_placement[shard_md]
        tensor = gather_list[rank]
        tensor = tensor[rank_offset : rank_offset + shard_size(shard_md)]
        tensor = tensor.view(shard_md.shard_sizes)

        out_narrow_view = out
        for dim in range(dims):
            out_narrow_view = out_narrow_view.narrow(
                dim,
                shard_md.shard_offsets[dim],
                shard_md.shard_sizes[dim],
            )

        out_narrow_view.copy_(tensor)

    return out


def objects_are_equal(a: Any, b: Any) -> bool:
    if type(a) != type(b):
        return False
    if isinstance(a, ndarray):
        return np.array_equal(a, b)
    elif isinstance(a, Tensor):
        return torch.equal(a, b)
    else:
        return a == b


def unshard(input_dir: Union[str, Path], output_dir: Union[str, Path]) -> None:
    if isinstance(input_dir, str):
        input_dir = Path(input_dir)
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Monkeypatch torch's ShardedTensor, so we can unpickle without having torch.distributed set up.
    def _rebuild_from_type_v2_monkey(func, new_type, args, state):
        ret = func(*args)
        if type(ret) is not new_type:
            ret = ret.as_subclass(new_type)

        # Shortcut the construction of ShardedTensor
        # This is in the top 5 of my worst hacks.
        if isinstance(ret, ShardedTensor):
            ret._local_shards, ret._metadata, pg_state, ret._sharding_spec, ret._init_rrefs = state
            return ret

        # The rest of this function ought to be in the top 5 of somebody else's worst hacks.
        # Tensor does define __setstate__ even though it doesn't define
        # __getstate__. So only use __setstate__ if it is NOT the one defined
        # on Tensor
        if (
            getattr(ret.__class__, "__setstate__", Tensor.__setstate__)
            is not Tensor.__setstate__
        ):
            ret.__setstate__(state)
        else:
            ret = torch._utils._set_obj_state(ret, state)
        return ret
    torch._tensor._rebuild_from_type_v2 = _rebuild_from_type_v2_monkey

    shards_dict = {}
    for shard_name in input_dir.glob("rank*.pt"):
        logger.info("Loading %s ...", shard_name)
        shard_number = int(shard_name.name[4:-3])  # shard names look like "rankXX.pt"
        shards_dict[shard_number] = torch.load(shard_name, map_location="cpu")
    shards = [None] * len(shards_dict)
    for rank, shard in shards_dict.items():
        shards[rank] = shard
    assert all(shard is not None for shard in shards)
    del shards_dict

    logger.info("Unsharding from %d shards ...", len(shards))
    def unshard_object(os: List[Any]) -> Any:
        rank0_item = os[0]
        assert all(type(o) == type(rank0_item) for o in os)
        if isinstance(rank0_item, str):
            assert all(o == rank0_item for o in os)
            return rank0_item
        elif isinstance(rank0_item, (list, tuple, set)):
            assert all(len(o) == len(rank0_item) for o in os)
            return rank0_item.__class__(unshard_object(o) for o in zip(*os))
        elif isinstance(rank0_item, dict):
            assert all(o.keys() == rank0_item.keys() for o in os)
            return {key: unshard_object([o[key] for o in os]) for key in rank0_item.keys()}
        elif isinstance(rank0_item, ShardedTensor):
            return gather(os)
        else:
            assert all(objects_are_equal(o, rank0_item) for o in os)
            return rank0_item

    unsharded_state_dict = unshard_object(shards)
    # At this point in time we need 2x memory :-(
    del shards

    # model
    model_output = str(output_dir / "model.pt")
    logger.info("Saving model state to %s", model_output)
    model_state_dict = unsharded_state_dict.pop("model")
    torch.save(model_state_dict, model_output)
    del model_state_dict

    # optimizer
    optim_output = str(output_dir / "optim.pt")
    logger.info("Saving optimizer state to %s", optim_output)
    optim_state_dict = unsharded_state_dict.pop("optim")
    torch.save(optim_state_dict, optim_output)
    del optim_state_dict

    # whatever is left
    other_output = str(output_dir / "other.pt")
    logger.info("Saving everything else to %s", other_output)
    torch.save(unsharded_state_dict, other_output)
    del unsharded_state_dict

    logger.info("Copying config.yaml to %s", output_dir)
    shutil.copy(input_dir / "config.yaml", output_dir)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.stderr.write("Usage: unshard.py <input dir> <output dir>")
        sys.exit(1)
    else:
        logging.basicConfig(level=logging.INFO)
        unshard(sys.argv[1], sys.argv[2])
