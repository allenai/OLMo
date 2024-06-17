import gc
import os
from typing import Optional, TypeVar

import torch
import torch.distributed as dist

T = TypeVar("T")


def seed_all(seed: int):
    """Seed all rng objects."""
    import random

    import numpy as np

    if seed < 0 or seed > 2**32 - 1:
        raise ValueError(f"Seed {seed} is invalid. It must be on [0; 2^32 - 1]")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.manual_seed may call manual_seed_all but calling it again here
    # to make sure it gets called at least once
    torch.cuda.manual_seed_all(seed)


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_node_rank() -> int:
    return int(os.environ.get("NODE_RANK") or (get_global_rank() - get_local_rank()) // get_local_world_size())


def get_world_size() -> int:
    if is_distributed():
        return dist.get_world_size()
    else:
        return 1


def get_local_world_size() -> int:
    return int(os.environ.get("LOCAL_WORLD_SIZE") or 1)


def get_global_rank() -> int:
    if is_distributed():
        return int(os.environ.get("RANK") or dist.get_rank())
    else:
        return 0


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK") or 0)


def get_fs_local_rank() -> int:
    """Get the local rank per filesystem, meaning that, regardless of the number of nodes,
    if all ranks share the same filesystem then `get_fs_local_rank()` will be equivalent to `get_global_rank()`,
    but if nodes do not share the same filesystem then `get_fs_local_rank()` will be equivalent to `get_local_rank()`.
    """
    if os.environ.get("OLMO_SHARED_FS"):
        return int(os.environ.get("FS_LOCAL_RANK") or get_global_rank())
    else:
        return int(os.environ.get("FS_LOCAL_RANK") or get_local_rank())


def move_to_device(o: T, device: torch.device) -> T:
    if isinstance(o, torch.Tensor):
        return o.to(device)  # type: ignore[return-value]
    elif isinstance(o, dict):
        return {k: move_to_device(v, device) for k, v in o.items()}  # type: ignore[return-value]
    elif isinstance(o, list):
        return [move_to_device(x, device) for x in o]  # type: ignore[return-value]
    elif isinstance(o, tuple):
        return tuple((move_to_device(x, device) for x in o))  # type: ignore[return-value]
    else:
        return o


def ensure_finite_(x: torch.Tensor, check_neg_inf: bool = True, check_pos_inf: bool = False):
    """
    Modify ``x`` in place to replace ``float("-inf")`` with the minimum value of the dtype when ``check_neg_inf``
    is ``True`` and to replace ``float("inf")`` with the maximum value of the dtype when ``check_pos_inf`` is ``True``.
    """
    if check_neg_inf:
        x.masked_fill_(x == float("-inf"), torch.finfo(x.dtype).min)
    if check_pos_inf:
        x.masked_fill_(x == float("inf"), torch.finfo(x.dtype).max)


def get_default_device() -> torch.device:
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def peak_gpu_memory(reset: bool = False) -> Optional[float]:
    """
    Get the peak GPU memory usage in MB across all ranks.
    Only rank 0 will get the final result.
    """
    if not torch.cuda.is_available():
        return None

    device = torch.device("cuda")
    peak_mb = torch.cuda.max_memory_allocated(device) / 1000000
    if is_distributed():
        peak_mb_tensor = torch.tensor(peak_mb, device=device)
        dist.reduce(peak_mb_tensor, 0, dist.ReduceOp.MAX)
        peak_mb = peak_mb_tensor.item()

    if reset:
        # Reset peak stats.
        torch.cuda.reset_max_memory_allocated(device)

    return peak_mb


V = TypeVar("V", bool, int, float)


def synchronize_value(value: V, device: torch.device) -> V:
    if dist.is_available() and dist.is_initialized():
        value_tensor = torch.tensor(value, device=device)
        dist.broadcast(value_tensor, 0)
        return value_tensor.item()  # type: ignore
    else:
        return value


def synchronize_flag(flag: bool, device: torch.device) -> bool:
    return synchronize_value(flag, device)


def gc_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
