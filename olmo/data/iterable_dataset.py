import math
from typing import Iterator, Optional, Sequence, TypeVar

import torch
import torch.distributed as dist
import torch.utils.data

from ..util import global_rank

__all__ = ["IterableDataset"]


T = TypeVar("T")


class IterableDataset(torch.utils.data.IterableDataset[T]):
    def __init__(
        self,
        dataset: Sequence[T],
        *,
        seed: int = 0,
        start_step: int = 0,
        max_steps: Optional[int] = None,
        shuffle: bool = True,
        drop_last: bool = False
    ):
        self.dataset = dataset
        self.seed = seed
        self.start_step = start_step
        self.max_steps = max_steps
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rank = global_rank()
        self.world_size = dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.world_size != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.world_size) / self.world_size  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.world_size)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.world_size

    def __iter__(self) -> Iterator[T]:
        if self.shuffle:
            # Deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # Add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # Remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # Slice indices by rank.
        indices = indices[self.rank : self.total_size : self.world_size]

        if self.max_steps is not None:
            indices = indices[: self.max_steps]
        if self.start_step > 0:
            indices = indices[self.start_step :]

        # Lastly, slice by data loader worker rank.
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            indices = indices[worker_info.id :: worker_info.num_workers]

        return (self.dataset[idx] for idx in indices)
