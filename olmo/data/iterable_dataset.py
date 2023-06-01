import math
from typing import Iterator, Optional, Sequence, TypeVar

import torch
import torch.distributed as dist
import torch.utils.data

from ..util import global_rank

__all__ = ["IterableDataset"]


T = TypeVar("T")


class IterableDataset(torch.utils.data.IterableDataset[T]):
    """
    Adapted from PyTorch's DistributedSampler, this wraps a Dataset or arbitrary sequence
    as an IterableDataset that can be deterministically restarted at any point by setting `start_index`,
    which should be a multiple of your global batch size.
    Similarly `max_examples`, if set, should be a multiple of global batch size.
    """

    def __init__(
        self,
        dataset: Sequence[T],
        *,
        seed: int = 0,
        start_index: int = 0,
        max_examples: Optional[int] = None,
        shuffle: bool = True,
        drop_last: bool = False,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
    ):
        self.dataset = dataset
        self.seed = seed
        self.start_index = start_index
        self.max_examples = max_examples
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rank = rank if rank is not None else global_rank()
        self.world_size = (
            world_size
            if world_size is not None
            else (dist.get_world_size() if (dist.is_available() and dist.is_initialized()) else 1)
        )
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.world_size != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible by world size.
            # This is to ensure each rank receives the same amount of data.
            num_samples = math.ceil(
                (len(self.dataset) - self.world_size) / self.world_size  # type: ignore[arg-type]
            )
        else:
            num_samples = math.ceil(len(self.dataset) / self.world_size)  # type: ignore[arg-type]
        self.total_size = num_samples * self.world_size

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

        # Truncate to max_examples.
        if self.max_examples is not None:
            assert self.max_examples % self.world_size == 0
            indices = indices[: self.max_examples]

        # Start at the specified index.
        if self.start_index > 0:
            assert self.start_index % self.world_size == 0
            indices = indices[self.start_index :]

        # Slice indices by rank to avoid duplicates.
        indices = indices[self.rank : self.total_size : self.world_size]

        # Lastly, slice the indices by data loader worker rank to avoid duplicates.
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            indices = indices[worker_info.id :: worker_info.num_workers]

        return (self.dataset[idx] for idx in indices)
