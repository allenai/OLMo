import dataclasses
import logging
import math
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.utils.data

from ..aliases import PathOrStr
from ..torch_util import barrier, get_fs_local_rank, get_global_rank, get_world_size
from ..util import roundrobin, threaded_generator

__all__ = ["IterableDataset"]

log = logging.getLogger(__name__)


class IterableDataset(torch.utils.data.IterableDataset[Dict[str, Any]]):
    """
    Adapted from PyTorch's DistributedSampler, this wraps a Dataset or arbitrary sequence
    as an IterableDataset that can be deterministically restarted at any point by setting `start_index`,
    which should be a multiple of your global batch size.
    Similarly `max_examples`, if set, should be a multiple of global batch size.
    """

    def __init__(
        self,
        dataset: Union[Sequence[List[int]], Sequence[torch.Tensor], Sequence[Dict[str, Any]]],
        global_batch_size: int,
        *,
        seed: int = 0,
        epoch: int = 0,
        start_index: int = 0,
        max_examples: Optional[int] = None,
        shuffle: bool = True,
        drop_last: bool = False,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        fs_local_rank: Optional[int] = None,
        work_dir: Optional[PathOrStr] = None,
        num_threads: Optional[int] = None,
    ):
        self.dataset = dataset
        self.seed = seed
        self.epoch = epoch
        self.start_index = start_index
        self.max_examples = max_examples
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rank = rank if rank is not None else get_global_rank()
        self.fs_local_rank = fs_local_rank if fs_local_rank is not None else get_fs_local_rank()
        self.world_size = world_size if world_size is not None else get_world_size()
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
        self.num_threads = num_threads
        assert global_batch_size % self.world_size == 0
        self.device_batch_size = global_batch_size // self.world_size
        self.global_indices_file: Optional[Path] = None
        self.work_dir = work_dir

        if work_dir is not None:
            self._build_and_save_global_indices()

    def _build_and_save_global_indices(self):
        assert self.work_dir is not None
        self.global_indices_file = Path(self.work_dir) / "global_indices.npy"
        if self.fs_local_rank == 0:
            log.info("Saving global data order indices...")
            self.global_indices_file.parent.mkdir(parents=True, exist_ok=True)
            global_indices = self._build_global_indices()
            global_indices_mmap = np.memmap(
                self.global_indices_file, dtype=np.uint32, mode="w+", shape=(len(global_indices),)
            )
            global_indices_mmap[:] = global_indices
            global_indices_mmap.flush()
            del global_indices_mmap
            log.info("Global data order indices saved to '%s'", self.global_indices_file)
        barrier()

    def _build_global_indices(self) -> np.ndarray:
        assert len(self.dataset) < np.iinfo(np.uint32).max
        indices = np.arange(len(self.dataset), dtype=np.uint32)
        if self.shuffle:
            # Deterministically shuffle based on epoch and seed
            # Torch built-in randomness is not very random, so we use numpy.
            rng = np.random.Generator(np.random.PCG64(seed=self.seed + self.epoch))
            rng.shuffle(indices)

        if not self.drop_last:
            # Add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            arrays_to_concatenate = [indices]
            while padding_size > 0:
                array_to_concatenate = indices[: min(padding_size, len(indices))]
                arrays_to_concatenate.append(array_to_concatenate)
                padding_size -= len(array_to_concatenate)
                del array_to_concatenate
            indices = np.concatenate(arrays_to_concatenate)
        else:
            # Remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size
        return indices

    def get_global_indices(self) -> np.ndarray:
        if self.global_indices_file is not None:
            return np.memmap(self.global_indices_file, mode="r", dtype=np.uint32)  # type: ignore
        else:
            return self._build_global_indices()

    def reshuffle(self, epoch: int):
        self.epoch = epoch
        if self.work_dir is not None:
            self._build_and_save_global_indices()

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        indices = self.get_global_indices()

        # Truncate to max_examples.
        if self.max_examples is not None:
            assert self.max_examples % self.world_size == 0
            indices = indices[: self.max_examples]

        # Start at the specified index.
        if self.start_index > 0:
            #  assert self.start_index % self.world_size == 0
            indices = indices[self.start_index :]

        # Slice indices by rank to avoid duplicates.
        indices = indices[self.rank : self.total_size : self.world_size]

        # Separate from data loading workers (which use multiprocessing), we also have the option
        # to use multi-threading (within workers).
        num_threads = self.num_threads

        # Slice the indices by data loader worker rank to avoid duplicates.
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Note that each data loading worker gathers a whole batch at a time, and the workers
            # are called round-robin by rank. So to slice these up in a way that preserves order, regardless
            # of the number of workers, we should give worker 0 the first chunk of `device_batch_size` indices,
            # worker 1 the 2nd chunk of `device_train_batch_size` indices, etc...
            truncated_size = self.device_batch_size * (len(indices) // self.device_batch_size)
            left_overs = indices[truncated_size + worker_info.id :: worker_info.num_workers]
            indices = (
                indices[:truncated_size]
                .reshape((-1, self.device_batch_size))[worker_info.id :: worker_info.num_workers]  # type: ignore
                .reshape((-1,))
            )
            indices = np.concatenate([indices, left_overs])
        elif num_threads is None:
            # If `num_threads` hasn't been specified and we're not using multiprocessing we'll try to guess
            # a good number of threads.
            num_threads = 4

        # Finally, potentially slice by threads.
        if num_threads:
            # In order to stay ahead of training the total queue size (sum across all threads)
            # should be bigger than the batch size.
            queue_size = math.ceil(self.device_batch_size * 2 / num_threads)

            thread_generators = []
            for i in range(num_threads):
                generator = (self._get_dataset_item(int(idx)) for idx in indices[i::num_threads])
                thread_generators.append(
                    threaded_generator(generator, maxsize=queue_size, thread_name=f"data thread {i}")
                )

            return (x for x in roundrobin(*thread_generators))
        else:
            return (self._get_dataset_item(int(idx)) for idx in indices)

    def _get_dataset_item(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[idx]
        if isinstance(item, dict):
            return dict(**item, index=idx)
        elif dataclasses.is_dataclass(item):
            return dict(**dataclasses.asdict(item), index=idx)  # type: ignore
        else:
            return {"input_ids": item, "index": idx}
