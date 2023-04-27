from __future__ import annotations

from typing import List, Optional, Tuple, cast

import numpy as np
import torch
from torch.utils.data import Dataset

from ..aliases import PathOrStr

__all__ = ["MemMapDataset"]


class MemMapDataset(Dataset[torch.LongTensor]):
    """
    A PyTorch :class:`~torch.utils.data.Dataset` backed by one or more numpy memory-mapped arrays
    of token IDs. Token IDs are chunked together into contiguous blocks of ``chunk_size``
    to create instances.

    If the length of a memory-mapped array is not a multiple of ``chunk_size`` the
    remainder of the tokens will be ignored.

    No special tokens are added to the input IDs so it's assumed that if you want
    EOS tokens between documents, for example, those will already by in the memory-mapped array.

    :param paths: Paths to memory-mapped token arrays.
    :param chunk_size: The number of tokens to chunk together into a single instance.
        Generally this should correspond to your model's maximum input length.
    :param memmap_dtype: The numpy datatype of the memory-mapped array.
    """

    def __init__(self, *paths: PathOrStr, chunk_size: int = 1024, memmap_dtype=np.uint16):
        if not paths:
            raise ValueError("At least one path is required")
        self._memmap_paths = paths
        self._chunk_size = chunk_size
        self._mmaps: Optional[List[np.memmap]] = None
        self._mmap_offsets: Optional[List[Tuple[int, int]]] = None
        self._num_instances: Optional[int] = None
        self.dtype = memmap_dtype

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def max_seq_len(self) -> int:
        # For compatibility with composer's SpeedMonitor callback.
        return self.chunk_size

    @property
    def memmaps(self) -> List[np.memmap]:
        if self._mmaps is None:
            self._mmaps = []
            for path in self._memmap_paths:
                mmap = np.memmap(path, mode="r", dtype=self.dtype)
                self._mmaps.append(mmap)
        return self._mmaps

    @property
    def offsets(self) -> List[Tuple[int, int]]:
        if self._mmap_offsets is None:
            start_offset = 0
            self._mmap_offsets = []
            for mmap in self.memmaps:
                length = mmap.shape[0] // self._chunk_size
                end_offset = start_offset + length
                self._mmap_offsets.append((start_offset, end_offset))
                start_offset += length
        return self._mmap_offsets

    def __len__(self) -> int:
        if self._num_instances is None:
            self._num_instances = self.offsets[-1][1]
        return self._num_instances

    def __getitem__(self, index: int) -> torch.LongTensor:
        pos_index = index if index >= 0 else len(self) + index

        # The index of the memmap array within 'self.memmaps'
        memmap_index: Optional[int] = None
        # The 'index' relative to the corresponding memmap array.
        memmap_local_index: Optional[int] = None
        for i, (offset_start, offset_end) in enumerate(self.offsets):
            if offset_start <= pos_index < offset_end:
                memmap_index = i
                memmap_local_index = pos_index - offset_start

        if memmap_index is None or memmap_local_index is None:
            raise IndexError(f"{index} is out of bounds for dataset of size {len(self)}")

        memmap = self.memmaps[memmap_index]
        index_start = memmap_local_index * self._chunk_size
        index_stop = (memmap_local_index + 1) * self._chunk_size
        data = memmap[index_start:index_stop].astype(np.int_)
        return cast(torch.LongTensor, torch.tensor(data, dtype=torch.long))

    def __add__(self, other: MemMapDataset) -> MemMapDataset:
        """
        Concatenate one :class:`MemMapDataset` with another.
        """
        if not isinstance(other, MemMapDataset):
            raise NotImplementedError(f"Expected another MemMapDataset but got {type(other)}")
        return MemMapDataset(
            *(self._memmap_paths + other._memmap_paths), chunk_size=self._chunk_size, memmap_dtype=self.dtype
        )
