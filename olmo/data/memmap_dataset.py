from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from cached_path import cached_path
from torch.utils.data import Dataset

from ..aliases import PathOrStr
from ..util import file_size

__all__ = ["MemMapDataset"]


class MemMapDataset(Dataset[Dict[str, Any]]):
    """
    A PyTorch :class:`~torch.utils.data.Dataset` backed by one or more numpy memory-mapped arrays
    of token IDs. Token IDs are chunked together into contiguous blocks of ``chunk_size``
    to create instances.

    If the length of a memory-mapped array is not a multiple of ``chunk_size`` the
    remainder of the tokens will be ignored.

    No special tokens are added to the input IDs so it's assumed that if you want
    EOS tokens between documents, for example, those will already be in the memory-mapped array.

    :param paths: Paths to memory-mapped token arrays.
    :param chunk_size: The number of tokens to chunk together into a single instance.
        Generally this should correspond to your model's maximum input length.
    :param memmap_dtype: The numpy datatype of the memory-mapped array.
    :param metadata: Metadata to add to each item. This should be a dictionary or a list of dictionaries
        with the same number of items as there are paths.
    """

    def __init__(
        self,
        *paths: PathOrStr,
        chunk_size: int = 1024,
        memmap_dtype=np.uint16,
        metadata: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None,
    ):
        if not paths:
            raise ValueError("At least one path is required")
        if isinstance(metadata, list):
            if len(metadata) != len(paths):
                raise ValueError("'metadata' should have the same length as the number of file paths")
        else:
            metadata = [metadata or {}] * len(paths)
        self._memmap_paths = paths
        self._metadata = metadata
        self._chunk_size = chunk_size
        self._mmap_offsets: Optional[List[Tuple[int, int]]] = None
        self._num_instances: Optional[int] = None
        self.dtype = memmap_dtype
        self._item_size = self.dtype(0).itemsize

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def max_seq_len(self) -> int:
        # For compatibility with composer's SpeedMonitor callback.
        return self.chunk_size

    @property
    def offsets(self) -> List[Tuple[int, int]]:
        if self._mmap_offsets is None:
            start_offset = 0
            self._mmap_offsets = []
            for path in self._memmap_paths:
                length = file_size(path) // (self._item_size * self._chunk_size)
                end_offset = start_offset + length
                self._mmap_offsets.append((start_offset, end_offset))
                start_offset += length
        return self._mmap_offsets

    def _read_chunk_from_memmap(self, path: PathOrStr, index: int) -> torch.Tensor:
        index_start = index * self._item_size * self._chunk_size
        with open(cached_path(path), "rb") as f:
            f.seek(index_start)
            buffer = f.read(self._item_size * self._chunk_size)
            array = np.frombuffer(buffer, dtype=self.dtype)
        return torch.tensor(array.astype(np.int_), dtype=torch.long)

    def __len__(self) -> int:
        if self._num_instances is None:
            self._num_instances = self.offsets[-1][1]
        return self._num_instances

    def __getitem__(self, index: int) -> Dict[str, Any]:
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

        # Read the data from file.
        input_ids = self._read_chunk_from_memmap(self._memmap_paths[memmap_index], memmap_local_index)
        metadata = self._metadata[memmap_index]
        return {"input_ids": input_ids, "metadata": deepcopy(metadata)}

    def __add__(self, other: MemMapDataset) -> MemMapDataset:
        """
        Concatenate one :class:`MemMapDataset` with another.
        """
        if not isinstance(other, MemMapDataset):
            raise NotImplementedError(f"Expected another MemMapDataset but got {type(other)}")
        return MemMapDataset(
            *(self._memmap_paths + other._memmap_paths),
            chunk_size=self._chunk_size,
            memmap_dtype=self.dtype,
            metadata=self._metadata + other._metadata,
        )
