from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

from olmo.exceptions import OlmoEnvironmentError

from ..aliases import PathOrStr
from ..util import _get_s3_client, file_size, get_bytes_range

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
    :param include_instance_metadata: If ``True`` (the default), each instance returned from `__getitem__` will
        include the metadata from its source.
    """

    def __init__(
        self,
        *paths: PathOrStr,
        chunk_size: int = 1024,
        memmap_dtype=np.uint16,
        metadata: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = None,
        include_instance_metadata: bool = True,
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
        self._include_instance_metadata = include_instance_metadata

    @property
    def chunk_size(self) -> int:
        return self._chunk_size

    @property
    def max_seq_len(self) -> int:
        # For compatibility with composer's SpeedMonitor callback.
        return self.chunk_size

    @property
    def offsets(self) -> List[Tuple[int, int]]:
        # Create the global S3 client up front to work around a threading issue in boto.
        _get_s3_client("s3")
        try:
            _get_s3_client("r2")
        except OlmoEnvironmentError:
            # R2 might not be needed, so ignore this error. We will get an error
            # later if R2 is needed.
            pass

        if self._mmap_offsets is None:
            import concurrent.futures

            self._mmap_offsets = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                for path in self._memmap_paths:
                    future = executor.submit(self._get_file_length, path)
                    futures.append(future)

                path_to_length: Dict[PathOrStr, int] = {}
                for future in concurrent.futures.as_completed(futures):
                    path, length = future.result()
                    path_to_length[path] = length

            start_offset = 0
            for path in self._memmap_paths:
                length = path_to_length[path]
                end_offset = start_offset + length
                self._mmap_offsets.append((start_offset, end_offset))
                start_offset += length
        return self._mmap_offsets

    def _read_chunk_from_memmap(self, path: PathOrStr, index: int) -> torch.Tensor:
        bytes_start = index * self._item_size * self._chunk_size
        num_bytes = self._item_size * self._chunk_size
        buffer = get_bytes_range(path, bytes_start, num_bytes)
        array = np.frombuffer(buffer, dtype=self.dtype)
        return torch.tensor(array.astype(np.int_), dtype=torch.long)

    def _get_file_length(self, path) -> Tuple[PathOrStr, int]:
        return path, file_size(path) // (self._item_size * self._chunk_size)

    def __len__(self) -> int:
        if self._num_instances is None:
            self._num_instances = self.offsets[-1][1]
        return self._num_instances

    def __getitem__(self, index: int) -> Dict[str, Any]:
        index = int(index)  # in case this is a numpy int type.
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
        out: Dict[str, Any] = {"input_ids": input_ids}
        if self._include_instance_metadata:
            metadata = self._metadata[memmap_index]
            out["metadata"] = deepcopy(metadata)
        return out

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
