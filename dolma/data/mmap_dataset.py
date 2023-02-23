from typing import Optional, cast

import numpy as np
import torch
from torch.utils.data import Dataset

from ..aliases import PathOrStr

__all__ = ["MMapDataset"]


class MMapDataset(Dataset[torch.LongTensor]):
    """
    A PyTorch :class:`~torch.utils.data.Dataset` backed by a numpy memory-mapped array
    of token IDs. Token IDs are chunked together into contiguous blocks of ``chunk_size``
    to create instances.

    If the length of the memory-mapped array is not a multiple of ``chunk_size`` the
    remainder tokens will be ignored.

    No special tokens are added to the input IDs so it's assumed that if you want
    EOS tokens between documents, for example, those will already by in the memory-mapped array.

    :param tokens_fname: The path of the memory-mapped tokens array.
    :param chunk_size: The number of tokens to chunk together into a single instance.
        Generally this should correspond to your model's maximum input length.
    :param dtype: The numpy datatype of the memory-mapped array.
    """

    def __init__(self, tokens_fname: PathOrStr, chunk_size: int = 1024, mmap_dtype=np.uint16):
        self._tokens_fname = tokens_fname
        self._chunk_size = chunk_size
        self._token_ids: Optional[np.memmap] = None
        self._num_instances: Optional[int] = None
        self.dtype = mmap_dtype

    @property
    def token_ids(self) -> np.memmap:
        if self._token_ids is None:
            self._token_ids = np.memmap(self._tokens_fname, mode="r", dtype=self.dtype)
        return self._token_ids

    def __len__(self) -> int:
        if self._num_instances is None:
            self._num_instances = self.token_ids.shape[0] // self._chunk_size
        return self._num_instances

    def __getitem__(self, index: int) -> torch.LongTensor:
        index_start = index * self._chunk_size
        index_stop = (index + 1) * self._chunk_size
        data = self.token_ids[index_start:index_stop].astype(np.int_)
        return cast(torch.LongTensor, torch.tensor(data, dtype=torch.long))
