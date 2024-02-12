import random
from typing import Any, Dict, Iterator, Optional

import torch
import torch.utils.data

from ..torch_util import get_fs_local_rank, get_global_rank, get_world_size

__all__ = ["MultiModalIterableDataset"]


class MultiModalIterableDataset(torch.utils.data.IterableDataset[Dict[str, Any]]):
    def __init__(
        self,
        *,
        pad_token_id: int,
        max_sequence_length: int,
        vocab_size: int,
        patch_width: int,
        patch_height: int,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
        fs_local_rank: Optional[int] = None,
    ):
        self.pad_token_id = pad_token_id
        self.max_sequence_length = max_sequence_length
        self.vocab_size = vocab_size
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.rank = rank if rank is not None else get_global_rank()
        self.fs_local_rank = fs_local_rank if fs_local_rank is not None else get_fs_local_rank()
        self.world_size = world_size if world_size is not None else get_world_size()

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        index = self.rank
        while True:
            # Generate mock input IDs.
            input_ids = torch.randint(0, self.vocab_size, (self.max_sequence_length,))
            # Make sure there are no padding tokens so far.
            input_ids.masked_fill_(input_ids == self.pad_token_id, self.pad_token_id + 1)
            # Determine where to place image patches.
            image_offsets = torch.tensor(
                sorted(random.sample(range(self.max_sequence_length), random.randint(1, 5)))
            )
            # Mask out patch location in input IDs.
            input_ids.index_fill_(0, image_offsets, self.pad_token_id)
            # Generate mock image patches.
            image_patches = torch.rand(len(image_offsets), self.patch_width, self.patch_height, 3)
            yield {
                "index": index,
                "input_ids": input_ids,
                "label_mask": input_ids != self.pad_token_id,
                "image_offsets": image_offsets,
                "image_patches": image_patches,
            }
            index += self.world_size
