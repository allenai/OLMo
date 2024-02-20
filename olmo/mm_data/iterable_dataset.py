from concurrent.futures import ThreadPoolExecutor
from typing import List, Any, Dict, Optional

import torch

from olmo.mm_data.data_store import ExampleReader
from olmo.mm_data.sequence_index import get_idx_file, SequenceIndex
from olmo.torch_util import get_global_rank, get_world_size


class MMIterableDataset(torch.utils.data.IterableDataset[Dict[str, Any]]):
  def __init__(
      self,
      reader: ExampleReader,
      idx_dir: str,
      seeds: List[int],
      sequence_length: int,
      global_batch_size: int=1,
      start_index: int = 0,
      drop_last: bool = False,
      max_examples: Optional[int] = None,
      world_size: Optional[int] = None,
      rank: Optional[int] = None,
      num_threads: Optional[int] = None,
      segment_ids=False,
  ):
    self.sequence_length = sequence_length
    self.segment_ids = segment_ids
    self.seeds = seeds
    self.idx_dir = idx_dir
    self.world_size = world_size if world_size is not None else get_world_size()
    self.rank = rank if rank is not None else get_global_rank()
    self.start_index = start_index
    self.reader = reader
    self.max_examples = max_examples
    self.drop_last = drop_last

    self.num_threads = num_threads
    assert global_batch_size % self.world_size == 0
    self.device_batch_size = global_batch_size // self.world_size

    self._seed_idx = -1
    self._index = None
    self.reshuffle()

  def reshuffle(self):
    self._seed_idx += 1
    if self._seed_idx >= len(self.seeds):
      raise ValueError()
    index_file = get_idx_file(
      self.idx_dir, self.reader.data_files.values(), self.reader.image_sizer,
      self.sequence_length, self.seeds[self._seed_idx])
    self._index = SequenceIndex(index_file)

  def __iter__(self):
    global_end_sequence = self._index.num_sequences

    # pad or truncate to get a number of sequences divisible by world size
    remainder = global_end_sequence % self.world_size
    if remainder:
      if self.drop_last:
        global_end_sequence -= remainder
        if global_end_sequence == remainder:
          raise ValueError("Entire dataset was dropped")
      else:
        global_end_sequence += (self.world_size - remainder)

    if self.max_examples:
      assert self.max_examples % self.world_size == 0
      global_end_sequence = min(global_end_sequence, self.max_examples)

    if self.start_index:
      assert self.start_index % self.world_size == 0

    if hasattr(self, "worker_info"):  # for testing
      worker_info = self.worker_info
    else:
      worker_info = torch.utils.data.get_worker_info()

    # Compute global rank/worker count across all devices
    if worker_info is not None:
      global_workers = worker_info.num_workers*self.world_size
      worker_rank = self.rank + worker_info.id * self.world_size
    else:
      global_workers = self.world_size
      worker_rank = self.rank

    # Each worker reads one device batch and the skips examples other workers will read
    block_step = self.device_batch_size * (global_workers - 1)
    start = self.start_index + worker_rank * self.device_batch_size
    it = self._index.iter_blocks(
        start, global_end_sequence, block_size=self.device_batch_size, block_step=block_step)

    num_threads = self.num_threads
    if num_threads == 0:
      for sequence in it:
        yield self.reader.read_ranges(sequence, self.sequence_length, self.segment_ids)
    elif num_threads is None:
      raise NotImplementedError("default thread count")
    else:
      # TODO is there a reason not to use? IterableDataset doesn't
      with ThreadPoolExecutor(max_workers=num_threads) as pool:
        def _read(_seq):
          return self.reader.read_ranges(_seq, self.sequence_length, self.segment_ids)
        return pool.map(_read, it)

