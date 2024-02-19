import math
from typing import List, Any, Dict, Optional

import torch

from olmo.mm_data.data_store import ExampleReader
from olmo.mm_data.sequence_index import get_idx_file, SequenceIndex
from olmo.torch_util import get_global_rank, get_world_size


class MMDatasetIterator(torch.utils.data.IterableDataset[Dict[str, Any]]):
  def __init__(
      self,
      reader: ExampleReader,
      idx_dir: str,
      seeds: List[int],
      sequence_length: int,
      start_index: int = 0,
      drop_last: bool = False,
      max_examples: Optional[int] = None,
      world_size: Optional[int] = None,
      rank: Optional[int] = None,
      segment_ids=False,
  ):
    self._seed_idx = 0
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

    self._index = None
    self._start_sequence = None
    self._end_sequence = None

  def reshuffle(self):
    self._seed_idx += 1
    if self._seed_idx >= len(self.seeds):
      raise ValueError()
    self._start_sequence = None
    self._index = None

  def init_for_seed(self):
    if self._start_sequence is not None:
      return
    self._n_pad = 0
    index_file = get_idx_file(
      self.idx_dir, self.reader.data_files.values(), self.reader.image_sizer,
      self.sequence_length, self.seeds[self._seed_idx])
    self._index = SequenceIndex(index_file)
    num_sequences = self._index.num_sequences
    if self.max_examples:
      num_sequences = min(num_sequences, self.max_examples)

    num_samples = num_sequences // self.world_size
    remainder = num_sequences % self.world_size

    if remainder and not self.drop_last:
      raise NotImplementedError("Padding")

    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
      raise NotImplementedError("Multiple workers")
    self._start_sequence = self.rank*num_samples + self.start_index
    self._end_sequence = self._start_sequence + num_samples - self.start_index
    assert self._end_sequence <= num_sequences

  def __iter__(self):
    self.init_for_seed()
    for sequence in self._index.iter_from(self._start_sequence, self._end_sequence):
      yield self.reader.read_ranges(
        sequence, self.sequence_length, self.segment_ids)

