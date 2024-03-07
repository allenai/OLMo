import io
import itertools
import sys
from typing import Tuple, Iterator, List, Union, Optional

import numpy as np
from transformers.utils import logging

from olmo.aliases import PathOrStr
from olmo.mm_data.data_iteration import DOC_SEQUENCE_DTYPE
from olmo.mm_data.image_token_size import ImageTokenSizer
from olmo.util import file_size, get_bytes_range

logger = logging.get_logger(__name__)


def get_idx_file(image_sizer: Optional[ImageTokenSizer], sequence_length: int, seed: int):
    """Gets the filename that should be used to save the shuffled index"""
    sz_id = "none" if image_sizer is None else image_sizer.get_id()
    # v0: had no compression
    return f"index.{sz_id}.{sequence_length}.s{seed}.v1"


COMPRESSED_DTYPE = np.dtype([
    ('file_id', np.uint16),
    ('start_byte', np.uint8, (6,)),
    ('length', np.uint16),
    ('sequence_number', np.uint8, (6,)),
])
# How to store the index on disk, this reduces bytes per document from 22 to 16
# If we want to really try hard we could maybe only store the sequence number every
# 1000's document or so to cut out that as space as well


def compress_index(data: np.ndarray):
    """Convert array of DOC_SEQUENCE_DTYPE to COMPRESSED_DTYPE"""
    assert sys.byteorder == "little"
    compressed = np.zeros(len(data), COMPRESSED_DTYPE)
    compressed["file_id"] = data["doc_id"]["file_id"]

    assert np.all(data["doc_id"]["length"]) < np.iinfo(np.uint16).max
    compressed["length"] = data["doc_id"]["length"]

    start_byte = np.ascontiguousarray(data["doc_id"]["start_byte"])
    start_byte = start_byte.view(np.uint8).reshape(-1, 8)
    assert np.all(start_byte[:, -2:] == 0)
    compressed["start_byte"] = start_byte[:, :-2]
    del start_byte

    seq_num = np.ascontiguousarray(data["sequence_number"])
    seq_num = seq_num.view(np.uint8).reshape(-1, 8)
    assert np.all(seq_num[:, -2:] == 0)
    compressed["sequence_number"] = seq_num[:, :-2]
    return compressed


def decompress_index(compressed: np.ndarray) -> np.ndarray:
    """Convert array of COMPRESSED_DTYPE to DOC_SEQUENCE_DTYPE"""
    out = np.zeros(len(compressed), DOC_SEQUENCE_DTYPE)
    out["sequence_number"] = np.pad(compressed["sequence_number"], [[0, 0], [0, 2]]).view(np.uint64).flatten()
    doc_id = out["doc_id"]
    doc_id["start_byte"] = np.pad(compressed["start_byte"], [[0, 0], [0, 2]]).view(np.uint64).flatten()
    doc_id["file_id"] = compressed["file_id"]
    doc_id["length"] = compressed["length"]
    return out


def write_index(idx: np.ndarray, out_f):
    compress_index(idx).tofile(out_f)


class SequenceIndex:
    """Provides a way to iterate through a document ordering given as array, or a array saved as a file

    If the array is saved as a file that data will only be fetched incrementally
    """

    def __init__(self, data: Union[np.ndarray, PathOrStr]):
        if isinstance(data, np.ndarray):
            self.data = data
            self._idx_size = len(data)
            self._chunk_size = len(data)
        else:
            self.data = data
            self._chunk_size = 1024
        self._idx_size = None
        self._num_sequence = None

    @property
    def file_size(self):
        if self._idx_size is None:
            if isinstance(self.data, np.ndarray):
                self._idx_size = len(self.data)*DOC_SEQUENCE_DTYPE.itemsize
            else:
                self._idx_size = file_size(self.data)
        return self._idx_size

    @property
    def num_examples(self):
        return self.file_size // DOC_SEQUENCE_DTYPE.itemsize

    @property
    def num_sequences(self):
        if self._num_sequence is None:
            last_example = self._read(self.num_examples-1, 1)
            last_sequence_id = int(last_example["sequence_number"][0])
            self._num_sequence = last_sequence_id + 1
        return self._num_sequence

    def find_sequence_start(self, target_sequence, search_step=1024) -> int:
        if isinstance(self.data, np.ndarray):
            return np.searchsorted(self._read_seq_numbers(0, self.num_examples), target_sequence)
        return find_sequence_start(
            self._read_seq_numbers, target_sequence, self.num_examples, self.num_sequences, search_step)

    def iter_from(
        self,
        start_sequence=0,
        end_sequence=None,
    ) -> Iterator[List[Tuple[int, int, int]]]:
        """Iterator through the sequences in this index"""
        current_seq = start_sequence-1
        examples_in_sequence = []
        on = self.find_sequence_start(start_sequence)
        while on < self.num_examples:
            data = self._read(on, self._chunk_size)
            for example_id, next_seq in data:
                if current_seq is None:
                    current_seq = next_seq
                if next_seq != current_seq:
                    if examples_in_sequence:
                        yield examples_in_sequence
                    if end_sequence and next_seq >= end_sequence:
                        return
                    current_seq = next_seq
                    examples_in_sequence = []
                examples_in_sequence.append(example_id)
            on += self._chunk_size
        # Reached the end of the file
        yield examples_in_sequence

    def iter_blocks(
        self,
        start_sequence=0,
        end_sequence=None,
        block_size: int=1,
        block_step: int=0,
    ):
        """Iterate through blocks of sequences in this index

        start_sequence: First sequence to yield
        end_sequence: Maximum sequence to yield, if it is larger than this index treat the index
                      as if it was repeats enough times to contain `end_sequence`
        block_size: size of the block of examples to read at once
        block_skip: number of sequences to skip after reading a block
        chunk_Size: How large of chunks to read the from the index file
        """
        # This iteration pattern is used to slice data for the dataloader
        # TODO we could be clever here and skip readings parts of the file, but for now
        # we just stream everything
        if end_sequence is None:
            end_sequence = self.num_sequences

        if end_sequence > self.num_sequences:
            iterators = []
            while end_sequence > self.num_sequences:
                iterators.append(self.iter_from(0, self.num_sequences))
                end_sequence -= self.num_sequences
            if end_sequence:
                iterators.append(self.iter_from(0, end_sequence))
            it = itertools.chain(*iterators)
        else:
            it = self.iter_from(start_sequence, end_sequence)

        if block_step != 0:
            while True:
                for _ in range(block_size):
                    yield next(it)
                for _ in range(block_step):
                    next(it)
        else:
            for out in it:
                yield out

    def _read_seq_numbers(self, start, n):
        return self._read(start, n)["sequence_number"]

    def _read(self, start, n):
        if isinstance(self.data, np.ndarray):
            return self.data[start:start+n]
        else:
            data = get_bytes_range(self.data, start * COMPRESSED_DTYPE.itemsize, n * COMPRESSED_DTYPE.itemsize)
            data = np.frombuffer(data, COMPRESSED_DTYPE)
            return decompress_index(data)


def find_sequence_start(read_fn, target_sequence, num_examples_in_file, num_seq_in_file, chunk_size) -> int:
    """Return the index of the first example that starts with `target_sequence` in `idx_file`"""
    max_start = max(num_examples_in_file - chunk_size, 0)

    # Do a binary search, but instead of checking the midpoint in each step we guess
    # based on how close `target_sequence` is to the max/min sequence we have found so far,
    # which we expect will be quite accurate for index files

    assert target_sequence < num_seq_in_file

    # Lowest index that cannot be `target_sequence` and its sequence number
    low = (-1, -1)

    # highest index that cannot be `target_sequence` and its sequence number
    high = (num_examples_in_file+1, num_seq_in_file)

    while True:
        # Guess assuming sequences between low and high are roughly evenly spaced
        remaining_seq = high[1] - low[1] - 1
        percent = (target_sequence - low[1] + 1) / remaining_seq
        remaining_examples = high[0] - low[0]
        current_guess = low[0] + int(round(remaining_examples*percent))
        read_start = current_guess - chunk_size // 2

        if read_start <= low[0]:
            read_start = low[0] + 1
        elif (read_start + chunk_size) > high[0]:
            read_start = high[0] - chunk_size
        read_start = min(max(read_start, 0), max_start)
        seq_numbers = read_fn(read_start, chunk_size)
        if seq_numbers[-1] < target_sequence:
            low = (read_start + chunk_size - 1, seq_numbers[-1])
        elif seq_numbers[0] > target_sequence:
            high = (read_start, seq_numbers[0])
        elif seq_numbers[0] == target_sequence:
            # Found target_sequence, but there might be occurrences of it before this chunk
            # Scan backwards until we find a chunk that does not start with `target_sequence`
            # (or we reach the end of the file) under the assumption the sequence won't be super long
            while seq_numbers[0] == target_sequence and read_start != 0:
                read_start = max(0, read_start - chunk_size)
                seq_numbers = read_fn(read_start, chunk_size)
            return read_start + np.searchsorted(seq_numbers, target_sequence)
        else:
            return read_start + np.searchsorted(seq_numbers, target_sequence)


def find_sequence_start_scan(
        read_fn, target_sequence, num_examples_in_file, num_seq_in_file, search_step) -> int:
    """Return the index of the first example that starts with `target_sequence` in `idx_file`"""
    # Naive version the just scans from the first guess
    # TODO should benchmark on a large index file to see if the optimization is worth anything
    percent = target_sequence/num_seq_in_file
    initial_guess = int(round((percent*num_examples_in_file)))
    max_start = max(num_examples_in_file - search_step, 0)
    on = initial_guess-search_step//2
    on = min(max(on, 0), max_start)

    n_scan = 0
    while True:
        n_scan += 1
        seq_numbers = read_fn(on, search_step)
        prev_on = on
        if target_sequence < seq_numbers[0]:
            if on == 0:
                raise ValueError(f"{target_sequence} not in index")
            on = max(0, on-search_step)
        elif target_sequence > seq_numbers[-1]:
            if on == max_start:
                raise ValueError(f"{target_sequence} not in index")
            on = min(on+search_step, max_start)
        elif seq_numbers[0] == target_sequence and on != 0:
            on = max(0, on-search_step+1)
        else:
            idx = np.searchsorted(seq_numbers, target_sequence)
            return on + idx
