import hashlib
import itertools
from os.path import join, exists
from typing import Iterable, Tuple, Iterator, List, Dict

import numpy as np

from olmo.mm_data.image_token_size import ImageTokenSizer
from olmo.mm_data.structure_index import Indexer, get_index_file
from olmo.mm_data.data_store import ExampleReader
from olmo.util import file_size, get_bytes_range


EXAMPLE_ID_DTYPE = np.dtype([
    ('file_id', np.uint16),
    ('start_byte', np.uint64),
    ('length', np.uint32)
])

IN_MEM_SHUFFLE_DTYPE = np.dtype([
    ("example_id", EXAMPLE_ID_DTYPE),
    ('num_tokens', np.uint32)    # so we know how to group examples into sequences after shuffling
])

IDX_DTYPE = np.dtype([
    ('sequence_number', np.uint64),    # So we can find a specific sequences in the index
    ("example_id", EXAMPLE_ID_DTYPE)
])


def get_idx_file(reader: ExampleReader, sequence_length: int, seed: int):
    """Gets the filename that should be used to save the shuffled index"""
    # TODO Currently the client has to track what datafiles this index applies to,
    # should we fix that?
    return f"index.{reader.image_sizer.get_id()}.{sequence_length}.s{seed}.v0"


class SequenceIndex:
    """Provides a way to iterate over sequences of examples ids stored in an index file"""

    def __init__(self, idx_file):
        self.idx_file = idx_file
        self._idx_size = None
        self._num_sequence = None

    @property
    def file_size(self):
        if self._idx_size is None:
            self._idx_size = file_size(self.idx_file)
        return self._idx_size

    @property
    def num_examples(self):
        return self.file_size // IDX_DTYPE.itemsize

    @property
    def num_sequences(self):
        if self._num_sequence is None:
            last_example_data = get_bytes_range(
                self.idx_file, self.file_size - IDX_DTYPE.itemsize, IDX_DTYPE.itemsize)
            last_example = np.frombuffer(last_example_data, IDX_DTYPE)
            last_sequence_id = last_example["sequence_number"][0]
            self._num_sequence = last_sequence_id + 1
        return self._num_sequence

    def find_sequence_start(self, target_sequence, search_step=1024) -> int:
        return find_sequence_start(
            self.idx_file, target_sequence, self.num_examples,
            search_step, self.num_examples, self.num_sequences
        )

    def iter_from(
        self,
        start_sequence=0,
        end_sequence=None,
        chunk_size=1024,
    ) -> Iterator[List[Tuple[int, int, int]]]:
        """Iterator through the sequences in this index"""
        current_seq = start_sequence-1
        examples_in_sequence = []
        on = self.find_sequence_start(start_sequence)
        while on < self.num_examples:
            buf = get_bytes_range(self.idx_file, on*IDX_DTYPE.itemsize, chunk_size*IDX_DTYPE.itemsize)
            data = np.frombuffer(buf, dtype=IDX_DTYPE)
            for next_seq, example_id in data:
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
            on += chunk_size
        # Reached the end of the file
        yield examples_in_sequence

    def iter_blocks(
        self,
        start_sequence=0,
        end_sequence=None,
        block_size: int=1,
        block_step: int=0,
        chunk_size=1024,
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
                iterators.append(self.iter_from(0, self.num_sequences, chunk_size))
                end_sequence -= self.num_sequences
            if end_sequence:
                iterators.append(self.iter_from(0, end_sequence, chunk_size))
            it = itertools.chain(*iterators)
        else:
            it = self.iter_from(start_sequence, end_sequence, chunk_size)

        if block_step != 0:
            while True:
                for _ in range(block_size):
                    yield next(it)
                for _ in range(block_step):
                    next(it)
        else:
            for out in it:
                yield out

    def _read(self, start, n):
        data = get_bytes_range(self.idx_file, start*IDX_DTYPE.itemsize, n*IDX_DTYPE.itemsize)
        return np.frombuffer(data, IDX_DTYPE)


def find_sequence_start(idx_file, target_sequence, n_examples, chunk_size,
                        num_examples_in_file, num_seq_in_file) -> int:
    """Return the index of the first example that starts with `target_sequence` in `idx_file`"""
    max_start = max(n_examples - chunk_size, 0)

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
        chunk = get_bytes_range(idx_file, read_start * IDX_DTYPE.itemsize,
                                                        IDX_DTYPE.itemsize * chunk_size)
        seq_numbers = np.frombuffer(chunk, IDX_DTYPE)["sequence_number"]
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
                chunk = get_bytes_range(
                    idx_file, read_start * IDX_DTYPE.itemsize, IDX_DTYPE.itemsize * chunk_size)
                seq_numbers = np.frombuffer(chunk, IDX_DTYPE)["sequence_number"]
            return read_start + np.searchsorted(seq_numbers, target_sequence)
        else:
            return read_start + np.searchsorted(seq_numbers, target_sequence)


def find_sequence_start_scan(
    idx_file, target_sequence, n_examples, num_examples_in_file, num_seq_in_file, search_step) -> int:
    """Return the index of the first example that starts with `target_sequence` in `idx_file`"""
    # Naive version the just scans from the first guess
    # TODO should benchmark on a large index file to see if the optimization is worth anything
    percent = target_sequence/num_seq_in_file
    initial_guess = int(round((percent*num_examples_in_file)))
    max_start = max(n_examples - search_step, 0)
    on = initial_guess-search_step//2
    on = min(max(on, 0), max_start)

    while True:
        chunk = get_bytes_range(idx_file, on*IDX_DTYPE.itemsize, IDX_DTYPE.itemsize*search_step)
        seq_numbers = np.frombuffer(chunk, IDX_DTYPE)["sequence_number"]
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


def chunk_example(rng, num_tokens, seq_len, min_seq_len=16):
    """Split `num_tokens` in a number of chunks between min_seq_len and seq_len in length

    Returns: the lengths of the chunks
    """
    # TODO is this the best approach? Should we add more randomness?
    assert seq_len > min_seq_len*2
    remainder = num_tokens % seq_len
    lengths = [seq_len]*(num_tokens // seq_len)
    if remainder:
        if remainder < min_seq_len:
            lengths[-1] = min_seq_len
            remainder += (seq_len - min_seq_len)
        lengths.append(remainder)

    rng.shuffle(lengths)
    return lengths


def build_sequence_index(
        reader: ExampleReader,
        indexer: Indexer,
        seq_len: int,
        seed: int,
        output_file: str
):
    rng = np.random.RandomState(seed)

    examples = []
    for file_id, data_f in sorted(reader.data_files.items(), key=lambda x: x[1]):
        index_data = []
        index_f = get_index_file(data_f)
        file_id_bytes = np.array(file_id, np.uint16).tobytes()
        for ex in indexer.get_indices(index_f, file_id, reader):
            if ex.num_tokens > seq_len:
                if ex.pure_text:
                    # TODO untested
                    lengths = chunk_example(rng, ex.num_tokens, seq_len)
                    on = 0
                    for chunk_tokens in lengths:
                        chunk_start_byte = ex.start_byte+on*2
                        index_data.append(np.array(
                            ((file_id, chunk_start_byte, chunk_tokens*2), chunk_tokens), IN_MEM_SHUFFLE_DTYPE))
                        on += chunk_tokens*2
                else:
                    raise NotImplementedError("Chunking multi-modal documents")
            else:
                point = np.array(((file_id, ex.start_byte, ex.num_bytes), ex.num_tokens), IN_MEM_SHUFFLE_DTYPE)
                index_data.append(point)
        examples.append(np.stack(index_data))
    examples = np.concatenate(examples)

    rng.shuffle(examples)
    sequence_number = 0
    total_tokens = 0
    with open(output_file, "wb") as f:
        for (example_id, num_tokens) in examples:
            # TODO truncation or data sorting to make packing more efficient
            if total_tokens + num_tokens > seq_len:
                sequence_number += 1
                total_tokens = 0
            total_tokens += num_tokens
            f.write(np.array((sequence_number, example_id), dtype=IDX_DTYPE).tobytes())

