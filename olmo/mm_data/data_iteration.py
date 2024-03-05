"""How to iterate over a data in a set of datafiles"""
import collections
import dataclasses
import math
import multiprocessing
from dataclasses import dataclass
from itertools import repeat
from time import perf_counter
from typing import List, Optional, Union, Tuple

import numpy as np
from transformers.utils import logging

from olmo.config import DataSamplingConfig, SequenceBuilderConfig, SequenceBuilderKind
from olmo.mm_data.data_store import MMStorageConfig
from olmo.mm_data.image_token_size import ImageTokenSizer
from olmo.mm_data.structure_index import Indexer, VectorizedIndexer
from olmo.util import human_readable_number as hs

try:
    import pyximport
except ImportError:
    pyximport = None

logger = logging.get_logger(__name__)

data_iteration_cython = None


def try_import_cython():
    if pyximport is None:
        return False
    # Only import if needed so we don't compile the cython code if its not needed
    global data_iteration_cython
    if data_iteration_cython is None:
        pyximport.install(setup_args={'include_dirs': np.get_include()}, inplace=False)
        from olmo.mm_data import data_iteration_cython as cython
        data_iteration_cython = cython
    return True


DOC_ID_DTYPE = np.dtype([
    ('file_id', np.uint16),
    ('start_byte', np.uint64),
    ('length', np.uint32)
])
# data we need to the load a document from the data files

DOC_INFO_DTYPE = np.dtype([
    ("doc_id", DOC_ID_DTYPE),
    ('num_tokens', np.uint32),
    ('pure_text', np.uint8),
])
# Meta-data needed when shuffling/packing documents


DOC_TOKENS_DTYPE = np.dtype([
    ("doc_id", DOC_ID_DTYPE),
    ('num_tokens', np.uint32),
])
# Document token counts


DOC_SEQUENCE_DTYPE = np.dtype([
    ("doc_id", DOC_ID_DTYPE),
    ('sequence_number', np.uint64),
])
# dtype to specify an example and its sequence number, an array of this dtypes defines iteration order
# sequence numbers are consecutive and start at 0
# sequence numbers are there to make it easy to start at a particular sequence number, and to ensure all workers
# across all device stay in agreement about how examples should be grouped into sequence


# TODO are these dtypes are too conservative?


def shuffle(config: Optional[DataSamplingConfig], rng: np.random.RandomState, data: List[np.ndarray]) -> np.ndarray:
    """Shuffle together data from multiple data files"""
    if config is None or (config.resample is None and not config.stratify):
        data = np.concatenate(data)
        rng.shuffle(data)
        return data
    
    if config.group_data:
        assert len(config.group_data) == len(data)
        grouped_data = [[] for _ in range(len(config.group_data))]
        for grp, ex in zip(config.group_data, data):
            grouped_data[grp].append(ex)
        data = [np.concatenate(x) for x in grouped_data]
    
    if config.resample:
        sampled_data = []
        for w, data in zip(config.resample, data):
            frac, n = math.modf(w)
            n = int(n)
    
            if frac:
                n = round(len(data) * w)
                selection = rng.choice(data, n, replace=False)
            else:
                selection = None
    
            if n >= 1:
                sampled = np.tile(data[None, :], [n, 1])
                if config.stratify:
                    for row in sampled:
                        rng.shuffle(row)
                sampled = sampled.ravel()
                if selection is not None:
                    sampled = np.concatenate(sampled, selection)
            else:
                sampled = selection  # already shuffled due to rng.choice
            sampled_data.append(sampled)
        data = sampled_data
    
    if config.stratify:
        # each individual array is now shuffled, balanced merge to get stratified data
        data = balanced_merge(data)
    else:
        data = np.concatenate(data)
        rng.shuffle(data)
    return data
    

def balanced_merge(arrays: List[np.ndarray]) -> np.ndarray:
    """Concatenate `arrays` so that examples for each input list are evenly spread across the output list"""
    if len(arrays) == 1:
        return arrays[1]
    lens = np.array([len(x) for x in arrays])
    target_ratios = lens / lens.sum()
    current_counts = np.zeros(len(arrays), dtype=np.int32)
    out = np.zeros(lens.sum(), dtype=arrays[0].dtype)
    on = 0
    while True:
        # Draw an item from the most under-represented list in our output
        next_i = np.argmin(current_counts / on - target_ratios)
        current_counts[next_i] += 1
        out[on] = arrays[next_i][0]
        on += 1
        arrays[next_i] = arrays[next_i][1:]
        if len(arrays[next_i]) == 0:
            if len(arrays) == 1:
                assert on + len(arrays[0]) == len(out)
                out[on:] = arrays[0]
                return out
            target_ratios = np.delete(target_ratios, next_i)
            current_counts = np.delete(current_counts, next_i)
            arrays = arrays[:next_i] + arrays[next_i + 1:]


class SequenceBuilder:
    """Determines how to pack examples into sequences"""

    def __call__(self, examples: np.ndarray, max_seq_len: int, n_processes: int=None) -> np.ndarray:
        """
        examples: [n] DOC_TOKENS_DTYPE examples to turn into sequences, already shuffled
        max_seq_len: max sequence length

        out: sequences[n] DOC_SEQUENCE_DTYPE, examples and sequence numbers of the documents

        Must be deterministic
        """
        raise NotImplementedError()


def build_sequence_builder(config: Optional[SequenceBuilderConfig]) -> SequenceBuilder:
    if config is None:
        return Sequential()
    if config.kind == SequenceBuilderKind.sequential:
        base = Sequential()
    elif config.kind == SequenceBuilderKind.one_document_per_sequence:
        assert config.n_splits is None
        assert config.max_per_split is None
        return OneDocumentPerSequence()
    elif config.kind == SequenceBuilderKind.split_text:
        base = SequentialSplitText()
    elif config.kind == SequenceBuilderKind.optimize_last:
        base = OptimizeLast(config.pool_size)
    else:
        raise NotImplementedError()
    if config.n_splits or config.max_per_split:
        return ParallelizableSequenceBuilder(base, config.n_splits, config.max_per_split)
    else:
        return base


class ParallelizableSequenceBuilder(SequenceBuilder):
    def __init__(self, builder: SequenceBuilder, n_splits: int=None, max_per_split: int=None):
        assert n_splits is None or max_per_split is None
        self.max_per_split = max_per_split
        self.n_splits = n_splits
        self.builder = builder

    def __call__(self, examples: np.ndarray, max_seq_len: int, n_procs=None):
        if self.n_splits:
            n_splits = self.n_splits
        else:
            n_splits = (len(examples) + self.max_per_split - 1) // self.max_per_split
        n_per_part = len(examples) // n_splits
        remainder = len(examples) % n_splits
        splits = []
        on = 0
        for ix in range(n_splits):
            part_len = n_per_part + int(ix < remainder)
            splits.append(examples[on:on+part_len])
            on += part_len
        assert on == len(examples)

        if n_procs is not None and n_procs > 1:
            with multiprocessing.Pool(n_procs) as pool:
                results = pool.starmap(self.builder, [(g, max_seq_len) for g in splits])
        else:
            results = [self.builder(g, max_seq_len) for g in splits]

        out = [results[0]]
        for r in results[1:]:
            end_seq = int(out[-1]["sequence_number"][-1]) + 1
            r["sequence_number"] += end_seq
            out.append(r)
        out = np.concatenate(out)
        return out


class OneDocumentPerSequence(SequenceBuilder):
    """Each document is in its own sequence"""

    def __call__(self, documents: np.ndarray, max_seq_len: int, pool=None):
        assert np.all(documents["num_tokens"] <= max_seq_len)
        out = np.empty(len(documents), DOC_SEQUENCE_DTYPE)
        out["doc_id"] = documents["doc_id"]
        out["sequence_number"] = np.arange(len(documents), dtype=np.uint64)
        return out


class Sequential(SequenceBuilder):
    """Naive approach that arranges documents end-to-end and never splits or truncate documents"""

    def __call__(self, documents: np.ndarray, max_seq_len: int, pool=None):
        assert np.all(documents["num_tokens"] <= max_seq_len)
        if try_import_cython():
            return data_iteration_cython.sequential(documents, max_seq_len)
        total_tokens = 0
        out = np.zeros(len(documents), dtype=DOC_SEQUENCE_DTYPE)
        on = 0
        sequence_number = 0
        for (doc_id, num_tokens, _) in documents:
            if total_tokens + num_tokens > max_seq_len:
                sequence_number += 1
                total_tokens = 0
            total_tokens += num_tokens
            out[on] = (doc_id, sequence_number)
            on += 1
        return out


def segment_cumulative_sum(start_vals, segment_val, segment_starts, segment_lens):
    arr = np.full(segment_starts[-1]+segment_lens[-1], segment_val, np.int64)
    seq_delta = start_vals.astype(np.int64)
    seq_delta[1:] -= seq_delta[:-1]
    seq_delta[1:] -= (segment_lens[:-1]-1)*segment_val
    arr[segment_starts] = seq_delta
    np.cumsum(arr, out=arr)
    return arr


def merge_pure_text_vectorized(documents: np.ndarray, seq_len):
    # Goes to some heroic lengths to vectorize text splitting
    t0 = perf_counter()

    # Assign existing documents their sequence numbers
    total_tokens = documents["num_tokens"].astype(np.uint64)
    np.cumsum(total_tokens, out=total_tokens)
    seq_nums = (total_tokens - 1) // seq_len

    out = np.zeros(len(documents), DOC_SEQUENCE_DTYPE)
    out["doc_id"] = documents["doc_id"]
    out["sequence_number"] = seq_nums
    t0 = perf_counter()

    # Now we will insert new doc_ids into `out` for cases where a document must be split
    # sequence number changes for each document
    delta = np.empty(len(documents), dtype=np.int64)
    delta[1:] = seq_nums[1:] - seq_nums[:-1]
    delta[0] = seq_nums[0]

    t0 = perf_counter()

    to_insert = []
    to_insert_ix = []

    # Handle cases where a document must be split to fill in a sequence
    remainder = np.empty(len(documents), dtype=np.int64)
    remainder[1:] = (seq_nums[:-1]+1)*seq_len - total_tokens[:-1]
    remainder[0] = 0
    to_split = np.argwhere((delta > 0) & (remainder > 0))[:, 0]
    if len(to_split) > 0:
        remainder = remainder[to_split].astype(np.uint64)
        remainder *= 2
        copies = out[to_split]
        copies["sequence_number"] = out[to_split-1]["sequence_number"]
        copies["doc_id"]["length"] = remainder
        out["doc_id"]["start_byte"][to_split] += remainder
        out["doc_id"]["length"][to_split] -= remainder
        to_insert.append(copies)
        to_insert_ix.append(to_split)

    t0 = perf_counter()

    to_copy = np.argwhere(delta > 1)[:, 0]
    if len(to_copy) > 0:
        # Case where a document into a some `max_seq_len` chunks
        n_copies = delta[to_copy]
        n_copies -= 1
        starts = np.pad(n_copies, [1, 0])
        starts = np.cumsum(starts)
        copies = np.zeros(starts[-1], DOC_SEQUENCE_DTYPE)
        starts = starts[:-1]
        copy_from = out[to_copy]

        copies["doc_id"]["length"] = seq_len*2
        copies["doc_id"]["file_id"] = segment_cumulative_sum(
            copy_from["doc_id"]["file_id"], 0, starts, n_copies)

        copies["sequence_number"] = segment_cumulative_sum(
            copy_from["sequence_number"].astype(np.int64) - n_copies,
            1, starts, n_copies
        )
        copies["doc_id"]["start_byte"] = segment_cumulative_sum(
            copy_from["doc_id"]["start_byte"],
            2*seq_len, starts, n_copies
        )
        delta = (n_copies * (2 * seq_len)).astype(np.uint64)
        out["doc_id"]["start_byte"][to_copy] += delta
        out["doc_id"]["length"][to_copy] -= delta

        idx = segment_cumulative_sum(to_copy, 0, starts, n_copies)
        to_insert_ix.append(idx)
        to_insert.append(copies)

    if to_insert_ix:
        out = np.insert(out, np.concatenate(to_insert_ix), np.concatenate(to_insert))

    return out


class SequentialSplitText(SequenceBuilder):
    """Arranges documents end-to-end but split text-only documents to fill in gaps"""
    def __call__(self, documents: np.ndarray, max_seq_len: int, n_proc=None):
        if try_import_cython():
            # About 100x faster
            return data_iteration_cython.sequential_split_text(documents, max_seq_len)

        max_splits = int(documents["num_tokens"].sum()) // max_seq_len
        max_out = len(documents) + max_splits
        out = np.empty(max_out, DOC_SEQUENCE_DTYPE)
        out_ix = 0

        total_tokens = 0
        sequence_number = 0
        for (doc_id, num_tokens, pure_text) in documents:
            next_len = total_tokens + num_tokens
            if next_len < max_seq_len:
                out[out_ix] = (doc_id, sequence_number)
                out_ix += 1
                total_tokens += num_tokens
            elif next_len == max_seq_len:
                out[out_ix] = (doc_id, sequence_number)
                out_ix += 1
                total_tokens = 0
                sequence_number += 1
            elif not pure_text:
                # Start a new sequence and just put up with the padding since we can't split MM documents
                sequence_number += 1
                out[out_ix] = (doc_id, sequence_number)
                out_ix += 1
                total_tokens = num_tokens
            else:
                # Split tp fill in this sequence
                tokens_to_take = [max_seq_len - total_tokens]
                num_tokens -= tokens_to_take[0]
                # Split if the document is so long it will take many up many sequences
                n_full_length_splits = num_tokens // max_seq_len
                tokens_to_take += [max_seq_len]*n_full_length_splits
                num_tokens -= max_seq_len*n_full_length_splits
                for n_tok in tokens_to_take:
                    out[out_ix] = ((doc_id["file_id"], doc_id["start_byte"], n_tok*2), sequence_number)
                    out_ix += 1
                    sequence_number += 1
                    doc_id["start_byte"] += n_tok*2
                if num_tokens:
                    out[out_ix] = ((doc_id["file_id"], doc_id["start_byte"], num_tokens*2), sequence_number)
                    out_ix += 1
                total_tokens = num_tokens
        return out[:out_ix]


class DocumentPool:
    """Utility class of packing, maintains a `sequence_lengths->examples` mapping for a pool of documents"""

    def __init__(self, max_seq_len, pool_size):
        self.n_tokens_to_ex_id = [collections.deque() for _ in range(max_seq_len+1)]
        self.hist = np.zeros(max_seq_len+1, dtype=np.int32)
        self.hist_min = max_seq_len

    def add(self, ex):
        _n = ex["num_tokens"]
        self.n_tokens_to_ex_id[_n].append(ex["doc_id"])
        self.hist[_n] += 1
        self.hist_min = min(_n, self.hist_min)

    def pop(self, seq_len):
        self.hist[seq_len] -= 1
        if seq_len == self.hist_min and self.hist[seq_len] == 0:
            self.hist_min = self.hist_min + np.argmax(self.hist[self.hist_min:] > 0)
            if self.hist[self.hist_min] == 0:
                self.hist_min = len(self.hist)
        return self.n_tokens_to_ex_id[seq_len].popleft()

    def is_empty(self):
        return self.hist_min == len(self.hist)

    def find_at_most(self, ix):
        """Return the largest seq len that we have a document for and is <= `ix`, 0 if there is no such example"""
        # this function gets called a lot its important its fast
        if ix < self.hist_min:
            return 0
        else:
            return ix-np.argmax(self.hist[self.hist_min:ix+1][::-1] > 0)

    def get_first(self):
        if self.is_empty():
            raise ValueError()
        return self.find_at_most(len(self.hist))


class OptimizeLast(SequenceBuilder):
    """Streams through the data while keeping a pool of examples in reserve, places examples from the pool at the
    end of sequences when helpful to maximize the sequence length

    This aims to be fast and mostly preserve the order of the data stream
    """
    def __init__(self, pool_size):
        assert isinstance(pool_size, int)
        self.pool_size = pool_size

    def __call__(self, examples: np.ndarray, max_seq_len: int, n_procs=None):
        assert np.all(examples["num_tokens"] <= max_seq_len)
        if try_import_cython():
            return data_iteration_cython.optimize_last(examples, max_seq_len, self.pool_size)
        pool = DocumentPool(max_seq_len, self.pool_size)
        for ex in examples[:self.pool_size]:
            pool.add(ex)

        out = np.zeros(len(examples), dtype=DOC_SEQUENCE_DTYPE)
        out_ix = 0
        on_seq = 0

        in_progress = np.zeros(max_seq_len, dtype=DOC_TOKENS_DTYPE)
        on = 0  # in_progress[:on] is our in-progress sequence
        on_example = min(len(examples), self.pool_size)  # what example in `examples` we have consumed
        seq_len = 0  # length of the sequence we are in-progress on
        n_from_pool = 0  # how many in-progress examples are from the pool
        stop_add_to_pool = len(examples) - self.pool_size
        best_pool_only = pool.find_at_most(max_seq_len)
        best_candidate = (on, best_pool_only, best_pool_only)

        # Max sequence length docs might be very common (e.g., splitting a very long documents), to avoid having
        # to end an in-progress sequence early when encountering them we skip over them
        while True:
            if on_example == len(examples):
                if pool.is_empty():
                    if on == 0:
                        break  # nothing left and nothing in-progress
                    else:
                        new_tokens = max_seq_len + 1  # yield the in-progress sequence
                else:
                    new_tokens = pool.get_first()  # Ran out of examples, start consuming the pool
            else:
                new_tokens = examples[on_example]["num_tokens"]
                if new_tokens == max_seq_len:
                    # Effectively 0 length since we will write them out separately
                    new_tokens = 0

            next_len = seq_len + new_tokens

            if next_len > max_seq_len:
                # Write the best candidate we have found so far
                best_end, item_from_pool = best_candidate[:2]
                to_add = in_progress[:best_end]

                # max length documents get written out specially
                is_max_len = to_add["num_tokens"] == max_seq_len
                for ex in to_add[is_max_len]:
                    out[out_ix] = (ex["doc_id"], on_seq)
                    out_ix += 1
                    on_seq += 1
                to_add = to_add[~is_max_len]["doc_id"]

                remainder = on - best_end  # num in-progress examples we are discarding
                for k in in_progress[max(on-n_from_pool, best_end):on]:
                    # If our in-progress sequence used examples from the pool, move the unused example back to
                    # the pool. We need this since the best sequence might have used one of these examples
                    pool.add(k.copy())  # copy so it is not a view in `in_progress`
                    remainder -= 1
                on_example -= remainder

                out_start = out_ix
                out["doc_id"][out_ix:out_ix+len(to_add)] = to_add
                out_ix += len(to_add)
                if item_from_pool > 0:
                    out["doc_id"][out_ix] = pool.pop(item_from_pool)
                    out_ix += 1
                    if on_example < stop_add_to_pool:
                        pool.add(examples[on_example])  # re-fill the pool
                        on_example += 1

                out["sequence_number"][out_start:out_ix] = on_seq
                on_seq += 1
                best_pool_only = pool.find_at_most(max_seq_len)
                best_candidate = (0, best_pool_only, best_pool_only)
                on, seq_len, n_from_pool = 0, 0, 0
            else:
                # Add to our in-progress sequence
                seq_len = next_len
                if on_example == len(examples):
                    n_from_pool += 1
                    in_progress[on] = (pool.pop(new_tokens), new_tokens)
                else:
                    in_progress[on] = examples[on_example][["doc_id", "num_tokens"]]
                    on_example += 1
                on += 1
                # Build a new candidate using the optimal sequence in the pool
                remainder = max_seq_len - seq_len
                in_pool = pool.find_at_most(remainder)
                candidate = (on, in_pool, seq_len + in_pool)
                if candidate[-1] > best_candidate[-1]:
                    best_candidate = candidate

        assert out_ix == len(out)
        return out


def split_example(rng, num_tokens, seq_len, min_seq_len=16):
    """Split `num_tokens` in a number of chunks between min_seq_len and seq_len in length

    Returns: the lengths of the chunks
    """
    # Try to maximum the number of `seq_len` chunks while avoiding chunks < `min_seq_len`
    # TODO is this the best approach? Should we add more randomness?
    assert seq_len > min_seq_len*2
    assert num_tokens > seq_len
    remainder = num_tokens % seq_len
    lengths = [seq_len]*(num_tokens // seq_len)
    if remainder:
        if remainder < min_seq_len:
            lengths[-1] = min_seq_len
            remainder += (seq_len - min_seq_len)
        lengths.append(remainder)
    lengths = np.array(lengths)
    rng.shuffle(lengths)
    return lengths


def reorder_sequences(idx, sequence_ixs):
    """Re-arranges the sequences in `idx` according to `sequence_ixs`"""
    # In essence, we want to do
    #     new_seq_ids = sequence_ixs[idx["sequence_number"]]
    #     idx["sequence_number"] = new_seq_ids                # change to the new sequence number
    #     return idx[argsort(new_seq_ids)]                    # sort to fix the ordering
    # But that can become slow at the 1b+ scale, with some tricks we can use a cumulative
    # sum to find out how the new sequence should be ordered instead of a sort
    if try_import_cython():
        # cython version roughly 4x faster
        return data_iteration_cython.reorder_sequence(idx, sequence_ixs)

    counts = np.zeros(len(sequence_ixs), dtype=np.int64)
    np.add.at(counts, idx["sequence_number"], 1)

    # starts[i] is where sequence number `i` will start in the new `idx` matrix
    new_counts = np.empty_like(counts)
    new_counts[sequence_ixs] = counts
    starts = np.empty(len(new_counts), dtype=np.int64)
    starts[0] = 0
    np.cumsum(new_counts[:-1], out=starts[1:])

    # now starts[i] is where the previous sequence number `i` will start in the new `idx` matrix
    starts = starts[sequence_ixs]

    # We want to map idx[l] to the row `starts[idx[l].sequence_number]`, but also
    # offset to account for the fact multiple documents will have the same sequence number
    mapping = np.repeat(starts, counts)
    mapping += np.arange(len(idx))
    mapping[counts[0]:] -= np.repeat(counts[:-1].cumsum(), counts[1:])

    # Update the sequence numbers
    idx["sequence_number"] = sequence_ixs[idx["sequence_number"]]

    # Shift to the new order
    out = np.empty_like(idx)
    out[mapping] = idx
    return out


@dataclass
class IterationConfig:
    """Specifies how to iterate through data stored in a set of data files"""
    paths: List[str]
    sampler: Optional[DataSamplingConfig] = None
    sequence_builder: Optional[SequenceBuilderConfig] = None

    shuffle_sequences: bool = False
    """Re-shuffle sequence after packing"""

    presplit_text_documents: bool = False
    """Split long text document before shuffling examples and packing"""

    get_blocks: Union[bool, List[bool]] = False
    """Read `seq_len` chunks from these pure-text datafiles instead of using an index"""


@dataclass
class DataFileStats:
    n_docs: int
    n_tokens: int
    n_mm: int
    n_mm_truncate: int
    n_text_split: int

    @staticmethod
    def sum(stats: List['DataFileStats']):
        vals = [dataclasses.astuple(x) for x in stats]
        summed = [sum(x) for x in zip(*vals)]
        return DataFileStats(*summed)


def _load_index(file_id, doc_seed, data, indexer, seq_len, sizing,
                storage_config, index_files) -> Tuple[np.ndarray, DataFileStats]:
    """Loads an index as an array of DOC_INFO_TYPE and computes some statistics"""
    data_file_documents = []
    stats = collections.Counter()

    example_info = indexer.get_indices(data.paths[file_id], sizing, storage_config,
                                       None if index_files is None else index_files[file_id])
    n_doc = len(example_info)
    n_tokens = example_info["num_tokens"].sum()
    n_mm = (~example_info["pure_text"]).sum()
    too_long = example_info["num_tokens"] > seq_len
    mm_truncate = too_long & (~example_info["pure_text"])
    n_split = 0
    if np.any(mm_truncate):
        # We don't adjust byte length since doing so might be unpredictable (e.g., only read part of
        # image), instead we just leave the full byte length and let the data reader truncate after loading
        # This is not optimal since the data loader could truncate the document to < seq_len, if, for example,
        # an entire image gets truncated, but we currently don't have the meta-data to detect that here
        example_info[mm_truncate]["num_tokens"] = seq_len

    if data.presplit_text_documents:
        long_text = too_long & example_info["pure_text"]
        documents_to_split = example_info[long_text]
        no_preprocess = example_info[~long_text]
    else:
        no_preprocess = example_info
        documents_to_split = []
    out = np.zeros(len(no_preprocess), dtype=DOC_INFO_DTYPE)
    out[["num_tokens", "pure_text"]] = no_preprocess[["num_tokens", "pure_text"]]
    out["doc_id"][["start_byte", "length"]] = no_preprocess[["start_byte", "num_bytes"]]
    out["doc_id"]["file_id"] = file_id
    data_file_documents.append(out)

    if len(documents_to_split) > 0:
        rng = np.random.RandomState(doc_seed)
        n_split += len(documents_to_split)

        n_splits = (documents_to_split["num_tokens"] + seq_len - 1) // seq_len
        with_splits = np.zeros(n_splits.sum(), DOC_INFO_DTYPE)

        on = 0
        for (n, doc) in zip(n_splits, documents_to_split):
            lens = split_example(rng, doc["num_tokens"], seq_len)
            lens *= 2
            part = with_splits[on:on+n]
            part["doc_id"]["length"] = lens
            part["doc_id"]["start_byte"] = doc["start_byte"]
            part["doc_id"]["start_byte"][1:] = lens[:-1]
            np.cumsum(part["doc_id"]["start_byte"], out=part["doc_id"]["start_byte"])
            on += n

        with_splits["doc_id"]["file_id"] = file_id
        with_splits["num_tokens"] = with_splits["doc_id"]["length"]//2
        with_splits["pure_text"] = True
        data_file_documents.append(with_splits)
    all_docs = np.concatenate(data_file_documents)
    return all_docs, DataFileStats(n_doc, n_tokens, n_mm, mm_truncate.sum(), n_split)


def build_iteration_order(
    data: Union[IterationConfig, List[str]],
    seq_len: int,
    seed: int,
    sizing: ImageTokenSizer,
    n_procs: int,
    indexer: Optional[Indexer] = None,
    index_files: Optional[List[str]] = None,
    storage_config: Optional[MMStorageConfig] = None
) -> np.ndarray:
    """Builds an array of DOC_SEQUENCE_DTYPE specifying how to iterate through the `data`

    data: Data to iterate through
    seq_len: max sequence length for this iteration
    seed: seed to this iteration
    sizing: image -> number_of_tokens mapping, needed to split/pack documents
    n_processes: Use multiple-processes when possible to speed up loading
    index: Indexer use to read the index files for the data, if None defaults to `VectorizedIndex`
    index_files: List of index files if the tiles are not in the default location
    storage_config: Store config it not default
    """
    # Set up the defaults
    if isinstance(data, list):
        data = IterationConfig(data)
    if indexer is None:
        indexer = VectorizedIndexer()
    if storage_config is None:
        storage_config = MMStorageConfig()
    rng = np.random.RandomState(seed)

    logger.info(f"Loading meta-data for {len(data.paths)} data files")
    doc_seeds = rng.randint(0, np.iinfo(np.uint32).max, len(data.paths))
    load_args = zip(*([range(len(data.paths)), doc_seeds] +
                      [repeat(x) for x in [data, indexer, seq_len, sizing, storage_config, index_files]]))
    if n_procs > 1 and len(data.paths) > 1:
        with multiprocessing.Pool(min(n_procs, len(data.paths)//2)) as pool:
            results = pool.starmap(_load_index, load_args)
    else:
        results = [_load_index(*x) for x in load_args]
    documents_per_data_file = [x[0] for x in results]
    stats = DataFileStats.sum(x[1] for x in results)

    logger.info(f"In {len(data.paths)} files, found {hs(stats.n_docs)} documents of which {hs(stats.n_mm)} are "
                f"multi-modal, with {hs(stats.n_tokens)} tokens (average {stats.n_tokens/stats.n_docs:0.1f} "
                f"tokens per document)")
    if stats.n_mm_truncate:
        logger.info(f"Truncated {hs(stats.n_mm_truncate)} multi-modal documents")
    if stats.n_text_split:
        logger.info(f"Split {hs(stats.n_text_split)} text documents")

    logger.info(f"Shuffling documents...")
    documents = shuffle(data.sampler, rng, documents_per_data_file)
    del documents_per_data_file

    logger.info(f"Building sequences...")
    t0 = perf_counter()
    sequence_builder = build_sequence_builder(data.sequence_builder)
    idx = sequence_builder(documents, seq_len, n_procs)
    del documents
    n_sequences = int(idx["sequence_number"][-1])+1
    logger.info(f"Built {hs(n_sequences)} sequences of len {seq_len} in {perf_counter()-t0:0.1f} seconds")
    tokens_per_seq = stats.n_tokens/n_sequences
    ex_per_seq = stats.n_docs/n_sequences
    logger.info(f"Packing efficiency={tokens_per_seq/seq_len:0.3f}, {tokens_per_seq:0.1f} tokens and {ex_per_seq:0.1f} "
                f"examples per a sequence on average.")
    if len(idx) != stats.n_docs:
        logger.info(f"Embedded {hs(len(idx))} docs ({len(idx)/stats.n_docs:0.2f} increase due to splitting)")

    if data.shuffle_sequences:
        logger.info(f"Shuffling sequences...")
        n_sequences = int(idx[-1]["sequence_number"]) + 1
        sequence_ixs = np.arange(n_sequences, dtype=np.int64)
        rng.shuffle(sequence_ixs)
        logger.info(f"Re-ordering index...")
        idx = reorder_sequences(idx, sequence_ixs)
    return idx

