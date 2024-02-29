"""How to iterate over a data in a set of datafiles"""
import collections
import math
import multiprocessing
from dataclasses import dataclass, asdict, field
from time import perf_counter
from typing import List, Optional, Union, Iterator, Tuple

import numpy as np
from transformers.utils import logging

from olmo.config import BaseConfig, DataSamplingConfig, SequenceBuilderConfig, SequenceBuilderKind
from olmo.mm_data.data_store import ExampleReader, MMStorageConfig
from olmo.mm_data.image_token_size import ImageTokenSizer
from olmo.mm_data.structure_index import Indexer, get_index_file, VectorizedIndexer
from olmo.util import NumpyList, StrEnum

logger = logging.get_logger(__name__)


DOC_ID_DTYPE = np.dtype([
    ('file_id', np.uint16),
    ('start_byte', np.uint64),
    ('length', np.uint32)
])
# data we need to the load the document from the data files

DOC_SEQUENCE_DTYPE = np.dtype([
    ("doc_id", DOC_ID_DTYPE),
    ('sequence_number', np.uint64),
])
# dtype to specify an example and its sequence number, an array of this dtypes defines iteration order
# sequence numbers are consecutive and start at 0
# sequence numbers are there to make it easy to start at a particular sequence number, and to ensure all workers
# across all device stay in agreement about how examples should be grouped into sequence


DOC_TOKENS_DTYPE = np.dtype([
    ("doc_id", DOC_ID_DTYPE),
    ('num_tokens', np.uint32),
])
# dtype for an example and its number of tokens, used when constructing an iteration order

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
    elif config.kind == SequenceBuilderKind.optimize_last:
        base = OptimizeLast(config.pool_size)
    else:
        raise NotImplementedError()
    if config.n_splits or config.max_splits:
        return ParallelizableSequenceBuilder(base, config.n_splits, config.max_splits)
    else:
        return base


class ParallelizableSequenceBuilder(SequenceBuilder):
    def __init__(self, builder: SequenceBuilder, n_splits: int=None, max_per_split: int=None):
        assert n_splits is None or max_per_split is None
        self.max_per_split = max_per_split
        self.n_splits = n_splits
        self.builder = builder

    def __call__(self, examples: np.ndarray, max_seq_len: int, n_proc: int=None):
        if self.n_splits:
            n_splits = self.n_splits
        else:
            n_splits = (len(examples) + self.max_per_split - 1) // self.max_per_split

        n_per_part = len(examples) // self.n_splits
        remainder = len(examples) % self.n_splits
        splits = []
        on = 0
        for ix in range(self.n_splits):
            part_len = n_per_part + int(ix < remainder)
            splits.append(examples[on:on+part_len])
            on += part_len
        assert on == len(examples)

        if n_proc and n_splits > 1:
            with multiprocessing.Pool(min(n_proc, n_splits)) as pool:
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


class Sequential(SequenceBuilder):
    """Naive approach that arranges documents end-to-end in the order they were given"""

    def __call__(self, documents: np.ndarray, max_seq_len: int, n_proc=None):
        total_tokens = 0
        out = np.zeros(len(documents), dtype=DOC_SEQUENCE_DTYPE)
        on = 0
        sequence_number = 0
        for (doc_id, num_tokens) in documents:
            if total_tokens + num_tokens > max_seq_len:
                sequence_number += 1
                total_tokens = 0
            total_tokens += num_tokens
            out[on] = (doc_id, sequence_number)
            on += 1
        return out


class DocumentPool:
    """Utility class of packing, maintains a `sequence_lengths->examples` mapping for a pool of documents"""

    def __init__(self, max_seq_len, pool_size):
        self.n_tokens_to_ex_id = [collections.deque() for _ in range(max_seq_len+1)]
        self.on = 0
        self.hist = np.zeros(max_seq_len+1, dtype=np.int32)
        self.hist_min = max_seq_len

    def add(self, ex):
        _n = ex["num_tokens"]
        self.n_tokens_to_ex_id[_n].append((self.on, ex["doc_id"]))
        self.hist[_n] += 1
        self.hist_min = min(_n, self.hist_min)
        self.on += 1

    def pop(self, seq_len):
        self.hist[seq_len] -= 1
        if seq_len == self.hist_min and self.hist[seq_len] == 0:
            self.hist_min = self.hist_min + np.argmax(self.hist[self.hist_min:] > 0)
            if self.hist[self.hist_min] == 0:
                self.hist_min = len(self.hist)
        return self.n_tokens_to_ex_id[seq_len].popleft()[1]

    def is_empty(self):
        return self.hist_min == len(self.hist)

    def find_at_most(self, ix):
        """Return the largest seq len that we have a document for and is <= `ix`, 0 if there is no such example"""
        # this function gets called a lot its important its fast
        if ix < self.hist_min:
            return 0
        else:
            return ix-np.argmax(self.hist[self.hist_min:ix+1][::-1])

    def get_first(self):
        if self.is_empty():
            raise ValueError()
        candidates = np.argwhere(self.hist > 0)[:, 0]
        return min(candidates, key=lambda x: self.n_tokens_to_ex_id[x][0][0])

    def get_all(self):
        """Return all document in the pool in the order they were entered"""
        examples = []
        ordering = []
        for ix, g in enumerate(self.n_tokens_to_ex_id):
            in_queue = list(g)
            data = np.zeros(len(in_queue), dtype=DOC_TOKENS_DTYPE)
            data["doc_id"] = [x[1] for x in in_queue]
            data["num_tokens"] = ix
            examples.append(data)
            ordering += [x[0] for x in in_queue]
        examples = np.concatenate(examples)
        examples = examples[np.argsort(ordering)]
        return examples


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
        to_skip = examples["num_tokens"] == max_seq_len

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

            next_len = seq_len + new_tokens

            if next_len > max_seq_len:
                # Write the best candidate we have found so far
                best_end, item_from_pool = best_candidate[:2]
                to_add = in_progress[:best_end]["doc_id"]
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
                    in_progress[on] = examples[on_example]
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
    remainder = num_tokens % seq_len
    lengths = [seq_len]*(num_tokens // seq_len)
    if remainder:
        if remainder < min_seq_len:
            lengths[-1] = min_seq_len
            remainder += (seq_len - min_seq_len)
        lengths.append(remainder)

    rng.shuffle(lengths)
    return lengths


@dataclass
class IterationConfig:
    """Specifies how to iterate through data stored in a set of data files"""
    paths: List[str]
    sampler: Optional[DataSamplingConfig] = None
    sequence_builder: Optional[SequenceBuilderConfig] = None


def build_iteration_order(
    data: Union[IterationConfig, List[str]],
    seq_len: int,
    seed: int,
    sizing: ImageTokenSizer,
    indexer: Optional[Indexer] = None,
    n_processes: Optional[int] = None
) -> np.ndarray:
    """Builds an array of DOC_SEQUENCE_DTYPE specifying how to iterate through the `data`

    data: Data to iterate through
    seq_len: max sequence length for this iteration
    seed: seed to this iteration
    sizing: image -> number_of_tokens mapping, needed to split/pack documents
    index: Indexer use to read the index files for the data, if None defaults to `VectorizedIndex`
    n_processes: Use multiple-processes when possible to speed up loading
    """
    if isinstance(data, list):
        data = IterationConfig(data)
    rng = np.random.RandomState(seed)
    if indexer is None:
        indexer = VectorizedIndexer()

    documents = []
    for file_id, data_f in enumerate(data.paths):
        data_file_documents = []

        logger.info(f"Reading data from {data_f}...")
        example_info = indexer.get_indices(data_f, sizing, MMStorageConfig())
        num_examples = len(example_info)

        is_below_seq_len = (example_info["num_tokens"] <= seq_len)
        no_preprocess = example_info[is_below_seq_len]
        out = np.zeros(len(no_preprocess), dtype=DOC_TOKENS_DTYPE)
        out["num_tokens"] = no_preprocess["num_tokens"]
        out["doc_id"]["start_byte"] = no_preprocess["start_byte"]
        out["doc_id"]["length"] = no_preprocess["num_bytes"]
        out["doc_id"]["file_id"] = file_id

        logger.info(f"Found {len(example_info)} examples with an "
                    f"average of {out['num_tokens'].mean():0.1f} tokens.")

        data_file_documents.append(out)

        example_info = example_info[~is_below_seq_len]
        if len(example_info) > 0:
            logger.info(f"Checking {len(example_info)} examples that are longer than {seq_len}...")
            n_truncated = 0
            split_data = NumpyList(DOC_TOKENS_DTYPE)
            t0 = perf_counter()
            for ex in example_info:
                if ex.pure_text:
                    lengths = split_example(rng, ex.num_tokens, seq_len)
                    on = 0
                    for chunk_tokens in lengths:
                        chunk_start_byte = ex.start_byte+on*2
                        split_data.append(((file_id, chunk_start_byte, chunk_tokens*2), chunk_tokens))
                        on += chunk_tokens*2
                else:
                    n_truncated += 1
                    split_data.append(((file_id, ex.start_byte, ex.num_bytes), ex.num_tokens))
            data_file_documents += split_data.get_blocks()
        documents.append(np.concatenate(data_file_documents))

    logger.info(f"Shuffling...")
    documents = shuffle(data.sampler, rng, documents)

    logger.info(f"Building sequences...")
    sequence_builder = build_sequence_builder(data.sequence_builder)
    idx = sequence_builder(documents, seq_len, n_processes)
    n_sequences = int(idx["sequence_number"][-1])+1
    n_tokens = documents["num_tokens"].astype(np.int64).sum()
    logger.info(f"Built {n_sequences} sequences of len {seq_len} from {len(documents)} examples and {n_tokens} tokens")
    tokens_per_seq = n_tokens/n_sequences
    ex_per_seq = len(documents)/n_sequences
    logger.info(f"Packing efficiency={tokens_per_seq/seq_len:0.3f}, {tokens_per_seq:0.1f} tokens and {ex_per_seq:0.1f} "
                f"examples per a sequence on average")
    return idx

