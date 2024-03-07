from pathlib import Path

import numpy as np

from olmo.mm_data.sequence_index import DOC_SEQUENCE_DTYPE, find_sequence_start, compress_index, decompress_index


def _test_seek(sequence_counts, to_test):
    examples = []
    for seq_num, c in enumerate(sequence_counts):
        examples += [seq_num] * c
    idx_arr = np.array(examples)

    def _read(on, n):
        return idx_arr[on:on+n]

    for target_seq, chunk_size in to_test:
        expected = np.searchsorted(examples, target_seq)
        actual = find_sequence_start(_read, target_seq, len(examples), len(sequence_counts), chunk_size)
        assert actual == expected


def test_seek_small():
    _test_seek([1, 5, 1, 3, 7, 4], [
        (2, 20), (4, 20), (4, 1), (5, 2)
    ])

    _test_seek([18, 1, 7, 3, 9, 4, 5], [
        (0, 8), (6, 4), (1, 3), (1, 2), (6, 2)
    ])


def test_seek_random():
    n_seq = 20
    n_targets = 5
    for seed in range(100):
        rng = np.random.RandomState(seed + 991741)
        counts = rng.randint(1, 10, (n_seq,))
        targets = rng.randint(0, n_seq, (n_targets,))
        step_sizes = rng.randint(1, 6, (n_targets,))
        _test_seek(counts, np.stack([targets, step_sizes], 1))


def test_compress():
    data = np.zeros(2, DOC_SEQUENCE_DTYPE)
    data["doc_id"]["file_id"][:] = [1231, 0]
    data["doc_id"]["length"][:] = [0, 2**15 + 10]
    data["doc_id"]["start_byte"][:] = [2^47, 2**48 - 1]
    data["sequence_number"][:] = [2**40, 2**48 - 1]
    actual = decompress_index(compress_index(data))
    assert np.all(actual == data)
