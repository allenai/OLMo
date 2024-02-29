from pathlib import Path

import numpy as np

from olmo.mm_data.sequence_index import DOC_SEQUENCE_DTYPE, find_sequence_start


def _test_seek(sequence_counts, idx_file, to_test):
    examples = []
    for seq_num, c in enumerate(sequence_counts):
        examples += [seq_num] * c
    examples = np.array(examples)
    idx_arr = []
    for seq_num in examples:
        idx_arr.append(np.array((tuple(np.random.randint(0, 1024, (3,))), seq_num), DOC_SEQUENCE_DTYPE))
    idx_arr = np.stack(idx_arr)
    with open(idx_file, "wb") as f:
        f.write(idx_arr.tobytes())

    for target_seq, chunk_size in to_test:
        expected = np.searchsorted(examples, target_seq)
        actual = find_sequence_start(idx_file, target_seq, len(examples), len(sequence_counts), chunk_size)
        assert actual == expected


def test_seek_small(tmp_path: Path):
    idx_file = tmp_path.joinpath("index.bin")
    _test_seek([1, 5, 1, 3, 7, 4], idx_file, [
        (2, 20), (4, 20), (4, 1), (5, 2)
    ])

    _test_seek([18, 1, 7, 3, 9, 4, 5], idx_file, [
        (0, 8), (6, 4), (1, 3), (1, 2), (6, 2)
    ])


def test_seek_random(tmp_path: Path):
    idx_file = tmp_path.joinpath("index.bin")
    n_seq = 20
    n_targets = 5
    for seed in range(100):
        rng = np.random.RandomState(seed + 991741)
        counts = rng.randint(1, 10, (n_seq,))
        targets = rng.randint(0, n_seq, (n_targets,))
        step_sizes = rng.randint(1, 6, (n_targets,))
        _test_seek(counts, idx_file, np.stack([targets, step_sizes], 1))
