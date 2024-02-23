from collections import Counter
from pathlib import Path

import numpy as np

from olmo.mm_data.sequence_index import IDX_DTYPE, find_sequence_start, chunk_example, balanced_merge, DataSampling


def _test_seek(sequence_counts, idx_file, to_test):
    examples = []
    for seq_num, c in enumerate(sequence_counts):
        examples += [seq_num] * c
    examples = np.array(examples)
    idx_arr = []
    for seq_num in examples:
        idx_arr.append(np.array((seq_num, tuple(np.random.randint(0, 1024, (3,)))), IDX_DTYPE))
    idx_arr = np.stack(idx_arr)
    with open(idx_file, "wb") as f:
        f.write(idx_arr.tobytes())

    for target_seq, chunk_size in to_test:
        expected = np.searchsorted(examples, target_seq)
        actual = find_sequence_start(
            idx_file, target_seq, len(examples), chunk_size,
            len(examples), len(sequence_counts)
        )
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


def test_chunk_random():
    for seed in range(50):
        rng = np.random.RandomState(1581 + seed)
        seq_len = rng.randint(3, 18)
        num_tokens = seq_len + rng.randint(1, 57)
        min_seq_len = rng.randint(1, (seq_len - 1) // 2 + 1)
        out = chunk_example(rng, num_tokens, seq_len, min_seq_len=min_seq_len)
        assert all(min_seq_len <= x <= seq_len for x in out)
        assert sum(out) == num_tokens


def test_balanced_merge_same_lens():
    out = balanced_merge([
        np.arange(0, 3),
        np.arange(3, 6),
        np.arange(6, 9)
    ])
    assert set(out[:3]) == {0, 3, 6}
    assert set(out[3:6]) == {1, 4, 7}
    assert set(out[6:9]) == {2, 5, 8}


def test_balanced_merge_different_lens():
    out = balanced_merge([
        np.arange(0, 6),
        np.arange(30, 33),
        np.arange(100, 118),
    ])
    assert set(out[:9]) == ({0, 1, 30} | set(range(100, 106)))
    assert set(out[9:18]) == ({2, 3, 31} | set(range(106, 112)))
    assert set(out[18:]) == ({4, 5, 32} | set(range(112, 118)))


def test_stratify():
    data1 = np.arange(8)
    data2 = np.arange(12) + 100
    out = DataSampling([3, 2], stratify=True)(np.random.RandomState(1231), [data1, data2])

    assert len(out) == 48

    from1 = out[out < 100]
    assert len(from1) == 24
    for i in range(3):
        assert set(from1[i*8:(i+1)*8]) == set(data1)

    from2 = out[out >= 100]
    assert len(from2) == 24
    assert set(from2[:12]) == set(data2)
    assert set(from2[12:]) == set(data2)

    # For every 2 examples, one should be in data1 and one should be in data2
    pairs = out.reshape(24, 2)
    assert np.all(np.sum(pairs >= 100, -1) == 1)
