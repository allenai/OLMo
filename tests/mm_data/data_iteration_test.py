import pytest

from olmo.mm_data.data_iteration import split_example, balanced_merge, DataSamplingConfig, OptimizeLast, \
    DOC_TOKENS_DTYPE, SequenceBuilder, Sequential, ParallelizableSequenceBuilder, shuffle

import numpy as np


def test_split_random():
    for seed in range(20):
        rng = np.random.RandomState(1581 + seed)
        seq_len = rng.randint(3, 18)
        num_tokens = seq_len + rng.randint(1, 57)
        min_seq_len = rng.randint(1, (seq_len - 1) // 2 + 1)
        out = split_example(rng, num_tokens, seq_len, min_seq_len=min_seq_len)
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
    out = shuffle(DataSamplingConfig([3, 2], stratify=True), np.random.RandomState(1231), [data1, data2])

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


def test_optimize_last():
    tokens = [12, 1, 7, 5, 12]
    data = np.zeros((len(tokens),), DOC_TOKENS_DTYPE)
    data["num_tokens"] = tokens
    data["doc_id"]["file_id"] = np.arange(len(tokens))
    out = OptimizeLast(pool_size=2)(data, 20)
    assert out["doc_id"]["file_id"][:2].tolist() == [2, 0]
    assert out["sequence_number"][:2].tolist() == [0, 0]

    assert out["doc_id"]["file_id"][2:].tolist() == [3, 4, 1]
    assert out["sequence_number"][2:].tolist() == [1, 1, 1]


@pytest.mark.parametrize("builder", [OptimizeLast(32), Sequential(),
                                     ParallelizableSequenceBuilder(OptimizeLast(7), n_splits=3)])
def test_sequence_building_random(builder: SequenceBuilder):
    for i in range(3):
        rng = np.random.RandomState(i*31 + 91193)
        seq_len = 1000 + rng.randint(0, 200, dtype=np.uint32)
        data = np.zeros((200,), DOC_TOKENS_DTYPE)
        data["doc_id"]["start_byte"] = np.arange(len(data), dtype=np.int64)
        data["num_tokens"] = rng.randint(100, 600, (len(data),), dtype=np.uint32)
        data["num_tokens"][rng.choice(len(data), 4, False)] = seq_len

        packed = builder(data, seq_len)

        # Must contain the original examples
        assert len(packed) == len(data)
        assert np.all(np.sort(packed["doc_id"]["start_byte"]) == np.arange(len(data), dtype=np.int64))

        # sequence numbers must be consecutive and start at 0
        seq_num = packed["sequence_number"].astype(np.int64)
        deltas = seq_num[1:] - seq_num[:-1]
        assert seq_num[0] == 0
        assert np.all(deltas <= 1)
        assert np.all(deltas >= 0)

        # sequences must avoid overflowing seq_len
        for k in range(seq_num[-1]+1):
            ixs = packed["doc_id"]["start_byte"][seq_num == k]
            assert data["num_tokens"][ixs].sum() <= seq_len
