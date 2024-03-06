import pytest

from olmo.mm_data.data_iteration import split_example, balanced_merge, DataSamplingConfig, OptimizeLast, \
    DOC_TOKENS_DTYPE, SequenceBuilder, Sequential, ParallelizableSequenceBuilder, shuffle, SequentialSplitText, \
    DOC_INFO_DTYPE, DOC_SEQUENCE_DTYPE, reorder_sequences

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


def _build_docs(tokens):
    data = np.zeros((len(tokens),), DOC_INFO_DTYPE)
    data["num_tokens"] = tokens
    data["doc_id"]["file_id"] = np.arange(len(tokens))
    data["doc_id"]["length"] = np.array(tokens)*2
    return data


def test_sequential_split_text():
    data = _build_docs([12, 3, 10, 25, 51])
    data["pure_text"] = True
    split = SequentialSplitText()(data, 20)
    expected = [
        [0, 1, 2],  # [12, 3, 5 from the 10]
        [2, 3],     # [5 from the 10, 15 from 25]
        [3, 4],     # [10 from the 25, 10 from 51]
        [4],        # [10-30 from 51]
        [4],        # [30-50 from 51]
        [4]         # [50-51 from 51]
    ]
    assert len(split) == sum(len(x) for x in expected)
    on = 0
    for seq_num, expected_seq in enumerate(expected):
        actual = split[on:on+len(expected_seq)]
        on += len(expected_seq)
        assert actual["doc_id"]["file_id"].tolist() == expected_seq
        assert np.all(actual["sequence_number"] == seq_num)
        if seq_num == (len(expected) - 1):
            assert actual["doc_id"]["length"].sum() // 2 == 1
        else:
            assert actual["doc_id"]["length"].sum() // 2 == 20


def test_reorder():
    data = np.zeros((3,), DOC_SEQUENCE_DTYPE)
    data["sequence_number"] = [0, 1, 1]
    idx = reorder_sequences(data, np.array([1, 0]))
    assert np.all(idx["sequence_number"] == [0, 0, 1])

    data = np.zeros((4,), DOC_SEQUENCE_DTYPE)
    data["sequence_number"] = [0, 1, 1, 2]
    idx = reorder_sequences(data, np.array([2, 0, 1]))
    assert np.all(idx["sequence_number"] == [0, 0, 1, 2])


def reorder_slow(idx, sequence_ixs):
    new_seq_ids = sequence_ixs[idx["sequence_number"]]
    idx["sequence_number"] = new_seq_ids                # change to the new sequence number
    return idx[np.argsort(new_seq_ids, kind="stable")]  # sort to fix the ordering


def test_reorder_rng():
    for i in range(3):
        rng = np.random.RandomState(i*31 + 91193)
        data = np.zeros((30,), DOC_SEQUENCE_DTYPE)
        seq_num = (rng.random(len(data)) > 0.7).astype(np.uint64)
        seq_num[0] = 0
        data["doc_id"]["start_byte"] = np.arange(len(data), dtype=np.uint64)
        data["sequence_number"] = np.cumsum(seq_num)
        n_seq = int(data["sequence_number"][-1]) + 1

        sequence_ixs = np.arange(int(data["sequence_number"][-1])+1, dtype=np.int64)
        rng.shuffle(sequence_ixs)

        actual = reorder_sequences(data.copy(), sequence_ixs)

        expected = reorder_slow(data, sequence_ixs)
        assert np.all(expected == actual)

        # Some sanity checks just to double-check
        seq_num = actual["sequence_number"].astype(np.int64)
        deltas = seq_num[1:] - seq_num[:-1]
        assert np.all(np.unique(seq_num) == np.arange(n_seq, dtype=np.int64))
        assert seq_num[0] == 0
        assert np.all(deltas <= 1)
        assert np.all(deltas >= 0)
        assert np.all(np.sort(actual["doc_id"]["start_byte"]) == np.arange(len(data), dtype=np.int64))


def test_optimize_last():
    data = _build_docs([12, 1, 7, 5, 12])
    out = OptimizeLast(pool_size=2)(data, 20)
    assert out["doc_id"]["file_id"][:2].tolist() == [2, 0]
    assert out["sequence_number"][:2].tolist() == [0, 0]

    assert out["doc_id"]["file_id"][2:].tolist() == [3, 4, 1]
    assert out["sequence_number"][2:].tolist() == [1, 1, 1]


@pytest.mark.parametrize("builder", [Sequential(), SequentialSplitText(), OptimizeLast(32),
                                     ParallelizableSequenceBuilder(OptimizeLast(7), n_splits=3)])
def test_sequence_building_random(builder: SequenceBuilder):
    for i in range(3):
        rng = np.random.RandomState(i*31 + 91193)
        seq_len = 1000 + rng.randint(0, 200, dtype=np.uint32)
        data = np.zeros((200,), DOC_INFO_DTYPE)
        data["doc_id"]["file_id"] = np.arange(len(data), dtype=np.int64)
        data["num_tokens"] = rng.randint(100, 600, (len(data),), dtype=np.uint32)
        data["num_tokens"][rng.choice(len(data), 4, False)] = seq_len
        data["pure_text"] = rng.random((len(data))) < 0.5

        packed = builder(data, seq_len)

        if not isinstance(builder, SequentialSplitText):
            # Must contain the original examples
            assert len(packed) == len(data)
            assert np.all(np.sort(packed["doc_id"]["file_id"]) == np.arange(len(data), dtype=np.int64))
        else:
            # Allowed to repeat examples
            assert len(packed) >= len(data)
            assert np.all(np.unique(packed["doc_id"]["file_id"]) == np.arange(len(data), dtype=np.int64))

        # sequence numbers must be consecutive and start at 0
        seq_num = packed["sequence_number"].astype(np.int64)
        deltas = seq_num[1:] - seq_num[:-1]
        assert seq_num[0] == 0
        assert np.all(deltas <= 1)
        assert np.all(deltas >= 0)

        # sequences must avoid overflowing seq_len
        for k in range(seq_num[-1]+1):
            assert packed["doc_id"]["length"][packed["sequence_number"] == k].sum()//2 <= seq_len
            if not isinstance(builder, SequentialSplitText):
                ixs = packed["doc_id"]["file_id"][seq_num == k]
                assert data["num_tokens"][ixs].sum() <= seq_len
