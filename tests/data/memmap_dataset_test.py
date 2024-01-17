from pathlib import Path
from typing import List

import numpy as np

from olmo.data.memmap_dataset import MemMapDataset
from olmo.tokenizer import Tokenizer


def test_mmap_dataset(tmp_path: Path):
    mmap1 = np.memmap(tmp_path / "mmap1.npy", mode="w+", dtype=np.uint16, shape=(16,))
    mmap1[:] = np.array(list(range(16)), dtype=np.uint16)
    mmap1.flush()

    mmap2 = np.memmap(tmp_path / "mmap2.npy", mode="w+", dtype=np.uint16, shape=(16,))
    mmap2[:] = np.array(list(range(16, 32)), dtype=np.uint16)
    mmap2.flush()

    ds = MemMapDataset(tmp_path / "mmap1.npy", tmp_path / "mmap2.npy", chunk_size=4)
    assert ds[0]["input_ids"].tolist() == [0, 1, 2, 3]
    assert ds[1]["input_ids"].tolist() == [4, 5, 6, 7]
    assert ds[7]["input_ids"].tolist() == [28, 29, 30, 31]


def test_mmap_dataset_with_label_mask(tmp_path: Path):
    mmap1 = np.memmap(tmp_path / "mmap1.npy", mode="w+", dtype=np.uint16, shape=(16,))
    mmap1[:] = np.array(list(range(16)), dtype=np.uint16)
    mmap1.flush()

    mask1 = [True] * 16
    mask1[1] = False
    mask_mmap1 = np.memmap(tmp_path / "mask_mmap1.npy", mode="w+", dtype=np.bool_, shape=(16,))
    mask_mmap1[:] = np.array(mask1, dtype=np.bool_)
    mask_mmap1.flush()

    mmap2 = np.memmap(tmp_path / "mmap2.npy", mode="w+", dtype=np.uint16, shape=(16,))
    mmap2[:] = np.array(list(range(16, 32)), dtype=np.uint16)
    mmap2.flush()

    mask2 = [True] * 16
    mask2[-1] = False
    mask_mmap2 = np.memmap(tmp_path / "mask_mmap2.npy", mode="w+", dtype=np.bool_, shape=(16,))
    mask_mmap2[:] = np.array(mask2, dtype=np.bool_)
    mask_mmap2.flush()

    ds = MemMapDataset(
        tmp_path / "mmap1.npy",
        tmp_path / "mmap2.npy",
        chunk_size=4,
        label_mask_paths=[tmp_path / "mask_mmap1.npy", tmp_path / "mask_mmap2.npy"],
    )
    assert ds[0]["input_ids"].tolist() == [0, 1, 2, 3]
    assert ds[0]["label_mask"].tolist() == [True, False, True, True]
    assert ds[1]["input_ids"].tolist() == [4, 5, 6, 7]
    assert ds[7]["input_ids"].tolist() == [28, 29, 30, 31]
    assert ds[7]["label_mask"].tolist() == [True, True, True, False]


def test_mmap_dataset_with_metadata(tokenizer: Tokenizer, tmp_path: Path, lorem_ipsum_docs: List[str]):
    chunk_size = 24

    # Tokenize input, adding the EOS token between documents.
    all_token_ids: List[int] = []
    for token_ids in tokenizer.encode_batch(lorem_ipsum_docs):
        all_token_ids.extend(token_ids)

    # Write tokens to memory-mapped array.
    tokens_fname = tmp_path / "tokens.npy"
    mmap = np.memmap(tokens_fname, dtype=np.uint16, mode="w+", shape=(len(all_token_ids),))
    mmap[:] = all_token_ids
    mmap.flush()
    del mmap

    # Now initialize the dataset and validate it.
    dataset = MemMapDataset(tokens_fname, chunk_size=chunk_size, metadata={"label": "test-data"})
    assert len(dataset) == len(all_token_ids) // chunk_size
    for idx in range(len(dataset)):
        x = dataset[idx]
        input_ids = x["input_ids"]
        assert input_ids.shape == (chunk_size,)
        assert x["metadata"]["label"] == "test-data"


def test_concat_mmap_datasets(tmp_path: Path):
    # Write some data to disk.
    mmap1 = np.memmap(tmp_path / "tokens1.npy", dtype=np.uint16, mode="w+", shape=(16,))
    mmap1[:] = list(range(16))
    mmap1.flush()
    mmap2 = np.memmap(tmp_path / "tokens2.npy", dtype=np.uint16, mode="w+", shape=(8,))
    mmap2[:] = list(range(8))
    mmap2.flush()
    del mmap1, mmap2

    # Initialize two datasets, one for each file.
    ds1 = MemMapDataset(tmp_path / "tokens1.npy", chunk_size=3, metadata={"label": "test1"})
    assert len(ds1) == 5
    ds2 = MemMapDataset(tmp_path / "tokens2.npy", chunk_size=3, metadata={"label": "test2"})
    assert len(ds2) == 2

    # Now concatenate them.
    ds = ds1 + ds2
    assert len(ds) == 7
    assert ds[0]["input_ids"].tolist() == [0, 1, 2]
    assert ds[0]["metadata"]["label"] == "test1"
    assert ds[6]["input_ids"].tolist() == [3, 4, 5]
    # Should get the same with negative index.
    assert ds[-1]["input_ids"].tolist() == [3, 4, 5]
    assert ds[-1]["metadata"]["label"] == "test2"
