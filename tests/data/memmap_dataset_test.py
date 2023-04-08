from pathlib import Path
from typing import List

import numpy as np

from olmo.data.memmap_dataset import MemMapDataset
from olmo.tokenizer import Tokenizer


def test_mmap_dataset(tokenizer: Tokenizer, tmp_path: Path, lorem_ipsum_docs: List[str]):
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
    dataset = MemMapDataset(tokens_fname, chunk_size=chunk_size)
    assert len(dataset) == len(all_token_ids) // chunk_size
    for idx in range(len(dataset)):
        x = dataset[idx]
        assert x.shape == (chunk_size,)


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
    ds1 = MemMapDataset(tmp_path / "tokens1.npy", chunk_size=3)
    assert len(ds1) == 5
    ds2 = MemMapDataset(tmp_path / "tokens2.npy", chunk_size=3)
    assert len(ds2) == 2

    # Now concatenate them.
    ds = ds1 + ds2
    assert len(ds) == 7
    assert ds[0].tolist() == [0, 1, 2]
    assert ds[6].tolist() == [3, 4, 5]
    # Should get the same with negative index.
    assert ds[-1].tolist() == [3, 4, 5]
