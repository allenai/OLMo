from pathlib import Path
from typing import List

import numpy as np

from dolma.data.mmap_dataset import MMapDataset
from dolma.data.tokenizer import Tokenizer


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
    dataset = MMapDataset(tokens_fname, chunk_size=chunk_size)
    assert len(dataset) == len(all_token_ids) // chunk_size
    for idx in range(len(dataset)):
        x = dataset[idx]
        assert x.shape == (chunk_size,)
