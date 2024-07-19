import argparse

import numpy as np


def create_sample_data(vocab_size: int = 50257, data_size: int = 2048 * 64, path: str = "sample-tokens.npy"):
    mmap = np.memmap(path, dtype=np.uint16, mode="w+", shape=(data_size,))
    mmap[:] = np.random.randint(0, vocab_size, (data_size,))
    mmap.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create sample data for mup testing",
    )

    parser.add_argument("--vocab-size", type=int, default=50257)
    parser.add_argument("--data-size", type=int, default=2048 * 64)
    parser.add_argument("--path", type=str, default="test_fixtures/sample-tokens.npy")

    args = parser.parse_args()

    create_sample_data(args.vocab_size, args.data_size, args.path)
