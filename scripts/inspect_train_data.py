"""
Use this script to inspect the data in given batches from a training run.
"""

import sys
from pathlib import Path

import numpy as np

from olmo.config import TrainConfig
from olmo.data import build_memmap_dataset
from olmo.exceptions import OlmoCliError
from olmo.tokenizer import Tokenizer
from olmo.util import clean_opt, prepare_cli_environment


def main(save_folder: Path, *steps: int):
    cfg = TrainConfig.load(save_folder / "config.yaml", overrides=[clean_opt("--evaluators=[]")])
    dataset = build_memmap_dataset(cfg, cfg.data)
    tokenizer = Tokenizer.from_train_config(cfg)

    global_indices = np.memmap(save_folder / "train_data" / "global_indices.npy", mode="r", dtype=np.uint64)

    for step in steps:
        indices = global_indices[cfg.global_train_batch_size * (step - 1) : cfg.global_train_batch_size * step]
        for i, index in enumerate(indices):
            token_ids = dataset[index]["input_ids"]
            example = tokenizer.decode(token_ids.tolist())
            print(f'[step={step}, example={i}] "{example}"\n')


if __name__ == "__main__":
    prepare_cli_environment()

    try:
        save_folder, steps = sys.argv[1], [int(i) for i in sys.argv[3:]]
    except (IndexError, ValueError):
        raise OlmoCliError(f"Usage: {sys.argv[0]} [SAVE_FOLDER] [STEP_NUMBER...]")

    main(Path(save_folder), *steps)
