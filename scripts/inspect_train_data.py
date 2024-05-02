"""
Use this script to inspect the data in given batches from a training run.
"""

import gzip
import sys
from pathlib import Path
from typing import Optional

from olmo.config import TrainConfig
from olmo.data import build_memmap_dataset
from olmo.exceptions import OLMoCliError
from olmo.tokenizer import Tokenizer
from olmo.util import clean_opt, prepare_cli_environment


def main(save_folder: Path, *steps: int, rank: Optional[int] = None):
    cfg = TrainConfig.load(save_folder / "config.yaml", overrides=[clean_opt("--evaluators=[]")])
    dataset = build_memmap_dataset(cfg, cfg.data)
    tokenizer = Tokenizer.from_train_config(cfg)

    if rank is None:
        world_size = len(list((save_folder / "data-indices").glob("*.tsv.gz")))
        indices_files = {
            rank: gzip.open(save_folder / "data-indices" / f"rank{rank}.tsv.gz", "rt")
            for rank in range(world_size)
        }
    else:
        indices_files = {rank: gzip.open(save_folder / "data-indices" / f"rank{rank}.tsv.gz", "rt")}

    try:
        for step in sorted(steps):
            for rank in sorted(indices_files.keys()):
                for line in indices_files[rank]:
                    if line.startswith(f"{step}\t"):
                        indices = [int(i) for i in line.strip().split("\t")[1:]]
                        for i, index in enumerate(indices):
                            token_ids = dataset[index]["input_ids"]
                            example = tokenizer.decode(token_ids.tolist())
                            print(f'[step={step}, rank={rank}, example={i}] "{example}"\n')
                    else:
                        continue
    finally:
        for f in indices_files.values():
            f.close()


if __name__ == "__main__":
    prepare_cli_environment()

    try:
        save_folder, rank, steps = sys.argv[1], int(sys.argv[2]), [int(i) for i in sys.argv[3:]]
    except (IndexError, ValueError):
        raise OLMoCliError(f"Usage: {sys.argv[0]} [SAVE_FOLDER] [RANK] [STEP_NUMBER...]")

    main(Path(save_folder), *steps, rank=rank if rank >= 0 else None)
