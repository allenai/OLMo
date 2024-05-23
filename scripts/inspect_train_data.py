"""
Use this script to inspect the data in given batches from a training run.
"""

import argparse
import gzip
from pathlib import Path
from typing import Optional

from olmo.config import TrainConfig
from olmo.data import build_memmap_dataset
from olmo.tokenizer import Tokenizer
from olmo.util import clean_opt, prepare_cli_environment


def main(
    run_path: str,
    *steps: int,
    world_size: Optional[int] = None,
    rank: Optional[int] = None,
    reference_step: Optional[int] = None,
    use_data_indices: bool = True,
):
    save_folder = Path(run_path)

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

    parser = argparse.ArgumentParser()

    parser.add_argument("run_path", help="Path to run of which you want to inspect training data")
    parser.add_argument(
        "rank",
        type=int,
        help="Device rank for which you want to see training data. Set to `-1` to get all ranks.",
    )
    parser.add_argument(
        "steps", nargs="+", type=int, help="Steps of run for which you want to see training data"
    )
    parser.add_argument(
        "--no_data_indices",
        action="store_false",
        dest="use_data_indices",
        help="If set, this script acts as if data indices are not present.",
    )
    parser.add_argument(
        "--checkpoint_num",
        type=int,
        help="Step number of checkpoint from which training state is to be obtained. Required when data indices are not present.",
    )
    parser.add_argument(
        "--world_size", type=int, help="World size. Required when data indices are not present."
    )

    args = parser.parse_args()

    main(
        args.run_path,
        *args.steps,
        world_size=args.world_size,
        rank=args.rank if args.rank >= 0 else None,
        reference_step=args.checkpoint_num if args.checkpoint_num >= 0 else None,
        use_data_indices=args.use_data_indices,
    )
