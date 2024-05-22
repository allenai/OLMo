"""
Use this script to inspect the data in given batches from a training run.
"""

import gzip
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

from cached_path import cached_path

from olmo.checkpoint import load_state_dict
from olmo.config import TrainConfig
from olmo.data import build_memmap_dataset, build_train_dataloader
from olmo.data.iterable_dataset import IterableDataset
from olmo.exceptions import OLMoCliError
from olmo.tokenizer import Tokenizer
from olmo.util import add_cached_path_clients, clean_opt, prepare_cli_environment


def get_global_train_examples_seen_at_step(step: int, trainer_state: dict, cfg: TrainConfig):
    global_step = trainer_state["global_step"]

    if global_step > step:
        raise ValueError(f"Step {step} must be after training first step {global_step}")

    global_train_examples_seen_this_epoch = trainer_state.get(
        "global_train_examples_seen_this_epoch",
        trainer_state.get(  # for backwards compatibility
            "global_train_examples_seen",
            trainer_state.get("global_data_step", global_step) * cfg.global_train_batch_size,
        ),
    )

    global_train_examples_seen_this_epoch += (step - global_step) * cfg.global_train_batch_size
    return global_train_examples_seen_this_epoch


def inspect_data_without_device_data_indices(run_path: str, *steps: int, world_size: int, ranks: List[int]):
    cfg = TrainConfig.load(
        cached_path(os.path.join(run_path, "config.yaml")), overrides=[clean_opt("--evaluators=[]"), clean_opt("--save_overwrite")]
    )
    cfg.data.num_workers = 1

    try:
        trainer_state = load_state_dict(run_path, "latest/train/rank0.pt", map_location="cpu")
    except FileNotFoundError:
        try:
            trainer_state = load_state_dict(run_path, "latest/train.pt", map_location="cpu")
        except FileNotFoundError:
            # Legacy checkpointing
            trainer_state = load_state_dict(run_path, "latest/rank0.pt", map_location="cpu")


    tokenizer = Tokenizer.from_train_config(cfg)

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg.save_folder = tmpdir

        for rank in ranks:
            # Set RANK env variable so that `get_local_rank()` returns the rank we want.
            os.environ["RANK"] = str(rank)

            for step in steps:
                dataloader = build_train_dataloader(cfg, world_size=world_size)
                assert isinstance(dataloader.dataset, IterableDataset)
                dataloader.dataset.start_index = get_global_train_examples_seen_at_step(step, trainer_state, cfg)

                batch = next(dataloader.__iter__())
                for i, batch_entry in enumerate(batch["input_ids"].tolist()):
                    example = tokenizer.decode(batch_entry)
                    print(f'[step={step}, rank={rank}, example={i}] "{example}"\n')


def main(run_path: str, *steps: int, world_size: Optional[int] = None, rank: Optional[int] = None):
    save_folder = Path(run_path)
    if not (save_folder / "data-indices").is_dir():
        assert world_size is not None
        ranks = [rank] if rank is not None else list(range(world_size))
        inspect_data_without_device_data_indices(run_path, *steps, world_size=world_size, ranks=ranks)
        return

    cfg = TrainConfig.load(save_folder / "config.yaml", overrides=[clean_opt("--evaluators=[]")])
    dataset = build_memmap_dataset(cfg, cfg.data)
    tokenizer = Tokenizer.from_train_config(cfg)

    if rank is None:
        num_indices_files = len(list((save_folder / "data-indices").glob("*.tsv.gz")))
        if world_size is not None and world_size != num_indices_files:
            raise ValueError(f"World size {world_size} does not match number of indices files {num_indices_files}")

        indices_files = {
            rank: gzip.open(save_folder / "data-indices" / f"rank{rank}.tsv.gz", "rt")
            for rank in range(num_indices_files)
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

    add_cached_path_clients()

    try:
        run_path, world_size, rank, steps = (
            sys.argv[1],
            int(sys.argv[2]),
            int(sys.argv[3]),
            [int(i) for i in sys.argv[4:]],
        )
    except (IndexError, ValueError):
        raise OLMoCliError(f"Usage: {sys.argv[0]} [RUN_PATH] [WORLD_SIZE] [RANK] [STEP_NUMBER...]")

    main(run_path, *steps, world_size=world_size, rank=rank if rank >= 0 else None)
