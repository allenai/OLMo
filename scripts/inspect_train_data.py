"""
Use this script to inspect the data in given batches from a training run.
"""

import argparse
import gzip
import os
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

from cached_path import cached_path

from olmo.checkpoint import load_state_dict
from olmo.config import TrainConfig
from olmo.data import build_memmap_dataset, build_train_dataloader
from olmo.data.iterable_dataset import IterableDataset
from olmo.tokenizer import Tokenizer
from olmo.util import add_cached_path_clients, clean_opt, prepare_cli_environment


def get_global_train_examples_seen_before_step(step: int, trainer_state: dict, cfg: TrainConfig):
    global_step = trainer_state["global_step"]

    if global_step >= step:
        raise ValueError(f"Step {step} must be after training first step {global_step}")

    global_train_examples_seen_this_epoch = trainer_state.get(
        "global_train_examples_seen_this_epoch",
        trainer_state.get(  # for backwards compatibility
            "global_train_examples_seen",
            trainer_state.get("global_data_step", global_step) * cfg.global_train_batch_size,
        ),
    )

    # Subtract 1 from step because we want to be just before that step
    global_train_examples_seen_this_epoch += (step - 1 - global_step) * cfg.global_train_batch_size
    return global_train_examples_seen_this_epoch


def _revert_data_mounts(cfg: TrainConfig, mounts: List[Tuple[str, str]]):
    if cfg.data.paths is None:
        return

    new_paths = []
    for path in cfg.data.paths:
        new_path = path

        # "Revert" the first mount that matches
        for source, target in mounts:
            if path.startswith(target):
                new_path = source + path[len(target) :]
                break

        new_paths.append(new_path)

    cfg.data.paths = new_paths


def inspect_data_without_device_data_indices(
    run_path: str,
    *steps: int,
    world_size: int,
    ranks: List[int],
    reference_step: int,
    mounts: Optional[List[Tuple[str, str]]] = None,
):
    cfg = TrainConfig.load(
        cached_path(os.path.join(run_path, f"step{reference_step}/config.yaml")),
        overrides=[clean_opt("--evaluators=[]"), clean_opt("--save_overwrite")],
    )
    cfg.data.num_workers = 1

    if mounts is not None:
        _revert_data_mounts(cfg, mounts)

    if cfg.global_train_batch_size % world_size != 0:
        raise ValueError(f"World size must divide global_train_batch_size {cfg.global_train_batch_size}")
    cfg.device_train_batch_size = cfg.global_train_batch_size // world_size

    try:
        trainer_state = load_state_dict(run_path, f"step{reference_step}/train/rank0.pt", map_location="cpu")
    except FileNotFoundError:
        try:
            # Unsharded checkpointing
            trainer_state = load_state_dict(run_path, f"step{reference_step}/train.pt", map_location="cpu")
        except FileNotFoundError:
            # Legacy checkpointing
            trainer_state = load_state_dict(run_path, f"step{reference_step}/rank0.pt", map_location="cpu")

    tokenizer = Tokenizer.from_train_config(cfg)

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg.save_folder = tmpdir

        # Build dataloader in rank 0 to generate the indices file
        os.environ["RANK"] = "0"
        dataloader = build_train_dataloader(cfg, world_size=world_size)

        for rank in ranks:
            os.environ["RANK"] = str(rank)
            # Set FS_LOCAL_RANK to a non-zero number so that global data indices are not rewritten
            os.environ["FS_LOCAL_RANK"] = "1"

            for step in steps:
                # With the current implementation, this does not rebuild the global data indices if the FS local rank is non-zero
                dataloader = build_train_dataloader(cfg, world_size=world_size)
                assert isinstance(dataloader.dataset, IterableDataset)
                dataloader.dataset.start_index = get_global_train_examples_seen_before_step(
                    step, trainer_state, cfg
                )
                batch = next(iter(dataloader))
                for i, (batch_entry, instance_mask) in enumerate(
                    zip(batch["input_ids"].tolist(), batch["instance_mask"].tolist())
                ):
                    masked_instance = not instance_mask
                    example = tokenizer.decode(batch_entry)
                    print(f'[step={step}, rank={rank}, example={i}, masked={masked_instance}] "{example}"\n')


def main(
    run_path: str,
    *steps: int,
    world_size: Optional[int] = None,
    rank: Optional[int] = None,
    reference_step: Optional[int] = None,
    use_data_indices: bool = True,
    mounts: Optional[List[Tuple[str, str]]] = None,
):
    save_folder = Path(run_path)
    if not use_data_indices or not (save_folder / "data-indices").is_dir():
        assert world_size is not None
        assert reference_step is not None
        ranks = [rank] if rank is not None else list(range(world_size))
        inspect_data_without_device_data_indices(
            run_path,
            *steps,
            world_size=world_size,
            ranks=ranks,
            reference_step=reference_step,
            mounts=mounts,
        )
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

    parser = argparse.ArgumentParser()

    parser.add_argument("run_path", help="Path to run of which you want to inspect training data")
    parser.add_argument(
        "rank",
        type=int,
        help="Device rank for which you want to see training data. Set to `-1` to get all ranks.",
    )
    parser.add_argument("steps", nargs="+", type=int, help="Steps of run for which you want to see training data")
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
    parser.add_argument("--world_size", type=int, help="World size. Required when data indices are not present.")
    parser.add_argument(
        "--mount",
        default=[],
        nargs=2,
        action="append",
        dest="mounts",
        help="Directory mounts used in the original run. Example: to indicate that 'weka://' was mounted to '/weka/', pass weka:// /weka/",
    )

    args = parser.parse_args()

    main(
        args.run_path,
        *args.steps,
        world_size=args.world_size,
        rank=args.rank if args.rank >= 0 else None,
        reference_step=args.checkpoint_num if args.checkpoint_num >= 0 else None,
        use_data_indices=args.use_data_indices,
        mounts=args.mounts,
    )
