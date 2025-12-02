"""
This script reproduces the actual training order for a run, taking into
account the world size and number of data_loader workers used.

It requires the config file for the run and the world size to be
specified. An existing global indices file can be specified
if available, otherwise it will be built from the config.

This is only necessary to recover order for training runs made prior to this commit--

https://github.com/allenai/LLM/pull/239/commits/3aa9af760dd6732cb8561a45dd649bacfebfc266

Example:

python scripts/recover_training_order.py \
    configs/v1-mix-medium-mcli.yaml \
    ./recovered_training_order.npy \
    32 \
    --global_indices  /tmp/run/data/global_indices.npy


Restarts aren't supported yet, but this script can be run separately
for individual segments of a run if the starting/stopping iteration numbers are known.
"""

from pathlib import Path

import click
import numpy as np
from tqdm import tqdm

from olmo.config import TrainConfig
from olmo.data import IterableDataset, build_memmap_dataset
from olmo.util import prepare_cli_environment


def build_global_indices(world_size: int, config: TrainConfig) -> np.memmap:
    work_dir = Path("/tmp/run")
    data_work_dir = work_dir / "data"
    data_work_dir.mkdir(exist_ok=True, parents=True)

    config.save_folder = str(work_dir)
    dataset = build_memmap_dataset(config, config.data)

    _ = IterableDataset(
        dataset,
        seed=config.seed,
        shuffle=True,
        drop_last=True,
        work_dir=data_work_dir,
        world_size=world_size,
        rank=0,
        fs_local_rank=0,
    )

    global_indices = np.memmap("/tmp/run/data/global_indices.npy", mode="r", dtype=np.uint64)
    return global_indices


def reproduce_train_order(global_indices: np.memmap, world_size: int, output_filename, config: TrainConfig):
    global_batch_size = config.global_train_batch_size
    n_workers = config.data.num_workers
    loader_chunk_size = global_batch_size * n_workers

    train_order = np.memmap(output_filename, dtype="uint64", mode="w+", shape=(len(global_indices),))

    pbar = tqdm(total=len(global_indices))
    i = 0
    while i < len(global_indices):
        if i % (loader_chunk_size * 10) == 0:
            pbar.update(loader_chunk_size * 10)
        loader_chunk = global_indices[i : i + loader_chunk_size]
        for j in range(n_workers):
            for k in range(world_size):
                per_device_slice = loader_chunk[k::world_size]
                per_worker_slice = per_device_slice[j::n_workers]
                train_order[i : i + len(per_worker_slice)] = per_worker_slice
                i += len(per_worker_slice)
    pbar.close()


@click.command()
@click.argument(
    "config_file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.argument(
    "output",
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.argument(
    "world_size",
    type=click.INT,
)
@click.option("-g", "--global_indices", type=click.Path(exists=True, dir_okay=False, path_type=Path))
def main(config_file: Path, output: Path, world_size: int, global_indices: Path):
    config = TrainConfig.load(config_file, validate_paths=False)

    if global_indices:
        global_indices_arr = np.memmap(global_indices, mode="r", dtype=np.uint64)
    else:
        global_indices_arr = build_global_indices(world_size, config)

    reproduce_train_order(global_indices_arr, world_size, output, config)


if __name__ == "__main__":
    prepare_cli_environment()

    main()
