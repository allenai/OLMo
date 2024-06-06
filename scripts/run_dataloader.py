import logging
from pathlib import Path
from typing import Dict

import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from tqdm import tqdm

from olmo.config import TrainConfig
from olmo.data import build_memmap_dataset
from olmo.data.collator import DataCollator
from olmo.data.iterable_dataset import IterableDataset
from olmo.torch_util import seed_all
from olmo.util import clean_opt, prepare_cli_environment

log = logging.getLogger("run_dataloader")


def main(cfg: TrainConfig, output_dir: Path) -> None:
    # Set seed
    seed_all(cfg.seed)

    # Set some additional settings
    if cfg.device_train_batch_size is None:
        cfg.device_train_batch_size = cfg.global_train_batch_size
    cfg.device_train_grad_accum = cfg.device_train_batch_size // cfg.device_train_microbatch_size
    cfg.data.num_workers = 4
    cfg.data.pin_memory = False
    cfg.data.prefetch_factor = 4

    # Construct data loader.
    collator = DataCollator(pad_direction=cfg.data.pad_direction, pad_token_id=cfg.model.pad_token_id)
    dataset = build_memmap_dataset(cfg, cfg.data, include_instance_metadata=False)
    seed = cfg.data.seed if cfg.data.seed is not None else cfg.seed
    train_loader = DataLoader(
        IterableDataset(
            dataset,  # type: ignore
            cfg.global_train_batch_size,
            seed=seed + (cfg.epoch or 0),
            shuffle=True,
            drop_last=cfg.data.drop_last,
            work_dir=None,
        ),
        batch_size=cfg.device_train_batch_size,
        drop_last=cfg.data.drop_last,
        collate_fn=collator,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        prefetch_factor=None if cfg.data.num_workers == 0 else cfg.data.prefetch_factor,
        persistent_workers=False if cfg.data.num_workers == 0 else cfg.data.persistent_workers,
        timeout=cfg.data.timeout,
    )

    batches_per_file = 1000
    batches_read = 0
    name_to_batches: Dict[str, np.array] = {}

    for batch_number, batch in enumerate(tqdm(train_loader)):
        for name, source_t in batch.items():
            source_t = source_t.numpy()
            if name == "input_ids":
                assert source_t.max() <= 2**16
                source_t = source_t.astype(np.uint16)
            try:
                target_t = name_to_batches[name]
            except KeyError:
                target_t = np.zeros((batches_per_file,) + source_t.shape, dtype=source_t.dtype)
                name_to_batches[name] = target_t
            target_t[batches_read] = source_t
        batches_read += 1

        if batches_read >= batches_per_file:
            file_start = batch_number - batches_per_file + 1
            file_end = batch_number + 1
            for name, t in name_to_batches.items():
                filename = output_dir / f"{name}-{file_start:07}-{file_end:07}.npy"
                np.save(filename, t[:batches_read])
            batches_read = 0


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="replay the dataloader and write batches out to files")
    parser.add_argument("-o", type=str, help="output directory")
    parser.add_argument("config_file", type=str, help="config file")
    args, other_args = parser.parse_known_args()
    output_dir = Path(args.o)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError as e:
        print(f"failed to set multiprocessing start method: {e}")

    dist.init_process_group(backend="gloo", world_size=1, rank=0, store=dist.HashStore())

    prepare_cli_environment()

    log.info(f"multiprocessing start method set to '{mp.get_start_method()}'")

    args_list = [clean_opt(s) for s in other_args]
    args_list.insert(0, "save_folder=runs/")

    cfg = TrainConfig.load(args.config_file, args_list)

    # If you have the data downloaded locally, uncomment this and fix the path for a massive speedup.
    # cfg.data.paths = [
    #    p.replace("s3://", "/mnt/tank/") for p in cfg.data.paths
    # ]

    main(cfg, output_dir)
