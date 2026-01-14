import logging
import time
from pathlib import Path
from typing import Dict, Optional

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


def main(cfg: TrainConfig, output_dir: Path, max_batches: Optional[int] = None) -> None:
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

    # Track throughput statistics
    start_time = time.time()
    total_bytes = 0

    progress_bar = tqdm(train_loader, total=max_batches, desc="Processing batches")
    for batch_number, batch in enumerate(progress_bar):
        # Check if we've reached the maximum number of batches
        if max_batches is not None and batch_number >= max_batches:
            log.info(f"Reached max_batches limit ({max_batches}), stopping.")
            break

        for name, source_t in batch.items():
            source_t = source_t.numpy()
            if name == "input_ids":
                assert source_t.max() <= 2**16
                source_t = source_t.astype(np.uint16)
                total_bytes += source_t.nbytes
            try:
                target_t = name_to_batches[name]
            except KeyError:
                target_t = np.zeros((batches_per_file,) + source_t.shape, dtype=source_t.dtype)
                name_to_batches[name] = target_t
            target_t[batches_read] = source_t
        batches_read += 1

        # Update progress bar with throughput stats
        elapsed = time.time() - start_time
        if elapsed > 0:
            batches_per_sec = (batch_number + 1) / elapsed
            mb_per_sec = (total_bytes / (1024 * 1024)) / elapsed
            progress_bar.set_postfix({"batches/s": f"{batches_per_sec:.2f}", "MB/s": f"{mb_per_sec:.2f}"})

        if batches_read >= batches_per_file:
            file_start = batch_number - batches_per_file + 1
            file_end = batch_number + 1
            for name, t in name_to_batches.items():
                filename = output_dir / f"{name}-{file_start:07}-{file_end:07}.npy"
                np.save(filename, t[:batches_read])
            batches_read = 0

    # Save any remaining batches
    if batches_read > 0:
        file_start = batch_number - batches_read + 1
        file_end = batch_number + 1
        for name, t in name_to_batches.items():
            filename = output_dir / f"{name}-{file_start:07}-{file_end:07}.npy"
            np.save(filename, t[:batches_read])
        log.info(f"Saved final {batches_read} batches.")

    # Print final statistics
    elapsed = time.time() - start_time
    log.info(
        f"Processed {batch_number + 1} batches in {elapsed:.1f}s ({(batch_number + 1) / elapsed:.2f} batches/s)"
    )
    log.info(
        f"Total data read: {total_bytes / (1024 * 1024):.2f} MB ({total_bytes / (1024 * 1024) / elapsed:.2f} MB/s)"
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="replay the dataloader and write batches out to files")
    parser.add_argument("-o", type=str, help="output directory")
    parser.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="Maximum number of batches to process (useful for testing or partial runs)",
    )
    parser.add_argument(
        "--local_data_root",
        type=str,
        default=None,
        help="Local directory root to substitute for remote paths (e.g., /mnt/tank/ replaces s3://)",
    )
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

    # Substitute remote paths with local paths if --local_data_root is provided
    if args.local_data_root:
        if cfg.data.paths:
            original_paths = cfg.data.paths
            cfg.data.paths = []
            for p in original_paths:
                for scheme in ("s3://", "r2://", "weka://", "gs://"):
                    if p.startswith(scheme):
                        # Extract bucket and key, then join with local root
                        local_path = args.local_data_root.rstrip("/") + "/" + p.split("://", 1)[1]
                        log.info(f"Substituting remote path: {p} -> {local_path}")
                        p = local_path
                        break
                cfg.data.paths.append(p)
    else:
        # Warn user about potential slow remote access
        if cfg.data.paths:
            remote_paths = [
                p for p in cfg.data.paths if any(p.startswith(s) for s in ("s3://", "r2://", "weka://", "gs://"))
            ]
            if remote_paths:
                log.warning(
                    f"Loading data from {len(remote_paths)} remote path(s). This may be very slow. "
                    "Consider using --local_data_root to use locally cached data for faster processing."
                )

    main(cfg, output_dir, max_batches=args.max_batches)
