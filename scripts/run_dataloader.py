import logging
import sys
import time

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from olmo.config import TrainConfig
from olmo.data import build_memmap_dataset
from olmo.data.collator import DataCollator
from olmo.data.iterable_dataset import IterableDataset
from olmo.exceptions import OLMoCliError
from olmo.torch_util import seed_all
from olmo.util import clean_opt, prepare_cli_environment

log = logging.getLogger("run_dataloader")


def main(cfg: TrainConfig) -> None:
    # Set seed.
    seed_all(cfg.seed)

    # Set some additional settings
    if cfg.device_train_batch_size is None:
        log.warning(
            "device_train_batch_size is not set, so we're assuming we're running on 8 GPUs. "
            "Set that value on the command line if this is not true."
        )
        cfg.device_train_batch_size = cfg.global_train_batch_size // 8

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

    # Warm up the data loader
    train_loader_iter = iter(train_loader)
    next(train_loader_iter)

    # Benchmark the dataloader
    start_time = time.time()
    last_log_time = start_time
    batches_loaded = 0
    for _ in train_loader_iter:
        batches_loaded += 1
        now = time.time()
        if now - last_log_time > 1:
            log.info(
                "Read %d batches in %.2f seconds, %.2f batches per second",
                batches_loaded,
                now - start_time,
                batches_loaded / (now - start_time),
            )
            last_log_time = now


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError as e:
        print(f"failed to set multiprocessing start method: {e}")

    dist.init_process_group(backend="gloo", world_size=1, rank=0, store=dist.HashStore())

    prepare_cli_environment()

    log.info(f"multiprocessing start method set to '{mp.get_start_method()}'")

    try:
        yaml_path, args_list = sys.argv[1], sys.argv[2:]
    except IndexError:
        raise OLMoCliError(f"Usage: {sys.argv[0]} [CONFIG_PATH] [OPTIONS]")

    args_list = [clean_opt(s) for s in args_list]
    args_list.insert(0, "save_folder=runs/")

    cfg = TrainConfig.load(yaml_path, args_list)
    main(cfg)
