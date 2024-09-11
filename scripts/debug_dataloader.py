import logging
from datetime import timedelta
from random import Random

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from olmo.config import TrainConfig
from olmo.data.collator import DataCollator
from olmo.torch_util import seed_all
from olmo.util import clean_opt, prepare_cli_environment

log = logging.getLogger("debug_dataloader")


class NewIterableDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        parent_world_size: int,
        parent_rank: int,
        global_batch_size: int,
        start_index: int = 0,
        seed: int = 6924
    ):
        self.parent_world_size = parent_world_size
        self.parent_rank = parent_rank
        self.global_batch_size = global_batch_size
        self.seed = seed
        super().__init__()

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            global_num_workers = 1
            global_worker_id = 0
            dist.init_process_group(
                backend="gloo",
                world_size=1,
                rank=0,
                store=dist.HashStore(),
                timeout=timedelta(seconds=30))
        else:
            global_num_workers = self.parent_world_size * worker_info.num_workers
            global_worker_id = self.parent_rank * worker_info.num_workers + worker_info.id
            dist_store = dist.TCPStore(
                "127.0.0.1",
                11234 + 1,
                self.parent_world_size * worker_info.num_workers,
                self.parent_rank == 0 and worker_info.id == 0
            )
            dist.init_process_group(
                backend="gloo",
                world_size=global_num_workers,
                rank=global_worker_id,
                store=dist_store,
                timeout=timedelta(seconds=30))
            del dist_store

        rng = Random(self.seed)

        total_dataset = [torch.tensor([i], dtype=torch.int32) for i in range(100)]

        # do one batch
        while len(total_dataset) > 0:
            indices_in_batch = rng.sample(
                range(len(total_dataset)),
                max(self.global_batch_size, len(total_dataset)))
            indices_chosen = []
            for i in range(global_worker_id, len(indices_in_batch), global_num_workers):
                index_chosen = indices_in_batch[i]
                # something expensive happens here
                yield {"input_ids": [total_dataset[index_chosen]]}
                indices_chosen.append(index_chosen)

            all_indices_chosen = [None] * global_num_workers
            dist.all_gather_object(all_indices_chosen, indices_chosen)
            del indices_chosen
            all_indices_chosen = [i for indices_chosen in all_indices_chosen for i in indices_chosen]
            all_indices_chosen.sort(reverse=True)
            for i in all_indices_chosen:
                del total_dataset[i]

        pass


def main(rank: int, world_size: int, cfg: TrainConfig) -> None:
    dist_store = dist.TCPStore(
        "127.0.0.1",
        11234,
        world_size,
        rank == 0
    )
    dist.init_process_group(backend="gloo", world_size=world_size, rank=rank, store=dist_store)

    # Set seed
    seed_all(cfg.seed)

    # Set some additional settings
    if cfg.device_train_batch_size is None:
        cfg.device_train_batch_size = cfg.global_train_batch_size // dist.get_world_size()
    cfg.device_train_grad_accum = cfg.device_train_batch_size // cfg.device_train_microbatch_size
    cfg.data.num_workers = 2
    cfg.data.pin_memory = False
    cfg.data.prefetch_factor = 1

    # Construct data loader.
    collator = DataCollator(pad_direction=cfg.data.pad_direction, pad_token_id=cfg.model.pad_token_id)
    seed = cfg.data.seed if cfg.data.seed is not None else cfg.seed
    train_loader = DataLoader(
        NewIterableDataset(
            parent_world_size=world_size,
            parent_rank=rank,
            global_batch_size=cfg.global_train_batch_size,
            seed=seed + (cfg.epoch or 0),
        ),
        batch_size=cfg.device_train_batch_size,
        drop_last=False, #cfg.data.drop_last,
        collate_fn=collator,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        prefetch_factor=None if cfg.data.num_workers == 0 else cfg.data.prefetch_factor,
        persistent_workers=False if cfg.data.num_workers == 0 else cfg.data.persistent_workers,
        timeout=cfg.data.timeout,
    )

    for batch_number, batch in enumerate(train_loader):
        print(f"{batch_number}: {batch['input_ids'].tolist()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="debug the dataloader and write batches out to files")
    parser.add_argument("--world-size", type=int, help="world size", default=4)
    parser.add_argument("config_file", type=str, help="config file")
    args, other_args = parser.parse_known_args()

    args_list = [clean_opt(s) for s in other_args]
    args_list.insert(0, "save_folder=runs/")

    cfg = TrainConfig.load(args.config_file, args_list)

    # If you have the data downloaded locally, uncomment this and fix the path for a massive speedup.
    # cfg.data.paths = [
    #    p.replace("s3://", "/mnt/tank/") for p in cfg.data.paths
    # ]

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError as e:
        print(f"failed to set multiprocessing start method: {e}")

    prepare_cli_environment()

    processes = []
    for rank in range(args.world_size):
        p = mp.Process(
            target=main,
            name=f"rank{rank}",
            args=(
                rank,
                args.world_size,
                cfg,
            )
        )
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
