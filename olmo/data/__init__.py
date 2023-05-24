import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from ..config import DataConfig, TrainConfig
from ..util import global_rank
from .collator import DataCollator
from .iterable_dataset import IterableDataset
from .memmap_dataset import MemMapDataset

__all__ = ["MemMapDataset", "DataCollator", "IterableDataset", "build_eval_dataloader", "build_train_dataloader"]


def build_eval_dataloader(
    train_config: TrainConfig,
    data_config: DataConfig,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    collator = DataCollator(pad_direction=data_config.pad_direction, pad_token_id=train_config.model.pad_token_id)
    dataset = MemMapDataset(*data_config.paths, chunk_size=train_config.model.max_sequence_length)
    sampler = DistributedSampler(
        dataset,
        drop_last=data_config.drop_last,
        shuffle=shuffle,
        num_replicas=dist.get_world_size(),
        rank=global_rank(),
        seed=train_config.seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=data_config.num_workers,
        sampler=sampler,
        pin_memory=data_config.pin_memory,
        prefetch_factor=data_config.prefetch_factor,
        persistent_workers=data_config.persistent_workers,
        timeout=data_config.timeout,
    )


def build_train_dataloader(train_config: TrainConfig) -> DataLoader:
    assert train_config.device_train_batch_size is not None
    collator = DataCollator(
        pad_direction=train_config.data.pad_direction, pad_token_id=train_config.model.pad_token_id
    )
    dataset = MemMapDataset(*train_config.data.paths, chunk_size=train_config.model.max_sequence_length)
    return DataLoader(
        IterableDataset(
            dataset,  # type: ignore
            seed=train_config.seed,
            shuffle=True,
            drop_last=train_config.data.drop_last,
            max_steps=train_config.device_train_batch_size * train_config.max_duration,
        ),
        batch_size=train_config.device_train_batch_size,
        collate_fn=collator,
        num_workers=train_config.data.num_workers,
        pin_memory=train_config.data.pin_memory,
        prefetch_factor=train_config.data.prefetch_factor,
        persistent_workers=train_config.data.persistent_workers,
        timeout=train_config.data.timeout,
    )
