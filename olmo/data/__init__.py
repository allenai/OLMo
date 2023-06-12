from typing import Any, Dict, List, Optional

import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from ..config import DataConfig, TrainConfig
from ..exceptions import OlmoConfigurationError
from ..util import global_rank
from .collator import DataCollator
from .iterable_dataset import IterableDataset
from .memmap_dataset import MemMapDataset

__all__ = ["MemMapDataset", "DataCollator", "IterableDataset", "build_eval_dataloader", "build_train_dataloader"]


def build_memmap_dataset(train_config: TrainConfig, data_config: DataConfig) -> MemMapDataset:
    paths: List[str]
    metadata: Optional[List[Dict[str, Any]]] = None
    if data_config.paths:
        if data_config.datasets:
            raise OlmoConfigurationError("DataConfig.paths is mutually exclusive with DataConfig.datasets")
        paths = data_config.paths
    elif data_config.datasets:
        paths = []
        metadata = []
        for label in sorted(data_config.datasets.keys()):
            label_paths = data_config.datasets[label]
            paths.extend(label_paths)
            metadata.extend([{"label": label}] * len(label_paths))
    else:
        raise OlmoConfigurationError("One of DataConfig.paths or DataConfig.datasets is required")
    return MemMapDataset(*paths, chunk_size=train_config.model.max_sequence_length, metadata=metadata)


def build_eval_dataloader(
    train_config: TrainConfig,
    data_config: DataConfig,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    dataset = build_memmap_dataset(train_config, data_config)
    collator = DataCollator(pad_direction=data_config.pad_direction, pad_token_id=train_config.model.pad_token_id)
    if data_config.drop_last:
        # Make sure batch size is small enough.
        samples_per_device = len(dataset) // dist.get_world_size()
        batch_size = min(batch_size, samples_per_device)
        assert batch_size > 0, f"dataset for {data_config.paths} is too small"
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
    dataset = build_memmap_dataset(train_config, train_config.data)
    return DataLoader(
        IterableDataset(
            dataset,  # type: ignore
            seed=train_config.seed,
            shuffle=True,
            drop_last=train_config.data.drop_last,
            max_examples=train_config.global_train_batch_size * train_config.max_duration,
        ),
        batch_size=train_config.device_train_batch_size,
        drop_last=train_config.data.drop_last,
        collate_fn=collator,
        num_workers=train_config.data.num_workers,
        pin_memory=train_config.data.pin_memory,
        prefetch_factor=train_config.data.prefetch_factor,
        persistent_workers=train_config.data.persistent_workers,
        timeout=train_config.data.timeout,
    )
