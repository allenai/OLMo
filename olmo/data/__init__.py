import importlib
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

from torch.utils.data import DataLoader, DistributedSampler

from ..aliases import PathOrStr
from ..config import DataConfig, TrainConfig
from ..exceptions import OLMoConfigurationError
from ..torch_util import barrier, get_global_rank, get_world_size
from .collator import CustomDatasetDataCollator, DataCollator
from .custom_datasets import build_custom_dataset, extract_module_and_class
from .iterable_dataset import IterableDataset
from .memmap_dataset import MemMapDataset

__all__ = ["MemMapDataset", "DataCollator", "IterableDataset", "build_eval_dataloader", "build_train_dataloader"]

LOGGER = logging.getLogger(__name__)


def build_memmap_dataset(
    train_config: TrainConfig, data_config: DataConfig, include_instance_metadata: bool = True
) -> MemMapDataset:
    paths: List[str]
    metadata: List[Dict[str, Any]] = []
    if data_config.paths:
        if data_config.datasets:
            raise OLMoConfigurationError("DataConfig.paths is mutually exclusive with DataConfig.datasets")
        paths = data_config.paths
        for path in paths:
            metadata.append({"path": str(path)})
    elif data_config.datasets:
        paths = []
        for label in sorted(data_config.datasets.keys()):
            label_paths = data_config.datasets[label]
            paths.extend(label_paths)
            metadata.extend([{"label": label}] * len(label_paths))
    else:
        raise OLMoConfigurationError("One of DataConfig.paths or DataConfig.datasets is required")
    return MemMapDataset(
        *paths,
        chunk_size=train_config.model.max_sequence_length,
        memmap_dtype=data_config.effective_memmap_dtype,
        metadata=metadata,
        include_instance_metadata=include_instance_metadata,
        pad_token_id=train_config.model.pad_token_id,
        eos_token_id=train_config.model.eos_token_id,
        generate_attention_mask=data_config.generate_attention_mask,
        generate_doc_lengths=data_config.generate_doc_lengths,
        label_mask_paths=cast(Optional[List[PathOrStr]], data_config.label_mask_paths),
        instance_filter_config=data_config.instance_filter,
    )


def build_collator(train_config: TrainConfig) -> DataCollator:
    """Returns a collator for the train dataloader. Either returns the default
    collator or a custom collator specified in the train config.

    :param train_config: OLMo train config
    :raises OLMoConfigurationError: Raises an error if the collate function is not found
    :return: Collator for the train dataloader
    """
    if train_config.data.custom_dataset:
        if train_config.data.custom_dataset.collate_fn:
            module, function = extract_module_and_class(train_config.data.custom_dataset.collate_fn)
            if module is None:
                if train_config.data.custom_dataset.module is None:
                    module, _ = extract_module_and_class(train_config.data.custom_dataset.name)
                else:
                    module = train_config.data.custom_dataset.module
            try:
                assert module is not None
                collator = getattr(importlib.import_module(module), function)
            except AttributeError:
                raise OLMoConfigurationError(
                    f"collate_fn {train_config.data.custom_dataset.collate_fn} not found in {module}. Please specify the full module path of the function."
                )
            return collator

        return CustomDatasetDataCollator(
            pad_direction=train_config.data.pad_direction,
            pad_token_id=train_config.model.pad_token_id,
            **train_config.data.custom_dataset.collate_config.asdict(),  # type: ignore
        )
    else:
        return DataCollator(
            pad_direction=train_config.data.pad_direction, pad_token_id=train_config.model.pad_token_id
        )


def build_eval_dataloader(
    train_config: TrainConfig,
    data_config: DataConfig,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    dataset = build_memmap_dataset(train_config, data_config, include_instance_metadata=True)
    collator = DataCollator(pad_direction=data_config.pad_direction, pad_token_id=train_config.model.pad_token_id)
    if data_config.drop_last:
        # Make sure batch size is small enough.
        samples_per_device = len(dataset) // get_world_size()
        batch_size = min(batch_size, samples_per_device)
        assert batch_size > 0, f"dataset for {data_config.paths} is too small"
    seed = data_config.seed if data_config.seed is not None else train_config.seed
    sampler = DistributedSampler(
        dataset,
        drop_last=data_config.drop_last,
        shuffle=shuffle,
        num_replicas=get_world_size(),
        rank=get_global_rank(),
        seed=seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=data_config.num_workers,
        sampler=sampler,
        pin_memory=data_config.pin_memory,
        prefetch_factor=None if data_config.num_workers == 0 else data_config.prefetch_factor,
        persistent_workers=False if data_config.num_workers == 0 else data_config.persistent_workers,
        timeout=data_config.timeout,
    )


def build_train_dataloader(
    train_config: TrainConfig,
    *,
    world_size: Optional[int] = None,
    rank: Optional[int] = None,
    fs_local_rank: Optional[int] = None,
    include_instance_metadata: bool = False,
) -> DataLoader:
    assert train_config.device_train_batch_size is not None
    seed = train_config.data.seed if train_config.data.seed is not None else train_config.seed
    collator = build_collator(train_config)
    if train_config.data.custom_dataset:
        if train_config.data.paths is not None or train_config.data.datasets is not None:
            raise OLMoConfigurationError(
                "custom_dataset_class is mutually exclusive with DataConfig.paths and DataConfig.datasets"
            )
        dataset = build_custom_dataset(train_config)
    else:
        dataset = build_memmap_dataset(
            train_config, train_config.data, include_instance_metadata=include_instance_metadata
        )
    work_dir = Path(train_config.save_folder) / "train_data"
    if get_global_rank() == 0:
        if work_dir.is_dir() and not train_config.save_overwrite:
            raise OLMoConfigurationError(
                "train data working directory already exists, use --save_overwrite to overwrite"
            )
        else:
            work_dir.mkdir(exist_ok=True, parents=True)
    dataset = IterableDataset(
        dataset,  # type: ignore
        train_config.global_train_batch_size,
        seed=seed,
        epoch=train_config.epoch or 0,
        shuffle=True,
        drop_last=train_config.data.drop_last,
        world_size=world_size,
        rank=rank,
        fs_local_rank=fs_local_rank,
        work_dir=work_dir,
    )
    barrier()
    out = DataLoader(
        dataset,
        batch_size=train_config.device_train_batch_size,
        drop_last=train_config.data.drop_last,
        collate_fn=collator,
        num_workers=train_config.data.num_workers,
        pin_memory=train_config.data.pin_memory,
        prefetch_factor=None if train_config.data.num_workers == 0 else train_config.data.prefetch_factor,
        persistent_workers=False if train_config.data.num_workers == 0 else train_config.data.persistent_workers,
        timeout=train_config.data.timeout,
    )
    return out
