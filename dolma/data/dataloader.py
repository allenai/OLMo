from torch.utils.data import DataLoader

from ..config import TrainConfig
from .collator import DataCollator
from .memmap_dataset import MemMapDataset

__all__ = ["build_dataloader"]


def build_dataloader(config: TrainConfig, batch_size: int) -> DataLoader:
    from composer.utils.dist import get_sampler

    collator = DataCollator.from_train_config(config)
    dataset = MemMapDataset.from_train_config(config)
    sampler = get_sampler(dataset, shuffle=True, drop_last=config.data.drop_last)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=config.data.num_workers,
        sampler=sampler,
        pin_memory=config.data.pin_memory,
        prefetch_factor=config.data.prefetch_factor,
        persistent_workers=config.data.persistent_workers,
        timeout=config.data.timeout,
    )
