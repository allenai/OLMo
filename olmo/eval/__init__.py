import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torchmetrics import MeanMetric

from ..config import EvaluatorConfig, EvaluatorType, TrainConfig
from ..tokenizer import Tokenizer
from ..util import cycle_through_epochs, global_rank
from .downstream import ICLMetric, label_to_task_map
from .evaluator import Evaluator

__all__ = ["Evaluator", "ICLMetric", "label_to_task_map", "build_downstream_evaluator", "build_evaluator"]


def build_downstream_evaluator(
    train_config: TrainConfig,
    eval_cfg: EvaluatorConfig,
    tokenizer: Tokenizer,
    device: torch.device,
    is_unit_test=False,
) -> Evaluator:
    task_class = label_to_task_map[eval_cfg.label]
    ds_eval_dataset = task_class(tokenizer=tokenizer)  # type: ignore
    data_config = eval_cfg.data
    if is_unit_test:
        ds_eval_sampler = None
    else:
        ds_eval_sampler = DistributedSampler(
            ds_eval_dataset,
            drop_last=data_config.drop_last,
            shuffle=False,
            num_replicas=dist.get_world_size(),
            rank=global_rank(),
            seed=train_config.seed,
        )
    ds_eval_dataloader = DataLoader(
        ds_eval_dataset,
        batch_size=eval_cfg.device_eval_batch_size or train_config.device_eval_batch_size,
        collate_fn=ds_eval_dataset.collate_fn,
        num_workers=data_config.num_workers,
        sampler=ds_eval_sampler,
        pin_memory=data_config.pin_memory,
        prefetch_factor=data_config.prefetch_factor,
        persistent_workers=data_config.persistent_workers,
        timeout=data_config.timeout,
    )
    metric = ICLMetric(metric_type=ds_eval_dataset.metric_type)

    evaluator = Evaluator(
        cfg=eval_cfg,
        eval_loader=ds_eval_dataloader,
        eval_batches=cycle_through_epochs(ds_eval_dataloader),
        eval_metric=metric.to(device),
    )
    return evaluator


def build_evaluator(
    train_config: TrainConfig, eval_config: EvaluatorConfig, tokenizer: Tokenizer, device: torch.device
) -> Evaluator:
    from ..data import build_eval_dataloader

    if eval_config.type == EvaluatorType.downstream:
        # Downstream evaluation.
        return build_downstream_evaluator(train_config, eval_config, tokenizer, device)
    elif eval_config.type == EvaluatorType.lm:
        # Language modeling evaluation.
        eval_loader = build_eval_dataloader(
            train_config,
            eval_config.data,
            eval_config.device_eval_batch_size or train_config.device_eval_batch_size,
        )
        return Evaluator(
            cfg=eval_config,
            eval_loader=eval_loader,
            eval_batches=cycle_through_epochs(eval_loader),
            eval_metric=MeanMetric(nan_strategy="error").to(device),
        )
    else:
        raise ValueError(f"Unexpected evaluator type '{eval_config.type}'")
