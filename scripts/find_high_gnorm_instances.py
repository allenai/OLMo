import gc
import logging
import math
import os
import sys
from pathlib import Path

from packaging import version
from typing import Dict, Any, List, Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from olmo import TrainConfig, OLMo
from olmo.aliases import PathOrStr
from olmo.checkpoint import load_state_dict
from olmo.data import DataCollator, build_memmap_dataset, IterableDataset
from olmo.torch_util import seed_all, move_to_device
from olmo.util import prepare_cli_environment, clean_opt

log = logging.getLogger(os.path.basename(__file__))


def _device_name() -> str:
    """Why is this not part of torch?"""
    if torch.cuda.device_count() <= 0:
        return "cpu"
    else:
        return "cuda"


def _split_batch(batch: Dict[str, Any]) -> List[Dict[str, Any]]:
    microbatch_size = 1
    batch_size = batch["input_ids"].shape[0]
    micro_batches = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            micro_batches[key] = value.split(microbatch_size, dim=0)
        elif isinstance(value, list):
            micro_batches[key] = [
                value[microbatch_size * i : microbatch_size * i + microbatch_size]
                for i in range(math.ceil(batch_size / microbatch_size))
            ]
        else:
            raise ValueError(f"unexpected item in batch: '{key}={value}'")
    return [
        {key: value[i] for key, value in micro_batches.items()}  # type: ignore
        for i in range(len(micro_batches["input_ids"]))
    ]


def tensor_checksum(t: torch.Tensor) -> int:
    t = t.flatten()
    r = t.clone().to(torch.int32)
    for i in range(13):
        r *= (i + 1)
        t = t.roll(i)
        r.bitwise_xor_(t)
    return r.sum().item()


def main(
    cfg: TrainConfig,
    output_file: Optional[PathOrStr],
    skip_batches: Optional[int],
    stop_after_batches: Optional[int]
) -> None:
    cfg.save_folder = "/tmp"        # should not be used

    cfg.model.precision = cfg.precision

    if cfg.device_train_batch_size is None:
        cfg.device_train_batch_size = cfg.global_train_batch_size
    cfg.device_train_grad_accum = cfg.device_train_batch_size // cfg.device_train_microbatch_size
    cfg.data.num_workers = 4
    cfg.data.pin_memory = False
    cfg.data.prefetch_factor = 4
    cfg.model.init_device = _device_name()

    seed_all(cfg.seed)

    if output_file is None:
        output_file_stream = sys.stdout
    else:
        output_file = Path(output_file)
        output_file_stream = output_file.open("wt", encoding="UTF-8")

    # make dataloader
    collator = DataCollator(pad_direction=cfg.data.pad_direction, pad_token_id=cfg.model.pad_token_id)
    seed = cfg.data.seed if cfg.data.seed is not None else cfg.seed
    train_loader = DataLoader(
        IterableDataset(
            build_memmap_dataset(cfg, cfg.data, include_instance_metadata=False),  # type: ignore
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

    max_epochs = 1
    if isinstance(cfg.max_duration, str) and cfg.max_duration.endswith("ep"):
        max_epochs = int(cfg.max_duration[:-2].strip())

    try:
        trainer_state = load_state_dict(cfg.load_path, "train.pt")
    except FileNotFoundError:
        # for backwards compatibility
        trainer_state = load_state_dict(cfg.load_path, "other.pt")
    checkpoint_epoch = trainer_state.get("epoch")
    if checkpoint_epoch is None:
        checkpoint_epoch = 0
    global_step = trainer_state["global_step"]
    global_train_examples_seen_this_epoch = trainer_state.get(
        "global_train_examples_seen_this_epoch",
        trainer_state.get(  # for backwards compatibility
            "global_train_examples_seen",
            trainer_state.get("global_data_step", global_step) * cfg.global_train_batch_size,
        ),
    )
    if global_train_examples_seen_this_epoch > 0:
        assert isinstance(train_loader.dataset, IterableDataset)
        train_loader.dataset.start_index = global_train_examples_seen_this_epoch
    if skip_batches is not None:
        train_loader.dataset.start_index += skip_batches * cfg.global_train_batch_size
    log.info(f"Data loader will start at instance index {global_train_examples_seen_this_epoch:,d}")

    if stop_after_batches is None:
        end_global_step = None
    else:
        end_global_step = global_step + stop_after_batches

    # make model
    model = OLMo(cfg.model)
    state_dict = load_state_dict(cfg.load_path, "model.pt", map_location=_device_name())
    model.load_state_dict(state_dict)
    del state_dict
    model.train()

    # make loss function
    from olmo.train import cross_entropy_loss
    loss_fn = cross_entropy_loss
    if cfg.fused_loss:
        import flash_attn
        from flash_attn.ops.triton.cross_entropy import cross_entropy_loss

        # The `ignored_index` parameter of `cross_entropy_loss` was changed to `ignore_index` in v2.5.8 with commit https://github.com/Dao-AILab/flash-attention/commit/ec6d22143b5d375e253b2ebfc563b26a43f43684
        ce_loss_use_ignore_index_param = version.parse(flash_attn.__version__) >= version.parse("2.5.8")

        def fused_loss_fn(
            logits, labels, ignore_index: int = -100, reduction: str = "mean", compute_z_loss: bool = False
        ):
            if ce_loss_use_ignore_index_param:
                ignore_index_kwarg = {"ignore_index": ignore_index}
            else:
                ignore_index_kwarg = {"ignored_index": ignore_index}

            loss, z_loss = cross_entropy_loss(
                logits,
                labels,
                label_smoothing=0.0,
                logit_scale=1.0,
                lse_square_scale=0.0,
                inplace_backward=False,
                process_group=None,
                **ignore_index_kwarg,
            )

            mask = labels != ignore_index

            if reduction == "mean":
                loss = loss.sum() / mask.sum()
            elif reduction == "sum":
                loss = loss.sum()
            else:
                loss = loss

            if not compute_z_loss:
                return loss, None

            if reduction == "mean":
                z_loss = z_loss.sum() / mask.sum()
            elif reduction == "sum":
                z_loss = z_loss.sum()
            else:
                z_loss = z_loss

            return loss, z_loss

        loss_fn = fused_loss_fn

    # run instances one by one
    for epoch in range(checkpoint_epoch, max_epochs):
        if end_global_step is not None and global_step >= end_global_step:
            break

        for batch in train_loader:
            batch_size, seq_len = batch["input_ids"].shape
            assert seq_len == cfg.model.max_sequence_length
            assert batch_size == cfg.device_train_batch_size

            micro_batches = _split_batch(batch)
            del batch

            for micro_batch_idx, micro_batch in enumerate(micro_batches):
                instance_checksum = tensor_checksum(micro_batch["input_ids"])
                batch_size_in_tokens = micro_batch["input_ids"].numel()
                micro_batch = move_to_device(micro_batch, torch.device(_device_name()))

                # Reset grads
                for p in model.parameters():
                    p.grad = None
                gc.collect()
                torch.cuda.empty_cache()

                with torch.autocast(_device_name(), enabled=True, dtype=cfg.autocast_precision):
                    # Run forward pass.
                    logits = model(
                        input_ids=micro_batch["input_ids"],
                        attention_mask=micro_batch.get("attention_mask"),
                        attention_bias=micro_batch.get("attention_bias"),
                    ).logits
                    logits_for_loss = logits[..., :-1, :].contiguous()
                    logits_for_loss = logits_for_loss.view(-1, logits_for_loss.size(-1))

                    # get labels
                    # Labels are just input IDs shifted to the left (first item is ignored).
                    labels, label_mask, attention_mask, instance_mask = (
                        micro_batch["input_ids"].clone(),
                        micro_batch.get("label_mask"),
                        micro_batch.get("attention_mask"),
                        micro_batch.get("instance_mask"),
                    )
                    if label_mask is not None:
                        labels.masked_fill_(~label_mask, -100)
                    if attention_mask is not None:
                        labels.masked_fill_(attention_mask == 0.0, -100)
                    if instance_mask is not None:
                        labels.masked_fill_(~instance_mask.unsqueeze(-1), value=-100)
                    labels = labels[..., 1:].contiguous()
                    labels = labels.view(-1)

                    ce_loss, z_loss = loss_fn(
                        logits_for_loss,
                        labels,
                        ignore_index=-100,
                        reduction="sum",
                        compute_z_loss=cfg.softmax_auxiliary_loss
                    )
                    ce_loss = ce_loss / batch_size_in_tokens

                    # Get loss to optimize for.
                    if cfg.softmax_auxiliary_loss:
                        assert z_loss is not None
                        z_loss = z_loss / batch_size_in_tokens
                        loss = ce_loss + z_loss
                    else:
                        loss = ce_loss

                # Run backward pass.
                loss.backward()

                # Calculate grad norm
                l1_gnorm = torch.tensor(0.0, dtype=torch.float32)
                l2_gnorm = torch.tensor(0.0, dtype=torch.float32)
                wte_gnorm = torch.tensor(0.0, dtype=torch.float32)
                for pname, p in model.named_parameters():
                    if p.grad is None:
                        log.warning("Parameter %s has no grad!", pname)
                        continue
                    grad = p.grad.to(torch.float32)
                    l1_gnorm += grad.abs().sum()
                    grad_squared_sum = (grad ** 2).sum()
                    l2_gnorm += grad_squared_sum
                    if pname == "transformer.wte.weight":
                        wte_gnorm = grad_squared_sum
                    del grad_squared_sum
                l2_gnorm = torch.sqrt(l2_gnorm)
                wte_gnorm = torch.sqrt(wte_gnorm)

                # Calculate norm of activated embeddings
                with torch.no_grad():
                    activated_embeddings_norm = \
                        torch.linalg.vector_norm(model.transformer.wte(micro_batch["input_ids"]))

                # print output
                output_line = "\t".join(map(str, [
                    global_step,
                    micro_batch_idx,
                    instance_checksum,
                    loss.item(),
                    l1_gnorm.item(),
                    l2_gnorm.item(),
                    wte_gnorm.item(),
                    activated_embeddings_norm.item()
                ]))
                output_file_stream.write(output_line)
                output_file_stream.write("\n")
                output_file_stream.flush()
                if output_file is not None:
                    log.info(output_line)

            global_step += 1
            if end_global_step is not None and global_step >= end_global_step:
                break

    if output_file is not None:
        output_file_stream.close()


if __name__ == "__main__":
    prepare_cli_environment()

    import argparse

    parser = argparse.ArgumentParser(description="run over a bunch of instances and record the grad norm of each of them")
    parser.add_argument("-o", type=str, help="output to file")
    parser.add_argument("--skip-batches", type=int, help="if specified, skip this many batches before starting")
    parser.add_argument("--stop-after-batches", type=int, help="if specified, stop after this many batches")
    parser.add_argument("config_file", type=str, help="config file")
    args, other_args = parser.parse_known_args()

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError as e:
        log.warning(f"failed to set multiprocessing start method: {e}")

    if torch.cuda.device_count() <= 0:
        dist_backend = "gloo"
    else:
        dist_backend = "nccl"
    dist.init_process_group(backend=dist_backend, world_size=1, rank=0, store=dist.HashStore())
    torch.set_default_device(_device_name())

    args_list = [clean_opt(s) for s in other_args]
    cfg = TrainConfig.load(args.config_file, args_list)

    main(cfg, args.o, args.skip_batches, args.stop_after_batches)
