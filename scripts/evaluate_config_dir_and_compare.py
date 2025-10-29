#!/usr/bin/env python3
"""Evaluate a directory of training configs and run paired significance tests."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from olmo.analysis.significance import (
    ComparisonConfig,
    EvalRow,
    compare_eval_rows,
)
from olmo.config import DistributedStrategy, TrainConfig
from olmo.data.collator import DataCollator
from olmo.data.memmap_dataset import MemMapDataset
from olmo.eval import build_evaluator
from olmo.model import OLMo
from olmo.tokenizer import Tokenizer
from olmo.torch_util import move_to_device


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", required=True, type=Path, help="Directory containing train configs.")
    parser.add_argument(
        "--eval-label",
        default="all-small-ppl-validation",
        help="Evaluator label to use when computing validation metrics (default: all-small-ppl-validation).",
    )
    parser.add_argument(
        "--checkpoint-name",
        default="latest-unsharded",
        help="Checkpoint directory name to evaluate (default: latest-unsharded).",
    )
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        default=None,
        help="Optional step number; if provided use step{N}-unsharded instead of --checkpoint-name.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on (default: cuda if available, else cpu).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Directory to write outputs (default: config directory).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Where to write the per-shard totals CSV (default: <config-dir>/val_totals.csv).",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Run name to treat as the baseline for significance comparisons (default: inferred).",
    )
    parser.add_argument(
        "--baseline-config",
        type=Path,
        default=None,
        help="Optional config file path whose run should be treated as the baseline.",
    )
    parser.add_argument(
        "--seeds",
        nargs="*",
        default=None,
        help="Optional subset of seeds to include in significance tests.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=10_000,
        help="Number of bootstrap samples for confidence intervals (default: 10000).",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=2025,
        help="Seed for the bootstrap RNG (default: 2025).",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level for intervals (default: 0.95).",
    )
    parser.add_argument(
        "--allow-missing-shards",
        action="store_true",
        help="Allow shard sets to differ between models by dropping unmatched shards.",
    )
    parser.add_argument(
        "--skip-significance",
        action="store_true",
        help="Only compute the CSV; skip statistical comparisons.",
    )
    parser.add_argument(
        "--batch-size-override",
        type=int,
        default=None,
        help="Override the evaluation batch size.",
    )
    return parser.parse_args(argv)


def find_eval_config(cfg: TrainConfig, label: str) -> Tuple[int, TrainConfig]:
    for idx, evaluator in enumerate(cfg.evaluators):
        if evaluator.label == label:
            return idx, evaluator
    raise ValueError(f"Evaluator '{label}' not found in config '{cfg.run_name}'.")


def resolve_checkpoint(cfg: TrainConfig, args: argparse.Namespace) -> Path:
    save_folder = Path(cfg.save_folder)
    if args.checkpoint_step is not None:
        ckpt = save_folder / f"step{args.checkpoint_step}-unsharded"
    else:
        ckpt = save_folder / args.checkpoint_name
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {ckpt}")
    return ckpt.resolve()


def infer_step(checkpoint_dir: Path) -> int:
    train_state = checkpoint_dir / "train.pt"
    if train_state.exists():
        state = torch.load(train_state, map_location="cpu", weights_only=False)
        step = int(state.get("global_step", -1))
        if step >= 0:
            return step
    name = checkpoint_dir.name
    if name.startswith("step") and "-unsharded" in name:
        try:
            return int(name[len("step") : name.index("-unsharded")])
        except ValueError:
            pass
    raise ValueError(f"Unable to determine step for checkpoint {checkpoint_dir}")


def build_memmap_dataloader(
    cfg: TrainConfig,
    evaluator_cfg,
    paths: List[str],
    label: str,
    *,
    batch_size: int,
) -> DataLoader:
    pad_token_id = cfg.model.pad_token_id
    dataset = MemMapDataset(
        *paths,
        chunk_size=cfg.model.max_sequence_length,
        memmap_dtype=evaluator_cfg.data.effective_memmap_dtype,
        metadata=[{"label": label, "path": path} for path in paths],
        include_instance_metadata=True,
        pad_token_id=pad_token_id,
        eos_token_id=cfg.model.eos_token_id,
        generate_attention_mask=evaluator_cfg.data.generate_attention_mask,
        generate_doc_lengths=evaluator_cfg.data.generate_doc_lengths,
        label_mask_paths=evaluator_cfg.data.label_mask_paths,
        instance_filter_config=evaluator_cfg.data.instance_filter,
    )
    collator = DataCollator(pad_direction=evaluator_cfg.data.pad_direction, pad_token_id=pad_token_id)
    num_workers = evaluator_cfg.data.num_workers
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        shuffle=False,
        drop_last=evaluator_cfg.data.drop_last,
        num_workers=num_workers,
        pin_memory=evaluator_cfg.data.pin_memory,
        prefetch_factor=None if num_workers == 0 else evaluator_cfg.data.prefetch_factor,
        persistent_workers=False if num_workers == 0 else evaluator_cfg.data.persistent_workers,
        timeout=evaluator_cfg.data.timeout,
    )


def resolve_shard_id(metadata_entry: Dict[str, object], default_label: str) -> str:
    if isinstance(metadata_entry, dict):
        shard = metadata_entry.get("path") or metadata_entry.get("label")
        if shard:
            return str(shard)
    return default_label


def evaluate_config(cfg_path: Path, args: argparse.Namespace) -> Tuple[str, List[EvalRow]]:
    cfg = TrainConfig.load(cfg_path)
    run_name = cfg.run_name or cfg_path.stem
    cfg.distributed_strategy = DistributedStrategy.single
    cfg.device_train_batch_size = cfg.global_train_batch_size
    device = torch.device(args.device)

    if args.device != "cuda":
        cfg.model.flash_attention = False
    cfg.model.init_device = "cpu"

    eval_idx, eval_cfg = find_eval_config(cfg, args.eval_label)
    batch_size = (
        args.batch_size_override
        or eval_cfg.device_eval_batch_size
        or cfg.device_eval_batch_size
        or cfg.device_train_microbatch_size
    )
    if batch_size is None:
        raise ValueError(f"Could not determine evaluation batch size for '{run_name}'.")

    checkpoint_dir = resolve_checkpoint(cfg, args)
    step = infer_step(checkpoint_dir)

    model = OLMo(cfg.model)
    state_dict = torch.load(checkpoint_dir / "model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    tokenizer = Tokenizer.from_train_config(cfg)
    evaluator = build_evaluator(cfg, eval_cfg, tokenizer, device)

    # Override the evaluator's dataloader to expose per-path metadata when datasets are grouped.
    if eval_cfg.data.datasets:
        merged_paths: List[str] = []
        metadata: List[Dict[str, str]] = []
        for label, paths in sorted(eval_cfg.data.datasets.items()):
            merged_paths.extend(paths)
            metadata.extend([{"label": label, "path": path} for path in paths])
        custom_loader = build_memmap_dataloader(cfg, eval_cfg, merged_paths, args.eval_label, batch_size=batch_size)
        evaluator = type(evaluator)(
            label=evaluator.label,
            type=evaluator.type,
            eval_loader=custom_loader,
            eval_metric=evaluator.eval_metric,
            subset_num_batches=evaluator.subset_num_batches,
        )

    totals: Dict[str, List[float]] = defaultdict(lambda: [0.0, 0.0])

    with torch.no_grad():
        for batch in evaluator.eval_loader:
            metadata_list = batch.get("metadata") or [{}] * len(batch["input_ids"])
            batch_device = {k: v for k, v in batch.items() if k != "metadata"}
            batch_device = move_to_device(batch_device, device)

            logits = model(
                input_ids=batch_device["input_ids"],
                attention_mask=batch_device.get("attention_mask"),
                attention_bias=batch_device.get("attention_bias"),
                doc_lens=batch_device.get("doc_lens"),
                max_doc_lens=batch_device.get("max_doc_lens"),
            ).logits

            labels = batch_device["input_ids"].clone()
            if "label_mask" in batch_device:
                labels.masked_fill_(~batch_device["label_mask"], -100)
            if "attention_mask" in batch_device:
                labels.masked_fill_(batch_device["attention_mask"] == 0, -100)
            if "instance_mask" in batch_device:
                instance_mask = batch_device["instance_mask"]
                if instance_mask.dim() == 1:
                    instance_mask = instance_mask.unsqueeze(-1)
                labels.masked_fill_(~instance_mask, -100)

            labels = labels[:, 1:]
            logits = logits[:, :-1, :].contiguous()

            losses = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
                reduction="none",
            ).reshape(labels.shape)

            mask = labels != -100
            loss_per_sample = (losses * mask).sum(dim=1)
            tokens_per_sample = mask.sum(dim=1)

            for idx, meta in enumerate(metadata_list):
                token_count = tokens_per_sample[idx].item()
                if token_count <= 0:
                    continue
                shard_id = resolve_shard_id(meta, args.eval_label)
                entry = totals[shard_id]
                entry[0] += loss_per_sample[idx].item()
                entry[1] += token_count

    torch.cuda.empty_cache() if device.type == "cuda" else None

    seed = str(cfg.seed)
    rows = [
        EvalRow(model=run_name, seed=seed, step=step, shard_id=shard, total_nll=nll, total_tokens=tokens)
        for shard, (nll, tokens) in sorted(totals.items())
    ]
    return run_name, rows


def write_csv(rows: Iterable[EvalRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=["model", "seed", "step", "shard_id", "total_nll", "total_tokens"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "model": row.model,
                    "seed": row.seed,
                    "step": row.step,
                    "shard_id": row.shard_id,
                    "total_nll": f"{row.total_nll:.8f}",
                    "total_tokens": int(row.total_tokens),
                }
            )


def aggregate_ce(rows: Iterable[EvalRow]) -> Dict[str, Dict[str, Tuple[float, float]]]:
    summary: Dict[str, Dict[str, Tuple[float, float]]] = defaultdict(dict)
    for row in rows:
        ce = row.total_nll / row.total_tokens
        summary[row.model][row.shard_id] = (ce, row.total_tokens)
    return summary


def choose_baseline(models: Sequence[str], explicit: Optional[str]) -> str:
    if explicit:
        if explicit not in models:
            raise ValueError(f"Baseline '{explicit}' not among evaluated models: {models}")
        return explicit
    for candidate in models:
        if "transformer" in candidate.lower() or "baseline" in candidate.lower():
            return candidate
    return sorted(models)[0]


def format_interval(low: float, high: float) -> str:
    return f"[{low:+.4f}, {high:+.4f}]"


def format_delta(delta: float) -> str:
    return f"{delta:+.4f}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    config_dir = args.config_dir
    if not config_dir.is_dir():
        raise FileNotFoundError(f"Config directory not found: {config_dir}")

    config_paths = sorted(p for p in config_dir.iterdir() if p.suffix in {".yaml", ".yml"})
    if not config_paths:
        raise ValueError(f"No YAML configs found in {config_dir}")

    output_root = (args.output_root or config_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    all_rows: List[EvalRow] = []
    model_order: List[str] = []

    baseline_config_resolved = args.baseline_config.resolve() if args.baseline_config else None
    baseline_run_from_config: Optional[str] = None

    for cfg_path in config_paths:
        run_name, rows = evaluate_config(cfg_path, args)
        model_order.append(run_name)
        all_rows.extend(rows)
        if baseline_config_resolved and cfg_path.resolve() == baseline_config_resolved:
            baseline_run_from_config = run_name

    if baseline_config_resolved and baseline_run_from_config is None:
        raise ValueError(
            f"Baseline config '{baseline_config_resolved}' was not found among evaluated configs: {config_paths}"
        )

    output_csv = args.output_csv or (output_root / "val_totals.csv")
    write_csv(all_rows, output_csv)
    print(f"Wrote per-shard totals to {output_csv}")

    summary = aggregate_ce(all_rows)
    print("\nPer-model shard averages (CrossEntropyLoss, tokens):")
    for model in model_order:
        shards = summary[model]
        shard_str = ", ".join(
            f"{shard}: CE={ce:.4f} ({int(tokens)} tok)" for shard, (ce, tokens) in sorted(shards.items())
        )
        print(f"  {model}: {shard_str}")

    if args.skip_significance or len(model_order) < 2:
        return 0

    explicit_baseline = args.baseline or baseline_run_from_config
    baseline = choose_baseline(model_order, explicit_baseline)
    print(f"\nBaseline model for comparisons: {baseline}")

    rows_iter = list(all_rows)
    summary_lines: List[str] = []
    for model in model_order:
        if model == baseline:
            continue
        cfg = ComparisonConfig(
            model_a=model,
            model_b=baseline,
            step=rows_iter[0].step,
            seeds=args.seeds,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=args.bootstrap_seed,
            confidence=args.confidence,
            allow_missing_shards=args.allow_missing_shards,
        )
        result = compare_eval_rows(rows_iter, cfg)
        line = (
            f"Compare {model} vs {baseline}: ΔCE={format_delta(result.mean_delta_ce)}; "
            f"CI={format_interval(result.ci_low, result.ci_high)}; "
            f"ppl≈{math.exp(result.mean_delta_ce):.3f}; significant={result.is_significant}"
        )
        print(f"\n{line}")
        summary_lines.append(line)

    if summary_lines:
        summary_path = output_root / "summary.txt"
        with summary_path.open("w") as fp:
            fp.write("Baseline: " + baseline + "\n")
            for line in summary_lines:
                fp.write(line + "\n")
        print(f"\nWrote comparison summary to {summary_path}")

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    try:
        sys.exit(main())
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

