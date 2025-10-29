#!/usr/bin/env python3
"""Evaluate statistical significance of validation cross-entropy deltas."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from olmo.analysis.significance import (
    ComparisonConfig,
    EvalRow,
    compare_eval_rows,
    percent_perplexity_change,
)


REQUIRED_COLUMNS = {
    "model",
    "seed",
    "step",
    "shard_id",
    "total_nll",
    "total_tokens",
}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path, help="Path to CSV containing evaluation totals.")
    parser.add_argument("--model-a", required=True, help="Identifier for model/run A.")
    parser.add_argument("--model-b", required=True, help="Identifier for model/run B.")
    parser.add_argument("--step", required=True, type=int, help="Training step to compare.")
    parser.add_argument(
        "--seeds",
        nargs="*",
        default=None,
        help="Optional list of seeds to include. If omitted, all overlapping seeds are used.",
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
        help="Proceed even if shard sets differ between models (drops unmatched shards).",
    )
    return parser.parse_args(argv)


def load_rows(csv_path: Path) -> List[EvalRow]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    rows: List[EvalRow] = []
    with csv_path.open("r", newline="") as fp:
        reader = csv.DictReader(fp)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")

        for line_num, raw in enumerate(reader, start=2):
            try:
                step = int(raw["step"])
            except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
                raise ValueError(f"Invalid step at line {line_num}: {raw['step']!r}") from exc

            try:
                total_nll = float(raw["total_nll"])
                total_tokens = float(raw["total_tokens"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid numeric values at line {line_num}: total_nll={raw['total_nll']!r}, total_tokens={raw['total_tokens']!r}"
                ) from exc

            if total_tokens <= 0:
                raise ValueError(
                    f"Non-positive total_tokens at line {line_num}: shard={raw['shard_id']!r}, tokens={total_tokens}"
                )

            rows.append(
                EvalRow(
                    model=str(raw["model"]),
                    seed=str(raw["seed"]),
                    step=step,
                    shard_id=str(raw["shard_id"]),
                    total_nll=total_nll,
                    total_tokens=total_tokens,
                )
            )

    return rows


def format_delta(delta: float) -> str:
    return f"{delta:+.4f}"


def format_interval(lower: float, upper: float) -> str:
    return f"[{lower:+.4f}, {upper:+.4f}]"


def run_comparison(rows: Iterable[EvalRow], args: argparse.Namespace) -> int:
    cfg = ComparisonConfig(
        model_a=args.model_a,
        model_b=args.model_b,
        step=args.step,
        seeds=args.seeds,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
        confidence=args.confidence,
        allow_missing_shards=args.allow_missing_shards,
    )

    result = compare_eval_rows(rows, cfg)

    print(f"Comparing A='{cfg.model_a}' vs B='{cfg.model_b}' at step {cfg.step}")
    print("Per-seed results (token-weighted ΔCE; CI from bootstrap; ppl change is derived from ΔCE):")
    print("seed\tdelta_ce\tbootstrap_CI\tpaired_t\tn_pairs\tppl_delta_%")

    for seed_result in result.seed_results:
        paired = seed_result.paired_t_stat
        paired_str = f"{paired:+.3f}" if math.isfinite(paired) else "nan"
        ppl_delta = percent_perplexity_change(seed_result.delta_ce)
        print(
            "\t".join(
                [
                    seed_result.seed,
                    format_delta(seed_result.delta_ce),
                    format_interval(seed_result.ci_low, seed_result.ci_high),
                    paired_str,
                    str(seed_result.n_pairs),
                    f"{ppl_delta:+.2f}",
                ]
            )
        )

    print()
    print("Across-seed summary:")
    print(
        f"Mean ΔCE = {format_delta(result.mean_delta_ce)}; {int(cfg.confidence * 100)}% CI "
        f"{format_interval(result.ci_low, result.ci_high)}; ≈ ppl change {result.ppl_delta_pct:+.2f}%"
    )

    if result.is_significant:
        print("Interpretation: confidence interval excludes 0 → difference is statistically significant.")
    else:
        print("Interpretation: confidence interval includes 0 → difference is not statistically significant.")

    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        rows = load_rows(args.csv)
        return run_comparison(rows, args)
    except Exception as exc:  # pragma: no cover - CLI convenience
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover - CLI entry
    sys.exit(main())

