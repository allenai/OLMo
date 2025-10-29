from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "EvalRow",
    "ComparisonConfig",
    "SeedComparison",
    "ComparisonResult",
    "token_weighted_delta",
    "bootstrap_weighted_delta",
    "paired_t_statistic",
    "percent_perplexity_change",
    "compare_eval_rows",
]


@dataclass(frozen=True)
class EvalRow:
    model: str
    seed: str
    step: int
    shard_id: str
    total_nll: float
    total_tokens: float


@dataclass(frozen=True)
class ComparisonConfig:
    model_a: str
    model_b: str
    step: int
    seeds: Optional[Sequence[str]] = None
    bootstrap_samples: int = 10_000
    bootstrap_seed: int = 2025
    confidence: float = 0.95
    allow_missing_shards: bool = False


@dataclass(frozen=True)
class SeedComparison:
    seed: str
    delta_ce: float
    ci_low: float
    ci_high: float
    paired_t_stat: float
    n_pairs: int


@dataclass(frozen=True)
class ComparisonResult:
    config: ComparisonConfig
    seed_results: List[SeedComparison]
    mean_delta_ce: float
    ci_low: float
    ci_high: float
    ppl_delta_pct: float

    @property
    def is_significant(self) -> bool:
        return self.ci_low > 0.0 or self.ci_high < 0.0


def _ensure_bootstrap_params(num_samples: int, confidence: float) -> None:
    if num_samples <= 0:
        raise ValueError("Bootstrap samples must be positive.")
    if not (0.0 < confidence < 1.0):
        raise ValueError("Confidence level must be between 0 and 1.")


def token_weighted_delta(nll_a: np.ndarray, nll_b: np.ndarray, tokens: np.ndarray) -> float:
    numerator = float(nll_a.sum() - nll_b.sum())
    denominator = float(tokens.sum())
    if denominator <= 0:
        raise ValueError("Total token count must be positive for delta computation.")
    return numerator / denominator


def bootstrap_weighted_delta(
    nll_a: np.ndarray,
    nll_b: np.ndarray,
    tokens: np.ndarray,
    *,
    num_samples: int,
    rng: np.random.Generator,
    confidence: float,
) -> Tuple[float, float, float]:
    _ensure_bootstrap_params(num_samples, confidence)
    if nll_a.size == 0:
        raise ValueError("No shards available for bootstrap computation.")

    indices = np.arange(nll_a.size)
    samples = np.empty(num_samples, dtype=np.float64)

    for i in range(num_samples):
        resample = rng.choice(indices, size=indices.size, replace=True)
        samples[i] = token_weighted_delta(nll_a[resample], nll_b[resample], tokens[resample])

    mean = float(samples.mean())
    alpha = 1.0 - confidence
    lower = float(np.quantile(samples, alpha / 2))
    upper = float(np.quantile(samples, 1.0 - alpha / 2))
    return mean, lower, upper


def paired_t_statistic(nll_a: np.ndarray, nll_b: np.ndarray, tokens: np.ndarray) -> Tuple[float, float, int]:
    ce_a = nll_a / tokens
    ce_b = nll_b / tokens
    deltas = ce_a - ce_b
    n = deltas.size
    if n < 2:
        return float(deltas.mean()), math.nan, n

    mean = float(deltas.mean())
    std = float(deltas.std(ddof=1))
    if std == 0.0:
        return mean, math.inf, n

    t_stat = mean / (std / math.sqrt(n))
    return mean, float(t_stat), n


def percent_perplexity_change(delta_ce: float) -> float:
    return 100.0 * (math.exp(delta_ce) - 1.0)


def compare_eval_rows(rows: Iterable[EvalRow], config: ComparisonConfig) -> ComparisonResult:
    rows_list = list(rows)
    if not rows_list:
        raise ValueError("No evaluation rows provided.")

    seeds_filter = {str(seed) for seed in config.seeds} if config.seeds is not None else None

    def filter_rows(model: str) -> List[EvalRow]:
        out = [
            row
            for row in rows_list
            if row.model == model and row.step == config.step and (seeds_filter is None or row.seed in seeds_filter)
        ]
        if not out:
            raise ValueError(f"No rows found for model '{model}' at step {config.step}.")
        return out

    rows_a = filter_rows(config.model_a)
    rows_b = filter_rows(config.model_b)

    def group(rows: List[EvalRow]) -> dict[Tuple[str, str], EvalRow]:
        return {(row.seed, row.shard_id): row for row in rows}

    grouped_a = group(rows_a)
    grouped_b = group(rows_b)

    shared_keys = sorted(set(grouped_a) & set(grouped_b))
    if not shared_keys:
        raise ValueError("No overlapping seeds and shards between the selected models.")

    if not config.allow_missing_shards:
        missing_a = sorted(set(grouped_b) - set(grouped_a))
        missing_b = sorted(set(grouped_a) - set(grouped_b))
        if missing_a or missing_b:
            raise ValueError(
                "Shard mismatch detected. Missing in model A: "
                + ", ".join(map(str, missing_a))
                + "; missing in model B: "
                + ", ".join(map(str, missing_b))
            )

    rng = np.random.default_rng(config.bootstrap_seed)
    per_seed_results: List[SeedComparison] = []
    per_seed_deltas: List[float] = []

    seeds = sorted({seed for seed, _ in shared_keys}, key=lambda s: (int(s) if s.isdigit() else s))
    for seed in seeds:
        shard_ids = [shard for s, shard in shared_keys if s == seed]
        shard_rows_a = [grouped_a[(seed, shard)] for shard in shard_ids]
        shard_rows_b = [grouped_b[(seed, shard)] for shard in shard_ids]

        nll_a = np.array([row.total_nll for row in shard_rows_a], dtype=np.float64)
        nll_b = np.array([row.total_nll for row in shard_rows_b], dtype=np.float64)
        tokens = np.array([row.total_tokens for row in shard_rows_a], dtype=np.float64)

        delta = token_weighted_delta(nll_a, nll_b, tokens)
        _, ci_low, ci_high = bootstrap_weighted_delta(
            nll_a,
            nll_b,
            tokens,
            num_samples=config.bootstrap_samples,
            rng=np.random.default_rng(rng.integers(2**32 - 1)),
            confidence=config.confidence,
        )
        _, t_stat, n_pairs = paired_t_statistic(nll_a, nll_b, tokens)

        per_seed_results.append(
            SeedComparison(
                seed=seed,
                delta_ce=delta,
                ci_low=ci_low,
                ci_high=ci_high,
                paired_t_stat=t_stat,
                n_pairs=n_pairs,
            )
        )
        per_seed_deltas.append(delta)

    per_seed_deltas_arr = np.array(per_seed_deltas, dtype=np.float64)
    mean_delta = float(per_seed_deltas_arr.mean())
    ppl_delta = percent_perplexity_change(mean_delta)

    # Bootstrap across seeds with equal weight per seed.
    seed_rng = np.random.default_rng(config.bootstrap_seed ^ 0xBAD5EED)
    _ensure_bootstrap_params(config.bootstrap_samples, config.confidence)
    seed_indices = np.arange(per_seed_deltas_arr.size)
    seed_samples = np.empty(config.bootstrap_samples, dtype=np.float64)
    for i in range(config.bootstrap_samples):
        resample = seed_rng.choice(seed_indices, size=seed_indices.size, replace=True)
        seed_samples[i] = float(per_seed_deltas_arr[resample].mean())

    alpha = 1.0 - config.confidence
    ci_low = float(np.quantile(seed_samples, alpha / 2))
    ci_high = float(np.quantile(seed_samples, 1.0 - alpha / 2))

    return ComparisonResult(
        config=config,
        seed_results=per_seed_results,
        mean_delta_ce=mean_delta,
        ci_low=ci_low,
        ci_high=ci_high,
        ppl_delta_pct=ppl_delta,
    )


