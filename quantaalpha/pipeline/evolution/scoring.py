"""
Multi-objective trajectory scoring.

When enabled, this score replaces single-metric RankIC sorting for:
- crossover parent selection
- best trajectory ranking
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


DEFAULT_WEIGHTS = {
    "rank_ic": 1.0,
    "ic": 0.5,
    "annualized_return": 0.5,
    "information_ratio": 0.5,
    "max_drawdown_penalty": 0.75,
    "complexity_penalty": 0.05,
}


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        if np.isnan(x) or np.isinf(x):
            return default
        return x
    except Exception:
        return default


def _clip(v: float, lo: float, hi: float) -> float:
    return float(min(hi, max(lo, v)))


def score_trajectory(trajectory: Any, weights: Dict[str, float] | None = None) -> float:
    """
    Compute multi-objective score from trajectory metrics.

    Positive objectives:
    - RankIC, IC, annualized_return, information_ratio

    Penalties:
    - |max_drawdown|
    - expression complexity (mean symbol length proxy)
    """
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update({k: float(v) for k, v in weights.items()})

    metrics = getattr(trajectory, "backtest_metrics", {}) or {}
    rank_ic = _safe_float(metrics.get("RankIC", metrics.get("Rank IC", 0.0)))
    ic = _safe_float(metrics.get("IC", 0.0))
    ann = _safe_float(metrics.get("annualized_return", 0.0))
    info = _safe_float(metrics.get("information_ratio", 0.0))
    mdd = abs(_safe_float(metrics.get("max_drawdown", 0.0)))

    # Normalize objectives to comparable scales.
    rank_ic_n = _clip(rank_ic / 0.1, -2.0, 2.0)      # 0.1 RankIC ~ strong signal
    ic_n = _clip(ic / 0.1, -2.0, 2.0)                # 0.1 IC ~ strong signal
    ann_n = _clip(ann / 0.5, -2.0, 2.0)              # 50% annualized excess as scale
    info_n = _clip(info / 2.0, -2.0, 2.0)            # IR=2 as strong
    mdd_n = _clip(mdd / 0.3, 0.0, 3.0)               # 30% drawdown is high penalty

    complexity = _estimate_complexity_penalty(trajectory)

    score = (
        w["rank_ic"] * rank_ic_n
        + w["ic"] * ic_n
        + w["annualized_return"] * ann_n
        + w["information_ratio"] * info_n
        - w["max_drawdown_penalty"] * mdd_n
        - w["complexity_penalty"] * complexity
    )
    return float(score)


def _estimate_complexity_penalty(trajectory: Any) -> float:
    factors = getattr(trajectory, "factors", []) or []
    if not factors:
        return 0.0
    expr_lens = [len((f.get("expression", "") or "")) for f in factors]
    if not expr_lens:
        return 0.0
    avg_len = float(np.mean(expr_lens))
    # 250 chars -> ~1 unit penalty; clipped to avoid dominating score.
    return _clip(avg_len / 250.0, 0.0, 4.0)
