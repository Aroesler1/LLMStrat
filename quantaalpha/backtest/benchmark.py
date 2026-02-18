#!/usr/bin/env python3
"""
Benchmark analysis helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd


@dataclass
class BenchmarkStats:
    alpha: float
    beta: float
    information_ratio: float
    tracking_error: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "information_ratio": self.information_ratio,
            "tracking_error": self.tracking_error,
        }


def compute_benchmark_stats(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> BenchmarkStats:
    r = strategy_returns.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    b = benchmark_returns.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    idx = r.index.intersection(b.index)
    r = r.reindex(idx).dropna()
    b = b.reindex(idx).dropna()
    if r.empty or b.empty:
        return BenchmarkStats(alpha=0.0, beta=0.0, information_ratio=0.0, tracking_error=0.0)

    excess = r - b
    te = float(excess.std(ddof=0) * np.sqrt(252))
    ir = float((excess.mean() * 252) / te) if te > 1e-12 else 0.0

    var_b = float(b.var(ddof=0))
    if var_b <= 1e-12:
        return BenchmarkStats(alpha=0.0, beta=0.0, information_ratio=ir, tracking_error=te)

    cov = float(np.cov(r.values, b.values, ddof=0)[0, 1])
    beta = cov / var_b
    alpha = float((r.mean() - beta * b.mean()) * 252)
    return BenchmarkStats(alpha=alpha, beta=float(beta), information_ratio=ir, tracking_error=te)
