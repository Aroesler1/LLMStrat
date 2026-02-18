#!/usr/bin/env python3
"""
Backtest metric utilities.

This module keeps metric calculation independent from the runner so that
walk-forward and live-paper monitoring can share one metric implementation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


TRADING_DAYS = 252


@dataclass
class BacktestMetrics:
    total_return: float
    cagr: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration_days: int
    calmar_ratio: float
    annual_volatility: float
    annual_turnover: float
    avg_holding_period_days: float
    win_rate: float
    profit_factor: float
    avg_win_loss_ratio: float
    total_costs_bps: float
    information_ratio: float
    alpha: float
    beta: float
    tracking_error: float
    avg_positions: float
    hit_rate_monthly: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def compute_backtest_metrics(
    strategy_returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    turnover: Optional[pd.Series] = None,
    costs: Optional[pd.Series] = None,
    num_positions: Optional[pd.Series] = None,
) -> BacktestMetrics:
    """
    Compute a full metric suite from daily returns.

    Args:
        strategy_returns: Daily strategy returns.
        benchmark_returns: Daily benchmark returns aligned to strategy index.
        turnover: Daily turnover ratio (0~2 typical long-only).
        costs: Daily cost in return terms (already scaled to NAV).
        num_positions: Daily count of active positions.
    """
    r = _clean_series(strategy_returns)
    b = _clean_series(benchmark_returns).reindex(r.index).fillna(0.0) if benchmark_returns is not None else pd.Series(0.0, index=r.index)
    t = _clean_series(turnover).reindex(r.index).fillna(0.0) if turnover is not None else pd.Series(0.0, index=r.index)
    c = _clean_series(costs).reindex(r.index).fillna(0.0) if costs is not None else pd.Series(0.0, index=r.index)
    npos = _clean_series(num_positions).reindex(r.index).fillna(0.0) if num_positions is not None else pd.Series(0.0, index=r.index)

    if r.empty:
        return BacktestMetrics(
            total_return=0.0,
            cagr=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            max_drawdown_duration_days=0,
            calmar_ratio=0.0,
            annual_volatility=0.0,
            annual_turnover=0.0,
            avg_holding_period_days=0.0,
            win_rate=0.0,
            profit_factor=0.0,
            avg_win_loss_ratio=0.0,
            total_costs_bps=0.0,
            information_ratio=0.0,
            alpha=0.0,
            beta=0.0,
            tracking_error=0.0,
            avg_positions=0.0,
            hit_rate_monthly=0.0,
        )

    eq = (1.0 + r).cumprod()
    total_return = float(eq.iloc[-1] - 1.0)
    years = max(len(r) / TRADING_DAYS, 1.0 / TRADING_DAYS)
    cagr = float(eq.iloc[-1] ** (1.0 / years) - 1.0)

    ann_vol = float(r.std(ddof=0) * np.sqrt(TRADING_DAYS))
    sharpe = _safe_div(float(r.mean() * TRADING_DAYS), ann_vol)

    downside = r[r < 0]
    downside_vol = float(downside.std(ddof=0) * np.sqrt(TRADING_DAYS)) if not downside.empty else 0.0
    sortino = _safe_div(float(r.mean() * TRADING_DAYS), downside_vol)

    max_dd, dd_duration = _max_drawdown(eq)
    calmar = _safe_div(cagr, abs(max_dd)) if max_dd < 0 else 0.0

    win_rate = float((r > 0).mean())
    gross_profit = float(r[r > 0].sum())
    gross_loss = float(-r[r < 0].sum())
    profit_factor = _safe_div(gross_profit, gross_loss)

    avg_win = float(r[r > 0].mean()) if (r > 0).any() else 0.0
    avg_loss = float(-r[r < 0].mean()) if (r < 0).any() else 0.0
    avg_win_loss_ratio = _safe_div(avg_win, avg_loss)

    annual_turnover = float(t.mean() * TRADING_DAYS)
    avg_holding = _safe_div(TRADING_DAYS, annual_turnover) if annual_turnover > 0 else 0.0
    total_costs_bps = float(c.sum() * 10000.0)

    excess = r - b
    tracking_error = float(excess.std(ddof=0) * np.sqrt(TRADING_DAYS))
    information_ratio = _safe_div(float(excess.mean() * TRADING_DAYS), tracking_error)

    alpha, beta = _alpha_beta(r, b)
    avg_positions = float(npos[npos > 0].mean()) if (npos > 0).any() else 0.0
    hit_rate_monthly = _monthly_hit_rate(r)

    return BacktestMetrics(
        total_return=total_return,
        cagr=cagr,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        max_drawdown_duration_days=dd_duration,
        calmar_ratio=calmar,
        annual_volatility=ann_vol,
        annual_turnover=annual_turnover,
        avg_holding_period_days=avg_holding,
        win_rate=win_rate,
        profit_factor=profit_factor,
        avg_win_loss_ratio=avg_win_loss_ratio,
        total_costs_bps=total_costs_bps,
        information_ratio=information_ratio,
        alpha=alpha,
        beta=beta,
        tracking_error=tracking_error,
        avg_positions=avg_positions,
        hit_rate_monthly=hit_rate_monthly,
    )


def _clean_series(s: Optional[pd.Series]) -> pd.Series:
    if s is None:
        return pd.Series(dtype=float)
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    out = s.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if not isinstance(out.index, pd.DatetimeIndex):
        try:
            out.index = pd.to_datetime(out.index)
        except Exception:
            pass
    return out.sort_index()


def _safe_div(a: float, b: float) -> float:
    if abs(b) <= 1e-12:
        return 0.0
    v = a / b
    if np.isnan(v) or np.isinf(v):
        return 0.0
    return float(v)


def _max_drawdown(eq: pd.Series) -> tuple[float, int]:
    if eq.empty:
        return 0.0, 0
    peak = eq.cummax()
    dd = eq / peak - 1.0
    max_dd = float(dd.min())
    # Duration: longest consecutive underwater period.
    underwater = dd < 0
    if not underwater.any():
        return max_dd, 0
    max_duration = 0
    cur = 0
    for flag in underwater.values:
        if flag:
            cur += 1
            max_duration = max(max_duration, cur)
        else:
            cur = 0
    return max_dd, int(max_duration)


def _alpha_beta(strategy_returns: pd.Series, benchmark_returns: pd.Series) -> tuple[float, float]:
    r = strategy_returns.reindex(benchmark_returns.index).dropna()
    b = benchmark_returns.reindex(r.index).dropna()
    if r.empty or b.empty:
        return 0.0, 0.0
    var_b = float(b.var(ddof=0))
    if var_b <= 1e-12:
        return 0.0, 0.0
    cov = float(np.cov(r.values, b.values, ddof=0)[0, 1])
    beta = cov / var_b
    alpha_daily = float(r.mean() - beta * b.mean())
    alpha_annual = alpha_daily * TRADING_DAYS
    return alpha_annual, float(beta)


def _monthly_hit_rate(returns: pd.Series) -> float:
    if returns.empty or not isinstance(returns.index, pd.DatetimeIndex):
        return 0.0
    monthly = (1.0 + returns).resample("ME").prod() - 1.0
    if monthly.empty:
        return 0.0
    return float((monthly > 0).mean())
