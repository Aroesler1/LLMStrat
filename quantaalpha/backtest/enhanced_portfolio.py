#!/usr/bin/env python3
"""
Enhanced, non-HFT portfolio construction utilities.

Features:
1. Optimizer choices: equal, mean-variance, risk-parity, kelly
2. Regime-aware signal adaptation (calm vs volatile)
3. Turnover and transaction-cost-aware simulation
4. Optional risk exposure neutralization
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from quantaalpha.log import logger
from .cost_model import CostModel, CostModelConfig
from .metrics import compute_backtest_metrics


@dataclass
class EnhancedPortfolioConfig:
    enabled: bool = False
    optimizer: str = "mvo"  # equal | mvo | risk_parity | kelly
    topk: int = 50
    rebalance_frequency: int = 5
    covariance_lookback: int = 60
    signal_power: float = 1.0
    transaction_cost_bps: float = 10.0
    max_weight: float = 0.08
    min_weight: float = 0.0
    shrinkage: float = 0.3
    ridge: float = 1e-4
    kelly_fraction: float = 0.3
    risk_neutralization_enabled: bool = False
    risk_exposure_file: Optional[str] = None
    risk_exposure_columns: Optional[list[str]] = None

    regime_enabled: bool = False
    regime_lookback: int = 63
    regime_high_vol_quantile: float = 0.7
    trend_window: int = 20
    reversal_window: int = 5
    momentum_window: int = 20
    volatile_reversal_blend: float = 0.35
    calm_momentum_blend: float = 0.15
    volatile_signal_scale: float = 0.6
    calm_signal_scale: float = 1.0

    @classmethod
    def from_dict(cls, cfg: Optional[Dict[str, Any]]) -> "EnhancedPortfolioConfig":
        cfg = cfg or {}
        regime = cfg.get("regime", {}) or {}
        risk = cfg.get("risk_neutralization", {}) or {}
        return cls(
            enabled=bool(cfg.get("enabled", False)),
            optimizer=str(cfg.get("optimizer", "mvo")).lower(),
            topk=int(cfg.get("topk", 50)),
            rebalance_frequency=max(1, int(cfg.get("rebalance_frequency", 5))),
            covariance_lookback=max(10, int(cfg.get("covariance_lookback", 60))),
            signal_power=float(cfg.get("signal_power", 1.0)),
            transaction_cost_bps=float(cfg.get("transaction_cost_bps", 10.0)),
            max_weight=float(cfg.get("max_weight", 0.08)),
            min_weight=float(cfg.get("min_weight", 0.0)),
            shrinkage=float(cfg.get("shrinkage", 0.3)),
            ridge=float(cfg.get("ridge", 1e-4)),
            kelly_fraction=float(cfg.get("kelly_fraction", 0.3)),
            risk_neutralization_enabled=bool(risk.get("enabled", False)),
            risk_exposure_file=risk.get("exposure_file"),
            risk_exposure_columns=risk.get("exposure_columns"),
            regime_enabled=bool(regime.get("enabled", False)),
            regime_lookback=max(10, int(regime.get("lookback", 63))),
            regime_high_vol_quantile=float(regime.get("high_vol_quantile", 0.7)),
            trend_window=max(5, int(regime.get("trend_window", 20))),
            reversal_window=max(2, int(regime.get("reversal_window", 5))),
            momentum_window=max(5, int(regime.get("momentum_window", 20))),
            volatile_reversal_blend=float(regime.get("volatile_reversal_blend", 0.35)),
            calm_momentum_blend=float(regime.get("calm_momentum_blend", 0.15)),
            volatile_signal_scale=float(regime.get("volatile_signal_scale", 0.6)),
            calm_signal_scale=float(regime.get("calm_signal_scale", 1.0)),
        )


class EnhancedPortfolioBacktester:
    def __init__(self, config: EnhancedPortfolioConfig):
        self.cfg = config
        self._risk_exposure_df = self._load_risk_exposures()
        self._cost_model = CostModel(CostModelConfig())

    def set_cost_model(self, config: CostModelConfig):
        self._cost_model = CostModel(config)

    def run(
        self,
        signal: pd.Series | pd.DataFrame,
        close_df: pd.DataFrame,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, float]]:
        """
        Simulate portfolio using predicted signal and close prices.

        Returns:
            (daily_report_df, metrics_dict)
        """
        signal_df = self._to_signal_matrix(signal)
        if signal_df.empty:
            raise ValueError("Empty signal input for enhanced portfolio backtest")

        close = close_df.sort_index().sort_index(axis=1)
        returns = close.pct_change(fill_method=None).shift(-1).replace([np.inf, -np.inf], np.nan)

        common_dates = signal_df.index.intersection(returns.index)
        common_assets = signal_df.columns.intersection(returns.columns)
        signal_df = signal_df.loc[common_dates, common_assets]
        returns = returns.loc[common_dates, common_assets]

        if benchmark_returns is None:
            benchmark_returns = pd.Series(0.0, index=common_dates)
        else:
            benchmark_returns = benchmark_returns.reindex(common_dates).fillna(0.0)

        regimes = self._classify_regime(benchmark_returns)
        weights_prev = pd.Series(0.0, index=common_assets)
        rebalance_freq = max(1, self.cfg.rebalance_frequency)

        rows = []
        nav = 1.0
        dates = list(common_dates)
        for i, dt in enumerate(dates):
            ret_t = returns.loc[dt].dropna()
            if ret_t.empty:
                continue

            bench_ret = float(benchmark_returns.loc[dt]) if dt in benchmark_returns.index else 0.0
            regime = regimes.get(dt, "unknown")
            do_rebalance = (i % rebalance_freq == 0) or (i == 0)

            if do_rebalance:
                scores = signal_df.loc[dt].reindex(ret_t.index).replace([np.inf, -np.inf], np.nan).dropna()
                hist_start = max(0, i - self.cfg.covariance_lookback)
                ret_hist = returns.iloc[hist_start:i].reindex(columns=scores.index).dropna(how="all", axis=1)
                scores = scores.reindex(ret_hist.columns).dropna()

                if not scores.empty:
                    scores = self._adapt_signal_by_regime(
                        scores=scores,
                        regime=regime,
                        returns_hist=ret_hist,
                    )
                    w_target = self._construct_weights(
                        scores=scores,
                        returns_hist=ret_hist,
                    )
                    w_target = self._neutralize_weights(w_target, dt)
                else:
                    w_target = weights_prev.copy()
            else:
                w_target = weights_prev.copy()

            # Keep only tradable assets for the day.
            w_target = w_target.reindex(ret_t.index).fillna(0.0).clip(lower=0.0)
            gross_exposure = float(w_target.sum())
            if gross_exposure > 1.0:
                w_target = w_target / gross_exposure
                gross_exposure = 1.0
            cash_weight = max(0.0, 1.0 - gross_exposure)

            turnover = float((w_target - weights_prev.reindex(w_target.index).fillna(0.0)).abs().sum())
            base_cost = turnover * (self.cfg.transaction_cost_bps / 10000.0)
            model_cost = self._cost_model.estimate_turnover_cost_rate(turnover)
            cost = max(base_cost, model_cost)
            port_ret = float((w_target * ret_t.reindex(w_target.index).fillna(0.0)).sum()) - cost
            excess = port_ret - bench_ret
            nav *= (1.0 + port_ret)

            rows.append(
                {
                    "date": dt,
                    "portfolio_return": port_ret,
                    "benchmark_return": bench_ret,
                    "excess_return": excess,
                    "turnover": turnover,
                    "cost": cost,
                    "regime": regime,
                    "nav": nav,
                    "gross_exposure": gross_exposure,
                    "cash_weight": cash_weight,
                    "num_positions": int((w_target > 1e-10).sum()),
                }
            )

            weights_prev = w_target.reindex(common_assets).fillna(0.0)

        if not rows:
            raise ValueError("Enhanced portfolio simulation produced no daily rows")

        daily = pd.DataFrame(rows).set_index("date").sort_index()
        metrics = self._compute_metrics(daily)
        return daily, metrics

    @staticmethod
    def _to_signal_matrix(signal: pd.Series | pd.DataFrame) -> pd.DataFrame:
        if isinstance(signal, pd.DataFrame):
            if signal.shape[1] == 1 and isinstance(signal.index, pd.MultiIndex):
                s = signal.iloc[:, 0]
            elif isinstance(signal.index, pd.MultiIndex):
                s = signal.iloc[:, 0]
            else:
                return signal.copy()
        else:
            s = signal

        if not isinstance(s.index, pd.MultiIndex):
            return pd.DataFrame()

        names = list(s.index.names)
        if "datetime" in names and "instrument" in names:
            return s.unstack("instrument").sort_index().sort_index(axis=1)

        # Fallback for unnamed index levels: assume (datetime, instrument) or inverse.
        idx0 = s.index.get_level_values(0)
        if pd.api.types.is_datetime64_any_dtype(idx0):
            mat = s.unstack(level=1)
            mat.index = pd.to_datetime(mat.index)
            return mat.sort_index().sort_index(axis=1)
        mat = s.unstack(level=0)
        mat.index = pd.to_datetime(mat.index)
        return mat.sort_index().sort_index(axis=1)

    def _classify_regime(self, benchmark_returns: pd.Series) -> Dict[pd.Timestamp, str]:
        if benchmark_returns.empty:
            return {}
        bench = benchmark_returns.fillna(0.0)
        if not self.cfg.regime_enabled:
            return {pd.Timestamp(dt): "neutral" for dt in bench.index}
        # Use only information available strictly before decision date (t-1 and earlier)
        # to avoid forward-looking regime labels.
        bench_lag = bench.shift(1).fillna(0.0)
        vol = bench_lag.rolling(
            self.cfg.regime_lookback,
            min_periods=max(10, self.cfg.regime_lookback // 3),
        ).std()
        trend = bench_lag.rolling(
            self.cfg.trend_window,
            min_periods=max(5, self.cfg.trend_window // 2),
        ).mean()
        # Time-varying quantile threshold, also lagged by one step.
        vol_q = vol.expanding(min_periods=max(20, self.cfg.regime_lookback // 2)).quantile(
            self.cfg.regime_high_vol_quantile
        )
        vol_thr = vol_q.shift(1)

        regime = {}
        for dt in bench.index:
            v = float(vol.loc[dt]) if pd.notna(vol.loc[dt]) else 0.0
            t = float(trend.loc[dt]) if pd.notna(trend.loc[dt]) else 0.0
            thr = float(vol_thr.loc[dt]) if pd.notna(vol_thr.loc[dt]) else 0.0
            volatile = v >= thr if thr > 0 else False
            uptrend = t >= 0
            if volatile and uptrend:
                regime[pd.Timestamp(dt)] = "volatile_up"
            elif volatile and not uptrend:
                regime[pd.Timestamp(dt)] = "volatile_down"
            elif not volatile and uptrend:
                regime[pd.Timestamp(dt)] = "calm_up"
            else:
                regime[pd.Timestamp(dt)] = "calm_down"
        return regime

    def _adapt_signal_by_regime(
        self,
        scores: pd.Series,
        regime: str,
        returns_hist: pd.DataFrame,
    ) -> pd.Series:
        if scores.empty:
            return scores

        z = self._zscore(scores)
        base = np.sign(z) * (np.abs(z) ** self.cfg.signal_power)

        if not self.cfg.regime_enabled or returns_hist.empty:
            return base

        rev = self._zscore(
            -returns_hist.tail(self.cfg.reversal_window).mean(skipna=True).reindex(base.index).fillna(0.0)
        )
        mom = self._zscore(
            returns_hist.tail(self.cfg.momentum_window).mean(skipna=True).reindex(base.index).fillna(0.0)
        )

        if regime.startswith("volatile"):
            b = float(np.clip(self.cfg.volatile_reversal_blend, 0.0, 1.0))
            out = (1.0 - b) * base + b * rev
            return out * self.cfg.volatile_signal_scale

        b = float(np.clip(self.cfg.calm_momentum_blend, 0.0, 1.0))
        out = (1.0 - b) * base + b * mom
        return out * self.cfg.calm_signal_scale

    def _construct_weights(self, scores: pd.Series, returns_hist: pd.DataFrame) -> pd.Series:
        scores = scores.dropna().sort_values(ascending=False)
        if scores.empty:
            return scores
        topk = max(1, min(self.cfg.topk, len(scores)))
        scores = scores.iloc[:topk]

        if self.cfg.optimizer == "equal":
            w = pd.Series(1.0 / len(scores), index=scores.index)
            return self._finalize_weights(w)

        ret_hist = returns_hist.reindex(columns=scores.index).dropna(how="all")
        if ret_hist.empty:
            w = pd.Series(1.0 / len(scores), index=scores.index)
            return self._finalize_weights(w)

        mu = self._zscore(scores).fillna(0.0).to_numpy(dtype=np.float64)
        cov = ret_hist.cov().replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)
        cov = self._shrink_covariance(cov)

        if self.cfg.optimizer == "risk_parity":
            vol = np.sqrt(np.clip(np.diag(cov), 1e-12, None))
            raw = 1.0 / vol
        else:
            inv_cov = np.linalg.pinv(cov)
            raw = inv_cov @ mu
            if self.cfg.optimizer == "kelly":
                raw = raw * max(0.0, self.cfg.kelly_fraction)

        raw = np.clip(raw, 0.0, None)
        if raw.sum() <= 0:
            raw = np.ones_like(raw) / len(raw)
        w = pd.Series(raw / raw.sum(), index=scores.index)
        return self._finalize_weights(w)

    def _finalize_weights(self, w: pd.Series) -> pd.Series:
        w = w.clip(lower=max(0.0, self.cfg.min_weight))
        if self.cfg.max_weight > 0:
            w = self._apply_weight_cap(w, cap=float(self.cfg.max_weight))
        total = float(w.sum())
        if total <= 0:
            return pd.Series(0.0, index=w.index)
        if total > 1.0:
            w = w / total
        return w

    def _shrink_covariance(self, cov: np.ndarray) -> np.ndarray:
        if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
            raise ValueError("Covariance matrix must be square")
        diag = np.diag(np.diag(cov))
        shrunk = (1.0 - self.cfg.shrinkage) * cov + self.cfg.shrinkage * diag
        shrunk = shrunk + np.eye(shrunk.shape[0]) * self.cfg.ridge
        return shrunk

    @staticmethod
    def _zscore(s: pd.Series) -> pd.Series:
        s = s.astype(float).replace([np.inf, -np.inf], np.nan)
        mu = float(s.mean(skipna=True))
        sd = float(s.std(skipna=True))
        if sd <= 1e-12:
            return pd.Series(0.0, index=s.index)
        return (s - mu) / sd

    @staticmethod
    def _apply_weight_cap(w: pd.Series, cap: float) -> pd.Series:
        cap = float(max(0.0, cap))
        if cap <= 0:
            return pd.Series(0.0, index=w.index)

        x = w.clip(lower=0.0, upper=cap).astype(float).copy()
        if x.empty:
            return x

        # Iteratively rescale only uncapped names to keep sum <= 1 while
        # preserving per-name cap constraints.
        for _ in range(8):
            total = float(x.sum())
            if total <= 1.0 + 1e-12:
                break
            capped_mask = x >= (cap - 1e-12)
            capped_sum = float(x[capped_mask].sum())
            free_idx = x.index[~capped_mask]
            remaining = 1.0 - capped_sum
            if remaining <= 0 or len(free_idx) == 0:
                x.loc[~capped_mask] = 0.0
                break
            free_sum = float(x.loc[free_idx].sum())
            if free_sum <= 1e-12:
                x.loc[free_idx] = remaining / len(free_idx)
            else:
                x.loc[free_idx] = x.loc[free_idx] / free_sum * remaining
            x = x.clip(lower=0.0, upper=cap)

        return x

    def _load_risk_exposures(self) -> Optional[pd.DataFrame]:
        if not self.cfg.risk_neutralization_enabled or not self.cfg.risk_exposure_file:
            return None
        path = Path(self.cfg.risk_exposure_file)
        if not path.exists():
            logger.warning(f"Risk exposure file not found: {path}")
            return None
        try:
            if path.suffix.lower() in (".parquet", ".pq"):
                df = pd.read_parquet(path)
            else:
                df = pd.read_csv(path)
            if not {"datetime", "instrument"}.issubset(df.columns):
                logger.warning("Risk exposure file must contain datetime and instrument columns")
                return None
            cols = self.cfg.risk_exposure_columns or [
                c for c in df.columns if c not in ("datetime", "instrument")
            ]
            if not cols:
                logger.warning("Risk exposure columns are empty; disabling neutralization")
                return None
            out = df[["datetime", "instrument", *cols]].copy()
            out["datetime"] = pd.to_datetime(out["datetime"])
            out["instrument"] = out["instrument"].astype(str)
            out = out.set_index(["datetime", "instrument"]).sort_index()
            return out
        except Exception as e:
            logger.warning(f"Failed to load risk exposure file: {e}")
            return None

    def _neutralize_weights(self, w: pd.Series, dt: pd.Timestamp) -> pd.Series:
        if self._risk_exposure_df is None or w.empty:
            return self._finalize_weights(w)

        dt = pd.Timestamp(dt)
        try:
            expo = self._risk_exposure_df.xs(dt, level="datetime")
        except Exception:
            return self._finalize_weights(w)

        expo = expo.reindex(w.index).dropna(how="all")
        if expo.empty:
            return self._finalize_weights(w)

        aligned = w.reindex(expo.index).fillna(0.0).to_numpy(dtype=np.float64)
        b = expo.to_numpy(dtype=np.float64)
        if b.ndim != 2 or b.shape[0] == 0 or b.shape[1] == 0:
            return self._finalize_weights(w)

        b = b - np.nanmean(b, axis=0, keepdims=True)
        b = np.nan_to_num(b, nan=0.0)
        try:
            bt_b = b.T @ b + np.eye(b.shape[1]) * 1e-6
            proj = b @ np.linalg.solve(bt_b, b.T @ aligned)
            neutral = np.clip(aligned - proj, 0.0, None)
        except Exception:
            return self._finalize_weights(w)

        out = w.copy()
        out.loc[expo.index] = neutral
        return self._finalize_weights(out)

    @staticmethod
    def _compute_metrics(daily: pd.DataFrame) -> Dict[str, float]:
        port = daily["portfolio_return"].fillna(0.0)
        bench = daily["benchmark_return"].fillna(0.0)
        turnover = daily["turnover"].fillna(0.0)
        cost = daily["cost"].fillna(0.0)
        npos = daily["num_positions"].fillna(0.0) if "num_positions" in daily.columns else None

        suite = compute_backtest_metrics(
            strategy_returns=port,
            benchmark_returns=bench,
            turnover=turnover,
            costs=cost,
            num_positions=npos,
        )
        out = suite.to_dict()
        # Backward-compatible aliases used by existing UI/export code.
        out["annualized_return"] = out["cagr"]
        out["avg_turnover"] = float(turnover.mean()) if len(turnover) > 0 else 0.0
        if "cash_weight" in daily.columns:
            out["avg_cash_weight"] = float(daily["cash_weight"].mean())
        if "gross_exposure" in daily.columns:
            out["avg_gross_exposure"] = float(daily["gross_exposure"].mean())
        out["num_days"] = float(len(daily))
        return out
