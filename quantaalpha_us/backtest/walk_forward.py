from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from quantaalpha_us.backtest.costs import TransactionCostModel
from quantaalpha_us.backtest.universe import SP500Universe
from quantaalpha_us.pipeline.signal_generator import SignalConfig, generate_signals


@dataclass
class FoldWindow:
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        for k in out:
            out[k] = str(pd.Timestamp(out[k]).date())
        return out


@dataclass
class WalkForwardSummary:
    num_folds: int
    num_observations: int
    start_date: str
    end_date: str
    gross_sharpe: float
    net_sharpe: float
    avg_turnover: float
    max_drawdown: float
    cumulative_net_return: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class WalkForwardResult:
    returns: pd.DataFrame
    folds: list[FoldWindow]
    summary: WalkForwardSummary

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary.to_dict(),
            "folds": [fold.to_dict() for fold in self.folds],
            "rows": int(len(self.returns)),
        }


def _annualized_sharpe(returns: pd.Series) -> float:
    s = pd.to_numeric(returns, errors="coerce").dropna()
    if s.empty:
        return 0.0
    std = float(s.std(ddof=1))
    if std <= 0:
        return 0.0
    return float(s.mean()) / std * math.sqrt(252.0)


def _max_drawdown(returns: pd.Series) -> float:
    s = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    if s.empty:
        return 0.0
    equity = (1.0 + s).cumprod()
    peak = equity.cummax()
    dd = 1.0 - equity / peak.replace(0, pd.NA)
    return float(dd.max()) if len(dd) else 0.0


class WalkForwardRunner:
    """Deterministic walk-forward runner for the US baseline strategy."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        wf = config.get("walk_forward", {}) if isinstance(config.get("walk_forward"), dict) else {}
        portfolio = config.get("portfolio", {}) if isinstance(config.get("portfolio"), dict) else {}
        costs = config.get("costs", {}) if isinstance(config.get("costs"), dict) else {}
        exec_cfg = config.get("execution_alignment", {}) if isinstance(config.get("execution_alignment"), dict) else {}

        self.initial_train_months = int(wf.get("initial_train_months", 60))
        self.validation_months = int(wf.get("validation_months", 12))
        self.test_months = int(wf.get("test_months", 3))
        self.embargo_days = int(wf.get("embargo_days", 5))
        self.expanding_window = bool(wf.get("expanding_window", True))
        self.warm_up_days = int(wf.get("warm_up_trading_days", 252))
        self.history_window_days = int(max(wf.get("history_window_days", 400), 120))
        self.signal_lag_days = int(exec_cfg.get("signal_lag_days", 1))
        self.label_horizon_days = int(exec_cfg.get("rebalance_frequency_days", 1))

        self.signal_config = SignalConfig(
            top_k=int(portfolio.get("top_k", 30)),
            max_weight=float(portfolio.get("max_weight_per_name", 0.05)),
            long_only=bool(portfolio.get("long_only", True)),
            max_turnover_daily=float(portfolio.get("max_daily_turnover", 0.20)),
            min_avg_dollar_volume=float(portfolio.get("min_avg_daily_volume_usd", 5_000_000)),
        )
        self.cost_model = TransactionCostModel(
            commission_per_share=float(costs.get("commission_per_share", 0.0)),
            half_spread_bps=float(costs.get("spread_bps", 1.5)),
            slippage_bps=float(costs.get("slippage_bps", 1.0)),
        )

    @staticmethod
    def _nearest_on_or_before(dates: pd.DatetimeIndex, target: pd.Timestamp) -> Optional[pd.Timestamp]:
        idx = dates.searchsorted(target, side="right") - 1
        if idx < 0 or idx >= len(dates):
            return None
        return pd.Timestamp(dates[idx]).normalize()

    @staticmethod
    def _nearest_on_or_after(dates: pd.DatetimeIndex, target: pd.Timestamp) -> Optional[pd.Timestamp]:
        idx = dates.searchsorted(target, side="left")
        if idx < 0 or idx >= len(dates):
            return None
        return pd.Timestamp(dates[idx]).normalize()

    @staticmethod
    def _shift_trading_days(dates: pd.DatetimeIndex, date_value: pd.Timestamp, n: int) -> Optional[pd.Timestamp]:
        idx = dates.searchsorted(pd.Timestamp(date_value).normalize(), side="left")
        out_idx = idx + int(n)
        if out_idx < 0 or out_idx >= len(dates):
            return None
        return pd.Timestamp(dates[out_idx]).normalize()

    def build_folds(self, trading_dates: pd.DatetimeIndex) -> list[FoldWindow]:
        dates = pd.DatetimeIndex(pd.to_datetime(trading_dates)).normalize().unique().sort_values()
        if len(dates) < self.warm_up_days + 252:
            return []

        folds: list[FoldWindow] = []
        train_start = pd.Timestamp(dates[0]).normalize()
        train_end_target = train_start + pd.DateOffset(months=self.initial_train_months)
        fold_id = 0

        while True:
            train_end = self._nearest_on_or_before(dates, pd.Timestamp(train_end_target))
            if train_end is None:
                break

            valid_start = self._nearest_on_or_after(dates, train_end + pd.offsets.BDay(self.embargo_days + 1))
            if valid_start is None:
                break
            valid_end = self._nearest_on_or_before(dates, valid_start + pd.DateOffset(months=self.validation_months))
            if valid_end is None:
                break
            test_start = self._nearest_on_or_after(dates, valid_end + pd.offsets.BDay(1))
            if test_start is None:
                break
            test_end = self._nearest_on_or_before(dates, test_start + pd.DateOffset(months=self.test_months))
            if test_end is None or test_end <= test_start:
                break

            folds.append(
                FoldWindow(
                    fold_id=fold_id,
                    train_start=train_start,
                    train_end=train_end,
                    valid_start=valid_start,
                    valid_end=valid_end,
                    test_start=test_start,
                    test_end=test_end,
                )
            )
            fold_id += 1

            shift = pd.DateOffset(months=self.test_months)
            if self.expanding_window:
                train_end_target = train_end + shift
            else:
                train_start = self._nearest_on_or_after(dates, train_start + shift) or train_start
                train_end_target = train_end + shift

            if test_end >= pd.Timestamp(dates[-1]).normalize():
                break
            if fold_id > 200:
                break

        return folds

    @staticmethod
    def _normalize_bars(bars: pd.DataFrame) -> pd.DataFrame:
        work = bars.copy()
        work["date"] = pd.to_datetime(work["date"], errors="coerce").dt.normalize()
        work["symbol"] = work["symbol"].astype(str).str.upper()
        for col in ["open", "high", "low", "close", "adj_close", "volume", "dollar_volume"]:
            if col in work.columns:
                work[col] = pd.to_numeric(work[col], errors="coerce")
        if "adj_close" not in work.columns:
            work["adj_close"] = work["close"]
        work = work.dropna(subset=["date", "symbol", "open", "close"])
        work = work.sort_values(["date", "symbol"]).reset_index(drop=True)
        return work

    @staticmethod
    def _open_return_for_symbols(
        bars_entry: pd.DataFrame,
        bars_exit: pd.DataFrame,
        symbols: list[str],
    ) -> pd.DataFrame:
        e = bars_entry[bars_entry["symbol"].isin(symbols)][["symbol", "open"]].rename(columns={"open": "open_entry"})
        x = bars_exit[bars_exit["symbol"].isin(symbols)][["symbol", "open"]].rename(columns={"open": "open_exit"})
        merged = e.merge(x, on="symbol", how="inner")
        merged = merged[(merged["open_entry"] > 0) & (merged["open_exit"] > 0)]
        merged["ret"] = merged["open_exit"] / merged["open_entry"] - 1.0
        return merged

    def run(
        self,
        *,
        bars: pd.DataFrame,
        universe: SP500Universe,
        output_dir: Optional[Path] = None,
    ) -> WalkForwardResult:
        data = self._normalize_bars(bars)
        dates = pd.DatetimeIndex(sorted(data["date"].dropna().unique())).normalize()
        folds = self.build_folds(dates)
        if not folds:
            raise RuntimeError("Unable to build walk-forward folds from available data")

        records: list[dict[str, Any]] = []
        for fold in folds:
            fold_test_dates = dates[(dates >= fold.test_start) & (dates <= fold.test_end)]
            previous_weights: Optional[dict[str, float]] = None

            for as_of in fold_test_dates:
                history_start = pd.Timestamp(as_of) - pd.Timedelta(days=self.history_window_days)
                history = data[(data["date"] >= history_start) & (data["date"] <= as_of)]
                active = universe.get_members(as_of)
                if not active:
                    continue

                signal_df = generate_signals(
                    history,
                    config=self.signal_config,
                    as_of=as_of,
                    active_universe=active,
                    previous_weights=previous_weights,
                )
                if signal_df.empty:
                    continue

                entry_date = self._shift_trading_days(dates, as_of, self.signal_lag_days)
                exit_date = self._shift_trading_days(dates, entry_date, self.label_horizon_days) if entry_date is not None else None
                if entry_date is None or exit_date is None:
                    continue

                bars_entry = data[data["date"] == entry_date]
                bars_exit = data[data["date"] == exit_date]
                symbols = signal_df["symbol"].astype(str).str.upper().tolist()
                sym_rets = self._open_return_for_symbols(bars_entry, bars_exit, symbols)
                if sym_rets.empty:
                    continue

                merged = signal_df.merge(sym_rets[["symbol", "ret"]], on="symbol", how="inner")
                if merged.empty:
                    continue

                gross_return = float((merged["weight"] * merged["ret"]).sum())
                curr_weights = {str(r.symbol).upper(): float(r.weight) for r in merged.itertuples(index=False)}
                if previous_weights:
                    all_syms = set(previous_weights) | set(curr_weights)
                    turnover = float(sum(abs(curr_weights.get(s, 0.0) - previous_weights.get(s, 0.0)) for s in all_syms))
                else:
                    turnover = float(sum(abs(w) for w in curr_weights.values()))
                cost_return = turnover * (
                    (self.cost_model.half_spread_bps + self.cost_model.slippage_bps) / 10000.0
                )
                net_return = gross_return - cost_return

                # Baselines for gate-7.
                spy_ret = float("nan")
                spy_df = self._open_return_for_symbols(bars_entry, bars_exit, ["SPY"])
                if not spy_df.empty:
                    spy_ret = float(spy_df["ret"].iloc[0])

                active_rets = self._open_return_for_symbols(bars_entry, bars_exit, active)
                eqw_ret = float(active_rets["ret"].mean()) if not active_rets.empty else float("nan")

                mom_ret = float("nan")
                lookback = history[history["date"] <= as_of].copy()
                if not lookback.empty and "adj_close" in lookback.columns:
                    mom = (
                        lookback.sort_values(["symbol", "date"])
                        .groupby("symbol")["adj_close"]
                        .pct_change(252)
                    )
                    lookback = lookback.assign(mom_252=mom)
                    latest = lookback[lookback["date"] == as_of][["symbol", "mom_252"]].dropna()
                    if not latest.empty:
                        top = latest.nlargest(self.signal_config.top_k, "mom_252")["symbol"].astype(str).tolist()
                        top_rets = self._open_return_for_symbols(bars_entry, bars_exit, top)
                        if not top_rets.empty:
                            mom_ret = float(top_rets["ret"].mean())

                records.append(
                    {
                        "fold_id": fold.fold_id,
                        "as_of_date": str(pd.Timestamp(as_of).date()),
                        "entry_date": str(pd.Timestamp(entry_date).date()),
                        "exit_date": str(pd.Timestamp(exit_date).date()),
                        "positions": int(len(curr_weights)),
                        "turnover": turnover,
                        "gross_return": gross_return,
                        "cost_return": cost_return,
                        "net_return": net_return,
                        "baseline_spy": spy_ret,
                        "baseline_equal_weight": eqw_ret,
                        "baseline_momentum": mom_ret,
                    }
                )
                previous_weights = curr_weights

        returns = pd.DataFrame(records)
        if returns.empty:
            raise RuntimeError("Walk-forward run produced no test observations")

        gross_sharpe = _annualized_sharpe(returns["gross_return"])
        net_sharpe = _annualized_sharpe(returns["net_return"])
        cumulative = float((1.0 + pd.to_numeric(returns["net_return"], errors="coerce").fillna(0.0)).prod() - 1.0)
        summary = WalkForwardSummary(
            num_folds=len(folds),
            num_observations=int(len(returns)),
            start_date=str(returns["entry_date"].min()),
            end_date=str(returns["exit_date"].max()),
            gross_sharpe=gross_sharpe,
            net_sharpe=net_sharpe,
            avg_turnover=float(pd.to_numeric(returns["turnover"], errors="coerce").mean()),
            max_drawdown=_max_drawdown(returns["net_return"]),
            cumulative_net_return=cumulative,
        )
        result = WalkForwardResult(returns=returns, folds=folds, summary=summary)

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            returns_path = output_dir / "walk_forward_returns.csv"
            folds_path = output_dir / "walk_forward_folds.json"
            summary_path = output_dir / "walk_forward_summary.json"

            returns.to_csv(returns_path, index=False)
            folds_path.write_text(json.dumps([f.to_dict() for f in folds], indent=2), encoding="utf-8")
            summary_path.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")

        return result
