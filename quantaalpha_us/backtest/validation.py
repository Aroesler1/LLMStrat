from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd


def annualized_sharpe(returns: pd.Series, periods_per_year: int = 252) -> float:
    s = pd.to_numeric(returns, errors="coerce").dropna()
    if s.empty:
        return 0.0
    std = float(s.std(ddof=1))
    if std <= 0:
        return 0.0
    return float(s.mean()) / std * math.sqrt(periods_per_year)


def max_drawdown(returns: pd.Series) -> float:
    s = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    if s.empty:
        return 0.0
    equity = (1.0 + s).cumprod()
    peak = equity.cummax()
    dd = 1.0 - equity / peak.replace(0, pd.NA)
    return float(dd.max()) if len(dd) else 0.0


def deflated_sharpe(observed_sharpe: float, n_trials: int, years: float) -> float:
    trials = max(int(n_trials), 1)
    safe_years = max(float(years), 1e-9)
    penalty = math.sqrt(2.0 * math.log(trials)) / math.sqrt(safe_years)
    return float(observed_sharpe - penalty)


@dataclass
class GateResult:
    name: str
    passed: bool
    threshold: float | int
    value: float | int | None
    details: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "threshold": self.threshold,
            "value": self.value,
            "details": self.details,
        }


@dataclass
class GateReport:
    passed: bool
    gates: dict[str, GateResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "gates": {k: v.to_dict() for k, v in self.gates.items()},
        }


@dataclass
class BacktestValidationConfig:
    gate_1_deflated_sharpe: float = 0.5
    gate_2_positive_subperiods: int = 6
    gate_3_max_sector_pnl_share: float = 0.40
    gate_4_max_drawdown: float = 0.25
    gate_5_max_turnover: float = 0.30
    gate_6_min_importance_overlap: float = 0.50
    gate_8_min_net_sharpe: float = 0.4


class BacktestValidation:
    """Run the blueprint research promotion gates on stitched walk-forward returns."""

    def __init__(self, config: Optional[BacktestValidationConfig] = None) -> None:
        self.config = config or BacktestValidationConfig()

    @staticmethod
    def _split_subperiods(series: pd.Series, n: int = 8) -> list[pd.Series]:
        s = pd.to_numeric(series, errors="coerce").dropna()
        if s.empty:
            return []
        idx_chunks = pd.qcut(pd.Series(range(len(s))), q=min(n, len(s)), labels=False, duplicates="drop")
        chunks: list[pd.Series] = []
        for label in sorted(idx_chunks.unique()):
            mask = idx_chunks == label
            chunks.append(s[mask.values])
        return chunks

    @staticmethod
    def _baseline_sharpe(df: pd.DataFrame, col: str) -> float:
        if col not in df.columns:
            return float("-inf")
        return annualized_sharpe(df[col])

    def run_all_gates(
        self,
        *,
        returns_df: pd.DataFrame,
        n_trials: int,
        factor_overlap_score: Optional[float] = None,
        sector_pnl_share: Optional[dict[str, float]] = None,
    ) -> GateReport:
        if returns_df.empty or "net_return" not in returns_df.columns:
            empty_gate = GateResult(
                name="GATE-0_input",
                passed=False,
                threshold=1,
                value=0,
                details="returns_df is empty or missing net_return",
            )
            return GateReport(passed=False, gates={"GATE-0": empty_gate})

        net = pd.to_numeric(returns_df["net_return"], errors="coerce").dropna()
        years = max(len(net) / 252.0, 1e-9)
        net_sharpe = annualized_sharpe(net)
        dsr = deflated_sharpe(net_sharpe, n_trials=n_trials, years=years)

        gates: dict[str, GateResult] = {}

        gates["GATE-1"] = GateResult(
            name="Deflated Sharpe Ratio",
            passed=dsr > self.config.gate_1_deflated_sharpe,
            threshold=self.config.gate_1_deflated_sharpe,
            value=dsr,
            details="Deflated Sharpe > threshold",
        )

        subperiods = self._split_subperiods(net, n=8)
        positive = sum(1 for sp in subperiods if annualized_sharpe(sp) > 0)
        gates["GATE-2"] = GateResult(
            name="Subperiod Stability",
            passed=positive >= self.config.gate_2_positive_subperiods,
            threshold=self.config.gate_2_positive_subperiods,
            value=positive,
            details="Positive Sharpe count across 8 subperiods",
        )

        if sector_pnl_share:
            max_sector_share = max((float(v) for v in sector_pnl_share.values()), default=0.0)
            g3_passed = max_sector_share <= self.config.gate_3_max_sector_pnl_share
            g3_details = "Max sector PnL share must be below threshold"
        else:
            max_sector_share = None
            g3_passed = False
            g3_details = "Sector contribution data missing"
        gates["GATE-3"] = GateResult(
            name="Sector Concentration",
            passed=g3_passed,
            threshold=self.config.gate_3_max_sector_pnl_share,
            value=max_sector_share,
            details=g3_details,
        )

        mdd = max_drawdown(net)
        gates["GATE-4"] = GateResult(
            name="Max Drawdown",
            passed=mdd < self.config.gate_4_max_drawdown,
            threshold=self.config.gate_4_max_drawdown,
            value=mdd,
            details="Peak-to-trough drawdown on stitched OOS returns",
        )

        turnover = float(pd.to_numeric(returns_df.get("turnover"), errors="coerce").dropna().mean()) if "turnover" in returns_df.columns else 0.0
        gates["GATE-5"] = GateResult(
            name="Turnover Check",
            passed=turnover < self.config.gate_5_max_turnover,
            threshold=self.config.gate_5_max_turnover,
            value=turnover,
            details="Average daily turnover must be below threshold",
        )

        if factor_overlap_score is None:
            gates["GATE-6"] = GateResult(
                name="Factor Importance Stability",
                passed=False,
                threshold=self.config.gate_6_min_importance_overlap,
                value=None,
                details="Factor overlap score missing",
            )
        else:
            gates["GATE-6"] = GateResult(
                name="Factor Importance Stability",
                passed=float(factor_overlap_score) >= self.config.gate_6_min_importance_overlap,
                threshold=self.config.gate_6_min_importance_overlap,
                value=float(factor_overlap_score),
                details="Top-factor overlap across windows",
            )

        spy_sharpe = self._baseline_sharpe(returns_df, "baseline_spy")
        eq_sharpe = self._baseline_sharpe(returns_df, "baseline_equal_weight")
        mom_sharpe = self._baseline_sharpe(returns_df, "baseline_momentum")
        gates["GATE-7"] = GateResult(
            name="Baseline Comparison",
            passed=net_sharpe > spy_sharpe and net_sharpe > eq_sharpe and net_sharpe > mom_sharpe,
            threshold=0,
            value=net_sharpe - max(spy_sharpe, eq_sharpe, mom_sharpe),
            details="Strategy Sharpe must exceed all baseline Sharpes",
        )

        gates["GATE-8"] = GateResult(
            name="Net Sharpe",
            passed=net_sharpe > self.config.gate_8_min_net_sharpe,
            threshold=self.config.gate_8_min_net_sharpe,
            value=net_sharpe,
            details="Net-of-cost annualized Sharpe",
        )

        passed = all(g.passed for g in gates.values())
        return GateReport(passed=passed, gates=gates)
