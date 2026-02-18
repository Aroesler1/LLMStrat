"""
Risk limits and checks for retail trading runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RiskLimits:
    max_position_pct: float = 0.10
    max_drawdown_pct: float = 0.15
    max_daily_loss_pct: float = 0.05
    max_turnover_daily: float = 0.40
    max_positions: int = 50
    min_positions: int = 5
    max_leverage: float = 1.0
    min_cash_pct: float = 0.02
    max_portfolio_value: float = 25000.0
    max_single_order_value: float = 2500.0
    flatten_on_kill: bool = True

    @classmethod
    def from_dict(cls, cfg: Optional[Dict[str, Any]]) -> "RiskLimits":
        cfg = cfg or {}
        out = cls()
        for k in out.__dict__.keys():
            if k in cfg:
                setattr(out, k, cfg[k])
        return out


@dataclass
class RiskCheckResult:
    approved: bool
    violations: List[str] = field(default_factory=list)
    adjustments: Dict[str, float] = field(default_factory=dict)
    severity: str = "info"


class RiskEngine:
    def __init__(self, limits: RiskLimits):
        self.limits = limits

    def check_portfolio(
        self,
        target_weights: Dict[str, float],
        current_weights: Dict[str, float],
        portfolio_value: float,
        daily_pnl: float,
        peak_value: float,
        cash: float = 0.0,
    ) -> RiskCheckResult:
        violations: List[str] = []

        n_pos = sum(1 for w in target_weights.values() if w > 1e-8)
        if n_pos > self.limits.max_positions:
            violations.append(f"Too many positions: {n_pos} > {self.limits.max_positions}")
        if n_pos > 0 and n_pos < self.limits.min_positions:
            violations.append(f"Too few positions: {n_pos} < {self.limits.min_positions}")

        for sym, w in target_weights.items():
            if w > self.limits.max_position_pct:
                violations.append(f"Position cap exceeded: {sym}={w:.2%} > {self.limits.max_position_pct:.2%}")

        gross_exposure = sum(abs(float(w)) for w in target_weights.values())
        if gross_exposure > self.limits.max_leverage:
            violations.append(f"Leverage exceeded: {gross_exposure:.2f} > {self.limits.max_leverage:.2f}")

        turnover = sum(abs(target_weights.get(sym, 0.0) - current_weights.get(sym, 0.0)) for sym in set(target_weights) | set(current_weights))
        if turnover > self.limits.max_turnover_daily:
            violations.append(f"Turnover exceeded: {turnover:.2%} > {self.limits.max_turnover_daily:.2%}")

        if portfolio_value > self.limits.max_portfolio_value:
            violations.append(f"Portfolio value cap exceeded: {portfolio_value:.2f} > {self.limits.max_portfolio_value:.2f}")

        drawdown = (peak_value - portfolio_value) / peak_value if peak_value > 0 else 0.0
        if drawdown > self.limits.max_drawdown_pct:
            violations.append(f"Drawdown exceeded: {drawdown:.2%} > {self.limits.max_drawdown_pct:.2%}")

        if portfolio_value > 0:
            daily_loss = -daily_pnl / portfolio_value
            if daily_loss > self.limits.max_daily_loss_pct:
                violations.append(f"Daily loss exceeded: {daily_loss:.2%} > {self.limits.max_daily_loss_pct:.2%}")

        if portfolio_value > 0:
            cash_pct = cash / portfolio_value
            if cash_pct < self.limits.min_cash_pct:
                violations.append(f"Cash buffer below minimum: {cash_pct:.2%} < {self.limits.min_cash_pct:.2%}")

        approved = len(violations) == 0
        severity = "info"
        if not approved:
            critical_markers = (
                "Drawdown exceeded",
                "Daily loss exceeded",
            )
            severity = (
                "critical"
                if any(v.startswith(marker) for marker in critical_markers for v in violations)
                else "warning"
            )
        return RiskCheckResult(approved=approved, violations=violations, severity=severity)

    def check_kill_switch(
        self,
        current_value: float,
        peak_value: float,
        daily_pnl: float,
        portfolio_value_start_of_day: float,
    ) -> bool:
        if peak_value > 0:
            drawdown = (peak_value - current_value) / peak_value
            if drawdown > self.limits.max_drawdown_pct:
                return True
        if portfolio_value_start_of_day > 0:
            daily_return = daily_pnl / portfolio_value_start_of_day
            if daily_return < -self.limits.max_daily_loss_pct:
                return True
        return False

    def adjust_weights(self, weights: Dict[str, float], result: RiskCheckResult) -> Dict[str, float]:
        if result.approved:
            return weights
        adjusted = {}
        for sym, w in weights.items():
            adjusted[sym] = min(max(w, 0.0), self.limits.max_position_pct)
        total = sum(adjusted.values())
        if total > 1e-12:
            adjusted = {k: v / total for k, v in adjusted.items()}
        return adjusted
