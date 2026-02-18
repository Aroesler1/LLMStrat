"""
Portfolio state helpers: convert between shares, value and weight targets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from quantaalpha.trading.broker import PositionInfo


@dataclass
class PortfolioSnapshot:
    equity: float
    cash: float
    positions_value: float
    positions: Dict[str, float]


class PortfolioStateManager:
    @staticmethod
    def shares_from_positions(positions: List[PositionInfo]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for p in positions:
            qty = float(getattr(p, "qty", 0.0) or 0.0)
            if abs(qty) <= 1e-12:
                continue
            out[p.symbol] = qty
        return out

    @staticmethod
    def weights_from_positions(
        positions: List[PositionInfo],
        prices: Dict[str, float],
        equity: float,
    ) -> Dict[str, float]:
        if equity <= 0:
            return {}
        w: Dict[str, float] = {}
        for p in positions:
            sym = p.symbol
            qty = float(p.qty)
            px = float(prices.get(sym, p.current_price or 0.0) or 0.0)
            if px <= 0:
                continue
            w[sym] = (qty * px) / equity
        return w

    @staticmethod
    def target_shares_from_weights(
        target_weights: Dict[str, float],
        prices: Dict[str, float],
        equity: float,
        cash_buffer_pct: float = 0.02,
        fractional_shares: bool = True,
    ) -> Dict[str, float]:
        if equity <= 0:
            return {}

        investable = max(0.0, float(equity) * (1.0 - max(0.0, cash_buffer_pct)))
        out: Dict[str, float] = {}
        for sym, w in target_weights.items():
            px = float(prices.get(sym, 0.0) or 0.0)
            if px <= 0:
                continue
            target_dollar = investable * max(0.0, float(w))
            qty = target_dollar / px
            if not fractional_shares:
                qty = int(qty)
            if qty > 0:
                out[sym] = float(qty)
        return out
