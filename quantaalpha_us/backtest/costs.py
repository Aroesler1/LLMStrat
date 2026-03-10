from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class TransactionCostModel:
    """Simple daily-equity transaction cost model for S&P 500 execution."""

    commission_per_share: float = 0.0
    half_spread_bps: float = 1.5
    slippage_bps: float = 1.0
    impact_coefficient: float = 0.1

    def estimate_cost_fraction(
        self,
        *,
        trade_notional: float,
        adv_20d: float | None = None,
        daily_vol: float | None = None,
    ) -> float:
        if trade_notional <= 0:
            return 0.0

        spread_cost = self.half_spread_bps / 10000.0
        slippage_cost = self.slippage_bps / 10000.0

        impact_cost = 0.0
        if adv_20d and adv_20d > 0 and daily_vol and daily_vol > 0:
            participation = max(trade_notional / adv_20d, 0.0)
            if participation > 0:
                impact_cost = self.impact_coefficient * daily_vol * math.sqrt(participation)

        return spread_cost + slippage_cost + impact_cost

    def estimate_trade_cost(
        self,
        *,
        trade_notional: float,
        adv_20d: float | None = None,
        daily_vol: float | None = None,
    ) -> float:
        return trade_notional * self.estimate_cost_fraction(
            trade_notional=trade_notional,
            adv_20d=adv_20d,
            daily_vol=daily_vol,
        )
