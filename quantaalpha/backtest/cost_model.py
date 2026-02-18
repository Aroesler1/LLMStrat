#!/usr/bin/env python3
"""
Transaction cost and slippage model.

This is intentionally lightweight and parameterized from YAML to support:
- pure backtest cost deductions
- paper/live realized-vs-expected tracking
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class CostModelConfig:
    spread_bps: float = 5.0
    slippage_bps: float = 5.0
    commission_per_share: float = 0.0
    min_trade_cost: float = 0.0
    adv_impact_coefficient: float = 0.1

    @classmethod
    def from_dict(cls, cfg: Optional[Dict[str, Any]]) -> "CostModelConfig":
        cfg = cfg or {}
        return cls(
            spread_bps=float(cfg.get("spread_bps", 5.0)),
            slippage_bps=float(cfg.get("slippage_bps", 5.0)),
            commission_per_share=float(cfg.get("commission_per_share", 0.0)),
            min_trade_cost=float(cfg.get("min_trade_cost", 0.0)),
            adv_impact_coefficient=float(cfg.get("adv_impact_coefficient", 0.1)),
        )


class CostModel:
    """Simple cost estimator with spread + slippage + impact."""

    def __init__(self, config: CostModelConfig):
        self.cfg = config

    def estimate_trade_cost_dollars(
        self,
        shares: float,
        price: float,
        adv_dollars: Optional[float] = None,
    ) -> float:
        if shares <= 0 or price <= 0:
            return 0.0

        notional = float(shares * price)
        spread = notional * (self.cfg.spread_bps / 10000.0)
        slippage = notional * (self.cfg.slippage_bps / 10000.0)
        commission = float(shares) * self.cfg.commission_per_share

        impact = 0.0
        if adv_dollars is not None and adv_dollars > 0:
            participation = max(notional / adv_dollars, 0.0)
            impact = notional * self.cfg.adv_impact_coefficient * np.sqrt(participation) / 100.0

        cost = spread + slippage + commission + impact
        if cost < self.cfg.min_trade_cost:
            cost = self.cfg.min_trade_cost
        return float(cost)

    def estimate_turnover_cost_rate(
        self,
        turnover: float,
        avg_spread_bps: Optional[float] = None,
        avg_slippage_bps: Optional[float] = None,
    ) -> float:
        """
        Estimate cost directly from portfolio turnover as return-rate penalty.
        """
        if turnover <= 0:
            return 0.0
        spread = (avg_spread_bps if avg_spread_bps is not None else self.cfg.spread_bps) / 10000.0
        slip = (avg_slippage_bps if avg_slippage_bps is not None else self.cfg.slippage_bps) / 10000.0
        return float(turnover * (spread + slip))
