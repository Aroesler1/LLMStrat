"""
Order/position reconciliation utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

from quantaalpha.storage.database import DatabaseManager
from quantaalpha.trading.broker import BrokerAPI


@dataclass
class ReconcileReport:
    open_orders_checked: int
    open_orders_updated: int
    positions_synced: int
    drift: float
    needs_rebalance: bool


class Reconciler:
    def __init__(self, broker: BrokerAPI, db: DatabaseManager):
        self.broker = broker
        self.db = db

    def reconcile_orders(self) -> Dict[str, int]:
        open_orders = self.db.get_open_orders()
        checked = 0
        updated = 0

        for o in open_orders:
            oid = o.get("broker_order_id")
            if not oid:
                continue
            checked += 1
            try:
                st = self.broker.get_order_status(oid)
            except Exception:
                continue
            self.db.update_order_by_broker_id(
                oid,
                status=st.status,
                filled_qty=st.filled_qty,
                filled_avg_price=st.filled_avg_price,
                filled_at=st.filled_at,
            )
            updated += 1

        return {
            "checked": checked,
            "updated": updated,
        }

    def sync_positions(self) -> List[Dict]:
        positions = self.broker.get_positions()
        payload = []
        ts = datetime.utcnow()
        for p in positions:
            payload.append(
                {
                    "symbol": p.symbol,
                    "qty": float(p.qty),
                    "avg_cost": float(p.avg_entry_price),
                    "current_price": float(p.current_price),
                    "market_value": float(p.market_value),
                    "unrealized_pnl": float(p.unrealized_pl),
                    "updated_at": ts,
                }
            )
        self.db.replace_positions(payload)
        return payload

    @staticmethod
    def calculate_drift(
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
    ) -> float:
        all_syms = set(current_weights) | set(target_weights)
        drift = sum(abs(float(current_weights.get(s, 0.0)) - float(target_weights.get(s, 0.0))) for s in all_syms)
        return float(drift)
