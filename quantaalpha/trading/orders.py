"""
Order generation and execution helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import time
from typing import Dict, List, Optional, Set

from loguru import logger

from quantaalpha.storage.database import DatabaseManager
from quantaalpha.trading.broker import BrokerAPI


@dataclass
class ProposedOrder:
    symbol: str
    side: str
    qty: float
    order_type: str = "market"
    limit_price: Optional[float] = None


class OrderState(str, Enum):
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


VALID_TRANSITIONS: Dict[OrderState, Set[OrderState]] = {
    OrderState.PENDING: {
        OrderState.PENDING,
        OrderState.SUBMITTED,
        OrderState.CANCELLED,
        OrderState.REJECTED,
        OrderState.EXPIRED,
    },
    OrderState.SUBMITTED: {
        OrderState.SUBMITTED,
        OrderState.PARTIAL,
        OrderState.FILLED,
        OrderState.CANCELLED,
        OrderState.REJECTED,
        OrderState.EXPIRED,
    },
    OrderState.PARTIAL: {
        OrderState.PARTIAL,
        OrderState.FILLED,
        OrderState.CANCELLED,
        OrderState.REJECTED,
        OrderState.EXPIRED,
    },
    OrderState.FILLED: {OrderState.FILLED},
    OrderState.CANCELLED: {OrderState.CANCELLED},
    OrderState.REJECTED: {OrderState.REJECTED},
    OrderState.EXPIRED: {OrderState.EXPIRED},
}


ORDER_STATE_ALIASES: Dict[str, OrderState] = {
    "new": OrderState.SUBMITTED,
    "accepted": OrderState.SUBMITTED,
    "pending_new": OrderState.PENDING,
    "partially_filled": OrderState.PARTIAL,
    "partiallyfilled": OrderState.PARTIAL,
    "canceled": OrderState.CANCELLED,
}


class OrderManager:
    def __init__(
        self,
        broker: BrokerAPI,
        db: Optional[DatabaseManager] = None,
        rate_limit_per_second: float = 5.0,
    ):
        self.broker = broker
        self.db = db
        self.rate_limit_per_second = float(rate_limit_per_second)
        self._order_states: Dict[str, OrderState] = {}

    @staticmethod
    def generate_orders(
        current_shares: Dict[str, float],
        target_shares: Dict[str, float],
        prices: Dict[str, float],
        min_trade_dollars: float = 50.0,
    ) -> List[ProposedOrder]:
        symbols = sorted(set(current_shares) | set(target_shares))
        sells: List[ProposedOrder] = []
        buys: List[ProposedOrder] = []
        for sym in symbols:
            cur = float(current_shares.get(sym, 0.0))
            tar = float(target_shares.get(sym, 0.0))
            delta = tar - cur
            px = float(prices.get(sym, 0.0))
            if px <= 0:
                continue
            if abs(delta * px) < min_trade_dollars:
                continue
            if delta < 0:
                sells.append(ProposedOrder(symbol=sym, side="sell", qty=abs(delta)))
            elif delta > 0:
                buys.append(ProposedOrder(symbol=sym, side="buy", qty=abs(delta)))
        # Sell-before-buy for cash safety.
        return sells + buys

    def submit_orders(self, orders: List[ProposedOrder], rebalance_id: str) -> List[str]:
        ids: List[str] = []
        min_interval = 1.0 / self.rate_limit_per_second if self.rate_limit_per_second > 0 else 0.0
        last_submit_at: Optional[float] = None

        for o in orders:
            if min_interval > 0 and last_submit_at is not None:
                elapsed = time.monotonic() - last_submit_at
                if elapsed < min_interval:
                    time.sleep(min_interval - elapsed)

            oid = self.broker.submit_order(
                symbol=o.symbol,
                qty=o.qty,
                side=o.side,
                order_type=o.order_type,
                limit_price=o.limit_price,
            )
            last_submit_at = time.monotonic()
            ids.append(oid)
            self._order_states[oid] = OrderState.SUBMITTED
            if self.db is not None:
                self.db.save_order(
                    {
                        "broker_order_id": oid,
                        "symbol": o.symbol,
                        "side": o.side,
                        "qty": o.qty,
                        "order_type": o.order_type,
                        "limit_price": o.limit_price,
                        "status": OrderState.SUBMITTED.value,
                        "filled_qty": 0.0,
                        "submitted_at": datetime.utcnow(),
                        "rebalance_id": rebalance_id,
                    }
                )
        return ids

    def poll_terminal(self, order_ids: List[str], timeout_seconds: int = 300, poll_interval: float = 2.0):
        start = time.time()
        remaining = set(order_ids)
        terminal = {
            OrderState.FILLED,
            OrderState.CANCELLED,
            OrderState.REJECTED,
            OrderState.EXPIRED,
        }
        while remaining and (time.time() - start) < timeout_seconds:
            done = []
            for oid in list(remaining):
                st = self.broker.get_order_status(oid)
                next_state = self._normalize_state(st.status)
                if next_state is None:
                    logger.warning(
                        "Skipping unknown order state from broker: order_id={} status={}",
                        oid,
                        st.status,
                    )
                    continue

                if self._update_order_state(
                    oid=oid,
                    next_state=next_state,
                    filled_qty=st.filled_qty,
                    filled_avg_price=st.filled_avg_price,
                    filled_at=st.filled_at,
                ):
                    self._order_states[oid] = next_state

                if self._order_states.get(oid, next_state) in terminal:
                    done.append(oid)
            for oid in done:
                remaining.discard(oid)
            if remaining:
                time.sleep(poll_interval)

        for oid in list(remaining):
            try:
                self.broker.cancel_order(oid)
            except Exception as e:
                logger.exception("Failed to cancel stale order {}: {}", oid, e)
            self._update_order_state(oid=oid, next_state=OrderState.CANCELLED)

    @staticmethod
    def _normalize_state(state: str) -> Optional[OrderState]:
        raw = str(state or "").strip().lower()
        mapped = ORDER_STATE_ALIASES.get(raw, raw)
        try:
            return mapped if isinstance(mapped, OrderState) else OrderState(mapped)
        except ValueError:
            return None

    def _update_order_state(
        self,
        oid: str,
        next_state: OrderState,
        filled_qty: Optional[float] = None,
        filled_avg_price: Optional[float] = None,
        filled_at: Optional[datetime] = None,
    ) -> bool:
        current_state = self._order_states.get(oid, OrderState.SUBMITTED)
        valid_next_states = VALID_TRANSITIONS.get(current_state, {current_state})
        if next_state not in valid_next_states:
            logger.warning(
                "Invalid order state transition ignored: order_id={} {} -> {}",
                oid,
                current_state.value,
                next_state.value,
            )
            return False

        self._order_states[oid] = next_state
        if self.db is not None:
            self.db.update_order_by_broker_id(
                oid,
                status=next_state.value,
                filled_qty=filled_qty,
                filled_avg_price=filled_avg_price,
                filled_at=filled_at,
            )
        return True
