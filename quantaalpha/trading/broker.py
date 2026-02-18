"""
Broker abstraction and Alpaca implementation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from loguru import logger
import pandas as pd


@dataclass
class AccountInfo:
    equity: float
    cash: float
    buying_power: float
    daytrade_count: int = 0


@dataclass
class PositionInfo:
    symbol: str
    qty: float
    avg_entry_price: float
    market_value: float
    current_price: float
    unrealized_pl: float


@dataclass
class OrderStatus:
    broker_order_id: str
    symbol: str
    side: str
    qty: float
    filled_qty: float
    filled_avg_price: float
    status: str
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None


class BrokerAPI(ABC):
    @abstractmethod
    def get_account(self) -> AccountInfo:
        raise NotImplementedError

    @abstractmethod
    def get_positions(self) -> List[PositionInfo]:
        raise NotImplementedError

    @abstractmethod
    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "market",
        limit_price: Optional[float] = None,
    ) -> str:
        raise NotImplementedError

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_order_status(self, order_id: str) -> OrderStatus:
        raise NotImplementedError

    @abstractmethod
    def get_bars(self, symbols: List[str], timeframe: str, limit: int) -> Dict[str, pd.DataFrame]:
        raise NotImplementedError

    @abstractmethod
    def cancel_all_orders(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def liquidate_all(self) -> None:
        raise NotImplementedError


class AlpacaBroker(BrokerAPI):
    """
    Alpaca broker adapter.

    Requires `alpaca-py`.
    """

    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.data.historical.stock import StockHistoricalDataClient
        except Exception as e:
            raise ImportError("alpaca-py is required for AlpacaBroker") from e

        self.paper = paper
        self.trading = TradingClient(api_key=api_key, secret_key=secret_key, paper=paper)
        self.data = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)

    def get_account(self) -> AccountInfo:
        a = self.trading.get_account()
        daytrade_count = int(getattr(a, "daytrade_count", 0) or 0)
        return AccountInfo(
            equity=float(a.equity),
            cash=float(a.cash),
            buying_power=float(a.buying_power),
            daytrade_count=daytrade_count,
        )

    def get_positions(self) -> List[PositionInfo]:
        out = []
        for p in self.trading.get_all_positions():
            out.append(
                PositionInfo(
                    symbol=p.symbol,
                    qty=float(p.qty),
                    avg_entry_price=float(p.avg_entry_price),
                    market_value=float(p.market_value),
                    current_price=float(p.current_price),
                    unrealized_pl=float(p.unrealized_pl),
                )
            )
        return out

    def submit_order(
        self,
        symbol: str,
        qty: float,
        side: str,
        order_type: str = "market",
        limit_price: Optional[float] = None,
    ) -> str:
        from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        side_e = OrderSide.BUY if str(side).lower() == "buy" else OrderSide.SELL
        if order_type == "limit":
            req = LimitOrderRequest(
                symbol=symbol,
                qty=abs(float(qty)),
                side=side_e,
                time_in_force=TimeInForce.DAY,
                limit_price=float(limit_price) if limit_price is not None else None,
            )
        else:
            req = MarketOrderRequest(
                symbol=symbol,
                qty=abs(float(qty)),
                side=side_e,
                time_in_force=TimeInForce.DAY,
            )
        order = self.trading.submit_order(order_data=req)
        return str(order.id)

    def cancel_order(self, order_id: str) -> bool:
        try:
            self.trading.cancel_order_by_id(order_id)
            return True
        except Exception:
            return False

    def get_order_status(self, order_id: str) -> OrderStatus:
        o = self.trading.get_order_by_id(order_id)
        return OrderStatus(
            broker_order_id=str(o.id),
            symbol=o.symbol,
            side=str(o.side).lower(),
            qty=float(o.qty or 0),
            filled_qty=float(o.filled_qty or 0),
            filled_avg_price=float(o.filled_avg_price or 0) if o.filled_avg_price is not None else 0.0,
            status=str(o.status).lower(),
            submitted_at=o.submitted_at,
            filled_at=o.filled_at,
        )

    def get_bars(self, symbols: List[str], timeframe: str = "1Min", limit: int = 200) -> Dict[str, pd.DataFrame]:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        if timeframe.lower() in ("1min", "1m"):
            tf = TimeFrame(amount=1, unit=TimeFrameUnit.Minute)
        elif timeframe.lower() in ("1day", "1d", "day"):
            tf = TimeFrame(amount=1, unit=TimeFrameUnit.Day)
        else:
            tf = TimeFrame(amount=1, unit=TimeFrameUnit.Minute)

        req = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=tf,
            limit=limit,
        )
        bars = self.data.get_stock_bars(req).df
        result: Dict[str, pd.DataFrame] = {}
        if bars.empty:
            return result
        # alpaca bars index: multi-index(symbol, timestamp)
        if isinstance(bars.index, pd.MultiIndex):
            for sym in symbols:
                try:
                    df = bars.xs(sym, level=0).sort_index()
                    result[sym] = df
                except Exception:
                    continue
        return result

    def cancel_all_orders(self) -> None:
        try:
            self.trading.cancel_orders()
        except Exception as e:
            logger.exception("Failed to cancel all Alpaca orders: {}", e)

    def liquidate_all(self) -> None:
        positions = self.get_positions()
        for p in positions:
            side = "sell" if p.qty > 0 else "buy"
            qty = abs(float(p.qty))
            if qty <= 0:
                continue
            try:
                self.submit_order(symbol=p.symbol, qty=qty, side=side, order_type="market")
            except Exception as e:
                logger.exception(
                    "Failed to liquidate position: symbol={} qty={} error={}",
                    p.symbol,
                    qty,
                    e,
                )
                continue


class MockBroker(BrokerAPI):
    """
    In-memory broker for tests and dry runs.
    """

    def __init__(self, equity: float = 100000.0):
        self._equity = float(equity)
        self._cash = float(equity)
        self._positions: Dict[str, PositionInfo] = {}
        self._orders: Dict[str, OrderStatus] = {}
        self._order_counter = 0
        self._last_prices: Dict[str, float] = {}

    def set_price(self, symbol: str, price: float):
        self._last_prices[symbol] = float(price)

    def get_account(self) -> AccountInfo:
        mv = sum(v.market_value for v in self._positions.values())
        return AccountInfo(equity=self._cash + mv, cash=self._cash, buying_power=self._cash, daytrade_count=0)

    def get_positions(self) -> List[PositionInfo]:
        return list(self._positions.values())

    def submit_order(self, symbol: str, qty: float, side: str, order_type: str = "market", limit_price: Optional[float] = None) -> str:
        self._order_counter += 1
        oid = f"mock-{self._order_counter}"
        px = float(limit_price) if (order_type == "limit" and limit_price is not None) else self._last_prices.get(symbol, 100.0)
        qty = abs(float(qty))
        notion = qty * px
        if side.lower() == "buy":
            self._cash -= notion
            pos = self._positions.get(symbol)
            if pos is None:
                pos = PositionInfo(symbol=symbol, qty=0.0, avg_entry_price=px, market_value=0.0, current_price=px, unrealized_pl=0.0)
            new_qty = pos.qty + qty
            pos.avg_entry_price = ((pos.avg_entry_price * pos.qty) + notion) / new_qty if new_qty > 0 else px
            pos.qty = new_qty
            pos.current_price = px
            pos.market_value = pos.qty * px
            pos.unrealized_pl = (px - pos.avg_entry_price) * pos.qty
            self._positions[symbol] = pos
        else:
            pos = self._positions.get(symbol)
            if pos is not None:
                sell_qty = min(qty, abs(pos.qty))
                self._cash += sell_qty * px
                pos.qty -= sell_qty
                pos.current_price = px
                pos.market_value = pos.qty * px
                pos.unrealized_pl = (px - pos.avg_entry_price) * pos.qty
                if abs(pos.qty) <= 1e-9:
                    self._positions.pop(symbol, None)
                else:
                    self._positions[symbol] = pos
        status = OrderStatus(
            broker_order_id=oid,
            symbol=symbol,
            side=side.lower(),
            qty=qty,
            filled_qty=qty,
            filled_avg_price=px,
            status="filled",
            submitted_at=datetime.utcnow(),
            filled_at=datetime.utcnow(),
        )
        self._orders[oid] = status
        return oid

    def cancel_order(self, order_id: str) -> bool:
        st = self._orders.get(order_id)
        if st is None:
            return False
        st.status = "cancelled"
        return True

    def get_order_status(self, order_id: str) -> OrderStatus:
        return self._orders[order_id]

    def get_bars(self, symbols: List[str], timeframe: str, limit: int) -> Dict[str, pd.DataFrame]:
        out = {}
        now = pd.Timestamp.utcnow().floor("min")
        idx = pd.date_range(end=now, periods=limit, freq="min")
        for s in symbols:
            px = self._last_prices.get(s, 100.0)
            df = pd.DataFrame(
                {
                    "open": px,
                    "high": px,
                    "low": px,
                    "close": px,
                    "volume": 1000.0,
                },
                index=idx,
            )
            out[s] = df
        return out

    def cancel_all_orders(self) -> None:
        for st in self._orders.values():
            if st.status not in ("filled", "cancelled", "rejected"):
                st.status = "cancelled"

    def liquidate_all(self) -> None:
        for sym, p in list(self._positions.items()):
            if p.qty > 0:
                self.submit_order(sym, p.qty, "sell")
