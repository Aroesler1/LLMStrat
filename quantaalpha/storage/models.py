"""
SQLAlchemy models for trading state.
"""

from __future__ import annotations

from datetime import datetime, date

from sqlalchemy import Column, Date, DateTime, Float, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import declarative_base


Base = declarative_base()


class Order(Base):
    __tablename__ = "orders"
    id = Column(Integer, primary_key=True, autoincrement=True)
    broker_order_id = Column(String(128), unique=True, nullable=True)
    symbol = Column(String(32), nullable=False, index=True)
    side = Column(String(8), nullable=False)
    qty = Column(Float, nullable=False)
    order_type = Column(String(16), nullable=False, default="market")
    limit_price = Column(Float, nullable=True)
    status = Column(String(32), nullable=False, default="pending")
    filled_qty = Column(Float, nullable=False, default=0.0)
    filled_avg_price = Column(Float, nullable=True)
    submitted_at = Column(DateTime, nullable=True)
    filled_at = Column(DateTime, nullable=True)
    rebalance_id = Column(String(64), nullable=True, index=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class Position(Base):
    __tablename__ = "positions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(32), nullable=False, unique=True, index=True)
    qty = Column(Float, nullable=False, default=0.0)
    avg_cost = Column(Float, nullable=False, default=0.0)
    current_price = Column(Float, nullable=False, default=0.0)
    market_value = Column(Float, nullable=False, default=0.0)
    unrealized_pnl = Column(Float, nullable=False, default=0.0)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class DailySnapshot(Base):
    __tablename__ = "daily_snapshots"
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, unique=True, index=True)
    portfolio_value = Column(Float, nullable=False, default=0.0)
    cash = Column(Float, nullable=False, default=0.0)
    positions_value = Column(Float, nullable=False, default=0.0)
    daily_return = Column(Float, nullable=False, default=0.0)
    cumulative_return = Column(Float, nullable=False, default=0.0)
    peak_value = Column(Float, nullable=False, default=0.0)
    drawdown = Column(Float, nullable=False, default=0.0)
    num_positions = Column(Integer, nullable=False, default=0)
    turnover = Column(Float, nullable=False, default=0.0)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class EventLog(Base):
    __tablename__ = "event_log"
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    event_type = Column(String(64), nullable=False, index=True)
    severity = Column(String(16), nullable=False, default="info")
    message = Column(Text, nullable=False)
    data_json = Column(Text, nullable=True)


class RuntimeState(Base):
    __tablename__ = "runtime_state"
    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String(128), nullable=False)
    value = Column(Text, nullable=True)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    __table_args__ = (UniqueConstraint("key", name="uq_runtime_state_key"),)
