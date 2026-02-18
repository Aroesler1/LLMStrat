"""
Database manager for trading runtime.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from quantaalpha.storage.models import Base, DailySnapshot, EventLog, Order, Position, RuntimeState


class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = str(Path(db_path).expanduser())
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{self.db_path}", future=True)
        self._Session = sessionmaker(bind=self.engine, autoflush=False, autocommit=False, future=True)
        Base.metadata.create_all(self.engine)

    @contextmanager
    def session_scope(self) -> Iterator[Session]:
        s = self._Session()
        try:
            yield s
            s.commit()
        except Exception:
            s.rollback()
            raise
        finally:
            s.close()

    def upsert_runtime_state(self, key: str, value: Any):
        with self.session_scope() as s:
            row = s.execute(select(RuntimeState).where(RuntimeState.key == key)).scalar_one_or_none()
            if row is None:
                row = RuntimeState(key=key, value=json.dumps(value, ensure_ascii=False))
                s.add(row)
            else:
                row.value = json.dumps(value, ensure_ascii=False)
                row.updated_at = datetime.utcnow()

    def get_runtime_state(self, key: str, default: Any = None) -> Any:
        with self.session_scope() as s:
            row = s.execute(select(RuntimeState).where(RuntimeState.key == key)).scalar_one_or_none()
            if row is None:
                return default
            try:
                return json.loads(row.value) if row.value is not None else default
            except Exception:
                return row.value

    def save_order(self, data: Dict[str, Any]) -> int:
        with self.session_scope() as s:
            obj = Order(**data)
            s.add(obj)
            s.flush()
            return int(obj.id)

    def update_order_by_broker_id(self, broker_order_id: str, **kwargs):
        with self.session_scope() as s:
            row = s.execute(select(Order).where(Order.broker_order_id == broker_order_id)).scalar_one_or_none()
            if row is None:
                return
            for k, v in kwargs.items():
                if hasattr(row, k):
                    setattr(row, k, v)

    def replace_positions(self, positions: list[Dict[str, Any]]):
        with self.session_scope() as s:
            s.query(Position).delete()
            for p in positions:
                s.add(Position(**p))

    def upsert_daily_snapshot(self, data: Dict[str, Any]):
        with self.session_scope() as s:
            row = s.execute(select(DailySnapshot).where(DailySnapshot.date == data["date"])).scalar_one_or_none()
            if row is None:
                s.add(DailySnapshot(**data))
            else:
                for k, v in data.items():
                    if hasattr(row, k):
                        setattr(row, k, v)

    def log_event(self, event_type: str, severity: str, message: str, data_json: Optional[str] = None):
        with self.session_scope() as s:
            s.add(EventLog(event_type=event_type, severity=severity, message=message, data_json=data_json))

    def get_latest_snapshot(self) -> Optional[Dict[str, Any]]:
        with self.session_scope() as s:
            row = s.execute(select(DailySnapshot).order_by(DailySnapshot.date.desc())).scalar_one_or_none()
            if row is None:
                return None
            return {
                "date": row.date.isoformat(),
                "portfolio_value": row.portfolio_value,
                "cash": row.cash,
                "positions_value": row.positions_value,
                "daily_return": row.daily_return,
                "cumulative_return": row.cumulative_return,
                "peak_value": row.peak_value,
                "drawdown": row.drawdown,
                "num_positions": row.num_positions,
                "turnover": row.turnover,
            }

    def get_positions(self) -> list[Dict[str, Any]]:
        with self.session_scope() as s:
            rows = s.execute(select(Position)).scalars().all()
            return [
                {
                    "symbol": r.symbol,
                    "qty": r.qty,
                    "avg_cost": r.avg_cost,
                    "current_price": r.current_price,
                    "market_value": r.market_value,
                    "unrealized_pnl": r.unrealized_pnl,
                    "updated_at": r.updated_at.isoformat() if r.updated_at else None,
                }
                for r in rows
            ]

    def get_open_orders(self) -> list[Dict[str, Any]]:
        with self.session_scope() as s:
            rows = s.execute(select(Order).where(Order.status.in_(["pending", "submitted", "partial"]))).scalars().all()
            return [
                {
                    "id": r.id,
                    "broker_order_id": r.broker_order_id,
                    "symbol": r.symbol,
                    "side": r.side,
                    "qty": r.qty,
                    "status": r.status,
                    "filled_qty": r.filled_qty,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in rows
            ]

    def get_recent_events(self, limit: int = 100) -> list[Dict[str, Any]]:
        with self.session_scope() as s:
            rows = (
                s.execute(select(EventLog).order_by(EventLog.timestamp.desc()).limit(max(1, int(limit))))
                .scalars()
                .all()
            )
            return [
                {
                    "id": r.id,
                    "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                    "event_type": r.event_type,
                    "severity": r.severity,
                    "message": r.message,
                    "data_json": r.data_json,
                }
                for r in rows
            ]
