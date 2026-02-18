"""
Runtime health monitoring for trading engine.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from quantaalpha.storage.database import DatabaseManager
from quantaalpha.trading.alerts import AlertManager
from quantaalpha.trading.broker import BrokerAPI


class HealthMonitor:
    def __init__(
        self,
        db: DatabaseManager,
        alerts: Optional[AlertManager] = None,
        stale_seconds: int = 600,
        broker: Optional[BrokerAPI] = None,
    ):
        self.db = db
        self.alerts = alerts
        self.stale_seconds = int(stale_seconds)
        self.broker = broker

    def heartbeat(self, component: str, payload: Optional[Dict[str, Any]] = None):
        state = {
            "timestamp": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
            "payload": payload or {},
        }
        self.db.upsert_runtime_state(f"heartbeat.{component}", state)

    def check_health(self, emit_alerts: bool = False) -> Dict[str, Any]:
        now = datetime.utcnow().replace(tzinfo=timezone.utc)
        keys = [
            "heartbeat.engine",
            "heartbeat.scheduler",
        ]

        heartbeat = {}
        stale_components = []
        for key in keys:
            v = self.db.get_runtime_state(key)
            heartbeat[key] = v
            ts = None
            if isinstance(v, dict):
                ts = v.get("timestamp")
            if not ts:
                stale_components.append(key)
                continue
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                age = (now - dt).total_seconds()
                if age > self.stale_seconds:
                    stale_components.append(key)
            except Exception:
                stale_components.append(key)

        healthy = len(stale_components) == 0
        broker_connectivity: Dict[str, Any] = {"ok": True}
        if self.broker is not None:
            try:
                account = self.broker.get_account()
                broker_connectivity = {
                    "ok": True,
                    "equity": float(account.equity),
                    "cash": float(account.cash),
                }
            except Exception as e:
                broker_connectivity = {
                    "ok": False,
                    "error": str(e),
                }
                healthy = False
                stale_components.append("broker.connectivity")

        status = {
            "healthy": healthy,
            "stale_components": stale_components,
            "heartbeat": heartbeat,
            "broker_connectivity": broker_connectivity,
            "latest_snapshot": self.db.get_latest_snapshot(),
        }

        if emit_alerts and (not healthy) and self.alerts is not None:
            self.alerts.warning(
                message="Trading health check found stale components",
                data={"stale": stale_components},
                event_type="health",
            )
        return status
