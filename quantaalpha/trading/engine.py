"""
Trading runtime engine for paper/live retail deployment.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from quantaalpha.storage.database import DatabaseManager
from quantaalpha.storage.event_log import EventLogger
from quantaalpha.trading.alerts import AlertManager
from quantaalpha.trading.broker import AlpacaBroker, BrokerAPI, MockBroker
from quantaalpha.trading.market_data import MarketDataFetcher
from quantaalpha.trading.monitor import HealthMonitor
from quantaalpha.trading.orders import OrderManager
from quantaalpha.trading.portfolio_state import PortfolioStateManager
from quantaalpha.trading.reconciler import Reconciler
from quantaalpha.trading.risk import RiskEngine, RiskLimits
from quantaalpha.trading.scheduler import TradingScheduler
from quantaalpha.trading.session import MarketSession, SessionConfig
from quantaalpha.trading.signals import SignalConfig, SignalGenerator


@dataclass
class TradingEngineResult:
    status: str
    message: str
    payload: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "message": self.message,
            "payload": self.payload,
        }


class TradingConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

    broker: Dict[str, Any] = Field(default_factory=dict)
    storage: Dict[str, Any] = Field(default_factory=dict)
    session: Dict[str, Any] = Field(default_factory=dict)
    execution: Dict[str, Any] = Field(default_factory=dict)
    scheduler: Dict[str, Any] = Field(default_factory=dict)
    risk: Dict[str, Any] = Field(default_factory=dict)
    signals: Dict[str, Any] = Field(default_factory=dict)
    monitor: Dict[str, Any] = Field(default_factory=dict)


class TradingEngine:
    def __init__(self, config: Dict[str, Any], config_path: Optional[str] = None):
        self.config = config
        self.config_path = config_path

        storage_cfg = config.get("storage", {}) or {}
        db_path = str(storage_cfg.get("db_path", "data/results/trading/trading_state.sqlite"))
        self.db = DatabaseManager(db_path)
        self.events = EventLogger(self.db)
        self.alerts = AlertManager(event_logger=self.events, verbose=True)

        self.session = MarketSession(SessionConfig.from_dict(config.get("session")))

        self.broker = self._build_broker(config.get("broker") or {})
        self.market_data = MarketDataFetcher(self.broker)

        self.risk = RiskEngine(RiskLimits.from_dict(config.get("risk")))
        self.portfolio = PortfolioStateManager()
        self.signals = SignalGenerator(SignalConfig.from_dict(config.get("signals")), self.market_data)
        self.execution_cfg = config.get("execution", {}) or {}
        self.orders = OrderManager(
            self.broker,
            self.db,
            rate_limit_per_second=float(self.execution_cfg.get("rate_limit_per_second", 5.0)),
        )
        self.reconciler = Reconciler(self.broker, self.db)

        monitor_cfg = config.get("monitor", {}) or {}
        self.monitor = HealthMonitor(
            db=self.db,
            alerts=self.alerts,
            stale_seconds=int(monitor_cfg.get("stale_seconds", 600)),
            broker=self.broker,
        )

        self.scheduler_cfg = config.get("scheduler", {}) or {}
        self.scheduler = TradingScheduler(
            rebalance_callback=self.scheduled_rebalance,
            heartbeat_callback=self.scheduled_heartbeat,
            config=self.scheduler_cfg,
        )
        self._save_startup_metadata()

    @classmethod
    def from_yaml(
        cls,
        config_path: str,
        paper: Optional[bool] = None,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> "TradingEngine":
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Trading config not found: {config_path}")
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        if paper is not None:
            cfg.setdefault("broker", {})["paper"] = bool(paper)
            if bool(paper):
                cfg.setdefault("execution", {})["mode"] = "paper"

        if overrides:
            cls._deep_update(cfg, overrides)

        try:
            validated = TradingConfig.model_validate(cfg).model_dump()
        except ValidationError as e:
            raise ValueError(f"Invalid trading config: {e}") from e

        return cls(config=validated, config_path=str(path))

    @staticmethod
    def _deep_update(base: Dict[str, Any], patch: Dict[str, Any]):
        for k, v in patch.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                TradingEngine._deep_update(base[k], v)
            else:
                base[k] = v

    def _save_startup_metadata(self) -> None:
        config_blob = json.dumps(self.config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        config_hash = hashlib.sha256(config_blob.encode("utf-8")).hexdigest()
        git_hash: Optional[str] = None
        try:
            repo_root = Path(__file__).resolve().parents[2]
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=str(repo_root),
                text=True,
            ).strip()
        except Exception as e:
            self.alerts.warning(
                message="Unable to resolve git hash at startup",
                data={"error": str(e)},
                event_type="runtime",
            )

        self.db.upsert_runtime_state("config_hash", config_hash)
        self.db.upsert_runtime_state("git_hash", git_hash)
        self.db.upsert_runtime_state(
            "startup_metadata",
            {
                "config_hash": config_hash,
                "git_hash": git_hash,
                "config_path": self.config_path,
                "started_at": datetime.utcnow().isoformat(),
            },
        )

    def _build_broker(self, broker_cfg: Dict[str, Any]) -> BrokerAPI:
        provider = str(broker_cfg.get("provider", "mock")).lower()
        paper = bool(broker_cfg.get("paper", True))

        if provider == "alpaca":
            api_key = broker_cfg.get("api_key") or os.environ.get(str(broker_cfg.get("api_key_env", "ALPACA_API_KEY")))
            secret_key = broker_cfg.get("secret_key") or os.environ.get(str(broker_cfg.get("secret_key_env", "ALPACA_SECRET_KEY")))
            if not api_key or not secret_key:
                raise ValueError(
                    "Alpaca broker selected but API keys are missing. "
                    "Set ALPACA_API_KEY / ALPACA_SECRET_KEY or configure broker.api_key/secret_key."
                )
            timeout_seconds = float(broker_cfg.get("timeout_seconds", 30.0))
            return AlpacaBroker(
                api_key=api_key,
                secret_key=secret_key,
                paper=paper,
                timeout_seconds=timeout_seconds,
            )

        mock_equity = float(broker_cfg.get("mock_equity", 100000.0))
        slippage_bps = float(broker_cfg.get("slippage_bps", 0.0))
        fill_probability = float(broker_cfg.get("fill_probability", 1.0))
        return MockBroker(
            equity=mock_equity,
            slippage_bps=slippage_bps,
            fill_probability=fill_probability,
        )

    def enable_trading(self, enabled: bool):
        self.db.upsert_runtime_state("trading_enabled", bool(enabled))

    def scheduled_heartbeat(self):
        self.monitor.heartbeat("scheduler", payload={"config": self.scheduler_cfg})
        try:
            now = self.session.now()
            account = self.broker.get_account()
            daily_state, daily_pnl, peak_equity = self._build_risk_context(now=now, equity=float(account.equity))
            if self.risk.check_kill_switch(
                current_value=float(account.equity),
                peak_value=float(peak_equity),
                daily_pnl=float(daily_pnl),
                portfolio_value_start_of_day=float(daily_state["start_of_day_equity"]),
            ):
                flatten = bool(self.risk.limits.flatten_on_kill)
                self.alerts.critical(
                    message="Kill-switch triggered during heartbeat; stopping trading engine",
                    data={
                        "time": now.isoformat(),
                        "equity": float(account.equity),
                        "peak_equity": float(peak_equity),
                        "daily_pnl": float(daily_pnl),
                        "flatten": flatten,
                    },
                    event_type="risk",
                )
                self.stop(flatten=flatten)
                return
        except Exception as e:
            self.alerts.warning(
                message="Kill-switch heartbeat check failed",
                data={"error": str(e)},
                event_type="risk",
            )
        self.monitor.check_health(emit_alerts=True)

    def scheduled_rebalance(self):
        self.rebalance_once(reason="scheduled")

    def start(self, run_once: bool = False, with_scheduler: Optional[bool] = None) -> Dict[str, Any]:
        self.enable_trading(True)
        self.monitor.heartbeat("engine", payload={"event": "start"})

        if run_once:
            result = self.rebalance_once(reason="manual_once")
            return result.to_dict()

        scheduler_enabled = bool(self.scheduler_cfg.get("enabled", True))
        if with_scheduler is not None:
            scheduler_enabled = bool(with_scheduler)

        if scheduler_enabled:
            self.scheduler.start()
            self.alerts.info("Trading scheduler started", event_type="runtime")
        else:
            self.alerts.info("Scheduler disabled, engine idle", event_type="runtime")

        try:
            while True:
                enabled = bool(self.db.get_runtime_state("trading_enabled", True))
                if not enabled:
                    break
                self.monitor.heartbeat("engine", payload={"event": "loop"})
                time.sleep(1.0)
        except KeyboardInterrupt:
            self.alerts.warning("Received keyboard interrupt; stopping trading engine", event_type="runtime")
        finally:
            if self.scheduler.running:
                self.scheduler.stop(wait=False)
            self.monitor.heartbeat("engine", payload={"event": "stopped"})

        return {
            "status": "stopped",
            "message": "Trading loop exited",
            "payload": self.status(),
        }

    def stop(self, flatten: bool = False) -> Dict[str, Any]:
        self.enable_trading(False)
        if self.scheduler.running:
            self.scheduler.stop(wait=False)

        if flatten:
            for attempt in (1, 2):
                try:
                    self.broker.cancel_all_orders()
                    self.broker.liquidate_all()
                    break
                except Exception as e:
                    if attempt == 1:
                        self.alerts.warning(
                            message="Flatten request failed; retrying once",
                            data={"error": str(e), "attempt": attempt},
                            event_type="runtime",
                        )
                    else:
                        self.alerts.critical(
                            message="Flatten request failed after retry",
                            data={"error": str(e), "attempt": attempt},
                            event_type="runtime",
                        )

        self.alerts.info("Trading engine stop requested", event_type="runtime")
        return self.status()

    def rebalance_once(self, reason: str = "manual") -> TradingEngineResult:
        now = self.session.now()
        require_market_open = bool(self.execution_cfg.get("require_market_open", False))
        if require_market_open and (not self.session.is_market_open(now)):
            msg = "Market is closed; skipping rebalance"
            self.alerts.info(msg, data={"time": now.isoformat()}, event_type="rebalance")
            return TradingEngineResult(status="skipped", message=msg, payload={"time": now.isoformat()})

        account = self.broker.get_account()
        positions = self.broker.get_positions()

        daily_state, daily_pnl, peak_equity = self._build_risk_context(now=now, equity=float(account.equity))
        if self.risk.check_kill_switch(
            current_value=float(account.equity),
            peak_value=float(peak_equity),
            daily_pnl=float(daily_pnl),
            portfolio_value_start_of_day=float(daily_state["start_of_day_equity"]),
        ):
            flatten = bool(self.risk.limits.flatten_on_kill)
            msg = "Kill-switch triggered; rebalance blocked and trading stopped"
            self.alerts.critical(
                message=msg,
                data={
                    "time": now.isoformat(),
                    "reason": reason,
                    "equity": float(account.equity),
                    "peak_equity": float(peak_equity),
                    "daily_pnl": float(daily_pnl),
                    "flatten": flatten,
                },
                event_type="risk",
            )
            self.stop(flatten=flatten)
            return TradingEngineResult(
                status="blocked",
                message=msg,
                payload={
                    "reason": reason,
                    "time": now.isoformat(),
                    "equity": float(account.equity),
                    "peak_equity": float(peak_equity),
                    "daily_pnl": float(daily_pnl),
                },
            )

        signal_universe = self.signals.load_universe()
        current_shares = self.portfolio.shares_from_positions(positions)

        universe = sorted(set(signal_universe) | set(current_shares.keys()))
        if not universe:
            msg = "No universe symbols configured"
            self.alerts.warning(msg, event_type="rebalance")
            return TradingEngineResult(status="failed", message=msg, payload={})

        prices = self.market_data.get_latest_prices(universe)
        for p in positions:
            if p.symbol not in prices and p.current_price:
                prices[p.symbol] = float(p.current_price)

        target_weights = self.signals.generate_target_weights(as_of=now, universe=signal_universe or universe)
        if not target_weights:
            msg = "Signal generator produced empty target weights"
            self.alerts.warning(msg, event_type="rebalance")
            return TradingEngineResult(status="failed", message=msg, payload={})

        current_weights = self.portfolio.weights_from_positions(
            positions=positions,
            prices=prices,
            equity=float(account.equity),
        )

        risk_result = self.risk.check_portfolio(
            target_weights=target_weights,
            current_weights=current_weights,
            portfolio_value=float(account.equity),
            daily_pnl=float(daily_pnl),
            peak_value=float(peak_equity),
            cash=float(account.cash),
        )

        if (not risk_result.approved) and risk_result.severity == "critical":
            msg = "Rebalance blocked by critical risk violations"
            self.alerts.critical(
                message=msg,
                data={"violations": risk_result.violations, "reason": reason},
                event_type="risk",
            )
            return TradingEngineResult(
                status="blocked",
                message=msg,
                payload={
                    "reason": reason,
                    "severity": risk_result.severity,
                    "violations": risk_result.violations,
                },
            )

        if not risk_result.approved:
            target_weights = self.risk.adjust_weights(target_weights, risk_result)
            self.alerts.warning(
                message="Risk engine adjusted target weights",
                data={"violations": risk_result.violations, "reason": reason},
                event_type="risk",
            )

        cash_buffer_pct = float(self.execution_cfg.get("cash_buffer_pct", 0.02))
        fractional_shares = bool(self.execution_cfg.get("fractional_shares", True))
        target_shares = self.portfolio.target_shares_from_weights(
            target_weights=target_weights,
            prices=prices,
            equity=float(account.equity),
            cash_buffer_pct=cash_buffer_pct,
            fractional_shares=fractional_shares,
        )

        min_trade_dollars = float(self.execution_cfg.get("min_trade_dollars", 25.0))
        proposed_orders = self.orders.generate_orders(
            current_shares=current_shares,
            target_shares=target_shares,
            prices=prices,
            min_trade_dollars=min_trade_dollars,
        )
        max_single_order_value = float(getattr(self.risk.limits, "max_single_order_value", 0.0) or 0.0)
        if max_single_order_value > 0 and proposed_orders:
            bounded_orders = []
            for order in proposed_orders:
                px = float(prices.get(order.symbol, 0.0) or 0.0)
                if px <= 0:
                    continue
                notional = abs(float(order.qty) * px)
                if notional > max_single_order_value:
                    scaled_qty = max_single_order_value / px
                    if scaled_qty * px < min_trade_dollars:
                        continue
                    order.qty = float(scaled_qty)
                bounded_orders.append(order)
            proposed_orders = bounded_orders

        drift = self.reconciler.calculate_drift(current_weights=current_weights, target_weights=target_weights)

        dry_run = bool(self.execution_cfg.get("dry_run", False))
        if dry_run:
            msg = f"Dry run: generated {len(proposed_orders)} orders"
            self.alerts.info(msg, data={"reason": reason, "drift": drift}, event_type="rebalance")
            snapshot = self._save_snapshot(now, account.equity, account.cash, positions, drift)
            return TradingEngineResult(
                status="dry_run",
                message=msg,
                payload={
                    "orders": [o.__dict__ for o in proposed_orders],
                    "snapshot": snapshot,
                    "drift": drift,
                },
            )

        if not proposed_orders:
            msg = "No orders generated (portfolio already aligned)"
            snapshot = self._save_snapshot(now, account.equity, account.cash, positions, drift)
            self.alerts.info(msg, data={"reason": reason, "drift": drift}, event_type="rebalance")
            return TradingEngineResult(status="ok", message=msg, payload={"snapshot": snapshot, "drift": drift})

        rebalance_id = f"{now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        order_ids = self.orders.submit_orders(proposed_orders, rebalance_id=rebalance_id)
        self.orders.poll_terminal(
            order_ids=order_ids,
            timeout_seconds=int(self.execution_cfg.get("poll_timeout_seconds", 300)),
            poll_interval=float(self.execution_cfg.get("poll_interval_seconds", 2.0)),
        )

        order_sync = self.reconciler.reconcile_orders()
        synced_positions = self.reconciler.sync_positions()
        account_after = self.broker.get_account()

        self.db.upsert_runtime_state("last_rebalance", {
            "time": now.isoformat(),
            "reason": reason,
            "rebalance_id": rebalance_id,
            "orders_submitted": len(order_ids),
            "drift": drift,
        })

        self.db.upsert_runtime_state("peak_equity", max(peak_equity, float(account_after.equity)))
        snapshot = self._save_snapshot(
            now=now,
            equity=float(account_after.equity),
            cash=float(account_after.cash),
            positions=synced_positions,
            turnover=drift,
        )

        self.alerts.info(
            message=f"Rebalance completed: {len(order_ids)} orders",
            data={"reason": reason, "rebalance_id": rebalance_id, "order_sync": order_sync},
            event_type="rebalance",
        )

        return TradingEngineResult(
            status="ok",
            message="Rebalance completed",
            payload={
                "rebalance_id": rebalance_id,
                "orders_submitted": len(order_ids),
                "order_sync": order_sync,
                "drift": drift,
                "snapshot": snapshot,
            },
        )

    def status(self) -> Dict[str, Any]:
        runtime = {
            "trading_enabled": bool(self.db.get_runtime_state("trading_enabled", False)),
            "last_rebalance": self.db.get_runtime_state("last_rebalance", {}),
            "peak_equity": self.db.get_runtime_state("peak_equity", None),
        }

        account = None
        try:
            a = self.broker.get_account()
            account = {
                "equity": float(a.equity),
                "cash": float(a.cash),
                "buying_power": float(a.buying_power),
                "daytrade_count": int(a.daytrade_count),
            }
        except Exception as e:
            account = {"error": str(e)}

        health = self.monitor.check_health()
        return {
            "mode": str(self.execution_cfg.get("mode", "paper")),
            "broker": str((self.config.get("broker") or {}).get("provider", "mock")),
            "paper": bool((self.config.get("broker") or {}).get("paper", True)),
            "runtime": runtime,
            "account": account,
            "latest_snapshot": self.db.get_latest_snapshot(),
            "positions": self.db.get_positions(),
            "open_orders": self.db.get_open_orders(),
            "health": health,
        }

    def _ensure_daily_state(self, now: datetime, equity: float) -> Dict[str, float]:
        state = self.db.get_runtime_state("daily_state", {}) or {}
        day = now.date().isoformat()

        if state.get("day") != day:
            state = {
                "day": day,
                "start_of_day_equity": float(equity),
            }
            self.db.upsert_runtime_state("daily_state", state)

        return {
            "start_of_day_equity": float(state.get("start_of_day_equity", equity)),
        }

    def _save_snapshot(
        self,
        now: datetime,
        equity: float,
        cash: float,
        positions: list[Any],
        turnover: float,
    ) -> Dict[str, Any]:
        day_state = self._ensure_daily_state(now, equity)
        start_equity = float(self.db.get_runtime_state("portfolio_start_equity", equity) or equity)
        if not self.db.get_runtime_state("portfolio_start_equity"):
            self.db.upsert_runtime_state("portfolio_start_equity", float(equity))

        peak = float(max(float(equity), float(self.db.get_runtime_state("peak_equity", equity) or equity)))
        self.db.upsert_runtime_state("peak_equity", peak)

        if positions and isinstance(positions[0], dict):
            positions_value = float(sum(float(p.get("market_value", 0.0)) for p in positions))
            n_pos = int(sum(1 for p in positions if abs(float(p.get("qty", 0.0))) > 1e-12))
        else:
            positions_value = float(sum(float(getattr(p, "market_value", 0.0)) for p in positions))
            n_pos = int(sum(1 for p in positions if abs(float(getattr(p, "qty", 0.0))) > 1e-12))

        daily_return = 0.0
        if day_state["start_of_day_equity"] > 0:
            daily_return = float(equity / day_state["start_of_day_equity"] - 1.0)
        cumulative_return = 0.0
        if start_equity > 0:
            cumulative_return = float(equity / start_equity - 1.0)
        drawdown = float((equity / peak - 1.0) if peak > 0 else 0.0)

        payload = {
            "date": now.date(),
            "portfolio_value": float(equity),
            "cash": float(cash),
            "positions_value": float(positions_value),
            "daily_return": float(daily_return),
            "cumulative_return": float(cumulative_return),
            "peak_value": float(peak),
            "drawdown": float(drawdown),
            "num_positions": int(n_pos),
            "turnover": float(turnover),
        }
        self.db.upsert_daily_snapshot(payload)
        return {**payload, "date": payload["date"].isoformat()}

    def _build_risk_context(self, now: datetime, equity: float) -> tuple[Dict[str, float], float, float]:
        daily_state = self._ensure_daily_state(now, equity)
        daily_pnl = float(equity - daily_state["start_of_day_equity"])
        peak_equity = float(max(equity, self.db.get_runtime_state("peak_equity", equity)))
        return daily_state, daily_pnl, peak_equity
