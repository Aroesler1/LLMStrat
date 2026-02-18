from __future__ import annotations

from pathlib import Path
import re

import pytest

from quantaalpha.trading.engine import TradingEngine


def _engine_config(tmp_path: Path, max_drawdown_pct: float = 0.05, flatten_on_kill: bool = False):
    signal_file = tmp_path / "signals.csv"
    signal_file.write_text("symbol,weight\nAAPL,1.0\n", encoding="utf-8")
    return {
        "broker": {
            "provider": "mock",
            "paper": True,
            "mock_equity": 100000.0,
        },
        "storage": {
            "db_path": str(tmp_path / "trading_state.sqlite"),
        },
        "session": {
            "timezone": "America/New_York",
            "open_time": "09:30",
            "close_time": "16:00",
            "holidays": [],
        },
        "execution": {
            "mode": "paper",
            "require_market_open": False,
            "fractional_shares": True,
            "cash_buffer_pct": 0.0,
            "min_trade_dollars": 1.0,
            "rate_limit_per_second": 5.0,
            "dry_run": True,
            "poll_timeout_seconds": 2,
            "poll_interval_seconds": 0.1,
        },
        "scheduler": {
            "enabled": False,
        },
        "risk": {
            "max_position_pct": 1.0,
            "max_drawdown_pct": max_drawdown_pct,
            "max_daily_loss_pct": 0.5,
            "max_turnover_daily": 2.0,
            "max_positions": 10,
            "min_positions": 1,
            "max_leverage": 1.0,
            "min_cash_pct": 0.0,
            "max_portfolio_value": 1_000_000_000.0,
            "max_single_order_value": 1_000_000_000.0,
            "flatten_on_kill": flatten_on_kill,
        },
        "signals": {
            "mode": "file",
            "universe": ["AAPL"],
            "signal_file": str(signal_file),
            "symbol_column": "symbol",
            "weight_column": "weight",
            "max_weight": 1.0,
            "long_only": True,
            "date_column": None,
        },
        "monitor": {
            "stale_seconds": 600,
        },
    }


def test_rebalance_returns_blocked_on_critical_drawdown(tmp_path, monkeypatch):
    engine = TradingEngine(_engine_config(tmp_path=tmp_path, max_drawdown_pct=0.05))
    engine.db.upsert_runtime_state("peak_equity", 120000.0)

    # Force this path through check_portfolio for IQ-0016 coverage.
    monkeypatch.setattr(engine.risk, "check_kill_switch", lambda **kwargs: False)

    result = engine.rebalance_once(reason="test_drawdown_block")
    assert result.status == "blocked"
    assert "critical risk" in result.message.lower()


def test_scheduled_heartbeat_invokes_kill_switch_stop(tmp_path, monkeypatch):
    engine = TradingEngine(_engine_config(tmp_path=tmp_path, flatten_on_kill=True))
    called = {}

    monkeypatch.setattr(engine.risk, "check_kill_switch", lambda **kwargs: True)

    def _fake_stop(flatten: bool = False):
        called["flatten"] = flatten
        return {}

    monkeypatch.setattr(engine, "stop", _fake_stop)
    engine.scheduled_heartbeat()

    assert called["flatten"] is True


def test_stop_flatten_retries_once_on_failure(tmp_path, monkeypatch):
    engine = TradingEngine(_engine_config(tmp_path=tmp_path))
    attempts = {"liquidate": 0}

    monkeypatch.setattr(engine.broker, "cancel_all_orders", lambda: None)

    def _flaky_liquidate():
        attempts["liquidate"] += 1
        if attempts["liquidate"] == 1:
            raise RuntimeError("transient flatten failure")

    monkeypatch.setattr(engine.broker, "liquidate_all", _flaky_liquidate)
    engine.stop(flatten=True)

    assert attempts["liquidate"] == 2


def test_rebalance_uses_uuid_suffix_for_rebalance_id(tmp_path):
    cfg = _engine_config(tmp_path=tmp_path)
    cfg["execution"]["dry_run"] = False
    engine = TradingEngine(cfg)

    result = engine.rebalance_once(reason="uuid_check")

    assert result.status == "ok"
    rid = str(result.payload["rebalance_id"])
    assert re.match(r"^\d{8}_\d{6}_[0-9a-f]{8}$", rid)


def test_from_yaml_rejects_invalid_config_shape(tmp_path):
    cfg_path = tmp_path / "invalid_trading.yaml"
    cfg_path.write_text(
        "broker:\n  provider: mock\nexecution: []\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        TradingEngine.from_yaml(str(cfg_path))
