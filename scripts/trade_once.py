#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
US_ROOT = SCRIPT_DIR.parent
for _p in (str(US_ROOT), str(US_ROOT.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from quantaalpha_us.paths import resolve_from_us_root

from quantaalpha_us.backtest.universe import SP500Universe  # noqa: E402
from quantaalpha_us.trading.alpaca_rest import AlpacaRESTClient  # noqa: E402
from quantaalpha_us.trading.risk import (  # noqa: E402
    evaluate_post_trade,
    evaluate_pre_trade,
    load_risk_config,
)


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _build_client(cfg: dict) -> AlpacaRESTClient:
    broker = cfg.get("broker", {}) if isinstance(cfg.get("broker"), dict) else {}
    paper = bool(broker.get("paper", True))

    api_key = os.environ.get(str(broker.get("api_key_env", "ALPACA_PAPER_API_KEY")))
    api_secret = os.environ.get(str(broker.get("secret_key_env", "ALPACA_PAPER_API_SECRET")))
    if not api_key or not api_secret:
        raise RuntimeError("Missing Alpaca credentials in environment variables")

    base_url = broker.get("base_url")
    if not base_url:
        base_url = "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"

    timeout_seconds = float(broker.get("timeout_seconds", 30.0))
    return AlpacaRESTClient(api_key=api_key, api_secret=api_secret, base_url=base_url, timeout_seconds=timeout_seconds)


def _load_latest_signals(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Signal file not found: {path}")
    signals = pd.read_csv(path)
    required = {"symbol", "weight"}
    missing = required - set(signals.columns)
    if missing:
        raise RuntimeError(f"Signal file missing columns: {sorted(missing)}")

    if "date" in signals.columns:
        signals["date"] = pd.to_datetime(signals["date"], errors="coerce").dt.normalize()
        latest = signals["date"].max()
        signals = signals[signals["date"] == latest]

    signals["symbol"] = signals["symbol"].astype(str).str.upper()
    signals["weight"] = pd.to_numeric(signals["weight"], errors="coerce").fillna(0.0)
    signals = signals[signals["weight"] > 0].copy()
    return signals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Portable one-shot rebalance for US project using Alpaca REST")
    parser.add_argument("--config", default="configs/paper_sp500.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Override config and avoid submitting orders")
    parser.add_argument("--ignore-pause", action="store_true", help="Ignore PAUSE flag")
    parser.add_argument("--min-trade-dollars", type=float, default=None)
    parser.add_argument("--allow-risk-fail", action="store_true", help="Continue even if pre-trade checks fail")
    parser.add_argument("--membership-file", default="data/us_equities/reference/sp500_membership_daily.parquet")
    parser.add_argument("--bars-file", default="data/us_equities/processed/daily_bars.parquet")
    parser.add_argument("--as-of", default=None, help="Optional signal date override (YYYY-MM-DD)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_from_us_root(args.config, US_ROOT)
    cfg = _load_yaml(config_path)
    risk_cfg = load_risk_config(cfg)

    data_dir = resolve_from_us_root("data/results/trading", US_ROOT)
    data_dir.mkdir(parents=True, exist_ok=True)
    pause_flag = data_dir / "PAUSE_US.flag"
    lockout_flag = data_dir / "LOCKOUT_US.flag"

    if lockout_flag.exists():
        raise RuntimeError(f"LOCKOUT flag present: {lockout_flag}")
    if pause_flag.exists() and not args.ignore_pause:
        raise RuntimeError(f"PAUSE flag present: {pause_flag}. Use --ignore-pause to bypass")

    execution_cfg = cfg.get("execution", {}) if isinstance(cfg.get("execution"), dict) else {}
    signals_cfg = cfg.get("signals", {}) if isinstance(cfg.get("signals"), dict) else {}

    signal_file = resolve_from_us_root(str(signals_cfg.get("signal_file", "data/results/trading/latest_signals.csv")), US_ROOT)
    signals = _load_latest_signals(signal_file)
    if signals.empty:
        raise RuntimeError("No positive-weight signals available for trade run")

    if args.as_of:
        signal_date = pd.Timestamp(args.as_of).normalize()
    elif "date" in signals.columns and signals["date"].notna().any():
        signal_date = pd.Timestamp(signals["date"].max()).normalize()
    else:
        signal_date = pd.Timestamp.utcnow().normalize()

    membership_file = resolve_from_us_root(args.membership_file, US_ROOT)
    active_universe: set[str] = set()
    if membership_file.exists():
        universe = SP500Universe(str(membership_file))
        active_universe = set(universe.get_members(signal_date))

    bars_file = resolve_from_us_root(args.bars_file, US_ROOT)
    latest_data_date: pd.Timestamp | None = None
    bars = _load_table(bars_file)
    if not bars.empty and "date" in bars.columns:
        latest_data_date = pd.to_datetime(bars["date"], errors="coerce").dt.normalize().max()

    client = _build_client(cfg)
    account = client.get_account()
    positions = client.list_positions()

    equity = float(account.get("equity", 0.0) or 0.0)
    if equity <= 0:
        raise RuntimeError(f"Invalid equity from broker: {equity}")
    buying_power = float(account.get("buying_power", equity) or equity)
    cash = float(account.get("cash", 0.0) or 0.0)

    cash_buffer_pct = float(execution_cfg.get("cash_buffer_pct", 0.02) or 0.0)
    deployable = max(equity * (1.0 - cash_buffer_pct), 0.0)

    min_trade_dollars = (
        float(args.min_trade_dollars)
        if args.min_trade_dollars is not None
        else float(execution_cfg.get("min_trade_dollars", 25.0) or 25.0)
    )

    dry_run_cfg = bool(execution_cfg.get("dry_run", False))
    dry_run = bool(args.dry_run or dry_run_cfg)

    target_values = {
        str(row.symbol).upper(): float(row.weight) * deployable
        for row in signals.itertuples(index=False)
    }

    current_values: dict[str, float] = {}
    current_qty: dict[str, float] = {}
    current_px: dict[str, float] = {}
    for p in positions:
        sym = str(p.get("symbol", "")).upper()
        if not sym:
            continue
        mv = float(p.get("market_value", 0.0) or 0.0)
        qty = float(p.get("qty", 0.0) or 0.0)
        px = float(p.get("current_price", 0.0) or 0.0)
        current_values[sym] = mv
        current_qty[sym] = qty
        current_px[sym] = px

    symbols = sorted(set(target_values) | set(current_values))
    order_intents: list[dict[str, Any]] = []

    for sym in symbols:
        target = float(target_values.get(sym, 0.0))
        current = float(current_values.get(sym, 0.0))
        delta = target - current

        if abs(delta) < min_trade_dollars:
            continue

        side = "buy" if delta > 0 else "sell"
        intent: dict[str, Any] = {
            "symbol": sym,
            "side": side,
            "target_value": round(target, 2),
            "current_value": round(current, 2),
            "delta_value": round(delta, 2),
            "price": round(current_px.get(sym, 0.0), 6),
        }

        if side == "buy":
            intent["submit_mode"] = "notional"
            intent["notional"] = round(abs(delta), 2)
        else:
            px = current_px.get(sym, 0.0)
            if px > 0:
                qty = min(abs(delta) / px, abs(current_qty.get(sym, 0.0)))
                intent["submit_mode"] = "qty"
                intent["qty"] = round(qty, 6)
            else:
                intent["submit_mode"] = "notional"
                intent["notional"] = round(abs(delta), 2)

        order_intents.append(intent)

    pre_trade = evaluate_pre_trade(
        signals=signals[["symbol", "weight"]].copy(),
        order_intents=order_intents,
        equity=equity,
        buying_power=buying_power,
        risk_config=risk_cfg,
        active_universe=active_universe if active_universe else None,
        latest_data_date=latest_data_date,
        as_of_date=signal_date,
        kill_switch_engaged=lockout_flag.exists() or pause_flag.exists(),
        is_trading_day=True,
    )
    if not pre_trade.passed and not args.allow_risk_fail:
        raise RuntimeError(f"Pre-trade risk checks failed: {json.dumps(pre_trade.to_dict(), indent=2)}")

    responses: list[dict[str, Any]] = []
    if not dry_run:
        tif = str(execution_cfg.get("time_in_force", "day")).lower()
        for intent in order_intents:
            if intent.get("submit_mode") == "qty":
                resp = client.submit_market_order(
                    symbol=str(intent["symbol"]),
                    side=str(intent["side"]),
                    qty=float(intent["qty"]),
                    time_in_force=tif,
                    extended_hours=False,
                )
            else:
                resp = client.submit_market_order(
                    symbol=str(intent["symbol"]),
                    side=str(intent["side"]),
                    notional=float(intent["notional"]),
                    time_in_force=tif,
                    extended_hours=False,
                )
            responses.append({"intent": intent, "response": resp})

    positions_after = client.list_positions() if not dry_run else positions
    target_weights = {str(r.symbol).upper(): float(r.weight) for r in signals.itertuples(index=False)}
    post_trade = evaluate_post_trade(
        target_weights=target_weights,
        positions_after=positions_after,
        order_responses=responses,
        equity=equity,
        risk_config=risk_cfg,
        account_cash=cash,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = data_dir / f"trade_once_{ts}.json"
    payload = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "dry_run": dry_run,
        "equity": equity,
        "buying_power": buying_power,
        "cash": cash,
        "deployable": deployable,
        "signal_file": str(signal_file),
        "signal_date": str(signal_date.date()),
        "latest_data_date": str(latest_data_date.date()) if latest_data_date is not None and pd.notna(latest_data_date) else None,
        "signals_count": int(len(signals)),
        "orders_count": int(len(order_intents)),
        "order_intents": order_intents,
        "responses": responses,
        "risk_pre_trade": pre_trade.to_dict(),
        "risk_post_trade": post_trade.to_dict(),
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(out_path),
                "orders": len(order_intents),
                "dry_run": dry_run,
                "pre_trade_passed": pre_trade.passed,
                "post_trade_passed": post_trade.passed,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
