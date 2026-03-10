#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path
import sys
from typing import Any, Optional

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
US_ROOT = SCRIPT_DIR.parent
for _p in (str(US_ROOT), str(US_ROOT.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from quantaalpha_us.paths import resolve_from_us_root

from quantaalpha_us.data.quality import DataQualityGate  # noqa: E402


def _load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _sha256_file(path: Path) -> Optional[str]:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _latest_file(directory: Path, pattern: str) -> Optional[Path]:
    files = sorted(directory.glob(pattern))
    return files[-1] if files else None


def _load_latest_trade_log(trading_dir: Path) -> dict[str, Any]:
    latest = _latest_file(trading_dir, "trade_once_*.json")
    if latest is None:
        return {}
    try:
        return json.loads(latest.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _summarize_execution(trade_log: dict[str, Any]) -> dict[str, Any]:
    if not trade_log:
        return {"status": "not_available", "details": "No trade log found for execution summary"}
    responses = trade_log.get("responses", [])
    statuses = []
    for wrapper in responses:
        resp = wrapper.get("response", {}) if isinstance(wrapper, dict) else {}
        if isinstance(resp, dict):
            statuses.append(str(resp.get("status", "")).lower())
    return {
        "status": "available",
        "orders_submitted": int(len(responses)),
        "status_counts": {s: statuses.count(s) for s in sorted(set(statuses))},
        "pre_trade_passed": bool(trade_log.get("risk_pre_trade", {}).get("passed", False)),
        "post_trade_passed": bool(trade_log.get("risk_post_trade", {}).get("passed", False)),
    }


def _summarize_risk(trade_log: dict[str, Any]) -> dict[str, Any]:
    if not trade_log:
        return {"status": "not_available", "details": "No trade log found for risk summary"}
    return {
        "status": "available",
        "pre_trade": trade_log.get("risk_pre_trade"),
        "post_trade": trade_log.get("risk_post_trade"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build daily US strategy report artifact.")
    parser.add_argument("--bars-file", default="data/us_equities/processed/daily_bars.parquet")
    parser.add_argument("--membership-file", default="data/us_equities/reference/sp500_membership_daily.parquet")
    parser.add_argument("--signals-file", default="data/results/trading/latest_signals.csv")
    parser.add_argument("--signals-meta-file", default="data/results/trading/latest_signals_meta.json")
    parser.add_argument("--report-date", default=str(date.today()))
    parser.add_argument("--llm-budget-json", default=None, help="Optional path to LLM budget state JSON")
    parser.add_argument("--config", default="configs/backtest_sp500_research.yaml")
    parser.add_argument("--trading-dir", default="data/results/trading")
    parser.add_argument("--output-dir", default="data/results/reports")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    bars_path = resolve_from_us_root(args.bars_file, US_ROOT)
    membership_path = resolve_from_us_root(args.membership_file, US_ROOT)
    signals_path = resolve_from_us_root(args.signals_file, US_ROOT)
    signals_meta_path = resolve_from_us_root(args.signals_meta_file, US_ROOT)
    output_dir = resolve_from_us_root(args.output_dir, US_ROOT)
    llm_budget_path = resolve_from_us_root(args.llm_budget_json, US_ROOT) if args.llm_budget_json else None
    config_path = resolve_from_us_root(args.config, US_ROOT)
    trading_dir = resolve_from_us_root(args.trading_dir, US_ROOT)

    bars = _load_table(bars_path)
    membership = _load_table(membership_path)
    signals = _load_table(signals_path)

    if bars.empty:
        raise RuntimeError(f"Bars data not found: {bars_path}")

    bars["date"] = pd.to_datetime(bars["date"], errors="coerce").dt.normalize()
    if "date" in membership.columns:
        membership["date"] = pd.to_datetime(membership["date"], errors="coerce").dt.normalize()

    report_date = pd.Timestamp(args.report_date).normalize()
    latest_bars_date = bars["date"].max()
    if pd.notna(latest_bars_date) and latest_bars_date < report_date:
        report_date = latest_bars_date

    quality_gate = DataQualityGate()
    quality = quality_gate.run_all_checks(
        bars,
        date=report_date,
        membership_df=membership if not membership.empty else None,
        mode="strict",
    )

    signal_summary: dict[str, Any] = {"available": False}
    if not signals.empty and {"symbol", "weight"}.issubset(signals.columns):
        if "date" in signals.columns:
            signals["date"] = pd.to_datetime(signals["date"], errors="coerce").dt.normalize()
            sig_date = signals["date"].max()
            subset = signals[signals["date"] == sig_date]
        else:
            sig_date = None
            subset = signals

        subset = subset.copy()
        subset["weight"] = pd.to_numeric(subset["weight"], errors="coerce").fillna(0.0)
        subset["score"] = pd.to_numeric(subset.get("score"), errors="coerce")
        top = subset.nlargest(5, "weight")[["symbol", "weight", "score"]]
        bottom = subset.nsmallest(5, "weight")[["symbol", "weight", "score"]]
        signal_summary = {
            "available": True,
            "date": str(sig_date.date()) if pd.notna(sig_date) else None,
            "positions": int(len(subset)),
            "weights_sum": float(subset["weight"].sum()),
            "top_5": top.to_dict(orient="records"),
            "bottom_5": bottom.to_dict(orient="records"),
        }

    llm_budget = None
    if llm_budget_path and llm_budget_path.exists():
        llm_budget = json.loads(llm_budget_path.read_text(encoding="utf-8"))

    signal_meta = {}
    if signals_meta_path.exists():
        try:
            signal_meta = json.loads(signals_meta_path.read_text(encoding="utf-8"))
        except Exception:
            signal_meta = {}

    trade_log = _load_latest_trade_log(trading_dir)
    risk_summary = _summarize_risk(trade_log)
    execution_summary = _summarize_execution(trade_log)

    exceptions: list[str] = []
    if not quality.passed:
        exceptions.append(f"Data quality failed: {quality.halt_reason}")
    if risk_summary.get("status") == "available":
        pre_pass = bool(risk_summary.get("pre_trade", {}).get("passed", True))
        post_pass = bool(risk_summary.get("post_trade", {}).get("passed", True))
        if not pre_pass:
            exceptions.append("Pre-trade risk checks failed")
        if not post_pass:
            exceptions.append("Post-trade reconciliation checks failed")

    report = {
        "report_date": str(report_date.date()),
        "model_version_hash": signal_meta.get("model_version_hash") or _sha256_file(signals_path),
        "config_hash": _sha256_file(config_path),
        "data_quality": quality.to_dict(),
        "llm_budget": llm_budget,
        "signal_summary": signal_summary,
        "risk_summary": risk_summary,
        "execution_summary": execution_summary,
        "exceptions": exceptions,
        "source_files": {
            "bars": str(bars_path),
            "membership": str(membership_path),
            "signals": str(signals_path),
            "signals_meta": str(signals_meta_path),
            "config": str(config_path),
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"daily_report_{report_date.strftime('%Y%m%d')}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "output": str(out_path),
                "passed": quality.passed and len(exceptions) == 0,
                "exceptions": len(exceptions),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
