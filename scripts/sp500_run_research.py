#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import pandas as pd
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
US_ROOT = SCRIPT_DIR.parent
for _p in (str(US_ROOT), str(US_ROOT.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from quantaalpha_us.paths import resolve_from_us_root

from quantaalpha_us.backtest.universe import SP500Universe  # noqa: E402
from quantaalpha_us.backtest.validation import BacktestValidation  # noqa: E402
from quantaalpha_us.backtest.walk_forward import WalkForwardRunner  # noqa: E402


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run US walk-forward research and validation gates.")
    parser.add_argument("--config", default="configs/backtest_sp500_research.yaml")
    parser.add_argument("--bars-file", default=None)
    parser.add_argument("--membership-file", default=None)
    parser.add_argument("--ticker-mapping-file", default=None)
    parser.add_argument("--n-trials", type=int, default=500, help="Total factor trial count for deflated Sharpe")
    parser.add_argument("--factor-overlap", type=float, default=None, help="Optional top-factor overlap score [0,1]")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_from_us_root(args.config, US_ROOT)
    cfg = _load_yaml(config_path)

    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data"), dict) else {}
    bars_path = resolve_from_us_root(args.bars_file or str(data_cfg.get("bars_path", "data/us_equities/processed/daily_bars.parquet")), US_ROOT)
    membership_path = resolve_from_us_root(args.membership_file or str(data_cfg.get("membership_parquet", "data/us_equities/reference/sp500_membership_daily.parquet")), US_ROOT)
    ticker_mapping_file = resolve_from_us_root(args.ticker_mapping_file or str(data_cfg.get("ticker_mapping_file", "data/us_equities/reference/ticker_mapping.csv")), US_ROOT)

    if args.output_dir:
        output_dir = resolve_from_us_root(args.output_dir, US_ROOT)
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_dir = resolve_from_us_root(f"data/results/research/{ts}", US_ROOT)
    output_dir.mkdir(parents=True, exist_ok=True)

    bars = _load_table(bars_path)
    if bars.empty:
        raise RuntimeError(f"Bars data missing/empty: {bars_path}")

    universe = SP500Universe(str(membership_path), ticker_mapping_file=str(ticker_mapping_file))
    runner = WalkForwardRunner(cfg)
    wf_result = runner.run(bars=bars, universe=universe, output_dir=output_dir)

    validator = BacktestValidation()
    gate_report = validator.run_all_gates(
        returns_df=wf_result.returns,
        n_trials=max(int(args.n_trials), 1),
        factor_overlap_score=args.factor_overlap,
        sector_pnl_share=None,
    )

    gates_path = output_dir / "validation_gates.json"
    gates_path.write_text(json.dumps(gate_report.to_dict(), indent=2), encoding="utf-8")

    summary = {
        "config": str(config_path),
        "bars_path": str(bars_path),
        "membership_path": str(membership_path),
        "output_dir": str(output_dir),
        "walk_forward": wf_result.summary.to_dict(),
        "validation_passed": gate_report.passed,
        "validation_gates_path": str(gates_path),
    }
    summary_path = output_dir / "research_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
