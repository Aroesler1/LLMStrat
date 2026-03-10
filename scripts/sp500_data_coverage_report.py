#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
US_ROOT = SCRIPT_DIR.parent
for _p in (str(US_ROOT), str(US_ROOT.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from quantaalpha_us.paths import resolve_from_us_root


def _load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute S&P 500 member-day data coverage report.")
    parser.add_argument("--bars-file", default="data/us_equities/processed/daily_bars.parquet")
    parser.add_argument("--membership-file", default="data/us_equities/reference/sp500_membership_daily.parquet")
    parser.add_argument("--output-file", default="data/results/research/data_coverage_report.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bars_path = resolve_from_us_root(args.bars_file, US_ROOT)
    membership_path = resolve_from_us_root(args.membership_file, US_ROOT)
    output_path = resolve_from_us_root(args.output_file, US_ROOT)

    bars = _load_table(bars_path)
    membership = _load_table(membership_path)
    if bars.empty or membership.empty:
        raise RuntimeError("Bars and membership files are required for coverage report")

    bars["date"] = pd.to_datetime(bars["date"], errors="coerce").dt.normalize()
    bars["symbol"] = bars["symbol"].astype(str).str.upper()
    bars = bars.dropna(subset=["date", "symbol"])

    membership["date"] = pd.to_datetime(membership["date"], errors="coerce").dt.normalize()
    membership["symbol"] = membership["symbol"].astype(str).str.upper()
    if "active" in membership.columns:
        membership = membership[membership["active"] == True]  # noqa: E712

    m = membership[["date", "symbol"]].drop_duplicates()
    b = bars[["date", "symbol"]].drop_duplicates()
    merged = m.merge(b.assign(has_bar=True), on=["date", "symbol"], how="left")
    merged["has_bar"] = merged["has_bar"].fillna(False)

    total_member_days = int(len(merged))
    covered_member_days = int(merged["has_bar"].sum())
    coverage_pct = covered_member_days / total_member_days if total_member_days > 0 else 0.0

    latest_date = membership["date"].max()
    current_members = set(membership[membership["date"] == latest_date]["symbol"])
    historical_only = sorted(set(membership["symbol"]) - current_members)
    bars_symbols = set(bars["symbol"])
    historical_with_bars = sorted([s for s in historical_only if s in bars_symbols])

    by_date = (
        merged.groupby("date")["has_bar"]
        .agg(total="size", covered="sum")
        .reset_index()
    )
    by_date["coverage"] = by_date["covered"] / by_date["total"].replace(0, pd.NA)

    report = {
        "bars_file": str(bars_path),
        "membership_file": str(membership_path),
        "date_min": str(membership["date"].min().date()),
        "date_max": str(membership["date"].max().date()),
        "member_days_total": total_member_days,
        "member_days_covered": covered_member_days,
        "coverage_pct": coverage_pct,
        "daily_min_coverage_pct": float(by_date["coverage"].min()),
        "daily_median_coverage_pct": float(by_date["coverage"].median()),
        "daily_max_coverage_pct": float(by_date["coverage"].max()),
        "historical_only_symbol_count": len(historical_only),
        "historical_only_with_bars_count": len(historical_with_bars),
        "historical_only_with_bars_sample": historical_with_bars[:50],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "coverage_pct": coverage_pct}, indent=2))


if __name__ == "__main__":
    main()
