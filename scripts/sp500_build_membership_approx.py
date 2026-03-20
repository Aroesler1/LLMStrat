#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import date
from io import StringIO
from pathlib import Path
import sys
from urllib.request import Request, urlopen

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
US_ROOT = SCRIPT_DIR.parent
for _p in (str(US_ROOT), str(US_ROOT.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from quantaalpha_us.paths import resolve_from_us_root

from quantaalpha_us.data.membership import (  # noqa: E402
    build_constant_membership_from_snapshot,
    get_trading_days,
    save_dataframe,
)


WIKIPEDIA_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def _load_snapshot_from_wikipedia() -> pd.DataFrame:
    req = Request(
        WIKIPEDIA_URL,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        },
    )
    with urlopen(req, timeout=30) as resp:
        html = resp.read().decode("utf-8")

    tables = pd.read_html(StringIO(html))
    for table in tables:
        cols = {str(c).strip() for c in table.columns}
        if "Symbol" in cols and ("GICS Sector" in cols or "Sector" in cols):
            return table
    raise RuntimeError("Could not find S&P 500 constituents table on Wikipedia")


def _load_snapshot(path: Path | None) -> pd.DataFrame:
    if path is None:
        return _load_snapshot_from_wikipedia()
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an approximate S&P 500 membership artifact from a current constituent snapshot."
    )
    parser.add_argument(
        "--snapshot-file",
        default=None,
        help="Optional CSV/parquet snapshot. If omitted, fetch current constituents from Wikipedia.",
    )
    parser.add_argument("--start-date", default="2000-01-03")
    parser.add_argument("--end-date", default=str(date.today()))
    parser.add_argument("--membership-out", default="data/us_equities/reference/sp500_membership_daily.parquet")
    parser.add_argument("--sectors-out", default="data/us_equities/reference/gics_sectors.csv")
    parser.add_argument("--ticker-map-out", default="data/us_equities/reference/ticker_mapping.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot_path = resolve_from_us_root(args.snapshot_file, US_ROOT) if args.snapshot_file else None

    snapshot = _load_snapshot(snapshot_path)
    trading_days = get_trading_days(args.start_date, args.end_date)
    result = build_constant_membership_from_snapshot(
        snapshot,
        start_date=args.start_date,
        end_date=args.end_date,
        trading_days=trading_days,
    )

    membership_out = resolve_from_us_root(args.membership_out, US_ROOT)
    sectors_out = resolve_from_us_root(args.sectors_out, US_ROOT)
    ticker_out = resolve_from_us_root(args.ticker_map_out, US_ROOT)

    membership_path = save_dataframe(result.membership, membership_out)
    sectors_path = save_dataframe(result.sectors, sectors_out)
    ticker_path = save_dataframe(result.ticker_mapping, ticker_out)

    counts = result.membership.groupby("date")["symbol"].nunique() if not result.membership.empty else pd.Series(dtype=float)
    summary = {
        "mode": "approx_current_constituents",
        "source": str(snapshot_path) if snapshot_path else WIKIPEDIA_URL,
        "survivorship_bias": True,
        "snapshot_symbols": int(result.sectors["symbol"].nunique()) if not result.sectors.empty else 0,
        "membership_rows": int(len(result.membership)),
        "date_min": str(result.membership["date"].min().date()) if not result.membership.empty else None,
        "date_max": str(result.membership["date"].max().date()) if not result.membership.empty else None,
        "min_members": int(counts.min()) if not counts.empty else 0,
        "max_members": int(counts.max()) if not counts.empty else 0,
        "membership_path": str(membership_path),
        "sectors_path": str(sectors_path),
        "ticker_map_path": str(ticker_path),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
