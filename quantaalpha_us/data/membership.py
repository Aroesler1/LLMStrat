from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class MembershipBuildResult:
    membership: pd.DataFrame
    sectors: pd.DataFrame
    ticker_mapping: pd.DataFrame


def get_trading_days(start_date: str | pd.Timestamp, end_date: str | pd.Timestamp) -> pd.DatetimeIndex:
    """Return NYSE trading days when possible, fallback to business days."""
    start_ts = pd.Timestamp(start_date).normalize()
    end_ts = pd.Timestamp(end_date).normalize()

    try:
        import exchange_calendars as xcals  # type: ignore

        cal = xcals.get_calendar("XNYS")
        sessions = cal.sessions_in_range(start_ts, end_ts)
        return pd.DatetimeIndex(sessions).tz_localize(None)
    except Exception:
        return pd.bdate_range(start=start_ts, end=end_ts, freq="B")


def build_membership_daily(
    constituents: pd.DataFrame,
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    trading_days: Optional[pd.DatetimeIndex] = None,
) -> pd.DataFrame:
    """Expand constituent intervals into point-in-time daily membership rows."""
    if constituents.empty:
        return pd.DataFrame(columns=["date", "symbol", "active"])

    df = constituents.copy()
    if "Code" not in df.columns:
        raise ValueError("Constituents dataframe must include 'Code' column")

    df["Code"] = df["Code"].astype(str).str.upper()
    df["StartDate"] = pd.to_datetime(df.get("StartDate"), errors="coerce").dt.normalize()
    df["EndDate"] = pd.to_datetime(df.get("EndDate"), errors="coerce").dt.normalize()

    min_start = df["StartDate"].dropna().min()
    max_end = df["EndDate"].dropna().max()

    global_start = pd.Timestamp(start_date).normalize() if start_date else (min_start or pd.Timestamp("2000-01-03"))
    global_end = pd.Timestamp(end_date).normalize() if end_date else (max_end or pd.Timestamp.today().normalize())

    days = trading_days if trading_days is not None else get_trading_days(global_start, global_end)
    days = pd.DatetimeIndex(days).tz_localize(None)

    chunks: list[pd.DataFrame] = []
    for row in df.itertuples(index=False):
        symbol = str(getattr(row, "Code")).upper()
        start = getattr(row, "StartDate", pd.NaT)
        end = getattr(row, "EndDate", pd.NaT)

        start_ts = pd.Timestamp(start).normalize() if pd.notna(start) else global_start
        end_ts = pd.Timestamp(end).normalize() if pd.notna(end) else global_end
        if end_ts < global_start or start_ts > global_end:
            continue

        start_ts = max(start_ts, global_start)
        end_ts = min(end_ts, global_end)
        active_days = days[(days >= start_ts) & (days <= end_ts)]
        if active_days.empty:
            continue

        chunks.append(
            pd.DataFrame(
                {
                    "date": active_days,
                    "symbol": symbol,
                    "active": True,
                }
            )
        )

    if not chunks:
        return pd.DataFrame(columns=["date", "symbol", "active"])

    out = pd.concat(chunks, ignore_index=True)
    out = out.drop_duplicates(subset=["date", "symbol"], keep="last").sort_values(["date", "symbol"])
    out["active"] = out["active"].astype(bool)
    return out.reset_index(drop=True)


def extract_sector_table(constituents: pd.DataFrame) -> pd.DataFrame:
    """Build a symbol->sector table from constituent payload."""
    cols = [c for c in ["Code", "Sector", "Industry", "Name"] if c in constituents.columns]
    if not cols:
        return pd.DataFrame(columns=["symbol", "sector", "industry", "name"])

    sector_df = constituents[cols].copy()
    sector_df = sector_df.rename(
        columns={
            "Code": "symbol",
            "Sector": "sector",
            "Industry": "industry",
            "Name": "name",
        }
    )
    sector_df["symbol"] = sector_df["symbol"].astype(str).str.upper()
    return sector_df.drop_duplicates(subset=["symbol"]).reset_index(drop=True)


def default_ticker_mapping() -> pd.DataFrame:
    """Return an empty ticker mapping table with the expected schema."""
    return pd.DataFrame(columns=["old_symbol", "new_symbol", "effective_date", "reason"])


def save_dataframe(df: pd.DataFrame, path: str | Path) -> Path:
    """Save dataframe as parquet when possible, otherwise CSV fallback."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if output.suffix.lower() == ".parquet":
        try:
            df.to_parquet(output, index=False)
            return output
        except Exception:
            fallback = output.with_suffix(".csv")
            df.to_csv(fallback, index=False)
            return fallback

    df.to_csv(output, index=False)
    return output


def load_membership(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if source.suffix.lower() == ".parquet":
        return pd.read_parquet(source)
    return pd.read_csv(source, parse_dates=["date"])
