#!/usr/bin/env python3
"""
Universe utilities for backtesting/trading.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert nullable config values to float safely."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class UniverseConfig:
    source: str = "custom"  # custom | csv | sp500
    symbols: Optional[list[str]] = None
    symbols_file: Optional[str] = None
    csv_path: Optional[str] = None
    use_historical_universe: bool = False
    historical_universe_file: Optional[str] = None
    min_price: float = 0.0
    max_price: float = 0.0
    min_adv: float = 0.0
    min_avg_dollar_volume: float = 0.0
    exclude_sectors: Optional[list[str]] = None
    survivorship_free: bool = False

    @classmethod
    def from_dict(cls, cfg: Optional[Dict[str, Any]]) -> "UniverseConfig":
        cfg = cfg or {}
        return cls(
            source=str(cfg.get("source", "custom")),
            symbols=cfg.get("symbols"),
            symbols_file=cfg.get("symbols_file"),
            csv_path=cfg.get("csv_path") or cfg.get("symbols_file"),
            use_historical_universe=bool(cfg.get("use_historical_universe", False)),
            historical_universe_file=cfg.get("historical_universe_file"),
            min_price=_safe_float(cfg.get("min_price", 0.0)),
            max_price=_safe_float(cfg.get("max_price", 0.0)),
            min_adv=_safe_float(cfg.get("min_adv", cfg.get("min_avg_dollar_volume", 0.0))),
            min_avg_dollar_volume=_safe_float(cfg.get("min_avg_dollar_volume", cfg.get("min_adv", 0.0))),
            exclude_sectors=cfg.get("exclude_sectors"),
            survivorship_free=bool(cfg.get("survivorship_free", False)),
        )


def load_universe(cfg: UniverseConfig) -> list[str]:
    if cfg.symbols:
        return sorted(set(str(s).upper() for s in cfg.symbols))

    if cfg.symbols_file:
        path = Path(cfg.symbols_file).expanduser()
        if not path.exists():
            return []
        if path.suffix.lower() == ".csv":
            cfg = UniverseConfig(source="csv", csv_path=str(path))
        else:
            symbols = []
            for line in path.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if s and not s.startswith("#"):
                    symbols.append(s.upper())
            return sorted(set(symbols))

    if cfg.source == "csv" and cfg.csv_path:
        path = Path(cfg.csv_path).expanduser()
        if not path.exists():
            return []
        df = pd.read_csv(path)
        for col in ("symbol", "ticker", "instrument"):
            if col in df.columns:
                return sorted(set(df[col].astype(str).str.upper().tolist()))
        return []

    # Placeholder for S&P 500 fallback: expects a bundled CSV path if user configures.
    if cfg.source == "sp500":
        if cfg.csv_path:
            return load_universe(UniverseConfig(source="csv", csv_path=cfg.csv_path))
        return []

    return []


def apply_liquidity_filters(
    symbols: Iterable[str],
    close_prices: pd.DataFrame,
    dollar_volume: Optional[pd.DataFrame] = None,
    min_price: float = 0.0,
    max_price: float = 0.0,
    min_adv: float = 0.0,
) -> list[str]:
    keep = []
    for sym in symbols:
        if sym not in close_prices.columns:
            continue
        px = close_prices[sym].dropna()
        if px.empty:
            continue
        if float(px.iloc[-1]) < min_price:
            continue
        if max_price > 0 and float(px.iloc[-1]) > max_price:
            continue
        if dollar_volume is not None and sym in dollar_volume.columns and min_adv > 0:
            adv = float(dollar_volume[sym].dropna().tail(20).mean()) if not dollar_volume[sym].dropna().empty else 0.0
            if adv < min_adv:
                continue
        keep.append(sym)
    return keep


def load_historical_universe_mask(
    cfg: UniverseConfig,
    dates: pd.Index,
    symbols: Iterable[str],
) -> Optional[pd.DataFrame]:
    if not cfg.use_historical_universe or not cfg.historical_universe_file:
        return None

    path = Path(cfg.historical_universe_file).expanduser()
    if not path.exists():
        return None

    hist = pd.read_csv(path)
    date_col = next((c for c in ("date", "datetime", "trade_date") if c in hist.columns), None)
    symbol_col = next((c for c in ("symbol", "ticker", "instrument") if c in hist.columns), None)
    if date_col is None or symbol_col is None:
        raise ValueError(
            f"Historical universe file must include date/symbol columns: {path}"
        )

    hist = hist[[date_col, symbol_col]].dropna()
    if hist.empty:
        return None
    hist[date_col] = pd.to_datetime(hist[date_col]).dt.normalize()
    hist[symbol_col] = hist[symbol_col].astype(str).str.upper()
    hist["active"] = True

    membership = (
        hist.pivot_table(index=date_col, columns=symbol_col, values="active", aggfunc="max", fill_value=False)
        .astype(bool)
    )

    target_dates = pd.DatetimeIndex(pd.to_datetime(dates))
    target_symbols = [str(s).upper() for s in symbols]
    aligned = membership.reindex(
        index=target_dates.normalize(),
        columns=target_symbols,
        fill_value=False,
    )
    aligned.index = target_dates
    return aligned.astype(bool)
