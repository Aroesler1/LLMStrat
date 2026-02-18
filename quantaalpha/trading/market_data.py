"""
Market data adapter for trading runtime.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from quantaalpha.trading.broker import BrokerAPI


class MarketDataFetcher:
    def __init__(self, broker: Optional[BrokerAPI] = None):
        self.broker = broker

    def get_bars(self, symbols: List[str], timeframe: str = "1Min", limit: int = 200) -> Dict[str, pd.DataFrame]:
        if self.broker is not None:
            try:
                bars = self.broker.get_bars(symbols=symbols, timeframe=timeframe, limit=limit)
                if bars:
                    return bars
            except Exception:
                pass
        return self._get_bars_yfinance(symbols=symbols, interval=timeframe, limit=limit)

    def get_latest_prices(self, symbols: List[str]) -> Dict[str, float]:
        out = {}
        bars = self.get_bars(symbols=symbols, timeframe="1Min", limit=2)
        for sym, df in bars.items():
            if df is None or df.empty:
                continue
            col = "close" if "close" in df.columns else ("Close" if "Close" in df.columns else None)
            if col is None:
                continue
            out[sym] = float(df[col].iloc[-1])
        return out

    @staticmethod
    def _get_bars_yfinance(symbols: List[str], interval: str = "1Min", limit: int = 200) -> Dict[str, pd.DataFrame]:
        try:
            import yfinance as yf
        except Exception:
            return {}
        out = {}
        yf_interval = "1m" if interval.lower() in ("1min", "1m") else "1d"
        period = "7d" if yf_interval == "1m" else "1y"
        for sym in symbols:
            try:
                df = yf.download(sym, period=period, interval=yf_interval, progress=False, auto_adjust=False)
                if df is None or df.empty:
                    continue
                df = df.rename(columns={c: c.lower() for c in df.columns})
                if limit > 0:
                    df = df.tail(limit)
                out[sym] = df
            except Exception:
                continue
        return out
