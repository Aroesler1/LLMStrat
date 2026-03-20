from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

from .crsp_client import CRSPClient
from .eodhd_client import EODHDClient


logger = logging.getLogger(__name__)


class MarketDataClient:
    """CRSP-first, EODHD-fallback client for membership and bars."""

    def __init__(self, primary: Any, fallback: Any | None = None) -> None:
        self.primary = primary
        self.fallback = fallback

    @property
    def source_name(self) -> str:
        return getattr(self.primary, "source_name", "unknown")

    def _call_with_fallback(self, method: str, *args, **kwargs):
        last_error: Optional[Exception] = None
        for client in [self.primary, self.fallback]:
            if client is None:
                continue
            fn = getattr(client, method, None)
            if fn is None:
                continue
            try:
                result = fn(*args, **kwargs)
                if isinstance(result, pd.DataFrame) and result.empty:
                    continue
                return result
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning("Data source %s failed for %s: %s", getattr(client, "source_name", "unknown"), method, exc)
        if last_error is not None:
            raise last_error
        return pd.DataFrame()

    def get_eod_history(self, *args, **kwargs) -> pd.DataFrame:
        return self._call_with_fallback("get_eod_history", *args, **kwargs)

    def get_bulk_eod(self, *args, **kwargs) -> pd.DataFrame:
        return self._call_with_fallback("get_bulk_eod", *args, **kwargs)

    def get_sp500_constituents_historical(self, *args, **kwargs) -> pd.DataFrame:
        return self._call_with_fallback("get_sp500_constituents_historical", *args, **kwargs)

    def get_eod_history_batch(self, *args, **kwargs) -> pd.DataFrame:
        return self._call_with_fallback("get_eod_history_batch", *args, **kwargs)

    def get_ticker_mapping(self, *args, **kwargs) -> pd.DataFrame:
        for client in [self.primary, self.fallback]:
            if client is None:
                continue
            fn = getattr(client, "get_ticker_mapping", None)
            if fn is None:
                continue
            try:
                result = fn(*args, **kwargs)
                if isinstance(result, pd.DataFrame):
                    return result
            except Exception as exc:  # noqa: BLE001
                logger.warning("Ticker mapping load failed from %s: %s", getattr(client, "source_name", "unknown"), exc)
        return pd.DataFrame(columns=["old_symbol", "new_symbol", "effective_date", "reason"])


def build_market_data_client(
    *,
    source: str = "auto",
    eodhd_api_token: Optional[str] = None,
    eodhd_cache_dir: str = "data/us_equities/raw/cache",
):
    source = str(source).strip().lower()
    crsp = CRSPClient()
    eodhd = EODHDClient(api_token=eodhd_api_token, cache_dir=eodhd_cache_dir) if (eodhd_api_token or source == "eodhd") else None

    if source == "crsp":
        if not crsp.is_configured():
            raise RuntimeError("Source 'crsp' requested but CRSP credentials are not configured.")
        return MarketDataClient(primary=crsp, fallback=eodhd)

    if source == "eodhd":
        if eodhd is None:
            raise RuntimeError("Source 'eodhd' requested but no EODHD API token is configured.")
        return MarketDataClient(primary=eodhd)

    if crsp.is_configured():
        return MarketDataClient(primary=crsp, fallback=eodhd)
    if eodhd is None:
        raise RuntimeError("No data source is configured. Set CRSP credentials or EODHD_API_TOKEN.")
    return MarketDataClient(primary=eodhd)
