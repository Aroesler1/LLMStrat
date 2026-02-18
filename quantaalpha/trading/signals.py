"""
Signal generation for retail trading runtime.

Supports:
- momentum mode: rank by trailing returns from market data
- file mode: read target weights/scores from CSV/JSON/Parquet
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from quantaalpha.trading.market_data import MarketDataFetcher


@dataclass
class SignalConfig:
    mode: str = "momentum"  # momentum | file
    topk: int = 20
    lookback_bars: int = 20
    min_price: float = 2.0
    max_weight: float = 0.10
    long_only: bool = True
    universe: List[str] = field(default_factory=list)
    universe_file: Optional[str] = None
    signal_file: Optional[str] = None
    symbol_column: str = "symbol"
    score_column: str = "score"
    weight_column: str = "weight"
    date_column: Optional[str] = "date"

    @classmethod
    def from_dict(cls, cfg: Optional[Dict]) -> "SignalConfig":
        cfg = cfg or {}
        return cls(
            mode=str(cfg.get("mode", "momentum")).lower(),
            topk=int(cfg.get("topk", 20)),
            lookback_bars=max(2, int(cfg.get("lookback_bars", 20))),
            min_price=float(cfg.get("min_price", 2.0)),
            max_weight=float(cfg.get("max_weight", 0.10)),
            long_only=bool(cfg.get("long_only", True)),
            universe=list(cfg.get("universe", []) or []),
            universe_file=cfg.get("universe_file"),
            signal_file=cfg.get("signal_file"),
            symbol_column=str(cfg.get("symbol_column", "symbol")),
            score_column=str(cfg.get("score_column", "score")),
            weight_column=str(cfg.get("weight_column", "weight")),
            date_column=cfg.get("date_column", "date"),
        )


class SignalGenerator:
    def __init__(self, config: SignalConfig, market_data: MarketDataFetcher):
        self.cfg = config
        self.market_data = market_data

    def load_universe(self) -> List[str]:
        symbols = set(s.upper() for s in self.cfg.universe if s)
        if self.cfg.universe_file:
            path = Path(self.cfg.universe_file)
            if path.exists():
                if path.suffix.lower() == ".csv":
                    df = pd.read_csv(path)
                    col = "symbol" if "symbol" in df.columns else df.columns[0]
                    for s in df[col].dropna().astype(str):
                        symbols.add(s.upper())
                else:
                    for line in path.read_text(encoding="utf-8").splitlines():
                        s = line.strip()
                        if s and not s.startswith("#"):
                            symbols.add(s.upper())
        return sorted(symbols)

    def generate_target_weights(
        self,
        as_of: Optional[datetime] = None,
        universe: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        symbols = sorted(set(s.upper() for s in (universe or self.load_universe()) if s))
        if not symbols:
            return {}
        if self.cfg.mode == "file":
            w = self._from_file(as_of=as_of)
        else:
            w = self._from_momentum(symbols)
        return self._cap_and_normalize(w)

    def _from_momentum(self, symbols: List[str]) -> Dict[str, float]:
        bars = self.market_data.get_bars(
            symbols=symbols,
            timeframe="1Day",
            limit=self.cfg.lookback_bars + 1,
        )

        scores: Dict[str, float] = {}
        for sym, df in bars.items():
            if df is None or df.empty:
                continue
            close_col = "close" if "close" in df.columns else ("Close" if "Close" in df.columns else None)
            if close_col is None:
                continue
            s = df[close_col].astype(float).dropna()
            if len(s) < 2:
                continue
            last_px = float(s.iloc[-1])
            if last_px < self.cfg.min_price:
                continue
            ret = float(s.iloc[-1] / s.iloc[0] - 1.0)
            scores[sym] = ret

        if not scores:
            return {}

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[: max(1, self.cfg.topk)]
        if self.cfg.long_only:
            ranked = [(k, v) for k, v in ranked if v > 0]
        if not ranked:
            return {}

        # Convert to non-negative weights proportional to score magnitude.
        raw = {k: max(0.0, float(v)) for k, v in ranked}
        if sum(raw.values()) <= 1e-12:
            raw = {k: 1.0 for k, _ in ranked}
        total = sum(raw.values())
        return {k: v / total for k, v in raw.items()}

    def _from_file(self, as_of: Optional[datetime]) -> Dict[str, float]:
        if not self.cfg.signal_file:
            return {}
        path = Path(self.cfg.signal_file)
        if not path.exists():
            return {}

        if path.suffix.lower() in (".json",):
            df = pd.read_json(path)
        elif path.suffix.lower() in (".parquet", ".pq"):
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)

        if df.empty:
            return {}

        sym_col = self.cfg.symbol_column if self.cfg.symbol_column in df.columns else "symbol"
        if sym_col not in df.columns:
            return {}

        # If signal file contains multiple dates, use latest <= as_of.
        date_col = self.cfg.date_column
        if date_col and date_col in df.columns:
            ref = pd.Timestamp(as_of) if as_of is not None else pd.Timestamp.utcnow()
            if ref.tzinfo is not None:
                ref = ref.tz_convert(None)
            ts = pd.to_datetime(df[date_col], errors="coerce")
            if getattr(ts.dt, "tz", None) is not None:
                ts = ts.dt.tz_convert(None)
            df = df.assign(__ts=ts)
            df = df[df["__ts"].notna() & (df["__ts"] <= ref)]
            if df.empty:
                return {}
            latest = df["__ts"].max()
            df = df[df["__ts"] == latest]

        score_col = self.cfg.score_column
        weight_col = self.cfg.weight_column

        if weight_col in df.columns:
            out = {
                str(sym).upper(): float(w)
                for sym, w in zip(df[sym_col], df[weight_col])
                if pd.notna(sym) and pd.notna(w)
            }
            out = {k: max(0.0, v) for k, v in out.items()}
            return out

        if score_col not in df.columns:
            return {}

        ranked = (
            df[[sym_col, score_col]]
            .dropna()
            .sort_values(score_col, ascending=False)
            .head(max(1, self.cfg.topk))
        )
        if ranked.empty:
            return {}

        scores = {
            str(sym).upper(): float(sc)
            for sym, sc in zip(ranked[sym_col], ranked[score_col])
        }
        if self.cfg.long_only:
            scores = {k: max(0.0, v) for k, v in scores.items() if v > 0}
        if not scores:
            return {}

        vals = np.array(list(scores.values()), dtype=float)
        vals = vals - float(np.nanmin(vals))
        if vals.sum() <= 1e-12:
            vals = np.ones_like(vals)
        vals = vals / vals.sum()
        return {k: float(v) for k, v in zip(scores.keys(), vals)}

    def _cap_and_normalize(self, weights: Dict[str, float]) -> Dict[str, float]:
        w = {k: max(0.0, float(v)) for k, v in weights.items() if v is not None}
        if not w:
            return {}

        if self.cfg.max_weight > 0:
            cap = float(self.cfg.max_weight)
            # Iterative cap + renormalize.
            for _ in range(4):
                total = sum(w.values())
                if total <= 1e-12:
                    break
                w = {k: v / total for k, v in w.items()}
                over = {k: v for k, v in w.items() if v > cap}
                if not over:
                    break
                excess = sum(v - cap for v in over.values())
                under_keys = [k for k, v in w.items() if v <= cap]
                w = {k: min(v, cap) for k, v in w.items()}
                if excess <= 0 or not under_keys:
                    continue
                under_total = sum(w[k] for k in under_keys)
                if under_total <= 1e-12:
                    add = excess / len(under_keys)
                    for k in under_keys:
                        w[k] += add
                else:
                    for k in under_keys:
                        w[k] += excess * (w[k] / under_total)

        total = sum(w.values())
        if total <= 1e-12:
            return {}
        return {k: v / total for k, v in w.items()}
