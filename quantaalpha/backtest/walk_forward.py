#!/usr/bin/env python3
"""
Walk-forward backtesting engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import pandas as pd


@dataclass
class WalkForwardConfig:
    enabled: bool = False
    train_months: int = 36
    test_months: int = 3
    step_months: int = 3
    validation_pct: float = 0.0
    expanding_window: bool = False
    retrain_model: bool = True

    @classmethod
    def from_dict(cls, cfg: Optional[Dict]) -> "WalkForwardConfig":
        cfg = cfg or {}
        return cls(
            enabled=bool(cfg.get("enabled", False)),
            train_months=int(cfg.get("train_months", 36)),
            test_months=int(cfg.get("test_months", 3)),
            step_months=int(cfg.get("step_months", 3)),
            validation_pct=max(0.0, min(0.8, float(cfg.get("validation_pct", 0.0)))),
            expanding_window=bool(cfg.get("expanding_window", False)),
            retrain_model=bool(cfg.get("retrain_model", True)),
        )


@dataclass
class WalkForwardWindow:
    idx: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


class WalkForwardEngine:
    def __init__(self, config: WalkForwardConfig):
        self.config = config

    def generate_windows(self, dates: pd.DatetimeIndex) -> List[WalkForwardWindow]:
        if len(dates) == 0:
            return []
        dates = pd.DatetimeIndex(sorted(pd.to_datetime(dates.unique())))
        start = pd.Timestamp(dates.min()).normalize()
        end = pd.Timestamp(dates.max()).normalize()

        windows: List[WalkForwardWindow] = []
        i = 0
        cursor = start
        while True:
            train_start = start if self.config.expanding_window else cursor
            train_end = train_start + pd.DateOffset(months=self.config.train_months) - pd.Timedelta(days=1)
            test_start = train_end + pd.Timedelta(days=1)
            test_end = test_start + pd.DateOffset(months=self.config.test_months) - pd.Timedelta(days=1)
            if test_end > end:
                break
            windows.append(
                WalkForwardWindow(
                    idx=i,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                )
            )
            cursor = cursor + pd.DateOffset(months=self.config.step_months)
            i += 1
            if cursor > end:
                break
        return windows

    def run(
        self,
        features: pd.DataFrame,
        label: pd.Series,
        fit_predict_fn: Callable[[pd.DataFrame, pd.Series, pd.DataFrame], pd.Series],
        evaluate_fn: Callable[[pd.Series, WalkForwardWindow], Dict],
    ) -> Dict:
        """
        Execute walk-forward:
        1) split by windows
        2) fit on train
        3) predict on test
        4) evaluate on test
        """
        if features.empty or label.empty:
            return {"windows": [], "oos_returns": pd.Series(dtype=float)}

        idx_dates = _extract_dates(features.index)
        windows = self.generate_windows(pd.DatetimeIndex(idx_dates))
        results = []
        all_oos = []

        for w in windows:
            train_mask = (idx_dates >= w.train_start) & (idx_dates <= w.train_end)
            test_mask = (idx_dates >= w.test_start) & (idx_dates <= w.test_end)

            x_train = features.loc[train_mask]
            y_train = label.loc[train_mask].reindex(x_train.index).dropna()
            x_train = x_train.reindex(y_train.index)

            x_test = features.loc[test_mask]
            y_test = label.loc[test_mask].reindex(x_test.index).dropna()
            x_test = x_test.reindex(y_test.index)

            if x_train.empty or y_train.empty or x_test.empty:
                continue

            x_fit, y_fit, valid_meta = self._split_train_valid(x_train, y_train)
            if x_fit.empty or y_fit.empty:
                continue

            pred = fit_predict_fn(x_fit, y_fit, x_test)
            if pred is None or len(pred) == 0:
                continue

            eval_out = evaluate_fn(pred, w)
            if valid_meta is not None:
                eval_out["validation_window"] = valid_meta
            eval_out["window"] = w
            results.append(eval_out)
            if "oos_returns" in eval_out and isinstance(eval_out["oos_returns"], pd.Series):
                all_oos.append(eval_out["oos_returns"])

        oos = pd.concat(all_oos).sort_index() if all_oos else pd.Series(dtype=float)
        return {
            "windows": results,
            "oos_returns": oos,
        }

    def _split_train_valid(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> tuple[pd.DataFrame, pd.Series, Optional[Dict[str, str]]]:
        pct = float(self.config.validation_pct)
        if pct <= 0.0:
            return x_train, y_train, None

        train_dates = _extract_dates(x_train.index)
        unique_dates = pd.DatetimeIndex(sorted(pd.to_datetime(train_dates.unique())))
        if len(unique_dates) < 3:
            return x_train, y_train, None

        cut_idx = int(len(unique_dates) * (1.0 - pct))
        if cut_idx <= 0 or cut_idx >= len(unique_dates):
            return x_train, y_train, None

        fit_end = unique_dates[cut_idx - 1]
        fit_mask = train_dates <= fit_end
        x_fit = x_train.loc[fit_mask]
        y_fit = y_train.loc[fit_mask].reindex(x_fit.index).dropna()
        x_fit = x_fit.reindex(y_fit.index)
        if x_fit.empty or y_fit.empty:
            return x_train, y_train, None

        meta = {
            "fit_end": str(fit_end.date()),
            "valid_start": str(unique_dates[cut_idx].date()),
            "valid_end": str(unique_dates[-1].date()),
        }
        return x_fit, y_fit, meta


def _extract_dates(index: pd.Index) -> pd.Series:
    if isinstance(index, pd.MultiIndex):
        if "datetime" in index.names:
            return pd.to_datetime(index.get_level_values("datetime"))
        return pd.to_datetime(index.get_level_values(0))
    return pd.to_datetime(index)
