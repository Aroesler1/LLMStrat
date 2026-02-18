#!/usr/bin/env python3
"""
Factor post-processing utilities for stricter novelty control.

Includes:
1. Correlation-based greedy filtering
2. Orthogonalization (residual or PCA)
3. Rolling out-of-sample IC decay pruning
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from quantaalpha.log import logger


@dataclass
class FactorPostprocessConfig:
    enabled: bool = False
    correlation_threshold: float = 0.5
    orthogonalization: str = "none"  # none | residual | pca
    pca_explained_variance: float = 0.95
    rolling_ic_window: int = 63
    min_abs_rolling_ic: float = 0.005
    min_ic_observations: int = 40
    max_factors: Optional[int] = None

    @classmethod
    def from_dict(cls, cfg: Optional[Dict[str, Any]]) -> "FactorPostprocessConfig":
        cfg = cfg or {}
        return cls(
            enabled=bool(cfg.get("enabled", False)),
            correlation_threshold=float(cfg.get("correlation_threshold", 0.5)),
            orthogonalization=str(cfg.get("orthogonalization", "none")).lower(),
            pca_explained_variance=float(cfg.get("pca_explained_variance", 0.95)),
            rolling_ic_window=int(cfg.get("rolling_ic_window", 63)),
            min_abs_rolling_ic=float(cfg.get("min_abs_rolling_ic", 0.005)),
            min_ic_observations=int(cfg.get("min_ic_observations", 40)),
            max_factors=(
                int(cfg["max_factors"]) if cfg.get("max_factors") is not None else None
            ),
        )


class FactorPostProcessor:
    def __init__(self, config: FactorPostprocessConfig):
        self.cfg = config

    def process(
        self,
        factor_df: pd.DataFrame,
        label_df: Optional[pd.DataFrame] = None,
        fit_index: Optional[pd.Index] = None,
    ) -> pd.DataFrame:
        """
        Run full post-processing pipeline and return processed factors.

        `fit_index` can be supplied to force all selection/fitting statistics to be
        computed on an in-sample subset only (e.g., train+valid), then applied to
        the full matrix. This avoids look-ahead bias in OOS backtests.
        """
        if factor_df is None or factor_df.empty:
            return pd.DataFrame()
        if not self.cfg.enabled:
            return factor_df

        out = factor_df.copy()
        out = out.loc[:, ~out.columns.duplicated(keep="first")]
        out = self._drop_constant_columns(out)
        if out.empty:
            return out

        label = self._extract_label_series(label_df)
        fit_df = self._subset_fit_dataframe(out, fit_index)
        if fit_df.empty:
            logger.warning(
                "Factor postprocess: fit subset is empty, fallback to full sample "
                "(this may introduce look-ahead if fit_index was intended)"
            )
            fit_df = out
        fit_label = label.reindex(fit_df.index) if label is not None else None
        priority = self._build_priority(fit_df, fit_label)

        if self.cfg.max_factors is not None and self.cfg.max_factors > 0 and len(out.columns) > self.cfg.max_factors:
            keep = priority[: self.cfg.max_factors]
            out = out[keep]
            fit_df = fit_df.reindex(columns=keep)
            logger.info(f"Factor postprocess: limited factors to top {len(keep)} by priority")

        out, fit_df = self._correlation_filter(out, fit_df, priority)
        if out.empty:
            return out

        if self.cfg.orthogonalization == "residual":
            out = self._residual_orthogonalize(
                full_df=out,
                fit_df=fit_df,
                priority=[c for c in priority if c in out.columns],
            )
            fit_df = out.loc[out.index.intersection(fit_df.index)]
        elif self.cfg.orthogonalization == "pca":
            out = self._pca_orthogonalize(full_df=out, fit_df=fit_df)
            fit_df = out.loc[out.index.intersection(fit_df.index)]

        if fit_label is not None:
            fit_label = fit_label.reindex(fit_df.index)
            out = self._prune_by_rolling_ic(full_df=out, fit_df=fit_df, fit_label=fit_label)

        out = out.replace([np.inf, -np.inf], np.nan).fillna(0)
        return out

    @staticmethod
    def _subset_fit_dataframe(df: pd.DataFrame, fit_index: Optional[pd.Index]) -> pd.DataFrame:
        if fit_index is None:
            return df
        try:
            idx = df.index.intersection(fit_index)
            return df.loc[idx]
        except Exception:
            return df

    @staticmethod
    def _drop_constant_columns(df: pd.DataFrame) -> pd.DataFrame:
        keep = []
        for col in df.columns:
            s = df[col]
            if s.count() == 0:
                continue
            if float(s.std(skipna=True)) <= 1e-12:
                continue
            keep.append(col)
        if len(keep) < len(df.columns):
            logger.info(f"Factor postprocess: dropped {len(df.columns) - len(keep)} constant/empty factors")
        return df[keep]

    @staticmethod
    def _extract_label_series(label_df: Optional[pd.DataFrame]) -> Optional[pd.Series]:
        if label_df is None or label_df.empty:
            return None
        if isinstance(label_df, pd.Series):
            return label_df
        col = label_df.columns[0]
        return label_df[col]

    def _build_priority(self, df: pd.DataFrame, label: Optional[pd.Series]) -> list[str]:
        if label is None:
            scores = {c: float(df[c].var(skipna=True)) for c in df.columns}
        else:
            scores = {}
            for c in df.columns:
                ic = self._mean_daily_ic(df[c], label)
                scores[c] = abs(ic) if np.isfinite(ic) else 0.0
        ordered = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)
        return ordered

    def _correlation_filter(
        self,
        full_df: pd.DataFrame,
        fit_df: pd.DataFrame,
        priority: list[str],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        thr = self.cfg.correlation_threshold
        if thr <= 0 or len(full_df.columns) <= 1:
            return full_df, fit_df

        corr = fit_df.corr(method="spearman").abs().fillna(0)
        kept: list[str] = []
        for col in priority:
            if col not in corr.columns:
                continue
            if not kept:
                kept.append(col)
                continue
            max_corr = float(corr.loc[col, kept].max()) if kept else 0.0
            if max_corr < thr:
                kept.append(col)

        if not kept:
            kept = [priority[0]] if priority else [full_df.columns[0]]
        if len(kept) < len(full_df.columns):
            logger.info(
                f"Factor postprocess: correlation filter kept {len(kept)}/"
                f"{len(full_df.columns)} factors (threshold={thr:.2f})"
            )
        return full_df[kept], fit_df[kept]

    def _residual_orthogonalize(
        self,
        full_df: pd.DataFrame,
        fit_df: pd.DataFrame,
        priority: list[str],
    ) -> pd.DataFrame:
        kept: list[str] = []
        transformed: Dict[str, pd.Series] = {}

        for col in priority:
            y_fit = fit_df[col].astype(float)
            y_full = full_df[col].astype(float)
            if not kept:
                transformed[col] = y_full
                kept.append(col)
                continue

            x_fit = fit_df[kept].astype(float)

            valid = y_fit.notna()
            for k in kept:
                valid &= x_fit[k].notna()

            if int(valid.sum()) <= len(kept) + 2:
                transformed[col] = y_full
                kept.append(col)
                continue

            x = x_fit.loc[valid].to_numpy(dtype=np.float64)
            yv = y_fit.loc[valid].to_numpy(dtype=np.float64)
            resid_full = y_full.copy()
            try:
                beta, *_ = np.linalg.lstsq(x, yv, rcond=None)
                x_full = full_df[kept].astype(float)
                valid_full = y_full.notna()
                for k in kept:
                    valid_full &= x_full[k].notna()
                if int(valid_full.sum()) > len(kept) + 2:
                    xv = x_full.loc[valid_full].to_numpy(dtype=np.float64)
                    resid_full.loc[valid_full] = y_full.loc[valid_full].to_numpy(dtype=np.float64) - xv @ beta
            except Exception:
                resid_full = y_full

            fit_resid = resid_full.reindex(fit_df.index)
            if float(fit_resid.std(skipna=True)) <= 1e-12:
                continue

            transformed[col] = resid_full
            kept.append(col)

        out = pd.DataFrame(transformed, index=full_df.index)
        logger.info(f"Factor postprocess: residual orthogonalization output {len(out.columns)} factors")
        return out

    def _pca_orthogonalize(self, full_df: pd.DataFrame, fit_df: pd.DataFrame) -> pd.DataFrame:
        if full_df.shape[1] <= 1:
            return full_df

        x_fit = fit_df.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
        fit_mean = x_fit.mean(axis=0, keepdims=True)
        x_fit = x_fit - fit_mean

        try:
            _, s, vt = np.linalg.svd(x_fit, full_matrices=False)
        except Exception as e:
            logger.warning(f"PCA orthogonalization failed: {e}")
            return full_df

        var = s ** 2
        if var.sum() <= 0:
            return full_df
        ratio = var / var.sum()
        cumsum = np.cumsum(ratio)
        k = int(np.searchsorted(cumsum, self.cfg.pca_explained_variance) + 1)
        k = max(1, min(k, vt.shape[0]))

        x_full = full_df.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
        x_full = x_full - fit_mean
        components = x_full @ vt[:k].T
        cols = [f"PCA_{i+1:03d}" for i in range(k)]
        out = pd.DataFrame(components, index=full_df.index, columns=cols)
        logger.info(
            "Factor postprocess: PCA orthogonalization kept "
            f"{k} components (explained={float(cumsum[k - 1]):.2f})"
        )
        return out

    def _prune_by_rolling_ic(
        self,
        full_df: pd.DataFrame,
        fit_df: pd.DataFrame,
        fit_label: pd.Series,
    ) -> pd.DataFrame:
        keep = []
        dropped = []
        for col in fit_df.columns:
            ic_ts = self._daily_ic_series(fit_df[col], fit_label)
            ic_ts = ic_ts.dropna()
            if len(ic_ts) < self.cfg.min_ic_observations:
                dropped.append(col)
                continue
            window = max(2, int(self.cfg.rolling_ic_window))
            min_periods = max(2, min(window, max(10, window // 3)))
            roll = ic_ts.rolling(window, min_periods=min_periods).mean().abs()
            roll = roll.dropna()
            if roll.empty:
                dropped.append(col)
                continue
            if float(roll.iloc[-1]) >= self.cfg.min_abs_rolling_ic:
                keep.append(col)
            else:
                dropped.append(col)

        if not keep:
            logger.warning(
                "Factor postprocess: rolling IC pruning removed all factors; keeping original set"
            )
            return full_df
        if dropped:
            logger.info(
                f"Factor postprocess: rolling IC pruning kept {len(keep)}/{len(fit_df.columns)} factors"
            )
        return full_df[keep]

    @staticmethod
    def _index_level(idx: pd.MultiIndex) -> Any:
        if "datetime" in idx.names:
            return "datetime"
        return idx.names[0] if idx.names and idx.names[0] is not None else 0

    def _mean_daily_ic(self, factor: pd.Series, label: pd.Series) -> float:
        ic = self._daily_ic_series(factor, label)
        ic = ic.dropna()
        if ic.empty:
            return 0.0
        return float(ic.mean())

    def _daily_ic_series(self, factor: pd.Series, label: pd.Series) -> pd.Series:
        pair = pd.concat([factor.rename("f"), label.rename("y")], axis=1).dropna()
        if pair.empty or not isinstance(pair.index, pd.MultiIndex):
            return pd.Series(dtype=float)

        level = self._index_level(pair.index)
        try:
            return pair.groupby(level=level).apply(
                lambda x: x["f"].corr(x["y"])
            )
        except Exception:
            return pd.Series(dtype=float)
