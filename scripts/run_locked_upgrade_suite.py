#!/usr/bin/env python3
"""
Run a preregistered, low-variance upgrade suite (no micro-parameter search).

Purpose:
- compare a fixed set of major architectural upgrades vs paper baseline
- avoid cherry-picking by using a locked variant list + fixed acceptance criteria
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml
from dotenv import load_dotenv

from quantaalpha.backtest.runner import BacktestRunner


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG_PATH = PROJECT_ROOT / "configs" / "backtest_paper_best.yaml"
OUT_DIR = PROJECT_ROOT / "data" / "results" / "locked_upgrade_suite"
CFG_DIR = OUT_DIR / "configs"
SUMMARY_PATH = OUT_DIR / "summary.json"


@dataclass(frozen=True)
class Variant:
    name: str
    notes: str
    patch: Dict[str, Any]


def _deep_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def variant_grid() -> List[Variant]:
    # Fixed set only. Do not add ad-hoc micro-variants.
    return [
        Variant(
            name="baseline_paper",
            notes="Paper-style TopkDropout baseline",
            patch={
                "backtest": {"mode": "qlib"},
                "factor_postprocess": {"enabled": False},
            },
        ),
        Variant(
            name="orthogonal_prune_only",
            notes="Strict orthogonalization + decay pruning; legacy portfolio",
            patch={
                "backtest": {"mode": "qlib"},
                "factor_postprocess": {
                    "enabled": True,
                    "correlation_threshold": 0.3,
                    "orthogonalization": "residual",
                    "rolling_ic_window": 63,
                    "min_abs_rolling_ic": 0.003,
                    "min_ic_observations": 40,
                    "max_factors": 120,
                },
            },
        ),
        Variant(
            name="portfolio_risk_parity",
            notes="Advanced portfolio construction only (risk parity)",
            patch={
                "backtest": {
                    "mode": "enhanced",
                    "enhanced": {
                        "enabled": True,
                        "optimizer": "risk_parity",
                        "topk": 30,
                        "rebalance_frequency": 5,
                        "covariance_lookback": 63,
                        "max_weight": 0.06,
                        "shrinkage": 0.4,
                        "ridge": 5e-4,
                        "transaction_cost_bps": 10,
                        "regime": {"enabled": False},
                    },
                },
                "factor_postprocess": {"enabled": False},
            },
        ),
        Variant(
            name="portfolio_rp_regime",
            notes="Risk parity + dynamic regime-aware weighting",
            patch={
                "backtest": {
                    "mode": "enhanced",
                    "enhanced": {
                        "enabled": True,
                        "optimizer": "risk_parity",
                        "topk": 30,
                        "rebalance_frequency": 5,
                        "covariance_lookback": 63,
                        "max_weight": 0.06,
                        "shrinkage": 0.4,
                        "ridge": 5e-4,
                        "transaction_cost_bps": 10,
                        "regime": {
                            "enabled": True,
                            "lookback": 63,
                            "high_vol_quantile": 0.7,
                            "trend_window": 20,
                            "reversal_window": 5,
                            "momentum_window": 20,
                            "volatile_reversal_blend": 0.35,
                            "calm_momentum_blend": 0.15,
                            "volatile_signal_scale": 0.6,
                            "calm_signal_scale": 1.0,
                        },
                    },
                },
                "factor_postprocess": {"enabled": False},
            },
        ),
        Variant(
            name="full_upgrade_locked",
            notes="Portfolio + regime + orthogonalization/pruning + MOEA-mined library",
            patch={
                "backtest": {
                    "mode": "enhanced",
                    "enhanced": {
                        "enabled": True,
                        "optimizer": "risk_parity",
                        "topk": 30,
                        "rebalance_frequency": 5,
                        "covariance_lookback": 63,
                        "max_weight": 0.06,
                        "shrinkage": 0.4,
                        "ridge": 5e-4,
                        "transaction_cost_bps": 10,
                        "regime": {
                            "enabled": True,
                            "lookback": 63,
                            "high_vol_quantile": 0.7,
                            "trend_window": 20,
                            "reversal_window": 5,
                            "momentum_window": 20,
                            "volatile_reversal_blend": 0.35,
                            "calm_momentum_blend": 0.15,
                            "volatile_signal_scale": 0.6,
                            "calm_signal_scale": 1.0,
                        },
                    },
                },
                "factor_postprocess": {
                    "enabled": True,
                    "correlation_threshold": 0.3,
                    "orthogonalization": "residual",
                    "rolling_ic_window": 63,
                    "min_abs_rolling_ic": 0.003,
                    "min_ic_observations": 40,
                    "max_factors": 120,
                },
            },
        ),
    ]


def _load_yearly_metrics(csv_path: Path) -> Dict[str, Any]:
    if not csv_path.exists():
        return {
            "years_present": [],
            "positive_years": 0,
            "median_yearly_excess": None,
            "min_yearly_excess": None,
        }
    df = pd.read_csv(csv_path, parse_dates=["date"]).set_index("date")
    if "daily_excess_return" not in df.columns or df.empty:
        return {
            "years_present": [],
            "positive_years": 0,
            "median_yearly_excess": None,
            "min_yearly_excess": None,
        }
    yr = df["daily_excess_return"].groupby(df.index.year).sum()
    return {
        "years_present": [int(y) for y in yr.index.tolist()],
        "positive_years": int((yr > 0).sum()),
        "median_yearly_excess": float(yr.median()),
        "min_yearly_excess": float(yr.min()),
        "yearly_excess": {str(int(k)): float(v) for k, v in yr.items()},
    }


def _accepted(row: Dict[str, Any], baseline: Dict[str, Any]) -> bool:
    # Preregistered acceptance criteria:
    # 1) positive excess return in at least 3 calendar years
    # 2) calmar improves vs baseline
    # 3) drawdown no worse than baseline
    pos_years = int(row.get("positive_years") or 0)
    calmar = row.get("calmar_ratio")
    base_calmar = baseline.get("calmar_ratio")
    mdd = row.get("max_drawdown")
    base_mdd = baseline.get("max_drawdown")
    if calmar is None or base_calmar is None or mdd is None or base_mdd is None:
        return False
    return (
        pos_years >= 3
        and float(calmar) > float(base_calmar)
        and float(mdd) >= float(base_mdd)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run locked, preregistered upgrade suite")
    parser.add_argument("--factor-json", required=True, help="Path to mined factor library JSON")
    parser.add_argument("--factor-source", default="combined", choices=["custom", "combined"])
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CFG_DIR.mkdir(parents=True, exist_ok=True)

    factor_json = str(Path(args.factor_json))
    if not Path(factor_json).exists():
        raise FileNotFoundError(f"factor json not found: {factor_json}")

    with open(BASE_CONFIG_PATH, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    runs: List[Dict[str, Any]] = []

    for v in variant_grid():
        cfg = copy.deepcopy(base_cfg)
        cfg.setdefault("experiment", {})
        cfg["experiment"]["output_dir"] = str(OUT_DIR)
        cfg["experiment"]["output_metrics_file"] = "backtest_metrics.json"
        cfg.setdefault("llm", {})
        cfg["llm"]["enabled"] = False
        cfg.setdefault("walk_forward", {})
        cfg["walk_forward"]["enabled"] = False
        _deep_merge(cfg, v.patch)

        cfg_path = CFG_DIR / f"{v.name}.yaml"
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)

        print(f"\nRunning {v.name}")
        runner = BacktestRunner(str(cfg_path))
        metrics = runner.run(
            factor_source=args.factor_source,
            factor_json=[factor_json],
            experiment_name=v.name,
            output_name=v.name,
            skip_uncached=False,
        )
        csv_path = OUT_DIR / f"{v.name}_cumulative_excess.csv"
        yr = _load_yearly_metrics(csv_path)

        runs.append(
            {
                "name": v.name,
                "notes": v.notes,
                "config_path": str(cfg_path),
                "annualized_return": metrics.get("annualized_return"),
                "information_ratio": metrics.get("information_ratio"),
                "sharpe_ratio": metrics.get("sharpe_ratio"),
                "max_drawdown": metrics.get("max_drawdown"),
                "calmar_ratio": metrics.get("calmar_ratio"),
                "avg_turnover": metrics.get("avg_turnover"),
                **yr,
            }
        )

    baseline = next((r for r in runs if r["name"] == "baseline_paper"), None)
    accepted = []
    if baseline is not None:
        accepted = [r for r in runs if r["name"] != "baseline_paper" and _accepted(r, baseline)]

    recommended = None
    if accepted:
        recommended = sorted(
            accepted,
            key=lambda r: (
                float(r.get("calmar_ratio") or -1e9),
                float(r.get("information_ratio") or -1e9),
                float(r.get("median_yearly_excess") or -1e9),
            ),
            reverse=True,
        )[0]

    summary = {
        "protocol": {
            "description": "locked major-change suite; no micro-tuning",
            "factor_source": args.factor_source,
            "factor_json": factor_json,
            "acceptance_criteria": [
                "positive_years >= 3",
                "calmar_ratio > baseline_paper",
                "max_drawdown >= baseline_paper (less severe)",
            ],
        },
        "runs": runs,
        "accepted": accepted,
        "recommended": recommended,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved summary: {SUMMARY_PATH}")
    if recommended:
        print(f"Recommended: {recommended['name']}")
    else:
        print("No variant passed preregistered acceptance criteria.")


if __name__ == "__main__":
    main()
