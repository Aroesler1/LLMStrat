#!/usr/bin/env python3
"""
Walk-forward OOS sweep runner for LLMStrat backtests.

Runs a grid of enhanced portfolio configs across selected factor sources,
saves per-run artifacts, and writes a ranked summary JSON.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml
from dotenv import load_dotenv

from quantaalpha.backtest.runner import BacktestRunner


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG_PATH = PROJECT_ROOT / "configs" / "backtest.yaml"
OUT_DIR = PROJECT_ROOT / "data" / "results" / "backtest_sweep"
CFG_DIR = OUT_DIR / "configs"
SUMMARY_PATH = OUT_DIR / "sweep_summary.json"


def candidate_grid() -> List[Dict[str, Any]]:
    """Targeted non-HFT parameter grid."""
    return [
        {
            "name": "rp30_r5",
            "enhanced": {
                "optimizer": "risk_parity",
                "topk": 30,
                "rebalance_frequency": 5,
                "max_weight": 0.06,
                "shrinkage": 0.40,
                "ridge": 5e-4,
                "transaction_cost_bps": 10,
            },
        },
        {
            "name": "rp20_r10",
            "enhanced": {
                "optimizer": "risk_parity",
                "topk": 20,
                "rebalance_frequency": 10,
                "max_weight": 0.07,
                "shrinkage": 0.45,
                "ridge": 1e-3,
                "transaction_cost_bps": 10,
            },
        },
        {
            "name": "mvo30_r5",
            "enhanced": {
                "optimizer": "mvo",
                "topk": 30,
                "rebalance_frequency": 5,
                "max_weight": 0.06,
                "shrinkage": 0.35,
                "ridge": 5e-4,
                "transaction_cost_bps": 10,
            },
        },
        {
            "name": "mvo20_r10",
            "enhanced": {
                "optimizer": "mvo",
                "topk": 20,
                "rebalance_frequency": 10,
                "max_weight": 0.07,
                "shrinkage": 0.45,
                "ridge": 1e-3,
                "transaction_cost_bps": 10,
            },
        },
        {
            "name": "kelly20_r10",
            "enhanced": {
                "optimizer": "kelly",
                "kelly_fraction": 0.20,
                "topk": 20,
                "rebalance_frequency": 10,
                "max_weight": 0.06,
                "shrinkage": 0.40,
                "ridge": 1e-3,
                "transaction_cost_bps": 10,
            },
        },
        {
            "name": "equal20_r10",
            "enhanced": {
                "optimizer": "equal",
                "topk": 20,
                "rebalance_frequency": 10,
                "max_weight": 0.07,
                "shrinkage": 0.40,
                "ridge": 1e-3,
                "transaction_cost_bps": 10,
            },
        },
    ]


def factor_sources() -> List[str]:
    return ["alpha158_20", "alpha158"]


def run_id(source: str, candidate_name: str) -> str:
    return f"{source}__{candidate_name}"


def prepare_config(base_cfg: Dict[str, Any], candidate: Dict[str, Any], source: str) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)

    cfg.setdefault("experiment", {})
    cfg["experiment"]["name"] = "backtest_oos_sweep"
    cfg["experiment"]["recorder"] = "backtest_oos_sweep"
    cfg["experiment"]["output_dir"] = str(OUT_DIR)
    cfg["experiment"]["output_metrics_file"] = "backtest_metrics.json"

    cfg.setdefault("walk_forward", {})
    cfg["walk_forward"].update(
        {
            "enabled": True,
            "train_months": 48,
            "test_months": 3,
            "step_months": 3,
            "expanding_window": True,
            "retrain_model": True,
        }
    )

    # Walk-forward path uses enhanced portfolio engine.
    cfg.setdefault("backtest", {})
    cfg["backtest"]["mode"] = "enhanced"
    cfg["backtest"].setdefault("enhanced", {})
    cfg["backtest"]["enhanced"].update(candidate["enhanced"])

    # Regime-aware defaults.
    cfg["backtest"]["enhanced"].setdefault("enabled", True)
    cfg["backtest"]["enhanced"].setdefault("regime", {})
    cfg["backtest"]["enhanced"]["regime"].update(
        {
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
        }
    )

    cfg.setdefault("cost_model", {})
    cfg["cost_model"].update(
        {
            "spread_bps": 7.0,
            "slippage_bps": 8.0,
            "commission_per_share": 0.0,
            "min_trade_cost": 0.0,
            "adv_impact_coefficient": 0.1,
        }
    )

    # Keep postprocess off for official factors; it's relevant for custom computed factors.
    cfg.setdefault("factor_postprocess", {})
    cfg["factor_postprocess"]["enabled"] = False

    # Keep llm off for official factor sources to avoid accidental API calls.
    cfg.setdefault("llm", {})
    cfg["llm"]["enabled"] = False

    # Carry source in config for traceability.
    cfg["_sweep"] = {"factor_source": source, "candidate": candidate["name"]}
    return cfg


def rank_results(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        rows,
        key=lambda r: (
            float(r.get("calmar_ratio") or -1e9),
            float(r.get("sharpe_ratio") or -1e9),
            float(r.get("annualized_return") or -1e9),
        ),
        reverse=True,
    )


def main() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CFG_DIR.mkdir(parents=True, exist_ok=True)

    with open(BASE_CONFIG_PATH, "r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    all_rows: List[Dict[str, Any]] = []
    total = len(factor_sources()) * len(candidate_grid())
    idx = 0

    for source in factor_sources():
        for cand in candidate_grid():
            idx += 1
            rid = run_id(source, cand["name"])
            print(f"\n[{idx}/{total}] Running {rid}")
            cfg = prepare_config(base_cfg=base_cfg, candidate=cand, source=source)
            cfg_path = CFG_DIR / f"{rid}.yaml"
            with open(cfg_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False)

            t0 = time.time()
            row: Dict[str, Any] = {
                "run_id": rid,
                "factor_source": source,
                "candidate": cand["name"],
                "config_path": str(cfg_path),
            }
            try:
                runner = BacktestRunner(str(cfg_path))
                metrics = runner.run(
                    factor_source=source,
                    factor_json=None,
                    experiment_name=rid,
                    output_name=rid,
                    skip_uncached=False,
                )
                row.update(
                    {
                        "status": "ok",
                        "annualized_return": metrics.get("annualized_return"),
                        "sharpe_ratio": metrics.get("sharpe_ratio"),
                        "max_drawdown": metrics.get("max_drawdown"),
                        "calmar_ratio": metrics.get("calmar_ratio"),
                        "information_ratio": metrics.get("information_ratio"),
                        "avg_turnover": metrics.get("avg_turnover"),
                        "walk_forward_windows": metrics.get("walk_forward_windows"),
                        "elapsed_seconds": metrics.get("elapsed_seconds", time.time() - t0),
                    }
                )
            except Exception as e:  # noqa: BLE001
                row.update(
                    {
                        "status": "error",
                        "error": str(e),
                        "elapsed_seconds": time.time() - t0,
                    }
                )

            all_rows.append(row)
            ranked = rank_results([r for r in all_rows if r.get("status") == "ok"])
            summary = {
                "runs": all_rows,
                "ranked_by_calmar": ranked,
                "best": ranked[0] if ranked else None,
            }
            SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            print(f"Saved summary: {SUMMARY_PATH}")
            if row["status"] == "ok":
                print(
                    "metrics:",
                    {
                        "ann": row.get("annualized_return"),
                        "sharpe": row.get("sharpe_ratio"),
                        "mdd": row.get("max_drawdown"),
                        "calmar": row.get("calmar_ratio"),
                        "ir": row.get("information_ratio"),
                    },
                )
            else:
                print(f"error: {row.get('error')}")

    ranked = rank_results([r for r in all_rows if r.get("status") == "ok"])
    print("\n=== Sweep finished ===")
    if ranked:
        print("Best run:", ranked[0])
    else:
        print("No successful runs.")


if __name__ == "__main__":
    main()
