#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
US_ROOT = SCRIPT_DIR.parent


def _run_cmd(cmd: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def _load_completed_stages(resume_log: Path) -> set[str]:
    if not resume_log.exists():
        return set()
    try:
        payload = json.loads(resume_log.read_text(encoding="utf-8"))
    except Exception:
        return set()
    stages = payload.get("stages", [])
    out = set()
    for stage in stages:
        if stage.get("status") == "success":
            out.add(str(stage.get("name")))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="US project daily orchestrator for research/paper/live modes.")
    parser.add_argument("--mode", choices=["research", "paper", "live"], default="research")
    parser.add_argument("--run-date", default=str(date.today()))
    parser.add_argument("--backtest-config", default="configs/backtest_sp500_research.yaml")
    parser.add_argument("--paper-config", default="configs/paper_sp500.yaml")
    parser.add_argument("--live-config", default="configs/live_sp500.yaml")
    parser.add_argument("--dry-run", action="store_true", help="Do not place live/paper orders")
    parser.add_argument("--resume-log", default=None, help="Previous orchestration JSON log path")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--n-trials", type=int, default=500)
    parser.add_argument("--run-factor-mining", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    python = sys.executable
    run_date = str(pd_timestamp(args.run_date))

    trade_cfg = args.paper_config if args.mode == "paper" else args.live_config
    report_cfg = args.backtest_config if args.mode == "research" else trade_cfg

    stages: list[dict[str, Any]] = []
    stage_cmds: list[tuple[str, list[str]]] = [
        (
            "ingest",
            [
                python,
                str(SCRIPT_DIR / "sp500_ingest_daily.py"),
                "--date",
                run_date,
            ],
        ),
        (
            "signals",
            [
                python,
                str(SCRIPT_DIR / "sp500_generate_signals.py"),
                "--config",
                args.backtest_config,
                "--as-of",
                run_date,
            ],
        ),
    ]
    if args.mode == "research":
        if args.run_factor_mining:
            stage_cmds.append(
                (
                    "factor_mining",
                    [
                        python,
                        str(SCRIPT_DIR / "sp500_run_factor_mining.py"),
                    ],
                )
            )
        stage_cmds.append(
            (
                "research_backtest",
                [
                    python,
                    str(SCRIPT_DIR / "sp500_run_research.py"),
                    "--config",
                    args.backtest_config,
                    "--n-trials",
                    str(args.n_trials),
                ],
            )
        )
    else:
        trade_cmd = [
            python,
            str(SCRIPT_DIR / "trade_once.py"),
            "--config",
            trade_cfg,
            "--as-of",
            run_date,
        ]
        if args.dry_run:
            trade_cmd.append("--dry-run")
        stage_cmds.append(("trade", trade_cmd))

    stage_cmds.append(
        (
            "report",
            [
                python,
                str(SCRIPT_DIR / "sp500_daily_report.py"),
                "--report-date",
                run_date,
                "--config",
                report_cfg,
            ],
        )
    )

    completed: set[str] = set()
    if args.resume_log:
        completed = _load_completed_stages(Path(args.resume_log))

    run_started = datetime.now(timezone.utc).isoformat()
    for stage_name, cmd in stage_cmds:
        if stage_name in completed:
            stages.append(
                {
                    "name": stage_name,
                    "status": "skipped_resume",
                    "command": cmd,
                }
            )
            continue

        start_ts = time.time()
        rc, out, err = _run_cmd(cmd)
        duration = time.time() - start_ts
        stage_payload = {
            "name": stage_name,
            "status": "success" if rc == 0 else "failed",
            "return_code": rc,
            "duration_seconds": round(duration, 3),
            "command": cmd,
            "stdout_tail": out[-4000:],
            "stderr_tail": err[-4000:],
        }
        stages.append(stage_payload)
        if rc != 0 and not args.continue_on_error:
            break

    orchestration_dir = US_ROOT / "data" / "results" / "orchestration"
    orchestration_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = orchestration_dir / f"orchestrator_{args.mode}_{ts}.json"
    payload = {
        "mode": args.mode,
        "run_date": run_date,
        "started_at_utc": run_started,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": bool(args.dry_run),
        "stages": stages,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    failed = [s for s in stages if s.get("status") == "failed"]
    print(
        json.dumps(
            {
                "output": str(out_path),
                "mode": args.mode,
                "run_date": run_date,
                "failed_stages": [s["name"] for s in failed],
                "success": len(failed) == 0,
            },
            indent=2,
        )
    )
    if failed:
        raise SystemExit(1)


def pd_timestamp(value: str) -> str:
    import pandas as pd

    return str(pd.Timestamp(value).normalize().date())


if __name__ == "__main__":
    main()
