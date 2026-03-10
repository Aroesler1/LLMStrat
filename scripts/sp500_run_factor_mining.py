#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any
from urllib.request import Request, urlopen

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
US_ROOT = SCRIPT_DIR.parent
for _p in (str(US_ROOT), str(US_ROOT.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from quantaalpha_us.paths import resolve_from_us_root

from quantaalpha_us.llm.budget import RunBudget  # noqa: E402
from quantaalpha_us.llm.mining import FactorMiningRuntime  # noqa: E402


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _dummy_call_model(model: str, prompt: str) -> Any:
    # Placeholder runtime hook: replace with your actual LLM client call.
    lines = [
        "TS_MEAN($close, 21) / (TS_STD($close, 21) + 1e-8)",
        "TS_SUM($return, 5) * (TS_MEAN($volume, 5) / (TS_MEAN($volume, 20) + 1e-8))",
        "import os",  # intentionally invalid to validate sanitizer flow
    ]
    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps({"factors": lines}),
                }
            }
        ],
        "usage": {"total_tokens": 128},
        "model": model,
    }


def _build_live_call_model(api_key: str, base_url: str):
    endpoint = base_url.rstrip("/") + "/chat/completions"

    def _call(model: str, prompt: str) -> Any:
        req_payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "Return JSON: {\"factors\":[...]} only."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.6,
            "max_tokens": 800,
        }
        body = json.dumps(req_payload).encode("utf-8")
        req = Request(
            endpoint,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        with urlopen(req, timeout=60) as resp:
            raw = resp.read().decode("utf-8")
        return json.loads(raw)

    return _call


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run US factor mining runtime with budget/sanitizer guards.")
    parser.add_argument("--config", default="configs/llm_sp500.yaml")
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--prompt", action="append", default=[], help="Optional mining prompt(s)")
    parser.add_argument("--live-call", action="store_true", help="Use OpenAI-compatible chat endpoint instead of dummy model")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--base-url-env", default="OPENAI_BASE_URL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = resolve_from_us_root(args.config, US_ROOT)
    cfg = _load_yaml(cfg_path)
    llm_cfg = cfg.get("llm", {}) if isinstance(cfg.get("llm"), dict) else {}
    budget_cfg = llm_cfg.get("budget", {}) if isinstance(llm_cfg.get("budget"), dict) else {}
    factor_cfg = llm_cfg.get("factor_generation", {}) if isinstance(llm_cfg.get("factor_generation"), dict) else {}

    prompt_file = llm_cfg.get("prompt_file")
    prompts = [p for p in args.prompt if str(p).strip()]
    if prompt_file:
        prompt_path = resolve_from_us_root(str(prompt_file), US_ROOT)
        if prompt_path.exists():
            prompts.append(prompt_path.read_text(encoding="utf-8"))
    if not prompts:
        prompts = ["Generate 5 robust US OHLCV factors for S&P 500."]

    models = [str(llm_cfg.get("primary_model", "kimi-k2.5"))]
    fallback = llm_cfg.get("fallback_models")
    if isinstance(fallback, list):
        models.extend(str(m) for m in fallback)
    models = [m for m in models if m]

    budget = RunBudget(
        max_requests=int(budget_cfg.get("max_requests_per_run", 250)),
        max_total_tokens=int(budget_cfg.get("max_total_tokens_per_run", 250000)),
        max_consecutive_failures=int(budget_cfg.get("max_consecutive_failures", 3)),
    )
    runtime = FactorMiningRuntime(
        budget=budget,
        max_batch_failure_rate=float(factor_cfg.get("max_batch_failure_rate", 0.50)),
    )

    if args.live_call:
        api_key = os.environ.get(args.api_key_env)
        base_url = os.environ.get(args.base_url_env, "https://api.openai.com/v1")
        if not api_key:
            raise RuntimeError(f"Missing API key in env var: {args.api_key_env}")
        call_model = _build_live_call_model(api_key=api_key, base_url=base_url)
    else:
        call_model = _dummy_call_model

    valid, stats = runtime.run(
        prompts=prompts,
        models=models,
        call_model=call_model,
        estimated_tokens_per_request=256,
    )

    payload = {
        "config": str(cfg_path),
        "models": models,
        "valid_factors": valid,
        "stats": stats.to_dict(),
    }

    if args.output_file:
        out_path = resolve_from_us_root(args.output_file, US_ROOT)
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = resolve_from_us_root(f"data/results/research/factor_mining_{ts}.json", US_ROOT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out_path), "valid_factors": len(valid), "stats": stats.to_dict()}, indent=2))


if __name__ == "__main__":
    main()
