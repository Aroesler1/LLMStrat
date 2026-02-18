# LLMStrat Research Protocol (Bias-Controlled)

This protocol is designed to improve from the paper baseline without introducing:
- forward-looking bias
- configuration overfitting
- cherry-picking from many small tweaks

## 1. Freeze Baseline First

Run the paper-style baseline end-to-end (LLM mining + OOS backtest) before any upgrades:

```bash
bash scripts/run_paper_best_repro.sh "Price-Volume Factor Mining" "paper_best_gpt52"
```

Baseline artifacts:
- factor library: `data/factorlib/all_factors_library_paper_best_gpt52.json`
- OOS metrics: `data/results/paper_best_reproduction/`

## 2. Run Locked Upgrade Suite (No Micro-Tuning)

Use a fixed set of major architectural variants only:

```bash
python scripts/run_locked_upgrade_suite.py \
  --factor-source combined \
  --factor-json data/factorlib/all_factors_library_paper_best_gpt52.json
```

The suite is preregistered in `scripts/run_locked_upgrade_suite.py`:
- `baseline_paper`
- `orthogonal_prune_only`
- `portfolio_risk_parity`
- `portfolio_rp_regime`
- `full_upgrade_locked`

Results:
- `data/results/locked_upgrade_suite/summary.json`

## 3. Selection Rule (Preregistered)

A variant is accepted only if all conditions hold:
1. Positive excess return in at least 3 calendar years
2. Calmar ratio improves vs baseline
3. Max drawdown is no worse than baseline

Recommended strategy is chosen only among accepted variants.

## 4. Forward-Looking Controls Implemented

- Regime classification is now lagged and uses expanding historical thresholds only.
- Factor postprocess (correlation filter / orthogonalization / IC decay pruning) is fit on train+valid only and then applied to full matrix.

## 5. What Not To Do

- Do not add ad-hoc micro-variants after seeing results.
- Do not pick a run solely because it has the highest single backtest metric.
- Do not re-optimize thresholds per market regime after viewing OOS outcomes.
