#!/usr/bin/env bash
set -euo pipefail

# One-command protocol:
# 1) paper-style baseline mining + OOS backtest
# 2) locked upgraded-suite evaluation (no micro-tuning)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Provide a portable timeout shim for macOS (used by LocalEnv shell commands).
if [[ -d "${PROJECT_ROOT}/scripts/bin" ]]; then
  export PATH="${PROJECT_ROOT}/scripts/bin:${PATH}"
fi

if [[ ! -f "${PROJECT_ROOT}/.env" ]]; then
  echo "Error: .env not found. Run: cp configs/.env.example .env"
  exit 1
fi

# shellcheck disable=SC1091
set -a
source "${PROJECT_ROOT}/.env"
set +a

DIRECTION="${1:-Price-Volume Factor Mining}"
RUN_SUFFIX="${2:-paper_best_gpt52}"
FACTOR_SOURCE="${3:-combined}"
FACTOR_JSON="data/factorlib/all_factors_library_${RUN_SUFFIX}.json"

echo "============================================================"
echo "Stage 1/2: Baseline paper-best reproduction"
echo "Direction: ${DIRECTION}"
echo "Run suffix: ${RUN_SUFFIX}"
echo "============================================================"
bash "${SCRIPT_DIR}/run_paper_best_repro.sh" "${DIRECTION}" "${RUN_SUFFIX}"

if [[ ! -f "${FACTOR_JSON}" ]]; then
  echo "Error: baseline completed but factor JSON not found: ${FACTOR_JSON}"
  exit 1
fi

echo "============================================================"
echo "Stage 2/2: Locked upgraded suite"
echo "Factor source: ${FACTOR_SOURCE}"
echo "Factor JSON: ${FACTOR_JSON}"
echo "============================================================"
python "${SCRIPT_DIR}/run_locked_upgrade_suite.py" \
  --factor-source "${FACTOR_SOURCE}" \
  --factor-json "${FACTOR_JSON}"

echo "Done."
echo "Baseline + upgraded summary:"
echo "  - Baseline results: data/results/paper_best_reproduction/"
echo "  - Upgrade summary: data/results/locked_upgrade_suite/summary.json"
