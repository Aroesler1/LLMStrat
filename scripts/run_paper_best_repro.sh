#!/usr/bin/env bash
set -euo pipefail

# End-to-end paper-best reproduction:
# 1) Run LLM-driven mining with paper-scale config
# 2) Run independent OOS backtests (custom + combined)

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
# Preserve explicit command-line overrides before loading .env
_preserve_var() {
  local _n="$1"
  eval "__OVR_${_n}_SET=\${${_n}+x}"
  eval "__OVR_${_n}_VAL=\${${_n}-}"
}
_restore_var() {
  local _n="$1"
  eval "local _set=\${__OVR_${_n}_SET-}"
  if [[ "${_set}" == "x" ]]; then
    eval "export ${_n}=\"\${__OVR_${_n}_VAL}\""
  fi
}
_preserve_var MAX_RETRY
_preserve_var LLM_MAX_REQUESTS_PER_RUN
_preserve_var LLM_MAX_TOTAL_TOKENS_PER_RUN
_preserve_var LLM_MAX_EMPTY_RESPONSES_PER_RUN
_preserve_var FACTOR_GEN_MAX_ATTEMPTS
_preserve_var CHAT_MODEL
_preserve_var REASONING_MODEL
_preserve_var QA_REASONING_EFFORT

set -a
source "${PROJECT_ROOT}/.env"
set +a

_restore_var MAX_RETRY
_restore_var LLM_MAX_REQUESTS_PER_RUN
_restore_var LLM_MAX_TOTAL_TOKENS_PER_RUN
_restore_var LLM_MAX_EMPTY_RESPONSES_PER_RUN
_restore_var FACTOR_GEN_MAX_ATTEMPTS
_restore_var CHAT_MODEL
_restore_var REASONING_MODEL
_restore_var QA_REASONING_EFFORT

# Safety defaults (can be overridden via env/.env)
: "${MAX_RETRY:=2}"
: "${LLM_MAX_REQUESTS_PER_RUN:=250}"
: "${LLM_MAX_TOTAL_TOKENS_PER_RUN:=250000}"
: "${LLM_MAX_EMPTY_RESPONSES_PER_RUN:=1}"
: "${QA_MAX_CONSECUTIVE_TASK_FAILURES:=3}"
: "${FACTOR_GEN_MAX_ATTEMPTS:=3}"
export MAX_RETRY LLM_MAX_REQUESTS_PER_RUN LLM_MAX_TOTAL_TOKENS_PER_RUN LLM_MAX_EMPTY_RESPONSES_PER_RUN QA_MAX_CONSECUTIVE_TASK_FAILURES FACTOR_GEN_MAX_ATTEMPTS

for required_var in OPENAI_API_KEY OPENAI_BASE_URL CHAT_MODEL REASONING_MODEL; do
  if [[ -z "${!required_var:-}" ]]; then
    echo "Error: ${required_var} is not set in .env"
    exit 1
  fi
done

if [[ -z "${QLIB_DATA_DIR:-}" || ! -d "${QLIB_DATA_DIR}" ]]; then
  echo "Error: QLIB_DATA_DIR is not set or does not exist: ${QLIB_DATA_DIR:-<unset>}"
  exit 1
fi

DIRECTION="${1:-Price-Volume Factor Mining}"
RUN_SUFFIX="${2:-paper_best}"
FACTOR_JSON="data/factorlib/all_factors_library_${RUN_SUFFIX}.json"

echo "============================================================"
echo "Paper-best reproduction run"
echo "Direction: ${DIRECTION}"
echo "Suffix: ${RUN_SUFFIX}"
echo "Config: configs/experiment_paper_best.yaml"
echo "Factor library target: ${FACTOR_JSON}"
echo "LLM chat model: ${CHAT_MODEL}"
echo "LLM reasoning model: ${REASONING_MODEL}"
echo "LLM safety caps: retry=${MAX_RETRY}, req=${LLM_MAX_REQUESTS_PER_RUN}, tokens=${LLM_MAX_TOTAL_TOKENS_PER_RUN}, empty=${LLM_MAX_EMPTY_RESPONSES_PER_RUN}, max_task_fail=${QA_MAX_CONSECUTIVE_TASK_FAILURES}, factor_attempts=${FACTOR_GEN_MAX_ATTEMPTS}"
echo "============================================================"

CONFIG_PATH="configs/experiment_paper_best.yaml" ./run.sh "${DIRECTION}" "${RUN_SUFFIX}"

if [[ ! -f "${FACTOR_JSON}" ]]; then
  echo "Error: factor library not found after mining: ${FACTOR_JSON}"
  exit 1
fi

echo "============================================================"
echo "Running OOS backtest (custom factors only)"
echo "============================================================"
python -m quantaalpha.backtest.run_backtest \
  -c configs/backtest_paper_best.yaml \
  --factor-source custom \
  --factor-json "${FACTOR_JSON}" \
  --experiment "paper_best_custom_${RUN_SUFFIX}"

echo "============================================================"
echo "Running OOS backtest (combined: alpha158_20 + custom)"
echo "============================================================"
python -m quantaalpha.backtest.run_backtest \
  -c configs/backtest_paper_best.yaml \
  --factor-source combined \
  --factor-json "${FACTOR_JSON}" \
  --experiment "paper_best_combined_${RUN_SUFFIX}"

echo "Done."
