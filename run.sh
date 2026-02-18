#!/bin/bash
# LLMStrat main experiment runner
#
# Usage:
#   ./run.sh "initial direction"                    # default experiment
#   ./run.sh "initial direction" "suffix"           # with factor library suffix
#   CONFIG=configs/experiment.yaml ./run.sh "direction"
#
# Examples:
#   ./run.sh "price-volume factor mining"
#   ./run.sh "momentum reversal factors" "exp_momentum"

# =============================================================================
# Locate project root
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Provide a portable timeout shim for macOS (used by LocalEnv shell commands).
if [ -d "${SCRIPT_DIR}/scripts/bin" ]; then
    export PATH="${SCRIPT_DIR}/scripts/bin:${PATH}"
fi

# =============================================================================
# Load .env configuration
# =============================================================================
if [ -f "${SCRIPT_DIR}/.env" ]; then
    # Preserve command-line overrides so ".env" does not clobber explicit run-time caps.
    _preserve_var() {
        local _n="$1"
        eval "__OVR_${_n}_SET=\${${_n}+x}"
        eval "__OVR_${_n}_VAL=\${${_n}-}"
    }
    _restore_var() {
        local _n="$1"
        eval "local _set=\${__OVR_${_n}_SET-}"
        if [ "${_set}" = "x" ]; then
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
    source "${SCRIPT_DIR}/.env"
    set +a

    _restore_var MAX_RETRY
    _restore_var LLM_MAX_REQUESTS_PER_RUN
    _restore_var LLM_MAX_TOTAL_TOKENS_PER_RUN
    _restore_var LLM_MAX_EMPTY_RESPONSES_PER_RUN
    _restore_var FACTOR_GEN_MAX_ATTEMPTS
    _restore_var CHAT_MODEL
    _restore_var REASONING_MODEL
    _restore_var QA_REASONING_EFFORT
else
    echo "Error: .env file not found"
    echo "Please run: cp configs/.env.example .env"
    exit 1
fi

# =============================================================================
# Activate conda environment
# =============================================================================
if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    conda activate "${CONDA_ENV_NAME:-quantaalpha}" 2>/dev/null || \
        source activate "${CONDA_ENV_NAME:-quantaalpha}" 2>/dev/null || true
else
    echo "Conda not found; using current Python environment."
fi

# Re-apply timeout shim after env activation and ensure a timeout binary exists in conda bin.
if [ -d "${SCRIPT_DIR}/scripts/bin" ]; then
    export PATH="${SCRIPT_DIR}/scripts/bin:${PATH}"
    if ! command -v timeout >/dev/null 2>&1 && [ -n "${CONDA_PREFIX:-}" ] && [ -d "${CONDA_PREFIX}/bin" ]; then
        if [ -x "${SCRIPT_DIR}/scripts/bin/timeout" ] && [ ! -e "${CONDA_PREFIX}/bin/timeout" ]; then
            ln -s "${SCRIPT_DIR}/scripts/bin/timeout" "${CONDA_PREFIX}/bin/timeout" 2>/dev/null || true
        fi
    fi
fi

# rdagent expects CONDA_DEFAULT_ENV to be set even in local/venv mode.
if [ -z "${CONDA_DEFAULT_ENV}" ]; then
    if [ -n "${CONDA_ENV_NAME}" ]; then
        export CONDA_DEFAULT_ENV="${CONDA_ENV_NAME}"
    elif [ -n "${VIRTUAL_ENV}" ]; then
        export CONDA_DEFAULT_ENV="$(basename "${VIRTUAL_ENV}")"
    else
        export CONDA_DEFAULT_ENV="quantaalpha"
    fi
fi

if ! command -v quantaalpha &> /dev/null; then
    echo "Error: quantaalpha command not found. Please install: pip install -e ."
    exit 1
fi

echo "Python: $(python --version)"
echo "LLMStrat: $(which quantaalpha)"
echo ""

# =============================================================================
# LLM runtime summary (for reproducibility)
# =============================================================================
echo "LLM base URL: ${OPENAI_BASE_URL:-<unset>}"
echo "LLM chat model: ${CHAT_MODEL:-<unset>}"
echo "LLM reasoning model: ${REASONING_MODEL:-<unset>}"
echo "LLM reasoning effort: ${QA_REASONING_EFFORT:-${REASONING_EFFORT:-none}}"
echo "LLM max retry: ${MAX_RETRY:-<default>}"
echo "LLM request cap/run: ${LLM_MAX_REQUESTS_PER_RUN:-<default>}"
echo "LLM token cap/run: ${LLM_MAX_TOTAL_TOKENS_PER_RUN:-<default>}"
echo "LLM empty-response cap/run: ${LLM_MAX_EMPTY_RESPONSES_PER_RUN:-<default>}"
echo "Max consecutive task failures: ${QA_MAX_CONSECUTIVE_TASK_FAILURES:-<default>}"
echo ""

# =============================================================================
# Experiment isolation
# =============================================================================
CONFIG_PATH=${CONFIG_PATH:-"configs/experiment.yaml"}

if [ -z "${EXPERIMENT_ID}" ]; then
    EXPERIMENT_ID="exp_$(date +%Y%m%d_%H%M%S)"
fi
export EXPERIMENT_ID

RESULTS_BASE="${DATA_RESULTS_DIR:-./data/results}"

if [ "${EXPERIMENT_ID}" != "shared" ]; then
    export WORKSPACE_PATH="${RESULTS_BASE}/workspace_${EXPERIMENT_ID}"
    export PICKLE_CACHE_FOLDER_PATH_STR="${RESULTS_BASE}/pickle_cache_${EXPERIMENT_ID}"
    mkdir -p "${WORKSPACE_PATH}" "${PICKLE_CACHE_FOLDER_PATH_STR}"
    echo "Experiment ID: ${EXPERIMENT_ID}"
    echo "Workspace: ${WORKSPACE_PATH}"
fi

# =============================================================================
# Validate Qlib data
# =============================================================================
QLIB_DATA="${QLIB_DATA_DIR:-}"
if [ -z "${QLIB_DATA}" ]; then
    echo "Error: QLIB_DATA_DIR not set. Please set Qlib data path in .env"
    echo "Example: QLIB_DATA_DIR=/path/to/qlib/cn_data"
    exit 1
fi
if [ ! -d "${QLIB_DATA}" ]; then
    echo "Error: Qlib data directory does not exist: ${QLIB_DATA}"
    echo "Please check QLIB_DATA_DIR path in .env"
    exit 1
fi
# Validate required subdirectories
for subdir in calendars features instruments; do
    if [ ! -d "${QLIB_DATA}/${subdir}" ]; then
        echo "Error: Qlib data directory missing ${subdir}/: ${QLIB_DATA}"
        echo "Valid Qlib data dir must contain calendars/, features/, instruments/"
        exit 1
    fi
done
echo "Qlib data validated: ${QLIB_DATA}"

# Ensure Qlib data symlink
if [ -n "${QLIB_DATA}" ]; then
    QLIB_SYMLINK_DIR="$HOME/.qlib/qlib_data"
    if [ ! -L "${QLIB_SYMLINK_DIR}/cn_data" ] || [ "$(readlink -f ${QLIB_SYMLINK_DIR}/cn_data 2>/dev/null)" != "$(readlink -f ${QLIB_DATA})" ]; then
        mkdir -p "${QLIB_SYMLINK_DIR}"
        ln -sfn "${QLIB_DATA}" "${QLIB_SYMLINK_DIR}/cn_data"
    fi
fi

# =============================================================================
# Parse arguments and run
# =============================================================================
DIRECTION="$1"
LIBRARY_SUFFIX="$2"

if [ -n "${LIBRARY_SUFFIX}" ]; then
    export FACTOR_LIBRARY_SUFFIX="${LIBRARY_SUFFIX}"
fi

echo ""
echo "Starting experiment..."
echo "Config: ${CONFIG_PATH}"
echo "Data: ${QLIB_DATA}"
echo "Results: ${RESULTS_BASE}"
echo "----------------------------------------"

if [ -n "${STEP_N}" ]; then
    quantaalpha mine --direction "${DIRECTION}" --step_n "${STEP_N}" --config_path "${CONFIG_PATH}"
else
    quantaalpha mine --direction "${DIRECTION}" --config_path "${CONFIG_PATH}"
fi
