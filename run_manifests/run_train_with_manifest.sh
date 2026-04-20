#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

STAGE="${STAGE:-Stage-1}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT="${OUTPUT:-${REPO_ROOT}/run_manifests/runs/${RUN_ID}.json}"
LOG_PATH="${LOG_PATH:-${REPO_ROOT}/logs/${RUN_ID}.txt}"

DATA_PATH="${DATA_PATH:-${REPO_ROOT}/data/datasets/fineweb10B_sp1024}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${REPO_ROOT}/data/tokenizers/fineweb_1024_bpe.model}"
TRAIN_PATTERN="${TRAIN_PATTERN:-${DATA_PATH}/fineweb_train_*.bin}"
VAL_PATTERN="${VAL_PATTERN:-${DATA_PATH}/fineweb_val_*.bin}"
N_GPUS="${N_GPUS:-${WORLD_SIZE:-1}}"
TRAIN_CMD_STRING="${TRAIN_CMD_STRING:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! [[ "${N_GPUS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "N_GPUS must be a positive integer, got: ${N_GPUS}" >&2
  exit 2
fi

if [[ ! -x "$(command -v "$PYTHON_BIN" 2>/dev/null || true)" ]]; then
  if [[ -x "$(command -v python 2>/dev/null || true)" ]]; then
    PYTHON_BIN=python
  else
    echo "No Python executable found (tried PYTHON_BIN='${PYTHON_BIN}', python)." >&2
    exit 2
  fi
fi

if [[ -n "${TRAIN_CMD_STRING}" && "$#" -eq 0 ]]; then
  # Robustly parse shell words (quotes, escapes) from TRAIN_CMD_STRING
  readarray -t TRAIN_CMD < <("$PYTHON_BIN" - <<'PY'
import os
import shlex
parts = shlex.split(os.environ.get("TRAIN_CMD_STRING", ""))
for p in parts:
    print(p)
PY
)
  if [[ "${#TRAIN_CMD[@]}" -eq 0 ]]; then
    echo "Failed to parse TRAIN_CMD_STRING='${TRAIN_CMD_STRING}'" >&2
    exit 2
  fi
else
  TRAIN_CMD=(python3 train_gpt.py)
fi

if [[ "$#" -gt 0 ]]; then
  TRAIN_CMD=("$@")
fi

if [[ ! -d "${DATA_PATH}" ]]; then
  echo "DATA_PATH does not exist: ${DATA_PATH}" >&2
  exit 2
fi

if [[ ! -f "${TOKENIZER_PATH}" ]]; then
  echo "TOKENIZER_PATH does not exist: ${TOKENIZER_PATH}" >&2
  exit 2
fi

shopt -s nullglob
train_glob=(${TRAIN_PATTERN})
val_glob=(${VAL_PATTERN})
if [[ ${#train_glob[@]} -eq 0 || ${#val_glob[@]} -eq 0 ]]; then
  echo "No data shards matched. TRAIN_PATTERN=${TRAIN_PATTERN} VAL_PATTERN=${VAL_PATTERN}" >&2
  exit 2
fi

mkdir -p "${REPO_ROOT}/logs"
mkdir -p "${REPO_ROOT}/run_manifests/runs"

export RUN_ID
export WORLD_SIZE="${N_GPUS}"
export DATA_PATH
export TOKENIZER_PATH
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"

if [[ "${N_GPUS}" -gt 1 ]]; then
  export LOCAL_RANK="${LOCAL_RANK:-0}"
  exec_cmd=(torchrun --standalone --nproc_per_node="${N_GPUS}" "${TRAIN_CMD[@]}")
else
  exec_cmd=("${TRAIN_CMD[@]}")
fi

cd "${REPO_ROOT}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/create_run_manifest.py" \
  --stage "${STAGE}" \
  --run-id "${RUN_ID}" \
  --output "${OUTPUT}" \
  --train-pattern "${TRAIN_PATTERN}" \
  --val-pattern "${VAL_PATTERN}" \
  --tokenizer-path "${TOKENIZER_PATH}" \
  --log-path "${LOG_PATH}" \
  --command "${exec_cmd[@]}"
