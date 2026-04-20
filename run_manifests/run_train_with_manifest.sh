#!/usr/bin/env bash
set -euo pipefail

STAGE="${STAGE:-Stage-1}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT="${OUTPUT:-run_manifests/runs/${RUN_ID}.json}"
LOG_PATH="${LOG_PATH:-logs/${RUN_ID}.txt}"

DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp1024}"
TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_1024_bpe.model}"
TRAIN_PATTERN="${TRAIN_PATTERN:-${DATA_PATH}/fineweb_train_*.bin}"
VAL_PATTERN="${VAL_PATTERN:-${DATA_PATH}/fineweb_val_*.bin}"
N_GPUS="${N_GPUS:-${WORLD_SIZE:-1}}"
TRAIN_CMD_STRING="${TRAIN_CMD_STRING:-python3 train_gpt.py}"
read -r -a TRAIN_CMD <<< "${TRAIN_CMD_STRING}"
if [[ ${#TRAIN_CMD[@]} -eq 0 ]]; then
  TRAIN_CMD=(python3 train_gpt.py)
fi
if [[ "$#" -gt 0 ]]; then
  TRAIN_CMD+=("$@")
fi

mkdir -p logs
mkdir -p run_manifests/runs

export RUN_ID
export WORLD_SIZE="${N_GPUS}"
export DATA_PATH
export TOKENIZER_PATH

if [[ "${N_GPUS}" -gt 1 ]]; then
  export LOCAL_RANK="${LOCAL_RANK:-0}"
  exec_cmd=(torchrun --standalone --nproc_per_node="${N_GPUS}" "${TRAIN_CMD[@]}")
else
  exec_cmd=("${TRAIN_CMD[@]}")
fi

python3 run_manifests/create_run_manifest.py \
  --stage "${STAGE}" \
  --run-id "${RUN_ID}" \
  --output "${OUTPUT}" \
  --train-pattern "${TRAIN_PATTERN}" \
  --val-pattern "${VAL_PATTERN}" \
  --tokenizer-path "${TOKENIZER_PATH}" \
  --log-path "${LOG_PATH}" \
  --command "${exec_cmd[@]}"
