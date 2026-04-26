#!/usr/bin/env bash
# Single H100 Parameter Golf run (600s train + eval)
# Adapted from 4090 sweep results and h100_preflight fixes
#
# Usage:
#   export DATA_PATH=/path/to/fineweb10B_sp1024
#   export TOKENIZER_PATH=/path/to/fineweb_1024_bpe.model
#   bash run_single_h100.sh

set -euo pipefail

: "${DATA_PATH:?Need to set DATA_PATH}"
: "${TOKENIZER_PATH:?Need to set TOKENIZER_PATH}"

export MATRIX_LR=0.025
export SCALAR_LR=0.025
export TIED_EMBED_LR=0.035
export MUON_MOMENTUM=0.99
export MUON_MOMENTUM_WARMUP_START=0.92
export MUON_MOMENTUM_WARMUP_STEPS=1500
export WARMDOWN_ITERS=3000
export MUON_WEIGHT_DECAY=0.04
export GRAD_CLIP_NORM=0.3
export EMA_DECAY=0.997
export QAT_THRESHOLD=0.15
export BIGRAM_VOCAB_SIZE=2048
export BIGRAM_DIM=128
export TRAIN_SEQ_LEN=2048
export TRAIN_BATCH_TOKENS=786432
export MAX_WALLCLOCK_SECONDS=600
export ITERATIONS=99999
export TORCH_COMPILE=1
export USE_BF16=1
export COMPILE_BACKEND=inductor
export VOCAB_SIZE=0

# Architecture / attention optimizations (frontier consensus)
export QK_GAIN_INIT=5.0
export XSA_LAST_N=4
export ROPE_DIMS=16
export LN_SCALE=1
export LEAKY_RELU_ALPHA=0.5

python3 train_gpt.py "$@"
