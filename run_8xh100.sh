#!/usr/bin/env bash
# 8x H100 Parameter Golf run (600s train + eval)
# Scaled from single-h100 branch for distributed multi-GPU training.
#
# Launch with:
#   export DATA_PATH="${DATA_PATH:-./data/datasets/datasets/fineweb10B_sp8192}"
#   export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/datasets/tokenizers/fineweb_8192_bpe.model}"
#   bash run_8xh100.sh
#
# Uses torchrun (Elastic Launch) with NCCL backend.
# TRAIN_BATCH_TOKENS is the *global* token count per optimizer step.
# With 8 GPUs and grad_accum_steps = 8 // 8 = 1, each rank processes
# 786432 / 8 = 98304 tokens per forward pass.

set -euo pipefail

: "${DATA_PATH:?Need to set DATA_PATH}"
: "${TOKENIZER_PATH:?Need to set TOKENIZER_PATH}"

export MATRIX_LR=0.022
export SCALAR_LR=0.022
export TIED_EMBED_LR=0.032
export MUON_MOMENTUM=0.99
export MUON_MOMENTUM_WARMUP_START=0.92
export MUON_MOMENTUM_WARMUP_STEPS=1500
export WARMDOWN_ITERS=3200
export MUON_WEIGHT_DECAY=0.095
export GRAD_CLIP_NORM=0.3
export EMA_DECAY=0.997
export QAT_THRESHOLD=0.15
export BIGRAM_VOCAB_SIZE=10240
export BIGRAM_DIM=128
export TRAIN_SEQ_LEN=2048
export TRAIN_BATCH_TOKENS=786432
export MAX_WALLCLOCK_SECONDS=540
export ITERATIONS=99999
export TORCH_COMPILE=1
export USE_BF16=1
export COMPILE_BACKEND=inductor
export VOCAB_SIZE=0

# Architecture / attention optimizations (frontier consensus)
export QK_GAIN_INIT=5.25
export XSA_LAST_N=11
export ROPE_DIMS=16
export LN_SCALE=1
export LEAKY_RELU_ALPHA=0.5
export BIGRAM_VOCAB_SIZE=10240
export MLP_MULT=3

# Frontier features
export USE_SDCLIP=1
export SDCLIP_K=12.85
export SDCLIP_HESSIAN_LAMBDA=0.0
export USE_BROTLI=0
export SKIP_GATE_TYPE=sigmoid
export USE_MUONEQR=1
export PARALLEL_START_LAYER=7
export INT5_NAME_PATTERNS=".fc.,.proj."

# Depth recurrence (11 physical → 17 virtual: encoder [0,1,2,3,4,5,3,4], decoder [5,3,4,5,6,7,8,9,10])
export ENCODER_SCHEDULE="0,1,2,3,4,5,3,4"
export DECODER_SCHEDULE="5,3,4,5,6,7,8,9,10"
export RECUR_START_FRAC=0.0

# TTT (Test-Time Training) for final evaluation
export TTT_ENABLED=0
export TTT_LR=0.005
export TTT_EPOCHS=3
export TTT_CHUNK_TOKENS=32768

# FlashAttention-2 is enabled by default in single-h100; keep it on for 8xH100.
export USE_FLASH_ATTN=1

torchrun \
    --standalone \
    --nproc_per_node=8 \
    train_gpt.py "$@"
