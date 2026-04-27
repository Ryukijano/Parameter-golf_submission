#!/usr/bin/env bash
# Run on 5×H100 (requires manual GRAD_ACCUM_STEPS)
set -euo pipefail

# Environment and reproducibility
export SEED="${SEED:-1337}"
export RUN_ID="${RUN_ID:-$(date +%s%N)}"
export DATA_PATH="${DATA_PATH:-./data/datasets/fineweb10B_sp4096}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-./data/tokenizers/fineweb_4096_bpe.model}"

# Training hyperparameters
export MATRIX_LR=0.025
export SCALAR_LR=0.025
export TIED_EMBED_LR=0.035
export MUON_MOMENTUM=0.99
export MUON_MOMENTUM_WARMUP_START=0.92
export MUON_MOMENTUM_WARMUP_STEPS=1500
export WARMDOWN_ITERS=3000
export MUON_WEIGHT_DECAY=0.09
export GRAD_CLIP_NORM=0.3
export EMA_DECAY=0.997
export QAT_THRESHOLD=0.15
export BIGRAM_DIM=128

# Training length and batching
export TRAIN_SEQ_LEN=2048
export TRAIN_BATCH_TOKENS=778240  # 5*2*2048*38 = divisible by all factors
export MAX_WALLCLOCK_SECONDS=600
export ITERATIONS=99999
export TORCH_COMPILE=1
export USE_BF16=1
export COMPILE_BACKEND=inductor
export VOCAB_SIZE=0

# Architecture / attention optimizations (frontier consensus)
export QK_GAIN_INIT=5.25
export XSA_LAST_N=4
export ROPE_DIMS=16
export LN_SCALE=1
export LEAKY_RELU_ALPHA=0.5
export BIGRAM_VOCAB_SIZE=10240
export MLP_MULT=4

# Frontier features
export USE_SDCLIP=1
export SDCLIP_K=12.85
export USE_BROTLI=1
export SKIP_GATE_TYPE=sigmoid
export USE_MUONEQR=1
export PARALLEL_START_LAYER=7
export INT5_NAME_PATTERNS=".fc.,.proj."

# Depth recurrence (11 physical → 17 virtual)
export ENCODER_SCHEDULE="0,1,2,3,4,5,3,4"
export DECODER_SCHEDULE="5,3,4,5,6,7,8,9,10"
export RECUR_START_STEP=3000

# For 5 GPUs, set grad_accum manually (e.g., 2 steps per GPU for effective batch=786432*2/5=314K per GPU)
# Or keep default and adjust TRAIN_BATCH_TOKENS per GPU.
export GRAD_ACCUM_STEPS=2

# FlashAttention-2
export USE_FLASH_ATTN=1

torchrun \
    --standalone \
    --nproc_per_node=5 \
    train_gpt.py "$@"
