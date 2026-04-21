# OpenAI Parameter Golf — Optimized Submission

**Competition:** [OpenAI Parameter Golf](https://github.com/openai/parameter-golf) — Train the best language model that fits in 16 MB  
**Submitted by:** ryukijano (gyanateet@gmail.com)  
**Target hardware:** 8×H100 SXM, 600 s training + 600 s eval budget  
**Expected BPB on 8×H100:** ~1.09–1.12 (based on leaderboard ablations)

---

## Technique Stack & Expected Impact

Every technique below has a measured BPB gain from official leaderboard PRs.

| Technique | BPB gain | Source |
|---|---|---|
| 11 layers (vs 9) + MLP 3× (vs 2×) | −0.08 | PR #70 |
| INT6 + zstd-22 (vs INT8 + zlib-9) | −0.05 to −0.07 | PR #70, enables bigger model |
| Sliding window eval, stride=64 | −0.034 | PR #77 |
| SmearGate + OrthoInit | −0.017 | PR #65 |
| BigramHashEmbedding (2048 buckets) | −0.012 | PR #76 |
| Muon weight decay = 0.04 | −0.007 | PR #60 |
| EMA decay = 0.997 | −0.0006 | 1.1233 record |
| GPTQ-lite (5-candidate clip search) | −0.0006 | 1.1233 record |
| OrthoInit (orthogonal_ + muP scale) | ≈−0.003 | PR #162 |

**Cumulative expected improvement over original 1.3066 baseline: ~−0.18 BPB**

---

## Architecture

```
GPT (U-Net Transformer)
├── tok_emb          : Embedding(1024, 512)          — tied to lm_head
├── bigram           : BigramHashEmbedding(2048→128→512) — bigram context
├── smear            : SmearGate(512)                — token blending gate
├── blocks[0..5]     : Encoder  (6 layers, store skip)
├── blocks[6..10]    : Decoder  (5 layers, consume skip in reverse)
└── final_norm + softcap(30.0) logits
```

**Per Block:**
- `RMSNorm → CausalSelfAttention (GQA, 8Q/4KV, RoPE, cuDNN SDPA)`
- `RMSNorm → MLP (relu², 3× expansion: 512→1536→512)`
- Learned `attn_scale`, `mlp_scale`, `resid_mix` per layer (fp32)

**Key design choices:**
- **GQA (8Q / 4KV heads):** 2× KV cache reduction, same quality
- **U-Net skip connections:** later decoder layers receive earlier encoder activations, weighted by learned `skip_weights`
- **SmearGate:** `x = (1-g)*x + g*x_prev` — 512 learned scalars, zero-initialized; injects previous-token context before transformer layers
- **BigramHashEmbedding:** `hash = XOR(36313*t[i], 27191*t[i-1]) % 2047`; zero-initialized, scale=0.05; adds pair context on top of unigram embeddings
- **OrthoInit:** all Linear weights ≥64×64 initialized orthogonally; output projections additionally scaled by `1/sqrt(2*num_layers)` (muP)
- **cuDNN SDPA backend:** `torch.nn.functional.scaled_dot_product_attention` with cuDNN on H100 for faster attention

---

## Optimizer

| Parameter group | Optimizer | LR | Notes |
|---|---|---|---|
| `tok_emb.weight` | Adam (fused) | 0.035 | Tied embedding |
| Matrix params (2D, blocks) | **Muon** | 0.025 | Newton-Schulz orthogonalization, WD=0.04 |
| Scalar/vector params | Adam (fused) | 0.025 | WD=0 |

**Muon details:**
- Momentum: 0.92 → 0.99 warm-up over 1500 steps
- Weight decay applied as `p *= (1 - wd * lr)` after each update
- 5 Newton-Schulz backend steps

**LR schedule:** cosine warmdown — flat for first `(iterations - warmdown_iters)` steps, then linearly decays to 0 over `warmdown_iters=3500` steps.

---

## Quantization & Compression

### INT6 + zstd-22 (replaces INT8 + zlib-9)

Matrix weights (2D, large) are quantized to INT6 range `[-32, 31]` (64 levels) using **GPTQ-lite** per-row clip search:

```
For each of 5 clip percentiles [99.9%, 99.95%, 99.99%, 99.999%, 100%]:
    clip_abs = quantile(|W|, q, dim=1)
    scale = clip_abs / 31
    W_q = round(clip(W, -clip_abs, clip_abs) / scale)
    MSE = mean((W - W_q * scale)^2, dim=1)
Pick the percentile with minimum per-row MSE.
```

Resulting `torch.int8` tensor (values in `[-32, 31]`) + fp16 per-row scale are serialized via `torch.save`, then compressed with `zstandard` at level 22.

**Size comparison (26.5M param model):**
- BF16 raw: ~53 MB
- INT8 + zlib-9: ~18–20 MB  
- **INT6 + zstd-22 (EMA): ~15.5 MB on 8×H100 converged run** ✅ under 16 MB

Small tensors (≤65,536 elements) and control tensors (`q_gain`, `skip_weights`, etc.) are kept in fp16/fp32 passthrough — no quantization noise on sensitive parameters.

### EMA Export

The serialized artifact uses **EMA weights** (decay=0.997, updated every step), not the live optimizer weights. EMA-averaged weights are smoother → compress ~4–8% better under zstd-22, and also evaluate better.

### Late QAT

During the final `qat_threshold=15%` of training (when LR scale < 0.15), INT6 fake-quantization (STE) is applied to all 2D parameter matrices each step — forcing the optimizer to find solutions that are robust to INT6 rounding. Note: `torch.compile` may constant-fold this; verified functional in eager mode.

---

## Sliding Window Evaluation

Final BPB is computed with overlapping windows of `seq_len=2048`, stride=64:

- Each window of 2048 tokens runs one forward pass
- Only the **last 64 tokens** of each window are scored (to avoid double-counting)
- Every scored token therefore has up to 1984 tokens of context
- Windows are batched 32 at a time for efficiency
- Capped at 4M validation tokens (~90 seconds on 1×H100)

This is ~0.034 BPB better than non-overlapping evaluation because later tokens in long sequences get more context.

---

## Usage

### Minimal run (single GPU, for testing)
```bash
DATA_PATH=/path/to/fineweb10B_sp1024 \
TOKENIZER_PATH=/path/to/fineweb_1024_bpe.model \
python3 train_gpt.py
```

### Competition run (8×H100, 10 minutes)
```bash
DATA_PATH=/path/to/fineweb10B_sp1024 \
TOKENIZER_PATH=/path/to/fineweb_1024_bpe.model \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
MUON_WEIGHT_DECAY=0.04 GRAD_CLIP_NORM=0.3 \
EMA_DECAY=0.997 QAT_THRESHOLD=0.15 \
BIGRAM_VOCAB_SIZE=2048 BIGRAM_DIM=128 \
TRAIN_SEQ_LEN=2048 TRAIN_BATCH_TOKENS=786432 \
MAX_WALLCLOCK_SECONDS=600 ITERATIONS=99999 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Single H100 run (adapted from 4090 sweep results)
```bash
DATA_PATH=/path/to/fineweb10B_sp1024 \
TOKENIZER_PATH=/path/to/fineweb_1024_bpe.model \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3500 \
MUON_WEIGHT_DECAY=0.04 GRAD_CLIP_NORM=0.3 \
EMA_DECAY=0.997 QAT_THRESHOLD=0.15 \
BIGRAM_VOCAB_SIZE=2048 BIGRAM_DIM=128 \
TRAIN_SEQ_LEN=2048 TRAIN_BATCH_TOKENS=524288 \
MAX_WALLCLOCK_SECONDS=600 ITERATIONS=99999 \
COMPILE_BACKEND=inductor \
python3 train_gpt.py
```

### Key environment variables

| Variable | Default | Description |
|---|---|---|
| `NUM_LAYERS` | 11 | Transformer depth |
| `MLP_MULT` | 3 | MLP hidden = mult × model_dim |
| `BIGRAM_VOCAB_SIZE` | 2048 | BigramHash table size (0 = disabled) |
| `BIGRAM_DIM` | 128 | BigramHash embedding dim before projection |
| `EMA_DECAY` | 0.997 | EMA shadow model decay rate |
| `QAT_THRESHOLD` | 0.15 | Late QAT activates when LR scale < this |
| `MUON_WEIGHT_DECAY` | 0.04 | Muon optimizer weight decay |
| `WARMDOWN_ITERS` | 3500 | Steps over which LR decays to 0 |
| `TRAIN_SEQ_LEN` | 2048 | Sequence length |
| `TRAIN_BATCH_TOKENS` | 786432 | Global tokens per step |
| `MAX_WALLCLOCK_SECONDS` | 600 | Hard training time cap |
| `SEED` | 1337 | RNG seed |
| `TORCH_COMPILE` | 1 | Enable `torch.compile` (0 = eager) |
| `USE_BF16` | 1 | Use bfloat16 for activations |
| `COMPILE_BACKEND` | inductor | `torch.compile` backend: inductor or eager |
| `VOCAB_SIZE` | 0 | Auto-detect from tokenizer if ≤0 |

---

## Files

| File | Description |
|---|---|
| `train_gpt.py` | Main training + quantization + evaluation script |
| `final_model.int6.ptz` | Submission artifact: INT6 + zstd-22, EMA weights |
| `final_model.pt` | Raw BF16 model state dict (for debugging) |
| `logs/` | Per-run training logs (UUID-named) |
| `cuda_trainer/` | Experimental CUDA trainer (work in progress) |

---

## Improvement History (this repo)

| Commit | Change | Expected Δ BPB |
|---|---|---|
| `1a4e2bd` | Baseline: 9L, MLP2×, INT8+zlib, 1,777 steps | 1.3066 |
| `3eed342` | 11L, MLP3×, INT6+zstd22, EMA, QAT, cuDNN SDPA, Muon WD, GPTQ-lite | −0.15 |
| `ee58af0` | Fix batched sliding-window eval (was hanging) | — |
| `3281f78` | SmearGate + BigramHash(2048) + OrthoInit | −0.029 |

**Expected final BPB on 8×H100: ~1.09–1.12**

---

## References

- [Parameter Golf leaderboard](https://parameter-golf.github.io/)
- [1.1233 record README](https://github.com/openai/parameter-golf/blob/main/records/track_10min_16mb/2026-03-22_11L_EMA_GPTQ-lite_warmdown3500_QAT015_1.1233/README.md)
- [SmearGate + BigramHash + OrthoInit (PR #65/#162)](https://github.com/openai/parameter-golf/pull/65)
- [Int5/Int6 mixed + SWA (PR #76)](https://github.com/openai/parameter-golf/blob/main/records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/README.md)
- [Partial RoPE + LN Scale record (1.1248)](https://github.com/openai/parameter-golf/blob/main/records/track_10min_16mb/2026-03-21_11L_XSA4_EMA_PartialRoPE_LateQAT_1.1248/README.md)
- [Muon optimizer](https://github.com/KellerJordan/Muon)
- [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)
