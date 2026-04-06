# OpenAI Parameter Golf Submission

**Competition:** OpenAI Parameter Golf - Train the best language model that fits in 16MB  
**Submitted by:** ryukijano (gyanateet@gmail.com)  
**Date:** April 2026  

## Results

| Metric | Value |
|--------|-------|
| **Validation BPB** | **1.3066** |
| **Training Time** | 600 seconds (10 minutes) |
| **Steps** | 1,777 |
| **Model Parameters** | 17,059,912 |
| **Compressed Size** | 14.65 MB |

## Features

- **FlashAttention-3** - Hopper-optimized attention (~75% FLOP util)
- **Exclusive Self Attention (XSA)** - arXiv:2603.09078 on last 3 layers
- **U-Net Transformer** - Skip connections, GQA (8Q/4KV heads)
- **Muon Optimizer** - Newton-Schulz orthogonalization
- **int8 Quantization** - Per-row quantization + zlib compression

## Usage

```bash
# Training with FA3 + XSA
USE_FA3=1 XSA_LAST_N=3 torchrun --nproc_per_node=1 train_gpt.py
```

## Files

- `train_gpt.py` - Training script with FA3/XSA
- `final_model.int8.ptz` - Submission artifact (14.65 MB)
- `final_model.pt` - Raw model (67.2 MB)

## Implementation

See code for FlashAttention-3 and XSA implementation details.
