# Parameter Golf Optimization Summary

## Objective
Optimize the Parameter Golf model to achieve val_bpb of approximately 1.08 using the "Maximum effort" approach with Full TTT during eval.

## Implemented Optimizations

### 1. Torch.compile Recompile Bug Fix
- **Problem**: `training_step` was a Python int attribute that changed every step, causing torch.compile to recompile the model repeatedly (~234ms/step)
- **Solution**: Moved schedule switching to a dedicated `update_schedule()` method called outside the compiled region
- **Result**: Step time improved from 234ms to 113ms (52% faster)

### 2. SP8192 Tokenizer + Dataset
- Switched from SP4096 to SP8192 tokenizer from `kevclark/parameter-golf` HuggingFace repo
- Data path: `./data/datasets/datasets/fineweb10B_sp8192`
- Tokenizer path: `./data/datasets/tokenizers/fineweb_8192_bpe.model`

### 3. Tuned Hyperparameters
| Parameter | Value |
|-----------|-------|
| Matrix LR | 0.022 |
| Scalar LR | 0.022 |
| Tied Embed LR | 0.032 |
| Weight Decay | 0.095 |
| EMA Decay | 0.9965 |
| Warmdown Iters | 2880 (72% of ~4000 steps) |
| QAT Threshold | 0.15 |
| Bigram Vocab Size | 10240 |

### 4. Progressive 3-Layer Recurrence
- Encoder schedule: `0,1,2,3,4,5,3,4` (8 virtual layers from 6 physical)
- Decoder schedule: `5,3,4,5,6,7,8,9,10` (9 virtual layers from 6 physical)
- Activation: `RECUR_START_FRAC=0.35` (~35% of training progress)

### 5. Legal Score-First TTT (Test-Time Training)
- Enabled via `TTT_ENABLED=1`
- Parameters: LR=0.005, Epochs=3, Chunk=32K tokens
- Integrated `eval_val_sliding_ttt()` function for final evaluation

### 6. Hessian-Aware SDClip
- Added `sdclip_hessian_lambda=0.175` parameter
- Uses gradient variance as proxy for Hessian diagonal to adjust clip factors
- Applied during Late QAT when LR scale < qat_threshold

## Results

| Metric | Value |
|--------|-------|
| Best val_bpb | 1.1555 |
| Step time | ~113-150ms |
| Submission size | ~12.25 MB (int6+zstd22) |
| Training steps | ~4000 (600s wallclock) |

## Files Modified
- `train_gpt.py` - Core training script with all optimizations
- `run_8xh100.sh` - 8-GPU run script with tuned hyperparameters

## Remaining Opportunities for Target ~1.08
1. Longer training time to enable full TTT evaluation
2. Further hyperparameter tuning (learning rates, warmdown fraction)
3. Additional regularization techniques
4. Ensemble or model merging strategies

## Git Commit
```
Optimize Parameter Golf: SP8192, TTT, Hessian-aware SDClip, tuned hyperparams

- Fix torch.compile recompile bug (training_step -> update_schedule)
- SP8192 tokenizer + dataset from kevclark/parameter-golf
- Tuned hyperparams: WD=0.095, MLR=0.022, EMA=0.9965, WARMDOWN=2880
- Progressive 3-layer recurrence with RECUR_START_FRAC=0.35
- Legal Score-First TTT evaluation (TTT_ENABLED=1)
- Hessian-aware SDClip (sdclip_hessian_lambda=0.175)
- Step time improved: 234ms -> 113ms (52% faster)
- Best val_bpb: 1.1555 (target ~1.08)
```

Branch: `h100`
