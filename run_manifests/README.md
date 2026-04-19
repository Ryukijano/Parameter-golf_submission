# Run manifest enforcement

This folder contains the lightweight manifest process for all planned
`train_gpt.py` experiments.

## Files

- `manifest_template.json`: field-complete schema for required run metadata.
- `create_run_manifest.py`: generates one manifest JSON and captures all
  reproducibility-critical details.
- `run_train_with_manifest.sh`: Linux launcher that writes manifest then runs
  `train_gpt.py`.
- `run_train_with_manifest.ps1`: Windows launcher for the same flow.

## Rule

No Stage-0/Stage-1/Stage-2 run is considered valid unless it has a matching
manifest file that records:

- git commit, working tree status, and branch
- data shard paths
- key model/optimizer hyperparameters
- library versions
- checks: NaN-free, Inf-free, quant roundtrip, deterministic rerun
- final artifact paths and checksums

## Recommended workflow

1. Prepare stage-specific arguments in `STAGE`, `RUN_ID`, and `OUTPUT`.
2. Start training with one of the launchers.
3. After run, fill `results` and `checks` in the generated JSON.
4. Save manifest + log together with tensorboard/console artifacts.

## Stage-0 smoke setup on local 4090

Generate deterministic smoke assets (dataset + tokenizer):

```bash
python run_manifests/prepare_stage0_smoke_data.py
```

Run a short 4090 smoke with full checks enforced:

```bash
$env:STAGE="Stage-0"; $env:RUN_ID="smoke_4090_01"; $env:MANIFEST_OUT="run_manifests/runs/smoke_4090_01.json"; $env:WORLD_SIZE=1; `
python run_manifests/create_run_manifest.py --stage Stage-0 --run-id smoke_4090_01 --output $env:MANIFEST_OUT --train-pattern smoke_stage0/data/datasets/fineweb10B_sp1024/fineweb_train_*.bin --val-pattern smoke_stage0/data/datasets/fineweb10B_sp1024/fineweb_val_*.bin --tokenizer-path smoke_stage0/data/tokenizers/fineweb_1024_bpe.model --iterations 20 --train-batch-tokens 16384 --train-seq-len 1024 --val-loss-every 0 --world-size 1 --local-rank 0
python run_manifests/run_train_with_manifest.ps1 -Stage Stage-0 -RunId smoke_4090_01 -ManifestOut $env:MANIFEST_OUT -TrainArgs @("python","train_gpt.py")
```

Set `PG_SMOKE_VOCAB_SIZE`, `PG_SMOKE_TRAIN_TOKENS`, and `PG_SMOKE_VAL_TOKENS` when regenerating smoke assets.
