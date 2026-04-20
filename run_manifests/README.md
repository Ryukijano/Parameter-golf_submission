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

1. Prepare `STAGE`, `RUN_ID`, and `OUTPUT` environment variables.
2. Start training with one launcher:
   - Linux: `run_manifests/run_train_with_manifest.sh`
   - Windows: `run_manifests/run_train_with_manifest.ps1`
3. `create_run_manifest.py` automatically fills `results` and `checks` from `train_gpt.py` log output.
4. Keep `manifests + logs + final int6 artifact` together with the run.

## Linux one-command launch (recommended)

```bash
STAGE=Stage-1 \
RUN_ID=stage1_$(date +%Y%m%d_%H%M%S) \
WORLD_SIZE=1 \
DATA_PATH=/workspace/data/fineweb10B_sp1024 \
TOKENIZER_PATH=/workspace/data/tokenizers/fineweb_1024_bpe.model \
N_GPUS=1 \
NCCL_DEBUG=INFO \
run_manifests/run_train_with_manifest.sh
```

For 8 GPUs on a single node:

```bash
N_GPUS=8 \
WORLD_SIZE=8 \
run_manifests/run_train_with_manifest.sh
```

## Windows launch

```powershell
$env:STAGE="Stage-1"
$env:RUN_ID="stage1_" + (Get-Date -Format "yyyyMMdd_HHmmss")
$env:WORLD_SIZE=1
$env:DATA_PATH=".\data\datasets\fineweb10B_sp1024"
$env:TOKENIZER_PATH=".\data\tokenizers\fineweb_1024_bpe.model"
.\run_manifests\run_train_with_manifest.ps1 -Stage Stage-1 -RunId $env:RUN_ID
```

## Notes

- `N_GPUS` controls both launch parallelism (`torchrun --nproc_per_node`) and manifest `WORLD_SIZE`.
- Pass extra train arguments as wrapper args (for example: `run_train_with_manifest.sh -- --batch-size 64`).
- Use `TRAIN_CMD_STRING="python3 train_gpt.py --help"` if you prefer to define the base command.
- Add a `data_manifest_hash` by regenerating runs with this tool and optional data-shard checksums if needed.
