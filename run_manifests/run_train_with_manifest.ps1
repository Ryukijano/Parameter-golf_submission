Param(
    [Parameter(Mandatory=$false)]
    [ValidateSet("Stage-0","Stage-1","Stage-2")]
    [string]$Stage = "Stage-1",

    [Parameter(Mandatory=$false)]
    [string]$RunId = ("stage1_" + (Get-Date -Format "yyyyMMdd_HHmmss")),

    [Parameter(Mandatory=$false)]
    [string]$Output = ("run_manifests/runs/{0}.json" -f $RunId),

    [Parameter(Mandatory=$false)]
    [string]$LogPath = ("logs/{0}.txt" -f $RunId),

    [Parameter(Mandatory=$false)]
    [string]$DataPath = "./data/datasets/fineweb10B_sp1024",

    [Parameter(Mandatory=$false)]
    [string]$TokenizerPath = "./data/tokenizers/fineweb_1024_bpe.model",

    [Parameter(Mandatory=$false)]
    [string]$TrainPattern = "",

    [Parameter(Mandatory=$false)]
    [string]$ValPattern = "",

    [Parameter(Mandatory=$false)]
    [int]$Ngpus = 1,

    [Parameter(ValueFromRemainingArguments=$true)]
    [string[]]$TrainArgs
)

if ([string]::IsNullOrWhiteSpace($TrainPattern)) {
    $TrainPattern = Join-Path $DataPath "fineweb_train_*.bin"
}
if ([string]::IsNullOrWhiteSpace($ValPattern)) {
    $ValPattern = Join-Path $DataPath "fineweb_val_*.bin"
}

New-Item -ItemType Directory -Force -Path "logs" | Out-Null
New-Item -ItemType Directory -Force -Path "run_manifests/runs" | Out-Null

$env:RUN_ID = $RunId
$env:WORLD_SIZE = $Ngpus
$env:DATA_PATH = $DataPath
$env:TOKENIZER_PATH = $TokenizerPath

$baseCommand = @("python", "train_gpt.py")
if ($Ngpus -gt 1) {
    $command = @(
        "torchrun",
        "--standalone",
        "--nproc_per_node=$Ngpus",
        "train_gpt.py"
    )
} else {
    $command = $baseCommand
}
if ($TrainArgs -ne $null -and $TrainArgs.Length -gt 0) {
    $command += $TrainArgs
}

python run_manifests/create_run_manifest.py `
  --stage $Stage `
  --run-id $RunId `
  --output $Output `
  --train-pattern $TrainPattern `
  --val-pattern $ValPattern `
  --tokenizer-path $TokenizerPath `
  --log-path $LogPath `
  --command $command
