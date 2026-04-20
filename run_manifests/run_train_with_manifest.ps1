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

$ScriptDir = Split-Path -Parent $PSCommandPath
$RepoRoot = Split-Path -Parent $ScriptDir

if ($Ngpus -lt 1) {
    throw "Ngpus must be a positive integer."
}

if ([string]::IsNullOrWhiteSpace($DataPath)) {
    throw "DataPath is required."
}
if ([string]::IsNullOrWhiteSpace($TokenizerPath)) {
    throw "TokenizerPath is required."
}
if ([string]::IsNullOrWhiteSpace($TrainPattern)) {
    $TrainPattern = Join-Path $DataPath "fineweb_train_*.bin"
}
if ([string]::IsNullOrWhiteSpace($ValPattern)) {
    $ValPattern = Join-Path $DataPath "fineweb_val_*.bin"
}

$Output = if ([System.IO.Path]::IsPathRooted($Output)) { $Output } else { Join-Path $RepoRoot $Output }
$LogPath = if ([System.IO.Path]::IsPathRooted($LogPath)) { $LogPath } else { Join-Path $RepoRoot $LogPath }
$DataPath = if ([System.IO.Path]::IsPathRooted($DataPath)) { $DataPath } else { Join-Path $RepoRoot $DataPath }
$TokenizerPath = if ([System.IO.Path]::IsPathRooted($TokenizerPath)) { $TokenizerPath } else { Join-Path $RepoRoot $TokenizerPath }
$TrainPattern = if ([System.IO.Path]::IsPathRooted($TrainPattern)) { $TrainPattern } else { Join-Path $RepoRoot $TrainPattern }
$ValPattern = if ([System.IO.Path]::IsPathRooted($ValPattern)) { $ValPattern } else { Join-Path $RepoRoot $ValPattern }
$Output = [System.IO.Path]::GetFullPath($Output)
$LogPath = [System.IO.Path]::GetFullPath($LogPath)

if (-not (Test-Path -PathType Container $DataPath)) {
    throw "Data path not found: $DataPath"
}
if (-not (Test-Path -PathType Leaf $TokenizerPath)) {
    throw "Tokenizer not found: $TokenizerPath"
}
if (-not (Get-ChildItem -Path $TrainPattern -ErrorAction SilentlyContinue)) {
    throw "No train shards matched: $TrainPattern"
}
if (-not (Get-ChildItem -Path $ValPattern -ErrorAction SilentlyContinue)) {
    throw "No val shards matched: $ValPattern"
}

New-Item -ItemType Directory -Force -Path (Join-Path $RepoRoot "logs") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $RepoRoot "run_manifests/runs") | Out-Null

$env:RUN_ID = $RunId
$env:WORLD_SIZE = $Ngpus
$env:DATA_PATH = $DataPath
$env:TOKENIZER_PATH = $TokenizerPath
if (-not $env:NCCL_DEBUG) { $env:NCCL_DEBUG = "INFO" }

if ($TrainArgs -and $TrainArgs.Length -gt 0) {
    $command = @($TrainArgs)
} else {
    $command = @("python", "train_gpt.py")
}
if ($Ngpus -gt 1) {
    $command = @(
        "torchrun",
        "--standalone",
        "--nproc_per_node=$Ngpus"
    ) + $command
}

python "$(Join-Path $ScriptDir "create_run_manifest.py")" `
  --stage $Stage `
  --run-id $RunId `
  --output $Output `
  --train-pattern $TrainPattern `
  --val-pattern $ValPattern `
  --tokenizer-path $TokenizerPath `
  --log-path $LogPath `
  --command $command
