#!/usr/bin/env python3
"""Run `train_gpt.py` with manifest capture and runtime health checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import time
import glob
from datetime import datetime, timezone
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent


def read_template() -> dict[str, Any]:
    template_path = Path(__file__).with_name("manifest_template.json")
    with open(template_path, "r", encoding="utf-8") as f:
        return json.load(f)


def git_info() -> dict[str, Any]:
    def _git(cmd: list[str]) -> str:
        try:
            out = subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=5)
            return out.strip()
        except Exception:
            return ""

    return {
        "commit": _git(["git", "rev-parse", "HEAD"]),
        "dirty": _git(["git", "status", "--porcelain"]) != "",
        "remote": _git(["git", "config", "--get", "remote.origin.url"]),
        "status": _git(["git", "status", "--short"]),
    }


def resolve_paths(path_pattern: str) -> tuple[list[str], int]:
    pattern = Path(path_pattern)
    if not pattern.is_absolute():
        pattern = REPO_ROOT / pattern
    files = sorted(glob.glob(str(pattern)))
    return files, len(files)


def manifest_data_hash(file_list: list[str]) -> str:
    hasher = hashlib.sha256()
    for path in sorted(file_list):
        hasher.update(path.encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def run_and_capture(command: list[str], log_path: Path, run_cwd: Path | None = None) -> tuple[int, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    with open(log_path, "w", encoding="utf-8") as f:
        working_dir = run_cwd or REPO_ROOT
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=os.environ.copy(),
            cwd=str(working_dir),
        )
        output_lines: list[str] = []
        assert process.stdout is not None
        try:
            for line in process.stdout:
                f.write(line)
                print(line, end="")
                output_lines.append(line)
        finally:
            process.stdout.close()
        rc = process.wait(timeout=None)
        elapsed = time.time() - start
        f.write(f"\nTRAIN_EXIT_CODE:{rc}\n")
        f.write(f"TRAIN_RUNTIME_SECONDS:{elapsed:.3f}\n")
    return rc, "".join(output_lines)


def parse_results(log_text: str) -> dict[str, Any]:
    def _find_float(pattern: str) -> float | None:
        m = re.search(pattern, log_text)
        if not m:
            return None
        try:
            return float(m.group(1))
        except Exception:
            return None

    val_bpb = _find_float(r"final_int6_zstd22_roundtrip val_loss:[0-9.]+\s+val_bpb:([0-9.]+)")
    if val_bpb is None:
        val_bpb = _find_float(r"final_int6_zstd22_roundtrip_exact val_loss:[0-9.]+\s+val_bpb:([0-9.]+)")
    train_loss = _find_float(r"step:\d+/\d+ train_loss:([0-9.]+)")
    if train_loss is not None:
        # pick last logged value
        for m in re.finditer(r"step:\d+/\d+ train_loss:([0-9.]+)", log_text):
            train_loss = float(m.group(1))
    step_avg = _find_float(r"step_avg:([0-9.]+)ms")
    if step_avg is None:
        step_avg = _find_float(r"train_loss:[0-9.]+\s+train_time:[0-9]+ms\s+step_avg:([0-9.]+)")
    peak_mem = _find_float(r"peak memory allocated:\s*([0-9]+)\s*MiB")

    checks = {
        "nan_free": bool(not re.search(r"\bnan\b", log_text, flags=re.I)),
        "inf_free": bool(not re.search(r"\binf\b", log_text, flags=re.I)),
        "roundtrip_ok": bool(re.search(r"final_int6_zstd22_roundtrip", log_text)),
        "deterministic_repro_ok": None,
    }

    return {
        "results": {
            "val_bpb": val_bpb,
            "val_loss_final": _find_float(r"final_int6_zstd22_roundtrip val_loss:([0-9.]+)"),
            "train_loss_final": train_loss,
            "step_latency_ms_avg": step_avg,
            "artifact_bytes": _find_float(r"Serialized model int6\+zstd22 \(EMA\):\s+([0-9]+)"),
            "peak_mem_mib": peak_mem,
            "notes": "",
        },
        "checks": checks,
    }


def _to_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    if isinstance(raw, str):
        raw = raw.strip()
    if raw == "":
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _to_env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    if isinstance(raw, str):
        raw = raw.strip()
    if raw == "":
        return default
    try:
        return float(raw)
    except Exception:
        return default


def _normalize_compile_backend(value: str | None) -> str:
    raw = "" if value is None else value
    normalized = raw.strip().strip("'\"").lower()
    if normalized in {"inductor", "eager"}:
        return normalized
    return "inductor"


def _to_repo_path(path_value: str) -> str:
    candidate = Path(path_value)
    if candidate.is_absolute():
        return str(candidate.as_posix())
    return str((REPO_ROOT / candidate).as_posix())


def build_manifest(args: argparse.Namespace, command: list[str], log_path: Path, exit_code: int, log_text: str) -> dict[str, Any]:
    m = read_template()
    host = platform.uname()
    try:
        import torch  # type: ignore

        torch_version = getattr(torch, "__version__", "")
        torch_cuda = getattr(torch.version, "cuda", "")
    except Exception:
        torch_version = ""
        torch_cuda = ""

    train_files, train_count = resolve_paths(args.train_pattern)
    val_files, val_count = resolve_paths(args.val_pattern)

    m["run_id"] = args.run_id
    m["timestamp_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    m["run_stage"] = args.stage
    m["branch"] = _git_field(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    m["git"] = git_info()
    m["host"] = {
        "hostname": host.node,
        "platform": host.system + "-" + host.release,
        "python": platform.python_version(),
        "cwd": str(REPO_ROOT),
    }

    m["environment"] = {
        "stage": args.stage,
        "command": shlex.join(command),
        "seed": _to_env_int("SEED", 1337),
        "world_size": _to_env_int("WORLD_SIZE", 1),
        "local_rank": _to_env_int("LOCAL_RANK", 0),
        "max_wallclock_seconds": _to_env_float("MAX_WALLCLOCK_SECONDS", 600),
        "train_batch_tokens": _to_env_int("TRAIN_BATCH_TOKENS", 524288),
        "train_seq_len": _to_env_int("TRAIN_SEQ_LEN", 1024),
        "iterations": _to_env_int("ITERATIONS", 20000),
        "warmdown_iters": _to_env_int("WARMDOWN_ITERS", 1200),
        "val_loss_every": _to_env_int("VAL_LOSS_EVERY", 1000),
        "train_log_every": _to_env_int("TRAIN_LOG_EVERY", 200),
        "warmup_steps": _to_env_int("WARMUP_STEPS", 20),
        "model_dim": _to_env_int("MODEL_DIM", 512),
        "num_layers": _to_env_int("NUM_LAYERS", 11),
        "num_heads": _to_env_int("NUM_HEADS", 8),
        "num_kv_heads": _to_env_int("NUM_KV_HEADS", 4),
        "mlp_mult": _to_env_int("MLP_MULT", 3),
        "qk_gain_init": _to_env_float("QK_GAIN_INIT", 1.5),
        "rope_base": _to_env_float("ROPE_BASE", 10000.0),
        "logit_softcap": _to_env_float("LOGIT_SOFTCAP", 30.0),
        "ema_decay": _to_env_float("EMA_DECAY", 0.997),
        "qat_threshold": _to_env_float("QAT_THRESHOLD", 0.15),
        "embed_lr": _to_env_float("EMBED_LR", 0.035),
        "head_lr": _to_env_float("HEAD_LR", 0.008),
        "tied_embed_lr": _to_env_float("TIED_EMBED_LR", 0.05),
        "matrix_lr": _to_env_float("MATRIX_LR", 0.04),
        "scalar_lr": _to_env_float("SCALAR_LR", 0.04),
        "muon_momentum": _to_env_float("MUON_MOMENTUM", 0.95),
        "muon_momentum_warmup_start": _to_env_float("MUON_MOMENTUM_WARMUP_START", 0.85),
        "muon_momentum_warmup_steps": _to_env_int("MUON_MOMENTUM_WARMUP_STEPS", 500),
        "muon_weight_decay": _to_env_float("MUON_WEIGHT_DECAY", 0.04),
        "grad_clip_norm": _to_env_float("GRAD_CLIP_NORM", 0.3),
        "torch_compile": _to_env_int("TORCH_COMPILE", 1),
        "compile_backend_requested": os.environ.get("COMPILE_BACKEND", "inductor"),
        "compile_backend": _normalize_compile_backend(os.environ.get("COMPILE_BACKEND", "inductor")),
        "xsa_last_n": _to_env_int("XSA_LAST_N", 0),
        "data_path": _to_repo_path(os.environ.get("DATA_PATH", str(REPO_ROOT / "data/datasets/fineweb10B_sp1024"))),
        "tokenizer_path": args.tokenizer_path,
        "train_pattern": _to_repo_path(args.train_pattern),
        "val_pattern": _to_repo_path(args.val_pattern),
        "train_files": [_to_repo_path(p) for p in train_files],
        "val_files": [_to_repo_path(p) for p in val_files],
        "train_batch_count": train_count,
        "val_batch_count": val_count,
        "torch_version": torch_version,
        "torch_cuda": torch_cuda,
        "numpy_version": _pip_version("numpy"),
        "sentencepiece_version": _pip_version("sentencepiece"),
        "zstandard_version": _pip_version("zstandard"),
        "libcudart": os.environ.get("LD_LIBRARY_PATH", ""),
    }

    m["artifacts"] = {
        "final_model_raw": "final_model.pt",
        "final_model_quant": "final_model.int6.ptz",
        "raw_bytes": _size_bytes("final_model.pt"),
        "quant_bytes": _size_bytes("final_model.int6.ptz"),
        "data_manifest_hash": manifest_data_hash(train_files + val_files),
        "log_path": str(log_path.as_posix()),
        "manifest_path": args.output,
    }

    parsed = parse_results(log_text)
    m["results"] = parsed["results"]
    m["checks"] = parsed["checks"]
    if parsed["results"]["artifact_bytes"] is None and m["artifacts"]["quant_bytes"]:
        m["results"]["artifact_bytes"] = float(m["artifacts"]["quant_bytes"])

    m["artifacts"]["exit_code"] = exit_code
    return m


def _git_field(args_list: list[str]) -> str:
    try:
        return subprocess.check_output(args_list, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def _pip_version(pkg: str) -> str:
    try:
        import importlib

        mod = importlib.import_module(pkg)
        return getattr(mod, "__version__", "")
    except Exception:
        return ""


def _size_bytes(path: str) -> int:
    p = Path(path)
    try:
        return p.stat().st_size
    except Exception:
        return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run train_gpt.py and capture manifest.")
    p.add_argument("--stage", required=True, choices=["Stage-0", "Stage-1", "Stage-2"])
    p.add_argument("--run-id", dest="run_id", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--train-pattern", required=True)
    p.add_argument("--val-pattern", required=True)
    p.add_argument("--tokenizer-path", required=True)
    p.add_argument("--train-id", dest="train_id", default=None)
    p.add_argument(
        "--command",
        nargs="+",
        required=True,
        help="Command to execute, e.g. python3 train_gpt.py",
    )
    p.add_argument("--log-path", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    output_path = Path(_to_repo_path(args.output))
    log_path = Path(_to_repo_path(args.log_path)) if args.log_path else REPO_ROOT / "logs" / f"{args.run_id}.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = read_template()
    manifest["run_id"] = args.run_id
    manifest["timestamp_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest["run_stage"] = args.stage
    manifest["environment"] = manifest.get("environment", {})
    manifest["environment"]["command"] = shlex.join(args.command)
    manifest["artifacts"] = manifest.get("artifacts", {})
    manifest["artifacts"]["log_path"] = str(log_path.as_posix())
    manifest["artifacts"]["manifest_path"] = str(output_path.as_posix())
    manifest["checks"] = manifest.get("checks", {"nan_free": None, "inf_free": None, "roundtrip_ok": None, "deterministic_repro_ok": None})
    manifest["results"] = manifest.get("results", {"val_bpb": None, "val_loss_final": None, "step_latency_ms_avg": None, "train_loss_final": None, "artifact_bytes": None, "peak_mem_mib": None, "notes": ""})
    manifest["git"] = git_info()
    manifest["branch"] = _git_field(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=False)
        f.write("\n")

    rc, log_text = run_and_capture(args.command, log_path, REPO_ROOT)
    final_manifest = build_manifest(args, args.command, log_path, rc, log_text)
    final_manifest["run_stage"] = args.stage

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_manifest, f, indent=2, sort_keys=False)
        f.write("\n")

    if rc != 0:
        raise SystemExit(rc)


if __name__ == "__main__":
    main()
