#!/usr/bin/env python3
"""Download SP4096 dataset and tokenizer from HuggingFace for Parameter Golf.

Usage:
    export HF_TOKEN=your_hf_token_here
    python3 download_hf_data.py

Requires: pip install huggingface-hub
"""
import os
from huggingface_hub import snapshot_download, hf_hub_download

REPO_ID = "kevclark/parameter-golf"
DATA_DIR = "./data"

def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("WARNING: HF_TOKEN not set. Using unauthenticated download (may hit rate limits).")
    
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Download dataset shards
    print(f"Downloading SP4096 dataset from {REPO_ID}...")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=["datasets/datasets/fineweb10B_sp4096/*"],
        local_dir=DATA_DIR,
        token=token,
    )
    
    # Download tokenizer
    print("Downloading SP4096 tokenizer...")
    hf_hub_download(
        repo_id=REPO_ID,
        filename="datasets/tokenizers/fineweb_4096_bpe.model",
        repo_type="dataset",
        local_dir=DATA_DIR,
        token=token,
    )
    
    print("Done. Files saved to ./data/")

if __name__ == "__main__":
    main()
