#!/usr/bin/env python
import os
import sys
from pathlib import Path
from huggingface_hub import HfApi

HF_USER = os.environ.get("HF_USER", "eslab1234")
ROOT = Path.home() / ".cache/huggingface/lerobot" / HF_USER

DATASET_NAME = sys.argv[1] if len(sys.argv) > 1 else "task1_hybrid_5blocks_v3_100ep_merged"
DATASET_PATH = ROOT / DATASET_NAME
REPO_ID = f"{HF_USER}/{DATASET_NAME}"

print("=" * 80)
print("🚀 [UPLOAD DATASET TO HUGGINGFACE HUB]")
print(f"Local Path: {DATASET_PATH}")
print(f"Repo ID:    {REPO_ID}")
print("=" * 80)

if not DATASET_PATH.exists():
    raise SystemExit(f"Dataset path not found: {DATASET_PATH}")

api = HfApi()
print(f"Creating / verifying dataset repo on Hugging Face: {REPO_ID} ...")
api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)

print("Uploading files to Hugging Face Hub...")
api.upload_folder(
    folder_path=str(DATASET_PATH),
    repo_id=REPO_ID,
    repo_type="dataset",
    commit_message=f"Upload merged 100-episode dataset: {DATASET_NAME}",
)
print("=" * 80)
print(f"🎉 Successfully uploaded to https://huggingface.co/datasets/{REPO_ID}")
print("=" * 80)
