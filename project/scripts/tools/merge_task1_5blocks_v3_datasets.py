#!/usr/bin/env python
"""Merges the 5 recording sessions of task1_hybrid_5blocks_v3 into a single 100-episode dataset."""

from pathlib import Path
import os
import shutil
import time
import torch
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset

HF_USER = os.environ.get("HF_USER", "eslab1234")
ROOT = Path.home() / ".cache/huggingface/lerobot" / HF_USER

OUT_NAME = os.environ.get("OUT_NAME", "task1_hybrid_5blocks_v3_100ep_merged")
OUT_REPO_ID = f"{HF_USER}/{OUT_NAME}"
OUT_ROOT = ROOT / OUT_NAME

SOURCE_NAMES = [
    "task1_hybrid_5blocks_v3_20260824_173659",  # 20 ep
    "task1_hybrid_5blocks_v3_20260824_203040",  # 20 ep
    "task1_hybrid_5blocks_v3_20260824_212608",  # 25 ep
    "task1_hybrid_5blocks_v3_20260824_224843",  # 25 ep
    "task1_hybrid_5blocks_v3_20260824_234835",  # 10 ep
]

AUTO_KEYS = {
    "index",
    "episode_index",
    "frame_index",
    "timestamp",
    "task_index",
}

def convert_val(val):
    if hasattr(val, "permute") and val.ndim == 3 and val.shape[0] in [1, 3]:
        return val.permute(1, 2, 0).cpu().numpy()
    elif isinstance(val, np.ndarray) and val.ndim == 3 and val.shape[0] in [1, 3]:
        return np.transpose(val, (1, 2, 0))
    elif hasattr(val, "cpu"):
        return val.cpu().numpy()
    return val

def main():
    print("=" * 80)
    print("📦 [MERGE TASK 1 DATASETS] Merging 5 Sessions into 100-Episode Dataset")
    print("=" * 80)
    print("OUT_REPO_ID:", OUT_REPO_ID)
    print("OUT_ROOT:   ", OUT_ROOT)
    print("=" * 80)

    if OUT_ROOT.exists():
        print(f"⚠️ Removing existing output folder: {OUT_ROOT}")
        shutil.rmtree(OUT_ROOT)

    # 1. Load base dataset
    base_name = SOURCE_NAMES[0]
    base_root = ROOT / base_name
    base = LeRobotDataset(
        repo_id=f"{HF_USER}/{base_name}",
        root=base_root,
        return_uint8=True,
    )

    frame_keys = [k for k in base.features.keys() if k not in AUTO_KEYS]
    task_prompt = "Pick and place 5 blocks in sequence (red, yellow, wood, green, blue)."

    print(f"Base dataset: {base_name}")
    print(f"FPS: {base.fps} | Features: {frame_keys}")

    # 2. Create destination dataset
    dst = LeRobotDataset.create(
        repo_id=OUT_REPO_ID,
        root=OUT_ROOT,
        fps=base.fps,
        features=base.features,
        robot_type=base.meta.robot_type,
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=4,
        encoder_threads=4,
    )

    merged_episodes = 0
    merged_frames = 0
    start_time = time.time()

    try:
        for s_idx, src_name in enumerate(SOURCE_NAMES, 1):
            src_root = ROOT / src_name
            src_repo_id = f"{HF_USER}/{src_name}"

            print("\n" + "=" * 80)
            print(f"📂 [{s_idx}/{len(SOURCE_NAMES)}] Processing {src_name}")
            print("=" * 80)

            if not src_root.exists():
                raise FileNotFoundError(f"Source folder not found: {src_root}")

            src = LeRobotDataset(
                repo_id=src_repo_id,
                root=src_root,
                return_uint8=True,
            )

            print(f"Source episodes: {src.num_episodes} | frames: {src.num_frames}")

            for ep_idx in range(src.num_episodes):
                ep_ds = LeRobotDataset(
                    repo_id=src_repo_id,
                    root=src_root,
                    episodes=[ep_idx],
                    return_uint8=True,
                )

                ep_len = len(ep_ds)
                print(f"  -> Merging episode {ep_idx+1:02d}/{src.num_episodes:02d} (global #{merged_episodes+1:03d}) | {ep_len} frames")

                for i in range(ep_len):
                    item = ep_ds[i]
                    frame = {k: convert_val(item[k]) for k in frame_keys}
                    frame["task"] = item.get("task", task_prompt)
                    dst.add_frame(frame)

                dst.save_episode()
                merged_episodes += 1
                merged_frames += ep_len

    finally:
        print("\n" + "=" * 80)
        print("💾 Finalizing merged dataset metadata & stats...")
        dst.finalize()

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print("🎉 [MERGE COMPLETE SUCCESSFUL!]")
    print(f"OUT_REPO_ID:     {OUT_REPO_ID}")
    print(f"OUT_ROOT:        {OUT_ROOT}")
    print(f"Total Episodes:  {merged_episodes} (Expected: 100)")
    print(f"Total Frames:    {merged_frames} ({merged_frames/30.0/60.0:.2f} min)")
    print(f"Time Taken:      {elapsed/60.0:.2f} minutes")
    print("=" * 80)

if __name__ == "__main__":
    main()
