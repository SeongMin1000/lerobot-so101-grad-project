#!/usr/bin/env python3
"""
Merges and trims RWFM autonomous rollout datasets for SO-101.

Processes 5 single-color recording sessions per task (10 episodes each = 50 episodes total)
in canonical color sequence: Red -> Yellow -> Wood -> Green -> Blue.

Trims trailing post-grasp stationary hold frames (where follower arm and gripper motion have ceased)
while preserving authoritatively synchronized `meta/rwfm_rollout_annotations.json` segment ranges.

Usage:
  # Task 1 (50 episodes)
  python project/scripts/tools/merge_and_trim_rwfm_rollouts.py --task=task1

  # Task 2 (50 episodes)
  python project/scripts/tools/merge_and_trim_rwfm_rollouts.py --task=task2

  # Both tasks sequentially (+ optional 100ep combined multitask)
  python project/scripts/tools/merge_and_trim_rwfm_rollouts.py --task=both --create-multitask
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

HF_USER = os.environ.get("HF_USER", "eslab1234")
CACHE_BASE = Path.home() / ".cache/huggingface/lerobot" / HF_USER

TASK1_CANONICAL_PROMPT = (
    "Pick up the 5 blocks in sequence (red, yellow, wood, green, blue), "
    "then place each block separately into its designated target position."
)

TASK2_CANONICAL_PROMPT = (
    "Pick up the 5 blocks in sequence (red, yellow, wood, green, blue), "
    "then hover over the target area and stack each block on top of the previous block."
)

# 10 Recorded Datasets (5 Task 1 + 5 Task 2, 10 episodes each)
TASK1_SOURCES = [
    ("red", "smolvla_task1_rwfm_rollout_822ep_50k_v1_20260927_235114"),
    ("yellow", "smolvla_task1_rwfm_rollout_822ep_50k_v1_20260928_001008"),
    ("wood", "smolvla_task1_rwfm_rollout_822ep_50k_v1_20260928_003041"),
    ("green", "smolvla_task1_rwfm_rollout_822ep_50k_v1_20260928_005933"),
    ("blue", "smolvla_task1_rwfm_rollout_822ep_50k_v1_20260928_011319"),
]

TASK2_SOURCES = [
    # Recorded with task1 prefix in repo_id, but designated and prompted as Task 2 Red
    ("red", "smolvla_task1_rwfm_rollout_822ep_50k_v1_20260928_012533"),
    ("yellow", "smolvla_task2_rwfm_rollout_822ep_50k_v1_20260928_014227"),
    ("wood", "smolvla_task2_rwfm_rollout_822ep_50k_v1_20260928_015323"),
    ("green", "smolvla_task2_rwfm_rollout_822ep_50k_v1_20260928_020031"),
    ("blue", "smolvla_task2_rwfm_rollout_822ep_50k_v1_20260928_021122"),
]

AUTO_KEYS = {
    "index",
    "episode_index",
    "frame_index",
    "timestamp",
    "task_index",
}


def convert_val(val: Any) -> Any:
    """Converts torch tensor or numpy array into numpy format suitable for LeRobotDataset."""
    if hasattr(val, "permute") and val.ndim == 3 and val.shape[0] in [1, 3]:
        return val.permute(1, 2, 0).cpu().numpy()
    elif isinstance(val, np.ndarray) and val.ndim == 3 and val.shape[0] in [1, 3]:
        return np.transpose(val, (1, 2, 0))
    elif hasattr(val, "cpu"):
        return val.cpu().numpy()
    return val


def find_rwfm_tail_cut(
    actions: np.ndarray,
    states: np.ndarray,
    buffer_frames: int = 2,
    arm_motion_tol: float = 0.35,
    grip_motion_tol: float = 0.40,
    min_ep_len: int = 30,
) -> Tuple[int, int, str]:
    """
    Identifies trailing stationary hold frames from the end of an RWFM rollout.

    Returns:
        tail_cut: ending frame index (exclusive, slice [0 : tail_cut])
        trimmed: number of trimmed frames (len(actions) - tail_cut)
        reason: diagnostic string
    """
    T = len(actions)
    if T <= min_ep_len:
        return T, 0, f"too_short (len={T})"

    # Follower joint motion deltas and 3-step moving average
    st_arm_d = np.zeros(T)
    st_arm_d[1:] = np.sum(np.abs(states[1:, :5] - states[:-1, :5]), axis=1)
    kernel = np.ones(3) / 3.0
    smooth_st_arm = np.convolve(st_arm_d, kernel, mode="same")

    st_grp_d = np.zeros(T)
    st_grp_d[1:] = np.abs(states[1:, 5] - states[:-1, 5])

    final_grp_st = states[-1, 5]
    is_closed = (final_grp_st <= 28.0)

    # 1. Gripper stability point
    t_grp_stable = T - 1
    if is_closed:
        for t in range(T - 1, 0, -1):
            if states[t, 5] <= max(final_grp_st + 1.5, 26.0) and st_grp_d[t] <= grip_motion_tol:
                t_grp_stable = t
            else:
                break
    else:
        for t in range(T - 1, 0, -1):
            if states[t, 5] >= min(final_grp_st - 1.5, 38.0) and st_grp_d[t] <= grip_motion_tol:
                t_grp_stable = t
            else:
                break

    # 2. Arm stationary point (follower arm motion ceased)
    t_arm_stop = T - 1
    for t in range(T - 1, 0, -1):
        if smooth_st_arm[t] <= arm_motion_tol:
            t_arm_stop = t
        else:
            break

    # Hold begins when BOTH arm stopped and gripper reached final stable state
    hold_start = max(t_grp_stable, t_arm_stop)

    # Preserve buffer_frames of the resting posture
    tail_cut = min(T, max(hold_start + buffer_frames, min_ep_len))
    trimmed = T - tail_cut
    reason = f"hold_start={hold_start}, grp_stable={t_grp_stable}, arm_stop={t_arm_stop}"
    return tail_cut, trimmed, reason


def merge_and_trim_task(
    sources: List[Tuple[str, str]],
    dst_repo_id: str,
    canonical_task: str,
    trim: bool = True,
    buffer_frames: int = 2,
    arm_motion_tol: float = 0.35,
    min_ep_len: int = 30,
    push_to_hub: bool = False,
    encoder_threads: int = 4,
) -> Path:
    """Merges the 5 single-color recording sessions into a single dataset, optionally trimming idle tails."""
    dst_root = CACHE_BASE / Path(dst_repo_id).name

    print("\n" + "=" * 80)
    print(f"📦 [RWFM MERGE & TRIM] Destination: {dst_repo_id}")
    print(f"   Destination path: {dst_root}")
    print(f"   Canonical task:   {canonical_task}")
    print(f"   Trim idle tail:   {trim} (buffer_frames={buffer_frames}, arm_tol={arm_motion_tol}°/step)")
    print(f"   Push to Hub:      {push_to_hub}")
    print("=" * 80)

    if dst_root.exists():
        print(f"⚠️ Destination directory exists: {dst_root}. Removing for clean rebuild...")
        shutil.rmtree(dst_root)

    # 1. Inspect first base dataset for schema
    base_color, base_name = sources[0]
    base_root = CACHE_BASE / base_name
    if not base_root.exists():
        raise FileNotFoundError(f"Source dataset not found: {base_root}")

    base_ds = LeRobotDataset(f"{HF_USER}/{base_name}", root=base_root, return_uint8=True)
    frame_keys = [k for k in base_ds.features.keys() if k not in AUTO_KEYS]

    print(f"Base dataset: {base_name} (color={base_color})")
    print(f"FPS: {base_ds.fps} | Features: {frame_keys}")

    # 2. Create destination dataset
    dst = LeRobotDataset.create(
        repo_id=dst_repo_id,
        root=dst_root,
        fps=base_ds.fps,
        features=base_ds.features,
        robot_type=base_ds.meta.robot_type,
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=encoder_threads,
        encoder_threads=encoder_threads,
    )

    merged_episodes = 0
    total_raw_frames = 0
    total_saved_frames = 0
    total_trimmed_frames = 0
    episodes_trimmed_count = 0

    merged_annotations = {
        "schema_version": "1.0",
        "episodes": {},
    }

    t0 = time.time()

    try:
        for s_idx, (color, src_name) in enumerate(sources, 1):
            src_root = CACHE_BASE / src_name
            src_repo_id = f"{HF_USER}/{src_name}"

            print("\n" + "-" * 80)
            print(f"📂 [{s_idx}/{len(sources)}] Source: {src_name} ({color.upper()} block, 10 episodes)")
            print("-" * 80)

            if not src_root.exists():
                raise FileNotFoundError(f"Source folder not found: {src_root}")

            src_ds = LeRobotDataset(src_repo_id, root=src_root, return_uint8=True)
            ann_path = src_root / "meta/rwfm_rollout_annotations.json"
            src_ann = {}
            if ann_path.exists():
                try:
                    src_ann = json.loads(ann_path.read_text()).get("episodes", {})
                except Exception as e:
                    print(f"⚠️ Warning: Could not read {ann_path}: {e}")

            for ep_idx in range(src_ds.num_episodes):
                ep_meta = src_ds.meta.episodes[ep_idx]
                raw_len = ep_meta["length"]
                from_idx = ep_meta["dataset_from_index"]
                to_idx = ep_meta["dataset_to_index"]

                actions = np.array(src_ds.hf_dataset[from_idx:to_idx]["action"])
                states = np.array(src_ds.hf_dataset[from_idx:to_idx]["observation.state"])

                if trim:
                    tail_cut, trimmed, desc = find_rwfm_tail_cut(
                        actions,
                        states,
                        buffer_frames=buffer_frames,
                        arm_motion_tol=arm_motion_tol,
                        min_ep_len=min_ep_len,
                    )
                else:
                    tail_cut = raw_len
                    trimmed = 0
                    desc = "no_trim"

                new_len = tail_cut
                total_raw_frames += raw_len
                total_saved_frames += new_len
                total_trimmed_frames += trimmed
                if trimmed > 0:
                    episodes_trimmed_count += 1

                global_ep_idx = merged_episodes
                ep_t0 = time.time()

                # Add frames
                for i in range(from_idx, from_idx + tail_cut):
                    item = src_ds[i]
                    frame = {k: convert_val(item[k]) for k in frame_keys}
                    frame["task"] = canonical_task
                    dst.add_frame(frame)

                dst.save_episode()
                ep_dur = time.time() - ep_t0

                # Synchronize annotation
                raw_ep_ann = src_ann.get(str(ep_idx), {})
                raw_segments = raw_ep_ann.get("segments", [])
                new_segments = []
                for seg in raw_segments:
                    s = seg["start"]
                    e = seg["end"]
                    if s >= new_len:
                        continue
                    new_segments.append({
                        "start": s,
                        "end": min(e, new_len),
                        "type": seg["type"],
                    })

                merged_annotations["episodes"][str(global_ep_idx)] = {
                    "episode_index": global_ep_idx,
                    "physical_trial": raw_ep_ann.get("physical_trial", ep_idx + 1),
                    "task": canonical_task,
                    "target_block": color,
                    "outcome": raw_ep_ann.get("outcome", "grasp_success"),
                    "segments": new_segments,
                }

                trim_str = f"(-{trimmed}f idle tail)" if trimmed > 0 else "(no trim)"
                print(
                    f"  [{merged_episodes + 1:02d}/50] {color.upper():6s} Ep {ep_idx:2d} -> Global Ep {global_ep_idx:02d}: "
                    f"{raw_len:3d} -> {new_len:3d} frames {trim_str:18s} | {desc} ({ep_dur:.2f}s)"
                )
                merged_episodes += 1

    finally:
        print("\n💾 Finalizing dataset metadata, statistics, and videos...")
        dst.finalize()

        # Save authoritative merged rwfm_rollout_annotations.json
        meta_dir = dst_root / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        ann_out_path = meta_dir / "rwfm_rollout_annotations.json"
        ann_out_path.write_text(json.dumps(merged_annotations, indent=2))
        print(f"📝 Wrote updated {len(merged_annotations['episodes'])} episodes to {ann_out_path}")

    elapsed = time.time() - t0
    print("\n" + "=" * 80)
    print(f"🎉 [MERGE COMPLETE] {dst_repo_id}")
    print(f"   Root path:             {dst_root}")
    print(f"   Total episodes:        {merged_episodes} (Expected: 50)")
    print(f"   Episodes trimmed:      {episodes_trimmed_count} / {merged_episodes} ({episodes_trimmed_count/merged_episodes*100:.1f}%)")
    print(f"   Total frames before:   {total_raw_frames:,}")
    print(f"   Total frames after:    {total_saved_frames:,}")
    print(f"   Total frames trimmed:  {total_trimmed_frames:,} ({total_trimmed_frames/30.0:.2f}s)")
    print(f"   Time taken:            {elapsed/60.0:.2f} min ({elapsed:.1f}s)")
    print("=" * 80)

    # 3. Verification check
    print("\n🔍 [VERIFICATION] Verifying merged dataset integrity...")
    verify_ds = LeRobotDataset(dst_repo_id, root=dst_root, return_uint8=True)
    assert verify_ds.num_episodes == merged_episodes, f"Episode count mismatch: {verify_ds.num_episodes} != {merged_episodes}"
    assert verify_ds.num_frames == total_saved_frames, f"Frame count mismatch: {verify_ds.num_frames} != {total_saved_frames}"

    verify_ann = json.loads(ann_out_path.read_text())
    assert len(verify_ann["episodes"]) == merged_episodes, "Annotation episode count mismatch"
    for ep_id_str, ep_info in verify_ann["episodes"].items():
        ep_i = int(ep_id_str)
        ep_len = verify_ds.meta.episodes[ep_i]["length"]
        for seg in ep_info["segments"]:
            assert 0 <= seg["start"] < seg["end"] <= ep_len, (
                f"Invalid segment [{seg['start']}, {seg['end']}] for ep {ep_i} (len={ep_len})"
            )
    print("✅ Verification passed: 100% video, frame, and annotation consistency confirmed.")

    if push_to_hub:
        print(f"\n🚀 [HUB PUSH] Uploading {dst_repo_id} to Hugging Face Hub...")
        dst.push_to_hub()
        print("✅ [HUB PUSH DONE]")

    return dst_root


def merge_multitask_100ep(
    task1_root: Path,
    task2_root: Path,
    dst_repo_id: str,
    push_to_hub: bool = False,
    encoder_threads: int = 4,
) -> Path:
    """Combines Task 1 (50ep) and Task 2 (50ep) into a 100-episode Multitask RWFM rollout dataset."""
    dst_root = CACHE_BASE / Path(dst_repo_id).name
    print("\n" + "=" * 80)
    print(f"📦 [MULTITASK 100EP MERGE] Combining Task 1 and Task 2 -> {dst_repo_id}")
    print(f"   Destination: {dst_root}")
    print("=" * 80)

    if dst_root.exists():
        print(f"⚠️ Destination exists: {dst_root}. Removing for clean rebuild...")
        shutil.rmtree(dst_root)

    ds1 = LeRobotDataset(f"{HF_USER}/{task1_root.name}", root=task1_root, return_uint8=True)
    ds2 = LeRobotDataset(f"{HF_USER}/{task2_root.name}", root=task2_root, return_uint8=True)
    frame_keys = [k for k in ds1.features.keys() if k not in AUTO_KEYS]

    dst = LeRobotDataset.create(
        repo_id=dst_repo_id,
        root=dst_root,
        fps=ds1.fps,
        features=ds1.features,
        robot_type=ds1.meta.robot_type,
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=encoder_threads,
        encoder_threads=encoder_threads,
    )

    ann1 = json.loads((task1_root / "meta/rwfm_rollout_annotations.json").read_text()).get("episodes", {})
    ann2 = json.loads((task2_root / "meta/rwfm_rollout_annotations.json").read_text()).get("episodes", {})

    merged_annotations = {"schema_version": "1.0", "episodes": {}}
    global_idx = 0
    t0 = time.time()

    try:
        # Task 1 episodes (0..49)
        for ep in range(ds1.num_episodes):
            ep_meta = ds1.meta.episodes[ep]
            from_i = ep_meta["dataset_from_index"]
            to_i = ep_meta["dataset_to_index"]
            for i in range(from_i, to_i):
                item = ds1[i]
                frame = {k: convert_val(item[k]) for k in frame_keys}
                frame["task"] = item["task"]
                dst.add_frame(frame)
            dst.save_episode()

            ep_ann = dict(ann1[str(ep)])
            ep_ann["episode_index"] = global_idx
            merged_annotations["episodes"][str(global_idx)] = ep_ann
            global_idx += 1

        # Task 2 episodes (50..99)
        for ep in range(ds2.num_episodes):
            ep_meta = ds2.meta.episodes[ep]
            from_i = ep_meta["dataset_from_index"]
            to_i = ep_meta["dataset_to_index"]
            for i in range(from_i, to_i):
                item = ds2[i]
                frame = {k: convert_val(item[k]) for k in frame_keys}
                frame["task"] = item["task"]
                dst.add_frame(frame)
            dst.save_episode()

            ep_ann = dict(ann2[str(ep)])
            ep_ann["episode_index"] = global_idx
            merged_annotations["episodes"][str(global_idx)] = ep_ann
            global_idx += 1

    finally:
        dst.finalize()
        meta_dir = dst_root / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        (meta_dir / "rwfm_rollout_annotations.json").write_text(json.dumps(merged_annotations, indent=2))

    print(f"🎉 [MULTITASK 100EP DONE] {global_idx} episodes merged in {time.time() - t0:.1f}s")
    if push_to_hub:
        dst.push_to_hub()
    return dst_root


def main():
    parser = argparse.ArgumentParser(description="Merge and trim RWFM rollout datasets.")
    parser.add_argument(
        "--task",
        choices=["task1", "task2", "both"],
        default="both",
        help="Task to merge (task1, task2, or both). Default: both.",
    )
    parser.add_argument(
        "--no-trim",
        dest="trim",
        action="store_false",
        default=True,
        help="Disable trimming of trailing stationary frames.",
    )
    parser.add_argument(
        "--buffer-frames",
        type=int,
        default=2,
        help="Buffer frames of stationary hold to preserve at end of episode (default: 2 = ~0.07s).",
    )
    parser.add_argument(
        "--arm-motion-tol",
        type=float,
        default=0.35,
        help="Max sum-of-abs arm joint delta for stationary classification in deg/step (default: 0.35).",
    )
    parser.add_argument(
        "--create-multitask",
        action="store_true",
        default=False,
        help="Also create a 100-episode combined multitask dataset from Task 1 and Task 2.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        default=False,
        help="Push merged dataset(s) to Hugging Face Hub.",
    )
    parser.add_argument(
        "--encoder-threads",
        type=int,
        default=4,
        help="Video encoder threads (default: 4).",
    )
    args = parser.parse_args()

    trim_tag = "_trimmed" if args.trim else ""
    t1_repo_id = f"{HF_USER}/smolvla_task1_rwfm_rollout_822ep_50k_50ep{trim_tag}_merged"
    t2_repo_id = f"{HF_USER}/smolvla_task2_rwfm_rollout_822ep_50k_50ep{trim_tag}_merged"

    t1_root = None
    t2_root = None

    if args.task in {"task1", "both"}:
        t1_root = merge_and_trim_task(
            sources=TASK1_SOURCES,
            dst_repo_id=t1_repo_id,
            canonical_task=TASK1_CANONICAL_PROMPT,
            trim=args.trim,
            buffer_frames=args.buffer_frames,
            arm_motion_tol=args.arm_motion_tol,
            push_to_hub=args.push_to_hub,
            encoder_threads=args.encoder_threads,
        )

    if args.task in {"task2", "both"}:
        t2_root = merge_and_trim_task(
            sources=TASK2_SOURCES,
            dst_repo_id=t2_repo_id,
            canonical_task=TASK2_CANONICAL_PROMPT,
            trim=args.trim,
            buffer_frames=args.buffer_frames,
            arm_motion_tol=args.arm_motion_tol,
            push_to_hub=args.push_to_hub,
            encoder_threads=args.encoder_threads,
        )

    if args.create_multitask:
        if t1_root is None:
            t1_root = CACHE_BASE / Path(t1_repo_id).name
        if t2_root is None:
            t2_root = CACHE_BASE / Path(t2_repo_id).name
        multi_repo_id = f"{HF_USER}/smolvla_multitask_rwfm_rollout_822ep_50k_100ep{trim_tag}_merged"
        merge_multitask_100ep(
            task1_root=t1_root,
            task2_root=t2_root,
            dst_repo_id=multi_repo_id,
            push_to_hub=args.push_to_hub,
            encoder_threads=args.encoder_threads,
        )


if __name__ == "__main__":
    main()
