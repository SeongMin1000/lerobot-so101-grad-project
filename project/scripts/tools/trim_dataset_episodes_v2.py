#!/usr/bin/env python3
"""
Trims trailing post-grasp / post-release stationary hold frames from HIL episodes
using backward state-action terminal hold detection, aligning with 575 Clean distribution.

Usage:
  python project/scripts/tools/trim_dataset_episodes_v2.py \
    --src-repo-id="eslab1234/smolvla_multitask_hil_285k_v1_220ep_trimmed" \
    --dst-repo-id="eslab1234/smolvla_multitask_hil_285k_v1_220ep_trimmed_v2" \
    --buffer-frames=5 \
    --push-to-hub
"""

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def find_dual_hold_cuts_v2(
    actions: np.ndarray,
    states: np.ndarray,
    buffer_frames: int = 5,
    arm_motion_tol: float = 0.30,
    arm_pose_tol: float = 0.50,
    min_ep_len: int = 30,
    trim_front: bool = True,
) -> tuple[int, int, int, str]:
    """
    actions: (T, 6)
    states:  (T, 6)
    buffer_frames: frames to preserve before motion starts / after motion stops (default: 5)
    arm_motion_tol: max sum-of-abs arm joint delta for stationary classification in deg/step (0.30)
    arm_pose_tol: max joint difference from initial pose for front stationary check (0.50 deg)
    min_ep_len: minimum length of trimmed episode in frames (30)
    trim_front: whether to trim leading stationary frames before motion begins

    Returns:
        front_cut: starting frame index (inclusive)
        tail_cut: ending frame index (exclusive, actions[front_cut:tail_cut])
        trimmed: total trimmed frames (front + tail)
        reason: description of cut event
    """
    T = len(actions)
    if T <= min_ep_len:
        return 0, T, 0, "too_short"

    # Step-to-step arm deltas and 5-frame moving average
    arm_deltas = np.zeros(T)
    arm_deltas[1:] = np.sum(np.abs(actions[1:, :5] - actions[:-1, :5]), axis=1)
    kernel = np.ones(5) / 5.0
    smooth_arm = np.convolve(arm_deltas, kernel, mode="same")

    # =========================================================================
    # 1. TAIL CUT (Backward scan from T-1)
    # =========================================================================
    final_grip_act = actions[-1, 5]
    is_closed = (final_grip_act <= 25.0)

    # Gripper stability point
    t_grip_stable = T - 1
    if is_closed:
        for t in range(T - 1, 0, -1):
            if actions[t, 5] <= max(final_grip_act + 2.0, 20.0):
                t_grip_stable = t
            else:
                break
    else:
        for t in range(T - 1, 0, -1):
            if actions[t, 5] >= min(final_grip_act - 2.0, 35.0):
                t_grip_stable = t
            else:
                break

    # Arm stationary point
    t_arm_stop = T - 1
    for t in range(T - 1, 0, -1):
        if smooth_arm[t] <= arm_motion_tol:
            t_arm_stop = t
        else:
            break

    # Terminal hold begins when BOTH gripper reached final pose AND arm motion stopped
    hold_start = max(t_grip_stable, t_arm_stop)

    # If closed tail, ensure follower state is firmly settled at the block (< 28.0 deg or stopped)
    if is_closed:
        st_deltas = np.zeros(T)
        st_deltas[1:] = np.abs(states[1:, 5] - states[:-1, 5])
        for t in range(hold_start, T):
            if states[t, 5] <= 28.0 or st_deltas[t] <= 0.2:
                hold_start = t
                break

    # Keep buffer_frames after hold_start
    tail_cut = min(T, max(hold_start + buffer_frames, min_ep_len))

    # =========================================================================
    # 2. FRONT CUT (Forward scan from 0)
    # =========================================================================
    if not trim_front:
        front_cut = 0
        motion_start = 0
    else:
        init_grip_act = actions[0, 5]
        init_is_closed = (init_grip_act <= 25.0)

        t_grip_start = 0
        if init_is_closed:
            for t in range(0, T - 1):
                if actions[t, 5] <= max(init_grip_act + 2.0, 20.0):
                    t_grip_start = t
                else:
                    break
        else:
            for t in range(0, T - 1):
                if actions[t, 5] >= min(init_grip_act - 2.0, 35.0):
                    t_grip_start = t
                else:
                    break

        init_arm_pose = np.median(actions[:min(5, T), :5], axis=0)
        arm_pose_diff = np.max(np.abs(actions[:, :5] - init_arm_pose), axis=1)

        t_arm_start = 0
        for t in range(0, T - 1):
            if smooth_arm[t] <= arm_motion_tol and arm_pose_diff[t] <= arm_pose_tol:
                t_arm_start = t
            else:
                break

        # Motion starts when EITHER arm moves OR gripper moves
        motion_start = min(t_grip_start, t_arm_start)
        # Preserve buffer_frames before motion starts
        front_cut = max(0, motion_start - buffer_frames)

    # Safety check: ensure min_ep_len is preserved
    if tail_cut - front_cut < min_ep_len:
        front_cut = max(0, tail_cut - min_ep_len)

    trimmed = front_cut + (T - tail_cut)
    cat = "grasp_hold" if is_closed else "release_hold"
    desc = f"{cat} | front: -{front_cut:2d}f (m_start={motion_start}), tail: -{T - tail_cut:2d}f (h_start={hold_start})"
    return front_cut, tail_cut, trimmed, desc


def find_terminal_hold_cut_v2(
    actions: np.ndarray,
    states: np.ndarray,
    buffer_frames: int = 5,
    arm_motion_tol: float = 0.30,
    min_ep_len: int = 30,
) -> tuple[int, int, str]:
    """Backward compatibility wrapper (tail cut only)."""
    _, tail_cut, trimmed, desc = find_dual_hold_cuts_v2(
        actions,
        states,
        buffer_frames=buffer_frames,
        arm_motion_tol=arm_motion_tol,
        min_ep_len=min_ep_len,
        trim_front=False,
    )
    return tail_cut, trimmed, desc


def parse_args():
    parser = argparse.ArgumentParser(description="Trim leading and trailing stationary hold frames from HIL episodes.")
    parser.add_argument(
        "--src-repo-id",
        type=str,
        default="eslab1234/smolvla_multitask_hil_285k_v1_220ep_trimmed",
        help="Source LeRobot dataset repo id.",
    )
    parser.add_argument(
        "--dst-repo-id",
        type=str,
        default="eslab1234/smolvla_multitask_hil_285k_v1_220ep_trimmed_v2",
        help="Destination LeRobot dataset repo id.",
    )
    parser.add_argument(
        "--buffer-frames",
        type=int,
        default=5,
        help="Number of buffer frames to preserve before/after motion (default: 5, matching 575 Clean median).",
    )
    parser.add_argument(
        "--arm-motion-tol",
        type=float,
        default=0.30,
        help="Max sum-of-abs arm joint delta for stationary classification in deg/step (default: 0.30).",
    )
    parser.add_argument(
        "--arm-pose-tol",
        type=float,
        default=0.50,
        help="Max deviation from initial arm pose for front stationary classification in deg (default: 0.50).",
    )
    parser.add_argument(
        "--trim-front",
        dest="trim_front",
        action="store_true",
        default=True,
        help="Trim leading stationary frames before motion starts (default: True).",
    )
    parser.add_argument(
        "--no-trim-front",
        dest="trim_front",
        action="store_false",
        help="Do not trim leading stationary frames.",
    )
    parser.add_argument(
        "--min-ep-len",
        type=int,
        default=30,
        help="Minimum length of trimmed episode in frames (default: 30).",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push resulting dataset to Hugging Face Hub.",
    )
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=4,
        help="Number of threads for async image writing.",
    )
    parser.add_argument(
        "--encoder-threads",
        type=int,
        default=4,
        help="Number of threads for video encoding.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cache_base = Path.home() / ".cache/huggingface/lerobot"

    src_root = cache_base / args.src_repo_id
    dst_root = cache_base / args.dst_repo_id

    print("=" * 80)
    print("✂️  [HIL V2 TERMINAL HOLD TRIMMER]")
    print(f"Source:            {args.src_repo_id} ({src_root})")
    print(f"Destination:       {args.dst_repo_id} ({dst_root})")
    print(f"Buffer Frames:     {args.buffer_frames} frames (0.17s at 30 FPS)")
    print(f"Arm Motion Tol:    {args.arm_motion_tol}°/step")
    print(f"Min Episode Len:   {args.min_ep_len} frames")
    print(f"Push to Hub:       {args.push_to_hub}")
    print("=" * 80)

    if not src_root.exists():
        print(f"[ERROR] Source dataset not found at {src_root}")
        sys.exit(1)

    if dst_root.exists():
        print(f"[WARN] Destination directory already exists: {dst_root}")
        print("Removing existing destination to ensure clean build...")
        shutil.rmtree(dst_root)

    t0 = time.time()
    src = LeRobotDataset(args.src_repo_id, root=src_root, return_uint8=True)
    num_episodes = src.num_episodes
    print(f"Loaded source dataset: {num_episodes} episodes, {src.num_frames} frames, FPS: {src.fps}")

    frame_keys = [
        k
        for k in src.features.keys()
        if k not in ["index", "episode_index", "frame_index", "timestamp", "task_index"]
    ]

    dst = LeRobotDataset.create(
        repo_id=args.dst_repo_id,
        root=dst_root,
        fps=src.fps,
        features=src.features,
        robot_type=src.meta.robot_type,
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=args.image_writer_threads,
        encoder_threads=args.encoder_threads,
    )

    total_trimmed_frames = 0
    total_saved_frames = 0
    trimmed_episodes_count = 0
    saved_episode_lengths = []

    try:
        for ep_idx in range(num_episodes):
            ep_meta = src.meta.episodes[ep_idx]
            orig_len = ep_meta["length"]
            from_idx = ep_meta["dataset_from_index"]
            to_idx = ep_meta["dataset_to_index"]

            # Load actions and states for this episode
            actions = np.array(src.hf_dataset[from_idx:to_idx]["action"])
            states = np.array(src.hf_dataset[from_idx:to_idx]["observation.state"])

            front_cut, tail_cut, cut_len, reason = find_dual_hold_cuts_v2(
                actions,
                states,
                buffer_frames=args.buffer_frames,
                arm_motion_tol=args.arm_motion_tol,
                arm_pose_tol=args.arm_pose_tol,
                min_ep_len=args.min_ep_len,
                trim_front=args.trim_front,
            )

            new_len = tail_cut - front_cut
            new_from_idx = from_idx + front_cut
            new_to_idx = from_idx + tail_cut

            if cut_len > 0:
                trimmed_episodes_count += 1

            total_trimmed_frames += cut_len
            total_saved_frames += new_len
            saved_episode_lengths.append(new_len)

            ep_t0 = time.time()
            for i in range(new_from_idx, new_to_idx):
                item = src[i]
                frame = {}
                for k in frame_keys:
                    val = item[k]
                    if "image" in k:
                        val = val.permute(1, 2, 0).numpy()
                    elif hasattr(val, "numpy"):
                        val = val.numpy()
                    frame[k] = val
                frame["task"] = item["task"]
                dst.add_frame(frame)

            dst.save_episode()
            ep_time = time.time() - ep_t0
            print(
                f"[{ep_idx + 1:3d}/{num_episodes}] Ep {ep_idx:3d}: {orig_len:3d} -> {new_len:3d} frames "
                f"(-{cut_len:2d}f, -{cut_len/src.fps:.2f}s) | {reason} ({ep_time:.2f}s)"
            )

    except Exception as e:
        print(f"\n[ERROR] Trimming failed at episode {ep_idx}: {e}")
        raise e
    finally:
        print("\n[FINALIZE] Finalizing dataset metadata and statistics...")
        dst.finalize()

        # Preserve episode interventions metadata if present
        src_intv_path = src_root / "meta/episode_interventions.json"
        if src_intv_path.exists():
            try:
                import json
                src_intvs = json.loads(src_intv_path.read_text())
                dst_intvs = {}
                for ep_str, data in src_intvs.items():
                    ep_i = int(ep_str)
                    if ep_i < len(saved_episode_lengths):
                        dst_intvs[ep_str] = dict(data)
                        dst_intvs[ep_str]["total_frames"] = saved_episode_lengths[ep_i]
                dst_intv_path = dst_root / "meta/episode_interventions.json"
                dst_intv_path.parent.mkdir(parents=True, exist_ok=True)
                dst_intv_path.write_text(json.dumps(dst_intvs, indent=2))
                print(f"📝 Copied & updated episode_interventions.json ({len(dst_intvs)} episodes)")
            except Exception as e:
                print(f"⚠️ Failed to update episode_interventions.json: {e}")

    elapsed = time.time() - t0
    print("\n" + "=" * 80)
    print("✅ [V2 TRIM COMPLETE]")
    print(f"Output Dataset:       {args.dst_repo_id}")
    print(f"Output Path:          {dst_root}")
    print(f"Total Episodes:       {num_episodes}")
    print(f"Trimmed Episodes:     {trimmed_episodes_count} / {num_episodes} ({trimmed_episodes_count/num_episodes*100:.1f}%)")
    print(f"Total Frames Before:  {src.num_frames:,}")
    print(f"Total Frames After:   {total_saved_frames:,}")
    print(f"Total Frames Trimmed: {total_trimmed_frames:,} ({total_trimmed_frames/src.fps:.1f}s)")
    print(f"Time Taken:           {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print("=" * 80)

    if args.push_to_hub:
        print(f"\n🚀 [HUB PUSH] Uploading {args.dst_repo_id} to Hugging Face Hub...")
        dst.push_to_hub()
        print("✅ [HUB PUSH DONE]")


if __name__ == "__main__":
    main()
