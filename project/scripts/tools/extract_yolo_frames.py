#!/usr/bin/env python3
"""Extract 6 boundary frames per episode (5,4,3,2,1,0 blocks remaining) from the
task1_pick_place_5blocks_125ep_merged_v1 dataset, using the gripper open/close
pattern to find pick boundaries. Saves top-camera JPEGs for YOLO labeling.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset

GRIPPER_CLOSED_THRESHOLD = 5.0  # gripper.pos below this = closed/grasping
MIN_SEGMENT_LEN = 8  # debounce: ignore transitions shorter than this many frames


def find_boundary_local_indices(gripper: np.ndarray) -> list[int]:
    """Return up to 6 local frame indices: episode start, then the frame right
    after each closed->open (release) transition, debounced against noise."""
    closed = gripper < GRIPPER_CLOSED_THRESHOLD

    # collapse into runs of (state, start, length)
    runs = []
    cur_state = closed[0]
    run_start = 0
    for i in range(1, len(closed)):
        if closed[i] != cur_state:
            runs.append((cur_state, run_start, i - run_start))
            cur_state = closed[i]
            run_start = i
    runs.append((cur_state, run_start, len(closed) - run_start))

    # drop short noisy runs by merging them into the previous run's state
    clean_runs = []
    for state, start, length in runs:
        if length < MIN_SEGMENT_LEN and clean_runs:
            continue  # treat as noise, keep previous state's run going
        clean_runs.append([state, start, length])

    boundaries = [0]  # frame 0 = 5 blocks remaining
    for idx in range(1, len(clean_runs)):
        prev_state, _, _ = clean_runs[idx - 1]
        state, start, _ = clean_runs[idx]
        if prev_state and not state:  # closed -> open transition
            boundaries.append(start)

    return boundaries[:6]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="eslab1234/task1_pick_place_5blocks_125ep_merged_v1")
    parser.add_argument("--camera", default="observation.images.top")
    parser.add_argument("--out-dir", default="var/yolo_frames")
    parser.add_argument("--limit-episodes", type=int, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = LeRobotDataset(args.repo_id)
    n_episodes = args.limit_episodes or ds.num_episodes
    print(f"dataset: {args.repo_id}, episodes to process: {n_episodes}/{ds.num_episodes}")

    total_saved = 0
    incomplete_episodes = []

    for ep_idx in range(n_episodes):
        ep = ds.meta.episodes[ep_idx]
        s, e = ep["dataset_from_index"], ep["dataset_to_index"]

        gripper = np.array([float(ds.hf_dataset[i]["action"][5]) for i in range(s, e)])
        local_boundaries = find_boundary_local_indices(gripper)

        if len(local_boundaries) != 6:
            incomplete_episodes.append((ep_idx, len(local_boundaries)))

        for blocks_remaining, local_idx in zip(
            range(5, 5 - len(local_boundaries), -1), local_boundaries
        ):
            global_idx = s + local_idx
            item = ds[global_idx]
            img_tensor = item[args.camera]  # CHW float [0,1]
            img = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            fname = out_dir / f"ep{ep_idx:03d}_blocks{blocks_remaining}.jpg"
            Image.fromarray(img).save(fname, quality=95)
            total_saved += 1

        if ep_idx % 10 == 0:
            print(f"  episode {ep_idx}: found {len(local_boundaries)} boundaries, saved so far: {total_saved}")

    print(f"\nDone. Total images saved: {total_saved} in {out_dir}")
    if incomplete_episodes:
        print(f"\n{len(incomplete_episodes)} episodes did NOT yield exactly 6 boundaries (check manually):")
        for ep_idx, n in incomplete_episodes:
            print(f"  episode {ep_idx}: {n} boundaries found")


if __name__ == "__main__":
    main()
