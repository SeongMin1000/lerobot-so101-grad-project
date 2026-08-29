#!/usr/bin/env python3
"""Mine (block pixel, true grasp position) pairs from a recorded teleop dataset.

The pixel->robot map the pipeline aims with is fitted from a dozen points a
person taught by hand in one sitting. Those scatter about +/-20mm, which is
most of a 25mm block, and no amount of downstream correction beats the data it
is built on. A recorded teleop dataset has the same information thousands of
times over: every episode drives the real arm onto a real block and closes on
it, so the arm's own forward kinematics at that instant IS the block's position.

Pairing is by pick ORDER, not geometry. The task drives the colours in a fixed
sequence, so the k-th grasp of an episode is the k-th colour -- a signal that
does not depend on the very transform being fitted. The existing transform is
used only to sanity-check each pairing, and any grasp whose nearest block is
ambiguous is dropped rather than guessed.

Each block is picked once per episode, from wherever it started, so a single
detection on the episode's first frame gives every block's true pre-pick
position.

Usage:
  mine_grasps_from_dataset.py --dataset_dir=DIR [--out=FILE] [--episodes=N]
"""

import argparse
import json
import subprocess
import tempfile
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from lerobot.grad_project.control.table_ik import load_kinematics
from lerobot.grad_project.paths import lerobot_root

PICK_ORDER = ["red", "yellow", "wood", "green", "blue"]
MIN_CLOSED_FRAMES = 8
GRASP_Z_RANGE = (-0.03, 0.06)
GRASP_REACH_RANGE = (0.12, 0.46)
# A pairing is only trusted when the runner-up block is clearly further away.
MIN_PAIRING_MARGIN_M = 0.07
MAX_PAIRING_DISTANCE_M = 0.10


def closed_runs(gripper: np.ndarray) -> list[tuple[int, int]]:
    """Frame spans where the jaws sat closed long enough to be holding something."""
    low, high = np.percentile(gripper, 5), np.percentile(gripper, 95)
    closed = gripper < low + 0.35 * (high - low)
    runs, index = [], 0
    while index < len(closed):
        if not closed[index]:
            index += 1
            continue
        end = index
        while end < len(closed) and closed[end]:
            end += 1
        if end - index >= MIN_CLOSED_FRAMES:
            runs.append((index, end))
        index = end
    return runs


def frame_at(video: Path, timestamp: float, out: Path) -> np.ndarray | None:
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{timestamp:.3f}",
         "-i", str(video), "-frames:v", "1", str(out), "-y"],
        check=False,
    )
    return cv2.imread(str(out)) if out.is_file() else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--out", default="project/config/grasp_samples_mined.json")
    parser.add_argument("--episodes", type=int, default=0, help="0 = all")
    parser.add_argument("--model", default="project/models/yolo_block_detector/best.pt")
    args = parser.parse_args()

    root = lerobot_root()
    dataset = Path(args.dataset_dir)
    from lerobot.grad_project.perception.yolo_block_detector import YoloBlockDetector

    detector = YoloBlockDetector.load(
        str(root / args.model), root / "project/config/detector.json", frame_color="bgr"
    )
    kinematics = load_kinematics()
    reference = np.array(
        json.loads((root / "project/config/grasp_pixel_to_robot.json").read_text())[
            "homography_pixel_to_robot_xy_m"
        ]
    )

    frames = pd.read_parquet(dataset / "data/chunk-000/file-000.parquet")
    episodes = pd.read_parquet(dataset / "meta/episodes/chunk-000/file-000.parquet")
    if args.episodes:
        episodes = episodes.head(args.episodes)

    pairs, skipped = [], {"no_frame": 0, "few_blocks": 0, "ambiguous": 0, "extra_runs": 0}
    with tempfile.TemporaryDirectory() as tmp:
        shot = Path(tmp) / "frame.jpg"
        for _, episode in episodes.iterrows():
            index = int(episode["episode_index"])
            video = dataset / (
                "videos/observation.images.top/chunk-"
                f"{int(episode['videos/observation.images.top/chunk_index']):03d}"
                f"/file-{int(episode['videos/observation.images.top/file_index']):03d}.mp4"
            )
            start = float(episode["videos/observation.images.top/from_timestamp"])
            image = frame_at(video, start, shot)
            if image is None:
                skipped["no_frame"] += 1
                continue

            detected = {b.color: b for b in detector.detect(image).blocks}
            if len(detected) < 5:
                skipped["few_blocks"] += 1
                continue
            predicted = {
                color: cv2.perspectiveTransform(
                    np.array([[[b.cx, b.cy]]], dtype=np.float64), reference
                )[0, 0]
                for color, b in detected.items()
            }

            episode_frames = frames[frames.episode_index == index].reset_index(drop=True)
            state = np.stack(episode_frames["observation.state"].values).astype(np.float64)
            grasps = []
            for begin, _end in closed_runs(state[:, 5]):
                joints = state[max(0, begin - 1)]
                xyz = kinematics.forward_kinematics(joints)[:3, 3]
                reach = float(np.hypot(xyz[0], xyz[1]))
                if not (GRASP_Z_RANGE[0] <= xyz[2] <= GRASP_Z_RANGE[1]):
                    continue
                if not (GRASP_REACH_RANGE[0] <= reach <= GRASP_REACH_RANGE[1]):
                    continue
                grasps.append((begin, xyz, joints))

            for order, (begin, xyz, joints) in enumerate(grasps):
                distances = {c: float(np.linalg.norm(xyz[:2] - p)) for c, p in predicted.items()}
                nearest = min(distances, key=distances.get)
                ordered = sorted(distances.values())
                margin = ordered[1] - ordered[0]
                if distances[nearest] > MAX_PAIRING_DISTANCE_M or margin < MIN_PAIRING_MARGIN_M:
                    skipped["ambiguous"] += 1
                    continue
                # Pick order is the independent check: if the sequence agrees
                # with geometry, the pairing is safe; if not, drop it rather
                # than trust either one alone.
                if order < len(PICK_ORDER) and PICK_ORDER[order] != nearest:
                    skipped["extra_runs"] += 1
                    continue
                block = detected[nearest]
                pairs.append(
                    {
                        "label": f"{nearest}@ep{index}f{begin}",
                        "color": nearest,
                        "episode": index,
                        "frame": int(begin),
                        "pixel": [float(block.cx), float(block.cy)],
                        "robot_xyz_m": [float(v) for v in xyz],
                        "joints_deg": joints.tolist(),
                        "reach_m": float(np.hypot(xyz[0], xyz[1])),
                        "pair_distance_mm": distances[nearest] * 1000,
                    }
                )

    output = root / args.out
    output.write_text(json.dumps({"samples": pairs}, indent=1), encoding="utf-8")
    print(f"mined {len(pairs)} pairs from {len(episodes)} episodes -> {output}")
    print(f"skipped: {skipped}")
    if pairs:
        distances = np.array([p["pair_distance_mm"] for p in pairs])
        print(f"pairing distance: median {np.median(distances):.0f}mm  p90 {np.percentile(distances, 90):.0f}mm")
        by_color = {}
        for p in pairs:
            by_color[p["color"]] = by_color.get(p["color"], 0) + 1
        print(f"per colour: {by_color}")


if __name__ == "__main__":
    main()
