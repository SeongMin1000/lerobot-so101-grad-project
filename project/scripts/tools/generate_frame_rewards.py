#!/usr/bin/env python3
"""
Helper script to generate frame-level / segment-level reward maps for Reward-Weighted Flow Matching (RWFM).

Supports:
1. Reading intervention metadata (meta/episode_interventions.json) from HIL recording.
2. Generating 0.8 (normal), 0.1 (mistake window), and 1.0 (human correction) segments.
3. Assigning 1.0 (or default) to clean expert demonstration episodes.
4. Exporting as compact segment format or dense frame arrays.

Usage:
    python project/scripts/tools/generate_frame_rewards.py \
        --repo_id eslab1234/multitask_5blocks_v3_704ep_hil_r1_merged \
        --output project/config/episode_frame_rewards.json \
        --normal_reward 0.8 \
        --failure_reward 0.1 \
        --correction_reward 1.0 \
        --mistake_window 45
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from lerobot.datasets import LeRobotDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Frame-Level Rewards for RWFM")
    parser.add_argument("--repo_id", type=str, required=True, help="LeRobot dataset repo ID or local path")
    parser.add_argument("--root", type=str, default=None, help="Optional root path to dataset")
    parser.add_argument(
        "--interventions_path",
        type=str,
        default=None,
        help="Optional path to episode_interventions.json (auto-detected if in dataset meta/)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="project/config/episode_frame_rewards.json",
        help="Output JSON path",
    )
    parser.add_argument("--normal_reward", type=float, default=0.8, help="Reward for normal autonomous driving")
    parser.add_argument("--failure_reward", type=float, default=0.1, help="Reward for failure / mistake window")
    parser.add_argument(
        "--correction_reward",
        type=float,
        default=1.0,
        help="Reward for human correction / recovery window",
    )
    parser.add_argument("--success_reward", type=float, default=1.0, help="Reward for clean demonstration episodes")
    parser.add_argument(
        "--mistake_window",
        type=int,
        default=45,
        help="Frames before intervention considered mistake/failure (default 45 = 1.5s at 30Hz)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["segment", "dense"],
        default="segment",
        help="Output format: 'segment' ([[start, end, r], ...]) or 'dense' ([r0, r1, ...])",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    logging.info("Loading dataset %s...", args.repo_id)
    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.root)
    num_episodes = dataset.num_episodes
    logging.info("Loaded dataset with %d episodes, %d total frames.", num_episodes, dataset.num_frames)

    # 1. Resolve interventions file
    interventions: dict[str, Any] = {}
    interv_path = None
    if args.interventions_path:
        interv_path = Path(args.interventions_path)
    elif dataset.root:
        cand = Path(dataset.root) / "meta" / "episode_interventions.json"
        if cand.exists():
            interv_path = cand

    if interv_path and interv_path.exists():
        with open(interv_path, "r", encoding="utf-8") as f:
            interventions = json.load(f)
        logging.info("Loaded intervention metadata for %d episodes from %s", len(interventions), interv_path)
    else:
        logging.info("No intervention metadata found; defaulting all episodes to success_reward=%.2f", args.success_reward)

    output_data: dict[str, Any] = {}
    total_mistake_frames = 0
    total_correction_frames = 0
    total_normal_frames = 0

    for ep_idx in range(num_episodes):
        ep_meta = dataset.meta.episodes[ep_idx]
        ep_from = ep_meta["dataset_from_index"]
        ep_to = ep_meta["dataset_to_index"]
        ep_len = ep_to - ep_from

        ep_key = str(ep_idx)
        interv_info = interventions.get(ep_key)

        segments = []
        if interv_info and "intervention_start_frame" in interv_info:
            interv_start = int(interv_info["intervention_start_frame"])
            if interv_start > 0:
                mistake_start = max(0, interv_start - args.mistake_window)
                # Segment 1: Normal autonomous driving before deviation
                if mistake_start > 0:
                    segments.append([0, mistake_start, args.normal_reward])
                    total_normal_frames += mistake_start
                # Segment 2: Mistake / failure deviation window
                segments.append([mistake_start, interv_start, args.failure_reward])
                total_mistake_frames += (interv_start - mistake_start)
                # Segment 3: Human correction / recovery to finish
                segments.append([interv_start, ep_len, args.correction_reward])
                total_correction_frames += (ep_len - interv_start)
            else:
                # 100% human correction (e.g. corrections_only mode)
                segments.append([0, ep_len, args.correction_reward])
                total_correction_frames += ep_len
        else:
            # Clean demonstration without intervention
            segments.append([0, ep_len, args.success_reward])
            total_normal_frames += ep_len

        if args.format == "segment":
            output_data[ep_key] = segments
        else:
            # Dense array
            dense_arr = [args.success_reward] * ep_len
            for seg in segments:
                s_f, e_f, r_val = seg
                for f_i in range(s_f, min(e_f, ep_len)):
                    dense_arr[f_i] = r_val
            output_data[ep_key] = dense_arr

    out_p = Path(args.output)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    logging.info("=" * 65)
    logging.info("✅ Frame rewards saved to %s", out_p)
    logging.info("   Total episodes processed: %d", len(output_data))
    logging.info("   Normal frames (r=%.2f):     %d", args.normal_reward, total_normal_frames)
    logging.info("   Mistake frames (r=%.2f):    %d (downweighted during Flow-Matching)", args.failure_reward, total_mistake_frames)
    logging.info("   Correction frames (r=%.2f): %d (upweighted during Flow-Matching)", args.correction_reward, total_correction_frames)
    logging.info("=" * 65)
    logging.info("Example lerobot-train command with Reward-Weighted Flow Matching:")
    logging.info(
        "  lerobot-train ... \\\n"
        "    --sample_weighting.type=reward_weighted \\\n"
        "    --sample_weighting.frame_reward_path=%s \\\n"
        "    --sample_weighting.temperature=0.5 \\\n"
        "    --sample_weighting.chunk_size=50",
        out_p,
    )


if __name__ == "__main__":
    main()
