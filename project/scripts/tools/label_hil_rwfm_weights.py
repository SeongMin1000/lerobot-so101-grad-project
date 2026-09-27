#!/usr/bin/env python3
"""
Assigns fixed RWFM human-correction reward weights (0.6) to an HIL dataset
(default: eslab1234/smolvla_hil_575_285k_v3_50ep_trimmed_merged).

Generates:
1. meta/episode_frame_rewards.json (segment-level chunk reward maps [[0, T, 0.6]])
2. meta/rwfm_rollout_annotations.json (authoritative RWFM annotations with type='human_correction')
3. Updates meta/episode_interventions.json with correction_reward=0.6
4. project/config/episode_rewards_hil_575_285k_v3_50ep.json (episode-level map for offline RL)

Usage:
    python project/scripts/tools/label_hil_rwfm_weights.py \
        --repo-id eslab1234/smolvla_hil_575_285k_v3_50ep_trimmed_merged \
        --reward 0.6
"""

import argparse
import json
import os
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Assign fixed RWFM weights to HIL dataset.")
    parser.add_argument(
        "--repo-id",
        type=str,
        default="eslab1234/smolvla_hil_575_285k_v3_50ep_trimmed_merged",
        help="Target LeRobot dataset repo id.",
    )
    parser.add_argument(
        "--reward",
        type=float,
        default=0.6,
        help="Fixed reward weight for HIL human corrections (default: 0.6).",
    )
    parser.add_argument(
        "--reward-type",
        type=str,
        default="human_correction",
        help="RWFM segment type (default: 'human_correction').",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cache_base = Path.home() / ".cache/huggingface/lerobot"
    ds_root = cache_base / args.repo_id

    if not ds_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {ds_root}")

    print("=" * 80)
    print("🏷️  [RWFM REWARD WEIGHT LABELER FOR HIL DATASET]")
    print(f"Dataset:       {args.repo_id}")
    print(f"Dataset Root:  {ds_root}")
    print(f"Fixed Reward:  {args.reward}")
    print(f"Segment Type:  {args.reward_type}")
    print("=" * 80)

    dataset = LeRobotDataset(args.repo_id, root=ds_root, return_uint8=True)
    num_episodes = dataset.num_episodes
    total_frames = dataset.num_frames
    print(f"Loaded dataset: {num_episodes} episodes, {total_frames} frames, FPS: {dataset.fps}")

    meta_dir = ds_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load existing interventions if available
    intvs_path = meta_dir / "episode_interventions.json"
    existing_intvs = {}
    if intvs_path.exists():
        try:
            existing_intvs = json.loads(intvs_path.read_text())
            print(f"Found existing episode_interventions.json ({len(existing_intvs)} episodes)")
        except Exception as e:
            print(f"Warning: Could not read existing interventions: {e}")

    # Data structures
    frame_rewards_map = {}
    rwfm_annotations = {
        "schema_version": "1.0",
        "episodes": {},
    }
    updated_intvs = {}
    episode_reward_map = {}

    for ep_idx in range(num_episodes):
        ep_meta = dataset.meta.episodes[ep_idx]
        ep_len = ep_meta["length"]
        ep_key = str(ep_idx)

        # 1. Frame / Segment reward map: [[0, ep_len, 0.6]]
        frame_rewards_map[ep_key] = [[0, ep_len, args.reward]]

        # 2. Episode reward map: 0.6
        episode_reward_map[ep_key] = args.reward

        # Existing intervention metadata for context
        intv_data = existing_intvs.get(ep_key, {})
        target_block = intv_data.get("target_block", "unknown")
        task_phase = intv_data.get("task_phase", "grasp")
        physical_trial = intv_data.get("parent_rollout_id", ep_idx + 1)

        # 3. RWFM Rollout Annotations
        rwfm_annotations["episodes"][ep_key] = {
            "episode_index": ep_idx,
            "physical_trial": physical_trial,
            "task": dataset[ep_meta["dataset_from_index"]]["task"],
            "target_block": target_block,
            "task_phase": task_phase,
            "outcome": "grasp_success",
            "segments": [
                {
                    "start": 0,
                    "end": ep_len,
                    "type": args.reward_type,
                    "reward": args.reward,
                }
            ],
        }

        # 4. Updated episode_interventions.json
        cur_intv = dict(intv_data) if intv_data else {}
        cur_intv["episode_index"] = ep_idx
        cur_intv["total_frames"] = ep_len
        cur_intv["correction_reward"] = args.reward
        cur_intv["success_reward"] = args.reward
        cur_intv["reward_type"] = args.reward_type
        updated_intvs[ep_key] = cur_intv

    # Save to meta/
    out_frame_rewards = meta_dir / "episode_frame_rewards.json"
    out_frame_rewards.write_text(json.dumps(frame_rewards_map, indent=2))
    print(f"✅ Saved: {out_frame_rewards} ({len(frame_rewards_map)} episodes)")

    out_rwfm_ann = meta_dir / "rwfm_rollout_annotations.json"
    out_rwfm_ann.write_text(json.dumps(rwfm_annotations, indent=2))
    print(f"✅ Saved: {out_rwfm_ann} ({len(rwfm_annotations['episodes'])} episodes)")

    intvs_path.write_text(json.dumps(updated_intvs, indent=2))
    print(f"✅ Updated: {intvs_path} ({len(updated_intvs)} episodes)")

    # Save to project/config/
    cfg_dir = Path("project/config")
    cfg_dir.mkdir(parents=True, exist_ok=True)

    base_name = Path(args.repo_id).name
    cfg_ep_rewards = cfg_dir / f"episode_rewards_{base_name}.json"
    cfg_ep_rewards.write_text(json.dumps(episode_reward_map, indent=2))
    print(f"✅ Saved: {cfg_ep_rewards} (episode-level map)")

    cfg_frame_rewards = cfg_dir / f"episode_frame_rewards_{base_name}.json"
    cfg_frame_rewards.write_text(json.dumps(frame_rewards_map, indent=2))
    print(f"✅ Saved: {cfg_frame_rewards} (frame-level map)")

    print("\n" + "=" * 80)
    print("🎉 [HIL RWFM REWARD WEIGHT LABELING COMPLETE]")
    print(f"All {num_episodes} episodes fixed to reward={args.reward} ({args.reward_type}).")
    print("Compatible with sample_weighting.type=reward_weighted and RWFM offline RL pipelines.")
    print("=" * 80)


if __name__ == "__main__":
    main()
