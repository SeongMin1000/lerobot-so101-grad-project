#!/usr/bin/env python
"""
Helper script to generate an episode reward map JSON for Offline RL training.

Usage:
    python project/scripts/tools/label_dataset_rewards.py \
        --repo_id eslab1234/smolvla_mixed_dataset \
        --output project/config/episode_rewards.json \
        --default_reward 1.0
"""

import argparse
import json
from pathlib import Path
from lerobot.datasets import LeRobotDataset


def main():
    parser = argparse.ArgumentParser(description="Label episode rewards for Offline RL")
    parser.add_argument("--repo_id", type=str, required=True, help="Hugging Face repo ID or local dataset ID")
    parser.add_argument("--root", type=str, default=None, help="Optional root path to dataset")
    parser.add_argument("--output", type=str, default="project/config/episode_rewards.json", help="Output JSON path")
    parser.add_argument("--default_reward", type=float, default=1.0, help="Default reward for episodes")
    args = parser.parse_args()

    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.root)
    num_episodes = dataset.num_episodes
    print(f"Dataset '{args.repo_id}' loaded with {num_episodes} episodes.")

    reward_map = {}
    for ep_idx in range(num_episodes):
        reward_map[ep_idx] = args.default_reward

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(reward_map, f, indent=2)

    print(f"✅ Saved reward map with {len(reward_map)} episodes to {out_path}")
    print("Example usage with lerobot-train:")
    print(f"  lerobot-train ... --sample_weighting.type=reward_weighted --sample_weighting.reward_map_path={out_path} --sample_weighting.temperature=0.5")


if __name__ == "__main__":
    main()
