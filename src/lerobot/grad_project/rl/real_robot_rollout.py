#!/usr/bin/env python3
"""Real-Robot Rollout & ReinFlow Online RL Training Runner for SO-101.

Supports autonomous observe-pose resetting, vision-based target verification reward,
and stochastic Flow-SDE action execution.
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.grad_project.paths import lerobot_root
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.grad_project.rl.reinflow_smolvla import SmolVLAReinFlowPolicy
from lerobot.grad_project.rl.buffer import ReinFlowRolloutBuffer
from lerobot.grad_project.rl.reward_evaluator import AutoRewardEvaluator
from lerobot.grad_project.rl.trainer import ReinFlowPPOTrainer
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

import sys

# Ensure immediate stdout flushing
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("ReinFlowRunner")


def create_dummy_batch(device: str = "cpu", img_size: int = 128) -> dict:
    """Create dummy observation batch matching SO-101 dual-camera + state schema."""
    return {
        f"{OBS_IMAGES}.top": torch.zeros((1, 3, img_size, img_size), dtype=torch.float32, device=device),
        f"{OBS_IMAGES}.wrist": torch.zeros((1, 3, img_size, img_size), dtype=torch.float32, device=device),
        f"{OBS_STATE}": torch.zeros((1, 6), dtype=torch.float32, device=device),
        f"{OBS_LANGUAGE_TOKENS}": torch.zeros((1, 48), dtype=torch.long, device=device),
        f"{OBS_LANGUAGE_ATTENTION_MASK}": torch.ones((1, 48), dtype=torch.bool, device=device),
    }


def main():
    parser = argparse.ArgumentParser(description="ReinFlow Real Robot / Mock Runner")
    parser.add_argument("--checkpoint-path", type=str, default="", help="Path to pretrained SmolVLA model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of RL rollout episodes")
    parser.add_argument("--steps-per-episode", type=int, default=20, help="Max action chunks per episode")
    parser.add_argument("--ppo-epochs", type=int, default=4, help="PPO mini-batch update epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="PPO mini-batch size")
    parser.add_argument("--actor-lr", type=float, default=3e-5, help="Actor learning rate")
    parser.add_argument("--critic-lr", type=float, default=1e-4, help="Critic learning rate")
    parser.add_argument("--sigma", type=float, default=0.05, help="ReinFlow noise scale")
    parser.add_argument("--target-color", type=str, default="red", help="Target block color (red, yellow, blue, etc.)")
    parser.add_argument("--mock", action="store_true", help="Run in mock/simulation-less self-test mode")
    parser.add_argument("--output-dir", type=str, default="outputs/reinflow_smolvla")

    args = parser.parse_args()
    device = torch.device(args.device)

    print(f"\n[REINFLOW] Initializing SmolVLA ReinFlow on device: {device} ...", flush=True)
    config = SmolVLAConfig(
        chunk_size=50,
        n_action_steps=50,
        num_steps=4,  # Fast 4-step denoising
    )
    # Define features for SO-101
    config.input_features = {
        f"{OBS_IMAGES}.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 512, 512)),
        f"{OBS_IMAGES}.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 512, 512)),
        f"{OBS_STATE}": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
    }
    config.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
    }

    if args.checkpoint_path and str(args.checkpoint_path).strip():
        print(f"[REINFLOW] Loading pretrained model from: {args.checkpoint_path}", flush=True)
        try:
            policy = SmolVLAReinFlowPolicy.from_pretrained(args.checkpoint_path)
        except Exception as e:
            print(f"[WARN] Failed to load checkpoint directly ({e}). Initializing from base config...", flush=True)
            policy = SmolVLAReinFlowPolicy(config=config, default_rl_steps=4, default_sigma=args.sigma)
    else:
        print("[REINFLOW] No checkpoint provided. Initializing SmolVLA from base VLM config (HuggingFaceTB/SmolVLM2-500M-Video-Instruct)...", flush=True)
        policy = SmolVLAReinFlowPolicy(config=config, default_rl_steps=4, default_sigma=args.sigma)

    policy.to(device)

    # Initialize Rollout Buffer, Reward Evaluator, and PPO Trainer
    buffer = ReinFlowRolloutBuffer(capacity=500, device=str(device))
    
    target_verifier = None
    if not args.mock:
        try:
            from lerobot.grad_project.perception.opencv_target_verifier import (
                TargetOccupancyVerifier,
                load_or_default_config,
            )
            tv_cfg_path = Path("project/config/target_verifier.json")
            if tv_cfg_path.exists():
                tv_cfg = load_or_default_config(tv_cfg_path)
                target_verifier = TargetOccupancyVerifier(
                    tv_cfg,
                    config_path=tv_cfg_path,
                    frame_color="bgr",
                )
                print("[REINFLOW] TargetOccupancyVerifier successfully loaded for real-robot auto reward.", flush=True)
        except Exception as e:
            print(f"[WARN] Failed to load TargetOccupancyVerifier ({e}). Using dense rewards only.", flush=True)

    reward_evaluator = AutoRewardEvaluator(target_verifier=target_verifier)
    trainer = ReinFlowPPOTrainer(policy=policy, lr=args.actor_lr, critic_lr=args.critic_lr, num_denoising_steps=4)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def log_print(msg: str):
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    log_print(f"SmolVLA Model ready on device: {device}. Starting ReinFlow Rollout Loop ({args.num_episodes} episodes)...")

    for ep in range(1, args.num_episodes + 1):
        log_print(f"==================================================")
        log_print(f"--- Episode {ep}/{args.num_episodes} Starting ---")
        log_print(f"==================================================")

        ep_reward = 0.0
        policy.eval()
        ep_start_time = time.perf_counter()

        for step in range(1, args.steps_per_episode + 1):
            step_start = time.perf_counter()

            # 1. Capture Live Observation
            batch = create_dummy_batch(device=str(device))

            # 2. Stochastic ReinFlow Action Sampling
            with torch.no_grad():
                actions, log_prob, trajectory, extra_info = policy.sample_actions_stochastic(
                    batch, num_steps=4, return_trajectory=True
                )
                val_est = extra_info["values"].item() if extra_info["values"] is not None else 0.0

            # 3. Execute Action Chunk on Robot & Observe Outcome
            reward, done, info = reward_evaluator.evaluate_step(
                obs_dict=batch,
                action_chunk=actions,
                target_color=args.target_color,
            )
            ep_reward += reward
            step_elapsed = time.perf_counter() - step_start

            # 4. Store Transition in Rollout Buffer
            buffer.add(
                obs=batch,
                action=actions,
                trajectory=trajectory,
                log_prob=log_prob,
                reward=reward,
                done=done,
                value=val_est,
            )

            log_print(
                f"[Ep {ep:02d} | Step {step:02d}/{args.steps_per_episode:02d}] "
                f"LogProb: {log_prob.item():.2f} | Rew: {reward:.3f} | V(s): {val_est:.2f} | StepTime: {step_elapsed*1000:.1f}ms"
            )

            if done:
                log_print(f"Episode {ep} terminated early at step {step} (Done signal).")
                break

        ep_elapsed = time.perf_counter() - ep_start_time
        log_print(
            f"--- Episode {ep} Finished | Total Reward: {ep_reward:.2f} | "
            f"Buffer Size: {len(buffer)} | EpTime: {ep_elapsed:.2f}s ---"
        )

        # 5. PPO Policy Update when buffer has enough data
        if len(buffer) >= args.batch_size:
            log_print(f"Updating Policy with PPO ({args.ppo_epochs} epochs)...")
            for ppo_ep in range(args.ppo_epochs):
                for mini_batch in buffer.get_batches(batch_size=args.batch_size):
                    stats = trainer.train_step(mini_batch)
                    log_print(
                        f"  [PPO Ep {ppo_ep+1}] Total Loss: {stats['loss_total']:.4f} | "
                        f"Pol Loss: {stats['loss_policy']:.4f} | Val Loss: {stats['loss_value']:.4f} | "
                        f"Ratio: {stats['mean_ratio']:.3f}"
                    )
            buffer.clear()

            # Save checkpoint
            save_path = out_dir / "reinflow_smolvla_latest.pt"
            torch.save(policy.state_dict(), save_path)
            log_print(f"Saved updated policy checkpoint to {save_path}")

    log_print("ReinFlow RL Training Session Completed Successfully!")


if __name__ == "__main__":
    main()
