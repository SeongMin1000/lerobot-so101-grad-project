#!/usr/bin/env python3
"""
train_smolvla_rwfm_pcgrad_combined.py

Unified "General Expert-Only RWFM + Same-Color PCGrad" SmolVLA Trainer.

Architecture:
  - Base Model: 865 full-finetuning checkpoint configured to Expert-Only
    (VLM & vision encoder frozen, state projection frozen, action expert trainable).
  - Normalization Fidelity: Base 865 normalizer safetensors strictly preserved.
  - Multi-Rate Dual Objective:
      A. General RWFM (every global step):
         - 725-episode merged dataset (0..574 clean575, 575..674 rollout100, 675..724 hil50).
         - Source-aware episode-balanced sampling (B32: 24 clean + 8 rollout, sparse HIL injection 22 clean + 8 rollout + 2 HIL).
         - Failure Action Mask: K=50 action loss masking for failure segments (effective_action_is_pad = pad | ~learn_mask).
         - Reward-Weighted Flow Matching: R_i computed over learnable non-padding actions, exp((R - max(R))/T), normalized.
      B. Placement PCGrad (every 6th global step):
         - Task 1 (120ep) & Task 2 (127ep) clean post-grasp branches.
         - Same-color paired batches (B16 T1 + B16 T2) with 5-step color cycle.
         - Paired FM randomness and canonical PCGrad projection when g_t1 . g_t2 < 0.
         - Standard FM loss on branches (NO RWFM applied to branch datasets).
      C. Unified Gradient Step:
         - Exactly ONE optimizer.step() and ONE scheduler.step() per global step.
         - g_final = GENERAL_WEIGHT * g_general + PCGRAD_WEIGHT * g_place (with optional anchor projection).
"""

import argparse
import collections
import csv
import json
import logging
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from huggingface_hub import HfApi
from safetensors import safe_open

# Repository root and train module resolution
REPO_ROOT = Path(__file__).resolve().parents[3]
REPO_SRC = REPO_ROOT / "src"
REPO_TRAIN = REPO_ROOT / "project/scripts/train"
for p in [REPO_SRC, REPO_TRAIN]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from train_smolvla_same_color_pcgrad import (
    COLOR_NAMES,
    TASK1_EPISODE_RANGES,
    TASK2_EPISODE_RANGES,
    PCGradWandBLogger,
    SameColorBalancedSampler,
    build_optimizer_and_scheduler,
    compute_accumulated_task_gradient,
    compute_pcgrad,
    load_model_exact_stats,
    prepare_torch_batch,
    push_final_checkpoint_to_hub,
    resolve_local_path,
    save_compatible_checkpoint,
    setup_logger,
)

from lerobot.common.train_utils import load_training_state
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.transforms.transforms import ImageTransforms, ImageTransformsConfig
from lerobot.utils.sample_weighting import RewardSampleWeighter, SampleWeightingConfig

DEFAULT_BASE_POLICY = "eslab1234/smolvla_multitask_5blocks_v3_865ep_trimmed_from575_285k_fullft_lr1e5_150k"
DEFAULT_GENERAL_DATASET = "eslab1234/smolvla_rwfm_general_masked575_topgrasp_755ep_v1"
DEFAULT_TASK1_DATASET = "eslab1234/smolvla_task1_hil_575_285k_v4_120ep_trimmed_merged"
DEFAULT_TASK2_DATASET = "eslab1234/smolvla_task2_hil_575_285k_v4_127ep_trimmed_merged"

# Authoritative 755-episode boundary partition
CLEAN_T1_RANGE = (0, 352)      # 353 episodes (reward 0.0, placement learn=false mask)
CLEAN_T2_RANGE = (353, 574)    # 222 episodes (reward 0.0, placement learn=false mask)
ROLLOUT100_RANGE = (575, 674)  # 100 episodes (normal 0.0, self_corr 0.4, failure -1.0)
TOPGRASP30_RANGE = (675, 704)  # 30 episodes (normal 0.0, self_corr 0.4, learn=true)
HIL_RANGE = (705, 754)         # 50 episodes (reward 0.6, learn=true)


class GeneralSourceBalancedSampler:
    """Manages source-aware, 50:50 clean balanced sampling across the 755-episode General dataset.

    Cadence:
      - Normal step (B32): Task1 Clean 11 + Task2 Clean 11 + Rollout100 6 + TopGrasp30 4 = 32
      - HIL step (B32):    Task1 Clean 10 + Task2 Clean 10 + Rollout100 6 + TopGrasp30 4 + HIL 2 = 32 (when step % hil_interval == 0)

    Guarantees:
      - Clean T1 and Clean T2 are sampled strictly 50:50 (not proportional to 353:222).
      - Inside each source, episodes are drawn uniformly without replacement before cycle repeat.
      - Within each chosen episode, a valid frame index is selected uniformly.
    """

    def __init__(
        self,
        episodes_parquet: Path,
        hil_interval: int = 3,
        seed: int = 42,
    ):
        self.hil_interval = hil_interval
        self.rng = np.random.default_rng(seed)

        ep_df = pd.read_parquet(episodes_parquet)
        total_episodes = len(ep_df)
        if total_episodes != 755:
            raise ValueError(f"General dataset expected 755 episodes, found {total_episodes}")

        self.clean_t1_pools: Dict[int, List[int]] = {}
        self.clean_t2_pools: Dict[int, List[int]] = {}
        self.rollout100_pools: Dict[int, List[int]] = {}
        self.topgrasp30_pools: Dict[int, List[int]] = {}
        self.hil_pools: Dict[int, List[int]] = {}

        for ep_idx in range(total_episodes):
            row = ep_df.iloc[ep_idx]
            f_start = int(row["dataset_from_index"])
            f_end = int(row["dataset_to_index"])
            # Exclude first 5 static dwell frames and guarantee at least 1 valid sample
            s = f_start + 5
            e = max(s + 1, f_end - 1)
            frames = list(range(s, e))

            if CLEAN_T1_RANGE[0] <= ep_idx <= CLEAN_T1_RANGE[1]:
                self.clean_t1_pools[ep_idx] = frames
            elif CLEAN_T2_RANGE[0] <= ep_idx <= CLEAN_T2_RANGE[1]:
                self.clean_t2_pools[ep_idx] = frames
            elif ROLLOUT100_RANGE[0] <= ep_idx <= ROLLOUT100_RANGE[1]:
                self.rollout100_pools[ep_idx] = frames
            elif TOPGRASP30_RANGE[0] <= ep_idx <= TOPGRASP30_RANGE[1]:
                self.topgrasp30_pools[ep_idx] = frames
            elif HIL_RANGE[0] <= ep_idx <= HIL_RANGE[1]:
                self.hil_pools[ep_idx] = frames
            else:
                raise ValueError(f"Unmapped episode index {ep_idx}")

        assert len(self.clean_t1_pools) == 353
        assert len(self.clean_t2_pools) == 222
        assert len(self.rollout100_pools) == 100
        assert len(self.topgrasp30_pools) == 30
        assert len(self.hil_pools) == 50

        # Cycle queues for episode-balanced sampling
        self.clean_t1_queue: List[int] = []
        self.clean_t2_queue: List[int] = []
        self.rollout100_queue: List[int] = []
        self.topgrasp30_queue: List[int] = []
        self.hil_queue: List[int] = []

    def _sample_from_pool(self, pools: Dict[int, List[int]], queue: List[int], count: int) -> List[int]:
        if count <= 0:
            return []
        all_eps = list(pools.keys())
        chosen_frames: List[int] = []
        for _ in range(count):
            if not queue:
                queue.extend(self.rng.permutation(all_eps).tolist())
            ep = queue.pop(0)
            avail_frames = pools[ep]
            frame_idx = self.rng.choice(avail_frames)
            chosen_frames.append(int(frame_idx))
        return chosen_frames

    def sample_step(self, step: int) -> Tuple[List[int], Dict[str, int], bool]:
        """Returns (sampled_indices, counts_dict, hil_injected_bool)."""
        hil_active = (step % self.hil_interval == 0)

        if hil_active:
            n_t1 = 10
            n_t2 = 10
            n_rollout = 6
            n_topgrasp = 4
            n_hil = 2
        else:
            n_t1 = 11
            n_t2 = 11
            n_rollout = 6
            n_topgrasp = 4
            n_hil = 0

        clean_t1_idx = self._sample_from_pool(self.clean_t1_pools, self.clean_t1_queue, n_t1)
        clean_t2_idx = self._sample_from_pool(self.clean_t2_pools, self.clean_t2_queue, n_t2)
        rollout_idx = self._sample_from_pool(self.rollout100_pools, self.rollout100_queue, n_rollout)
        topgrasp_idx = self._sample_from_pool(self.topgrasp30_pools, self.topgrasp30_queue, n_topgrasp)
        hil_idx = self._sample_from_pool(self.hil_pools, self.hil_queue, n_hil)

        all_indices = clean_t1_idx + clean_t2_idx + rollout_idx + topgrasp_idx + hil_idx
        assert len(all_indices) == 32

        counts = {
            "clean_t1": n_t1,
            "clean_t2": n_t2,
            "rollout": n_rollout,
            "topgrasp": n_topgrasp,
            "hil": n_hil,
            "total": len(all_indices),
        }
        return all_indices, counts, hil_active


class GeneralRWFMManager:
    """Manages failure masking, clean placement masking, and reward-weighted loss computation for General 755ep dataset."""

    def __init__(
        self,
        dataset_root: Path,
        temperature: float = 1.0,
        chunk_size: int = 50,
        device: torch.device = torch.device("cpu"),
    ):
        self.dataset_root = dataset_root
        self.temperature = temperature
        self.chunk_size = chunk_size
        self.device = device

        rew_path = dataset_root / "meta/episode_frame_rewards.json"
        ann_path = dataset_root / "meta/rwfm_rollout_annotations.json"
        learn_seg_path = dataset_root / "meta/rwfm_learn_segments.json"

        if not rew_path.exists():
            raise FileNotFoundError(f"Authoritative episode_frame_rewards.json missing: {rew_path}")
        if not ann_path.exists():
            raise FileNotFoundError(f"Authoritative rwfm_rollout_annotations.json missing: {ann_path}")

        self.frame_rewards = json.loads(rew_path.read_text())
        self.annotations = json.loads(ann_path.read_text()).get("episodes", {})

        # 1. Actual rollout failure intervals (episodes 575..674, etc.)
        self.failure_intervals: Dict[int, List[Tuple[int, int]]] = collections.defaultdict(list)
        for ep_str, info in self.annotations.items():
            ep_idx = int(ep_str)
            for seg in info.get("segments", []):
                if seg.get("type") == "failure":
                    self.failure_intervals[ep_idx].append((int(seg["start"]), int(seg["end"])))

        # 2. Clean placement intervals from meta/rwfm_learn_segments.json (episodes 0..574)
        self.clean_placement_intervals: Dict[int, List[Tuple[int, int]]] = collections.defaultdict(list)
        if learn_seg_path.exists():
            learn_seg_data = json.loads(learn_seg_path.read_text()).get("episodes", {})
            for ep_str, info in learn_seg_data.items():
                ep_idx = int(ep_str)
                for interval in info.get("masked_intervals", []):
                    self.clean_placement_intervals[ep_idx].append((int(interval[0]), int(interval[1])))

        # RewardSampleWeighter
        cfg = SampleWeightingConfig(
            type="reward_weighted",
            temperature=temperature,
            chunk_size=chunk_size,
            frame_reward_path=str(rew_path),
        )
        self.weighter = RewardSampleWeighter(cfg, device=device, dataset_root=dataset_root)

    def is_frame_failure(self, ep_idx: int, frame_idx: int) -> bool:
        if ep_idx not in self.failure_intervals:
            return False
        for s, e in self.failure_intervals[ep_idx]:
            if s <= frame_idx < e:
                return True
        return False

    def is_frame_clean_placement_masked(self, ep_idx: int, frame_idx: int) -> bool:
        if ep_idx not in self.clean_placement_intervals:
            return False
        for s, e in self.clean_placement_intervals[ep_idx]:
            if s <= frame_idx < e:
                return True
        return False

    def build_rwfm_learn_mask(
        self,
        episode_indices: List[int],
        frame_indices: List[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Constructs [B, K] boolean masks:
        - learn_mask: True = learnable, False = do not learn (either actual failure OR clean placement masked)
        - is_failure: True = actual rollout failure
        - is_clean_placement: True = clean placement mask
        """
        B = len(episode_indices)
        K = self.chunk_size
        learn_mask = torch.ones((B, K), dtype=torch.bool, device=self.device)
        is_failure = torch.zeros((B, K), dtype=torch.bool, device=self.device)
        is_clean_placement = torch.zeros((B, K), dtype=torch.bool, device=self.device)

        for b in range(B):
            ep_i = episode_indices[b]
            f_i = frame_indices[b]
            for k in range(K):
                target_f = f_i + k
                if self.is_frame_failure(ep_i, target_f):
                    learn_mask[b, k] = False
                    is_failure[b, k] = True
                elif self.is_frame_clean_placement_masked(ep_i, target_f):
                    learn_mask[b, k] = False
                    is_clean_placement[b, k] = True

        return learn_mask, is_failure, is_clean_placement

    def compute_weighted_general_loss(
        self,
        policy: SmolVLAPolicy,
        batch: Dict[str, torch.Tensor],
        raw_samples: List[Dict[str, Any]],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Executes failure & clean placement action loss masking and reward-weighted flow matching loss computation."""
        B = len(raw_samples)
        K = self.chunk_size

        ep_indices = [
            int(s["episode_index"].item() if isinstance(s["episode_index"], torch.Tensor) else s["episode_index"])
            for s in raw_samples
        ]
        f_indices = [
            int(s["frame_index"].item() if isinstance(s["frame_index"], torch.Tensor) else s["frame_index"])
            for s in raw_samples
        ]

        # 1. Build learn mask and component masks [B, K]
        learn_mask, is_failure, is_clean_placement = self.build_rwfm_learn_mask(ep_indices, f_indices)

        # 2. Formulate effective_action_is_pad = original_action_is_pad | (~learn_mask)
        orig_pad = batch.get("action_is_pad")
        if orig_pad is not None:
            effective_action_is_pad = orig_pad.to(self.device) | (~learn_mask)
        else:
            effective_action_is_pad = ~learn_mask

        batch["action_is_pad"] = effective_action_is_pad
        batch["rwfm_learn_mask"] = learn_mask
        batch["episode_index"] = torch.tensor(ep_indices, dtype=torch.long, device=self.device)
        batch["frame_index"] = torch.tensor(f_indices, dtype=torch.long, device=self.device)

        # 3. Compute per-sample losses via policy forward with reduction="none"
        per_sample_loss, loss_dict = policy.forward(batch, reduction="none")

        # 4. Compute reward weights (excluding failure, clean placement & padding actions)
        weights, weight_stats = self.weighter.compute_batch_weights(batch, learn_mask=learn_mask)

        # 5. Mask out entirely invalid samples (where all 50 actions are failure, placement, or pad)
        valid_sample_mask = (~effective_action_is_pad).any(dim=1)  # shape (B,)
        valid_weights = weights * valid_sample_mask.float()
        total_valid_weight = valid_weights.sum().clamp_min(1e-12)

        weighted_loss = (valid_weights * per_sample_loss).sum() / total_valid_weight

        # Metrics bookkeeping
        total_actions = B * K
        failure_actions = is_failure.sum().item()
        clean_placement_actions = is_clean_placement.sum().item()
        valid_actions = (~effective_action_is_pad).sum().item()

        stats = {
            "loss_general_raw": per_sample_loss.mean().item(),
            "loss_general_weighted": weighted_loss.item(),
            "reward_mean": weight_stats.get("mean_reward", 0.0),
            "reward_min": weight_stats.get("min_reward", 0.0),
            "reward_max": weight_stats.get("max_reward", 0.0),
            "weight_mean": weights.mean().item(),
            "weight_min": weights.min().item(),
            "weight_max": weights.max().item(),
            "valid_action_ratio": valid_actions / float(total_actions),
            "masked_failure_action_ratio": failure_actions / float(total_actions),
            "masked_clean_placement_action_ratio": clean_placement_actions / float(total_actions),
            "masked_sample_count": (B - valid_sample_mask.sum().item()),
        }

        return weighted_loss, stats


def parse_combined_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified General Expert-Only RWFM + Same-Color PCGrad SmolVLA Trainer.")

    # Model & Datasets
    parser.add_argument("--base-policy", type=str, default=DEFAULT_BASE_POLICY, help="Base 865 pretrained model.")
    parser.add_argument("--general-dataset", type=str, default=DEFAULT_GENERAL_DATASET, help="General 755ep dataset.")
    parser.add_argument("--task1-dataset", type=str, default=DEFAULT_TASK1_DATASET, help="Task 1 clean branch dataset (120ep).")
    parser.add_argument("--task2-dataset", type=str, default=DEFAULT_TASK2_DATASET, help="Task 2 clean branch dataset (127ep).")
    parser.add_argument("--job-name", type=str, default="smolvla_865base_rwfm755_masked575_samecolor_pcgrad_expertonly_30k", help="Job run name.")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for checkpoints and logs.")

    # Training Steps & Schedule
    parser.add_argument("--steps", type=int, default=30000, help="Total training steps (default 30,000).")
    parser.add_argument("--lr", type=float, default=1e-6, help="Peak learning rate (default 1e-6).")
    parser.add_argument("--final-lr", type=float, default=3e-7, help="Final learning rate (default 3e-7).")
    parser.add_argument("--warmup-steps", type=int, default=300, help="Linear warmup steps (default 300).")
    parser.add_argument("--save-freq", type=int, default=5000, help="Save frequency in global steps (default 5,000).")
    parser.add_argument("--log-freq", type=int, default=20, help="Console logging frequency (default 20).")
    parser.add_argument("--grad-clip-norm", type=float, default=10.0, help="Gradient clipping max norm (default 10.0).")

    # General RWFM Objective
    parser.add_argument("--general-batch-size", type=int, default=32, help="General objective batch size (default 32).")
    parser.add_argument("--hil-interval", type=int, default=3, help="HIL injection interval in global steps (default 3).")
    parser.add_argument("--temperature", type=float, default=1.0, help="RWFM softmax temperature (default 1.0).")
    parser.add_argument("--chunk-size", type=int, default=50, help="Action chunk size (default 50).")

    # Placement PCGrad Objective
    parser.add_argument("--pcgrad-interval", type=int, default=2, help="PCGrad invocation interval in global steps (default 2).")
    parser.add_argument("--microbatch-per-task", type=int, default=16, help="PCGrad batch size per task (default 16; total 32).")
    parser.add_argument("--micro-batch-size", type=int, default=8, help="Micro-batch chunk size for PCGrad gradient accumulation.")
    parser.add_argument("--use-pcgrad", type=lambda x: str(x).lower() in ("true", "1", "yes"), default=True, help="Apply PCGrad projection.")
    parser.add_argument("--paired-fm-randomness", type=lambda x: str(x).lower() in ("true", "1", "yes"), default=True, help="Paired noise/time for T1/T2.")

    # Gradient Combination & Anchor Projection
    parser.add_argument("--general-gradient-weight", type=float, default=1.0, help="Weight for General RWFM gradient (default 1.0).")
    parser.add_argument("--pcgrad-gradient-weight", type=float, default=1.0, help="Weight for Placement PCGrad gradient (default 1.0).")
    parser.add_argument(
        "--general-anchor-projection",
        type=lambda x: str(x).lower() in ("true", "1", "yes"),
        default=False,
        help="Optional: project g_place against g_general if dot < 0 (default False).",
    )

    # System & Tracking
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-wandb", type=lambda x: str(x).lower() in ("true", "1", "yes"), default=True)
    parser.add_argument("--wandb-project", type=str, default="lerobot-so101-grad")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--push-to-hub", type=lambda x: str(x).lower() in ("true", "1", "yes"), default=True)
    parser.add_argument("--hub-repo-id", type=str, default=None)
    parser.add_argument("--resume", type=lambda x: str(x).lower() in ("true", "1", "yes"), default=False)
    parser.add_argument("--allow-source-count-mismatch", action="store_true", default=False)

    return parser.parse_args()


def configure_expert_only_smolvla(policy: SmolVLAPolicy, logger: logging.Logger) -> List[torch.nn.Parameter]:
    """Configures SmolVLA policy strictly into Expert-Only mode matching project contract."""
    policy.config.train_expert_only = True
    policy.config.freeze_vision_encoder = True
    policy.config.train_state_proj = False

    # Apply requires_grad settings
    policy.model.vlm_with_expert.train_expert_only = True
    policy.model.vlm_with_expert.freeze_vision_encoder = True
    policy.model.vlm_with_expert.set_requires_grad()
    policy.model.set_requires_grad()

    trainable_params: List[torch.nn.Parameter] = []
    param_names: List[str] = []
    for name, param in policy.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
            param_names.append(name)

    logger.info("=" * 70)
    logger.info("🔒 [EXPERT-ONLY CONFIGURATION]")
    logger.info(f"Trainable Tensors Count: {len(trainable_params)} (Expected: 153)")
    logger.info(f"Trainable Parameters   : {sum(p.numel() for p in trainable_params):,} ({sum(p.numel() for p in trainable_params)/1e6:.2f}M)")
    prefixes = sorted(set(".".join(n.split(".")[:3]) for n in param_names))
    logger.info(f"Trainable Prefixes     : {prefixes}")
    logger.info("=" * 70)

    if len(trainable_params) != 153:
        logger.warning(f"Trainable tensors count is {len(trainable_params)}, expected 153 for standard SmolVLA Action Expert.")

    return trainable_params


def main():
    args = parse_combined_args()

    if args.output_dir is None:
        args.output_dir = f"outputs/train/{args.job_name}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out_dir)

    if args.hub_repo_id is None:
        args.hub_repo_id = f"eslab1234/{args.job_name}"

    logger.info("=" * 85)
    logger.info("🚀 SmolVLA Unified Trainer: General Expert-Only RWFM + Same-Color PCGrad")
    logger.info("=" * 85)
    logger.info(f"Base Policy          : {args.base_policy}")
    logger.info(f"General 755ep Dataset: {args.general_dataset}")
    logger.info(f"Task 1 Branch Dataset: {args.task1_dataset}")
    logger.info(f"Task 2 Branch Dataset: {args.task2_dataset}")
    logger.info(f"Output Directory     : {out_dir}")
    logger.info(f"Training Steps       : {args.steps}")
    logger.info(f"Learning Rate        : {args.lr} -> {args.final_lr} (warmup: {args.warmup_steps} steps)")
    logger.info(f"General Batch Size   : {args.general_batch_size} (normal: T1=11, T2=11, Rollout=6, TopGrasp=4 | HIL: T1=10, T2=10, Rollout=6, TopGrasp=4, HIL=2, interval={args.hil_interval})")
    logger.info(f"PCGrad Interval      : {args.pcgrad_interval}")
    logger.info(f"PCGrad Batch Size    : {args.microbatch_per_task} T1 + {args.microbatch_per_task} T2")
    logger.info(f"Gradient Weights     : General={args.general_gradient_weight}, PCGrad={args.pcgrad_gradient_weight}")
    logger.info(f"Anchor Projection    : {args.general_anchor_projection}")
    logger.info(f"Resume Mode          : {args.resume}")
    logger.info(f"Device               : {args.device}")
    logger.info(f"Seed                 : {args.seed}")
    logger.info("=" * 85)

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1. Resolve Dataset Paths
    p_gen = resolve_local_path(args.general_dataset)
    p_t1 = resolve_local_path(args.task1_dataset)
    p_t2 = resolve_local_path(args.task2_dataset)

    logger.info(f"Resolved General Path: {p_gen}")
    logger.info(f"Resolved Task 1 Path : {p_t1}")
    logger.info(f"Resolved Task 2 Path : {p_t2}")

    # 2. Image Transforms (ColorJitter/Sharpness only, NO RandomAffine geometric distortion)
    fps = 30.0
    delta_ts = {"action": [i / fps for i in range(args.chunk_size)]}
    tf_config = ImageTransformsConfig(enable=True, max_num_transforms=3, random_order=False)
    image_transforms = ImageTransforms(tf_config)

    # 3. Load Datasets
    logger.info("\nLoading General and Branch LeRobotDataset instances...")
    ds_gen = LeRobotDataset(args.general_dataset, root=p_gen, image_transforms=image_transforms, delta_timestamps=delta_ts, return_uint8=True)
    ds_t1 = LeRobotDataset(args.task1_dataset, root=p_t1, image_transforms=image_transforms, delta_timestamps=delta_ts, return_uint8=True)
    ds_t2 = LeRobotDataset(args.task2_dataset, root=p_t2, image_transforms=image_transforms, delta_timestamps=delta_ts, return_uint8=True)

    if ds_gen.num_episodes != 755 and not args.allow_source_count_mismatch:
        raise ValueError(f"General dataset expected 755 episodes, found {ds_gen.num_episodes}")
    if ds_t1.num_episodes != 120 and not args.allow_source_count_mismatch:
        raise ValueError(f"Task 1 branch expected 120 episodes, found {ds_t1.num_episodes}")
    if ds_t2.num_episodes != 127 and not args.allow_source_count_mismatch:
        raise ValueError(f"Task 2 branch expected 127 episodes, found {ds_t2.num_episodes}")

    logger.info(f"✓ General Dataset verified: {ds_gen.num_episodes} episodes ({ds_gen.num_frames:,} frames)")
    logger.info(f"✓ Task 1 Branch verified  : {ds_t1.num_episodes} episodes ({ds_t1.num_frames:,} frames)")
    logger.info(f"✓ Task 2 Branch verified  : {ds_t2.num_episodes} episodes ({ds_t2.num_frames:,} frames)")

    # 4. Samplers & Managers
    gen_ep_parquet = p_gen / "meta/episodes/chunk-000/file-000.parquet"
    gen_sampler = GeneralSourceBalancedSampler(
        episodes_parquet=gen_ep_parquet,
        hil_interval=args.hil_interval,
        seed=args.seed,
    )

    t1_parquet = p_t1 / "meta/episodes/chunk-000/file-000.parquet"
    t2_parquet = p_t2 / "meta/episodes/chunk-000/file-000.parquet"
    pcgrad_sampler = SameColorBalancedSampler(
        t1_parquet=t1_parquet,
        t2_parquet=t2_parquet,
        t1_ranges=TASK1_EPISODE_RANGES,
        t2_ranges=TASK2_EPISODE_RANGES,
        batch_size_per_task=args.microbatch_per_task,
        seed=args.seed,
    )

    rwfm_manager = GeneralRWFMManager(
        dataset_root=p_gen,
        temperature=args.temperature,
        chunk_size=args.chunk_size,
        device=device,
    )

    # 5. Load Base Model and Normalizer
    logger.info(f"\nLoading Base Policy: {args.base_policy} ...")
    policy = SmolVLAPolicy.from_pretrained(args.base_policy)
    policy.to(device)
    policy.train()

    trainable_params = configure_expert_only_smolvla(policy, logger)

    # Normalization Fidelity Check
    logger.info("\n" + "=" * 70)
    logger.info("✨ [NORMALIZER FIDELITY CHECK]")
    logger.info(f"Using BASE checkpoint preprocessor/normalizer from: {args.base_policy}")
    logger.info("Dataset stats will NOT override pretrained normalizer.")
    stats = load_model_exact_stats(args.base_policy, logger)
    preprocessor, _ = make_pre_post_processors(policy_cfg=policy.config, dataset_stats=stats)
    logger.info("Base normalizer successfully instantiated and locked.")
    logger.info("=" * 70 + "\n")

    # 6. Optimizer and Scheduler
    optimizer, scheduler = build_optimizer_and_scheduler(
        trainable_params=trainable_params,
        peak_lr=args.lr,
        final_lr=args.final_lr,
        warmup_steps=args.warmup_steps,
        total_steps=args.steps,
    )

    # WandB Logger
    wandb_logger = None
    if args.use_wandb:
        try:
            wandb_logger = PCGradWandBLogger(
                project=args.wandb_project,
                entity=args.wandb_entity,
                job_name=args.job_name,
                output_dir=out_dir,
                config=vars(args),
                logger=logger,
            )
        except Exception as e:
            logger.warning(f"Could not initialize WandB: {e}. Running without WandB.")

    # Resume Checkpoint if requested
    start_step = 1
    last_ckpt_link = out_dir / "checkpoints/last"
    if args.resume and last_ckpt_link.exists():
        logger.info(f"Resuming training state from {last_ckpt_link.resolve()} ...")
        start_step = load_training_state(last_ckpt_link.resolve(), optimizer, scheduler) + 1
        logger.info(f"Resumed successfully at global step {start_step}")

        # Fast-forward samplers to guarantee exact deterministic continuation
        logger.info(f"Fast-forwarding samplers to synchronize deterministic state to step {start_step} ...")
        for s in range(1, start_step):
            gen_sampler.sample_step(s)

        pcgrad_invocations_prior = sum(1 for s in range(1, start_step) if (s % args.pcgrad_interval == 0))
        for _ in range(pcgrad_invocations_prior):
            pcgrad_sampler.sample_step()

        logger.info(
            f"✓ Samplers synchronized: General={start_step-1} steps, PCGrad={pcgrad_invocations_prior} invocations advanced."
        )

    # CSV Logger
    csv_log_path = out_dir / "combined_training_metrics.csv"
    csv_file = open(csv_log_path, "a" if args.resume else "w", newline="", encoding="utf-8")
    csv_fields = [
        "step", "lr", "loss_general", "loss_general_weighted", "hil_injected",
        "pcgrad_active", "pcgrad_color", "loss_t1", "loss_t2",
        "pcgrad_cosine", "pcgrad_dot", "pcgrad_conflict",
        "gen_place_dot", "gen_place_cosine",
        "g_general_norm", "g_place_norm", "g_final_norm", "step_time_s", "gpu_mem_mb",
    ]
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    if not args.resume:
        csv_writer.writeheader()

    beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
    pcgrad_invocation_count = 0
    hil_injection_count = 0
    optimizer_step_count = 0
    last_saved_ckpt = None

    logger.info("--- Starting Unified Training Loop ---")
    t_start = time.time()

    for step in range(start_step, args.steps + 1):
        t0 = time.time()
        pcgrad_step = (step % args.pcgrad_interval == 0)

        # -------------------------------------------------------------
        # STEP A: General RWFM Forward & Loss Computation
        # -------------------------------------------------------------
        gen_indices, gen_counts, hil_injected = gen_sampler.sample_step(step)
        if hil_injected:
            hil_injection_count += 1

        raw_gen_samples = [ds_gen[i] for i in gen_indices]
        gen_batch = prepare_torch_batch(raw_gen_samples, preprocessor, device)

        L_general, rwfm_stats = rwfm_manager.compute_weighted_general_loss(
            policy=policy,
            batch=gen_batch,
            raw_samples=raw_gen_samples,
        )

        # -------------------------------------------------------------
        # STEP B: Gradient Evaluation & Multi-Rate Combination
        # -------------------------------------------------------------
        if not pcgrad_step:
            # Fast Path: Single Objective Backward
            policy.zero_grad(set_to_none=True)
            scaled_gen_loss = L_general * args.general_gradient_weight
            scaled_gen_loss.backward()

            final_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.grad_clip_norm).item()
            optimizer.step()
            scheduler.step()
            optimizer_step_count += 1
            optimizer.zero_grad(set_to_none=True)

            g_gen_norm = final_norm
            g_place_norm = 0.0
            raw_dot = 0.0
            raw_cos = 0.0
            is_conflict = False
            pc_color = "none"
            loss_t1 = 0.0
            loss_t2 = 0.0

        else:
            # PCGrad Path: Multi-Objective Gradient Projection
            pcgrad_invocation_count += 1

            # 1. Compute g_general via autograd.grad and release graph
            g_general = torch.autograd.grad(L_general, trainable_params, retain_graph=False, create_graph=False)
            g_general_tensors = [g.detach().clone() if g is not None else torch.zeros_like(p) for g, p in zip(g_general, trainable_params)]
            del L_general, gen_batch

            # 2. Sample same-color paired batches from Task 1 and Task 2
            pc_color, idx_t1, idx_t2 = pcgrad_sampler.sample_step()
            raw_t1 = [ds_t1[i] for i in idx_t1]
            raw_t2 = [ds_t2[i] for i in idx_t2]

            # 3. Paired Flow Matching Noise and Time Tensors
            if args.paired_fm_randomness:
                time_tensor = beta_dist.sample((args.microbatch_per_task,)).to(device=device, dtype=torch.float32) * 0.999 + 0.001
                noise_tensor = torch.randn((args.microbatch_per_task, args.chunk_size, 32), device=device, dtype=torch.float32)
                time_t1, time_t2 = time_tensor, time_tensor
                noise_t1, noise_t2 = noise_tensor, noise_tensor
            else:
                time_t1 = beta_dist.sample((args.microbatch_per_task,)).to(device=device, dtype=torch.float32) * 0.999 + 0.001
                time_t2 = beta_dist.sample((args.microbatch_per_task,)).to(device=device, dtype=torch.float32) * 0.999 + 0.001
                noise_t1 = torch.randn((args.microbatch_per_task, args.chunk_size, 32), device=device, dtype=torch.float32)
                noise_t2 = torch.randn((args.microbatch_per_task, args.chunk_size, 32), device=device, dtype=torch.float32)

            # 4. Compute Gradients for Task 1 and Task 2 independently
            g1, loss_t1 = compute_accumulated_task_gradient(
                policy=policy,
                trainable_params=trainable_params,
                raw_samples=raw_t1,
                preprocessor=preprocessor,
                noise_tensor=noise_t1,
                time_tensor=time_t1,
                micro_batch_size=args.micro_batch_size,
                device=device,
            )
            g2, loss_t2 = compute_accumulated_task_gradient(
                policy=policy,
                trainable_params=trainable_params,
                raw_samples=raw_t2,
                preprocessor=preprocessor,
                noise_tensor=noise_t2,
                time_tensor=time_t2,
                micro_batch_size=args.micro_batch_size,
                device=device,
            )

            # 5. Canonical Same-Color PCGrad Projection
            if args.use_pcgrad:
                g1_pc, g2_pc, raw_dot, raw_cos, is_conflict = compute_pcgrad(g1, g2)
            else:
                g1_pc, g2_pc = g1, g2
                is_conflict = False
                raw_dot, raw_cos = 0.0, 0.0

            g_place = [0.5 * (p1 + p2) for p1, p2 in zip(g1_pc, g2_pc)]

            # Evaluate dot and cosine between g_general and g_place (pre-anchor projection)
            g_gen_norm = math.sqrt(sum(torch.sum(p.float() ** 2).item() for p in g_general_tensors))
            g_place_norm = math.sqrt(sum(torch.sum(p.float() ** 2).item() for p in g_place))
            gen_place_dot = sum(torch.sum(pg.float() * pp.float()).item() for pg, pp in zip(g_general_tensors, g_place))
            gen_place_cosine = gen_place_dot / (g_gen_norm * g_place_norm + 1e-12) if (g_gen_norm > 1e-12 and g_place_norm > 1e-12) else 0.0

            # 6. Optional General Anchor Projection
            anchor_applied = False
            g_place_before_anchor_norm = g_place_norm
            if args.general_anchor_projection:
                dot_ag = gen_place_dot
                norm_gen_sq = g_gen_norm ** 2
                if dot_ag < 0.0 and norm_gen_sq > 1e-12:
                    scale_ag = dot_ag / norm_gen_sq
                    g_place = [(pp - scale_ag * pg) for pp, pg in zip(g_place, g_general_tensors)]
                    anchor_applied = True
                    g_place_norm = math.sqrt(sum(torch.sum(p.float() ** 2).item() for p in g_place))

            # 7. Final Combined Gradient: GENERAL_WEIGHT * g_general + PCGRAD_WEIGHT * g_place
            g_final = [
                args.general_gradient_weight * gg + args.pcgrad_gradient_weight * gp
                for gg, gp in zip(g_general_tensors, g_place)
            ]

            # Assign to parameter grad
            for param, g_val in zip(trainable_params, g_final):
                param.grad = g_val.to(dtype=param.dtype)

            # Exactly ONE optimizer.step() and scheduler.step() per global step
            final_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.grad_clip_norm).item()
            optimizer.step()
            scheduler.step()
            optimizer_step_count += 1
            optimizer.zero_grad(set_to_none=True)

        # -------------------------------------------------------------
        # STEP C: Metrics, Tracking & Logging
        # -------------------------------------------------------------
        step_time = time.time() - t0
        curr_lr = scheduler.get_last_lr()[0]
        gpu_alloc_mb = torch.cuda.memory_allocated(device) / (1024 * 1024) if torch.cuda.is_available() else 0.0

        # CSV record
        csv_writer.writerow({
            "step": step,
            "lr": f"{curr_lr:.3e}",
            "loss_general": round(rwfm_stats["loss_general_raw"], 5),
            "loss_general_weighted": round(rwfm_stats["loss_general_weighted"], 5),
            "hil_injected": int(hil_injected),
            "pcgrad_active": int(pcgrad_step),
            "pcgrad_color": pc_color,
            "loss_t1": round(loss_t1, 5) if pcgrad_step else 0.0,
            "loss_t2": round(loss_t2, 5) if pcgrad_step else 0.0,
            "pcgrad_cosine": round(raw_cos, 4) if pcgrad_step else 0.0,
            "pcgrad_dot": round(raw_dot, 4) if pcgrad_step else 0.0,
            "pcgrad_conflict": int(is_conflict) if pcgrad_step else 0,
            "gen_place_dot": round(gen_place_dot, 4) if pcgrad_step else 0.0,
            "gen_place_cosine": round(gen_place_cosine, 4) if pcgrad_step else 0.0,
            "g_general_norm": round(g_gen_norm, 3),
            "g_place_norm": round(g_place_norm, 3),
            "g_final_norm": round(final_norm, 3),
            "step_time_s": round(step_time, 3),
            "gpu_mem_mb": round(gpu_alloc_mb, 1),
        })
        if step % 50 == 0:
            csv_file.flush()

        # WandB Logging
        if wandb_logger is not None:
            wb_data = {
                "train/global_step": step,
                "train/lr": curr_lr,
                "train/general_loss": rwfm_stats["loss_general_raw"],
                "train/general_weighted_loss": rwfm_stats["loss_general_weighted"],
                "train/hil_injected": 1.0 if hil_injected else 0.0,
                "train/pcgrad_invocation_count": pcgrad_invocation_count,
                "train/step_time_s": step_time,
                "train/gpu_mem_alloc_mb": gpu_alloc_mb,
                # RWFM metrics
                "rwfm/reward_mean": rwfm_stats["reward_mean"],
                "rwfm/reward_min": rwfm_stats["reward_min"],
                "rwfm/reward_max": rwfm_stats["reward_max"],
                "rwfm/weight_mean": rwfm_stats["weight_mean"],
                "rwfm/weight_min": rwfm_stats["weight_min"],
                "rwfm/weight_max": rwfm_stats["weight_max"],
                "rwfm/valid_action_ratio": rwfm_stats["valid_action_ratio"],
                "rwfm/masked_failure_action_ratio": rwfm_stats["masked_failure_action_ratio"],
                "rwfm/masked_clean_placement_action_ratio": rwfm_stats["masked_clean_placement_action_ratio"],
                "rwfm/source_clean_t1_count": gen_counts["clean_t1"],
                "rwfm/source_clean_t2_count": gen_counts["clean_t2"],
                "rwfm/source_rollout100_count": gen_counts["rollout"],
                "rwfm/source_topgrasp30_count": gen_counts["topgrasp"],
                "rwfm/source_hil_count": gen_counts["hil"],
                # Gradient metrics
                "grad/general_norm": g_gen_norm,
                "grad/final_norm": final_norm,
            }
            if pcgrad_step:
                wb_data.update({
                    "pcgrad/active": 1.0,
                    f"pcgrad/color_{pc_color.lower()}": 1.0,
                    "pcgrad/task1_loss": loss_t1,
                    "pcgrad/task2_loss": loss_t2,
                    "pcgrad/raw_dot": raw_dot,
                    "pcgrad/raw_cosine": raw_cos,
                    "pcgrad/conflict": 1.0 if is_conflict else 0.0,
                    "pcgrad/g_place_norm": g_place_norm,
                    "grad/place_norm": g_place_norm,
                    "grad/general_place_dot": gen_place_dot,
                    "grad/general_place_cosine": gen_place_cosine,
                    "grad/effective_place_to_general_norm_ratio": g_place_norm / (g_gen_norm + 1e-12),
                })
            else:
                wb_data["pcgrad/active"] = 0.0
            wandb_logger.log_dict(wb_data, step=step)

        # Periodic Console Logging
        if step % args.log_freq == 0 or step == 1 or pcgrad_step or hil_injected:
            pc_str = f" | PCGrad({pc_color}): {'⚠️CONFLICT' if is_conflict else '✅ALIGNED'} (cos={raw_cos:+.2f})" if pcgrad_step else ""
            hil_str = " [+HIL]" if hil_injected else ""
            logger.info(
                f"Step {step:5d}/{args.steps} | GenLoss: {rwfm_stats['loss_general_weighted']:.4f} (raw={rwfm_stats['loss_general_raw']:.4f}){hil_str}{pc_str} | "
                f"||g_fin||: {final_norm:.2f} | LR: {curr_lr:.2e} | Rew: {rwfm_stats['reward_mean']:+.2f} | {step_time:.2f}s"
            )

        # -------------------------------------------------------------
        # STEP D: Checkpointing
        # -------------------------------------------------------------
        if step % args.save_freq == 0 or step == args.steps:
            last_saved_ckpt = save_compatible_checkpoint(
                out_dir=out_dir,
                total_steps=args.steps,
                step=step,
                policy=policy,
                optimizer=optimizer,
                scheduler=scheduler,
                base_policy_path=args.base_policy,
                logger=logger,
                wandb_logger=wandb_logger,
            )

    csv_file.close()
    elapsed = time.time() - t_start
    logger.info("=" * 85)
    logger.info(f"🎉 Combined Training Complete! Total time: {elapsed / 3600:.2f} hours ({elapsed:.1f}s)")
    logger.info(f"Total Optimizer Steps   : {optimizer_step_count}")
    logger.info(f"Total PCGrad Invocations: {pcgrad_invocation_count}")
    logger.info(f"Total HIL Injections    : {hil_injection_count}")
    if torch.cuda.is_available():
        peak_alloc = torch.cuda.max_memory_allocated(device) / (1024 * 1024 * 1024)
        peak_res = torch.cuda.max_memory_reserved(device) / (1024 * 1024 * 1024)
        logger.info(f"Peak Allocated VRAM     : {peak_alloc:.2f} GB ({peak_alloc * 1024:.1f} MB)")
        logger.info(f"Peak Reserved VRAM      : {peak_res:.2f} GB ({peak_res * 1024:.1f} MB)")
    logger.info(f"Checkpoints directory   : {out_dir / 'checkpoints'}")
    logger.info("=" * 85)

    # 7. Final Hub Push
    if args.push_to_hub and last_saved_ckpt is not None:
        push_final_checkpoint_to_hub(
            final_checkpoint_dir=last_saved_ckpt,
            hub_repo_id=args.hub_repo_id,
            logger=logger,
        )


if __name__ == "__main__":
    main()
