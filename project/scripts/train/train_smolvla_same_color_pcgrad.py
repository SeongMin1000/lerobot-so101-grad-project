#!/usr/bin/env python3
"""
train_smolvla_same_color_pcgrad.py

Same-Color Paired PCGrad Expert-Only Corrective Fine-Tuning for SmolVLA.
Preserves the primary 822ep model base while resolving cross-task gradient conflict
between Task 1 and Task 2 on identical block color contexts during post-grasp placement.

Key Architecture & Strategy:
  - Base Model: 822ep Expert-only checkpoint (eslab1234/smolvla_multitask_5blocks_v3_822ep_from865_expertonly_b32_lr5e6_50k)
  - Trainable Scope: Action Expert only (freeze_vision_encoder=True, train_expert_only=True, train_state_proj=False)
  - Normalization: 822 training-time statistics locked via serialized normalizer safetensors
  - Sampling: Same-color paired sampling across a 5-step balanced color cycle (Red, Yellow, Wood, Green, Blue)
  - Episode Balancing: Draw 16 samples per task across 16 distinct episodes without replacement per cycle
  - Batch Size: 16 Task 1 + 16 Task 2 = 32 effective samples per optimizer step
  - PCGrad: Evaluates g1 . g2. If negative, applies canonical orthogonal projection on both task gradients.
  - Final Gradient: 0.5 * (g1_pc + g2_pc), matching baseline 32-sample joint batch gradient scale.
  - Schedule: Warmup 300 steps (0 -> 2e-6), Cosine decay to 1e-6 at 15k steps. Checkpoints every 2,500 steps.
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
from draccus import decode
from huggingface_hub import HfApi, hf_hub_download
from safetensors import safe_open

from lerobot.common.train_utils import get_step_checkpoint_dir, save_training_state, update_last_checkpoint
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.transforms.transforms import ImageTransforms, ImageTransformsConfig
from lerobot.utils.collate import lerobot_collate_fn
from lerobot.utils.constants import CHECKPOINTS_DIR, PRETRAINED_MODEL_DIR, TRAINING_STATE_DIR, LAST_CHECKPOINT_LINK


COLOR_NAMES = ["Red", "Yellow", "Wood", "Green", "Blue"]

TASK1_EPISODE_RANGES: Dict[str, Tuple[int, int]] = {
    "Red": (0, 39),
    "Yellow": (40, 59),
    "Wood": (60, 79),
    "Green": (80, 99),
    "Blue": (100, 119),
}

TASK2_EPISODE_RANGES: Dict[str, Tuple[int, int]] = {
    "Yellow": (0, 19),
    "Blue": (20, 39),
    "Red": (40, 79),
    "Green": (80, 101),
    "Wood": (102, 126),
}


def setup_logger(output_dir: Optional[Path] = None) -> logging.Logger:
    logger = logging.getLogger("pcgrad_train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(output_dir / "train.log", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Same-Color Paired PCGrad Expert-Only Fine-Tuning for SmolVLA."
    )
    # Model & Datasets
    parser.add_argument(
        "--policy-path",
        type=str,
        default="eslab1234/smolvla_multitask_5blocks_v3_822ep_from865_expertonly_b32_lr5e6_50k",
        help="Base 822 checkpoint path or HuggingFace repo ID.",
    )
    parser.add_argument(
        "--task1-dataset",
        type=str,
        default="eslab1234/smolvla_task1_hil_575_285k_v4_120ep_trimmed_merged",
        help="Task 1 clean branch dataset path or repo ID.",
    )
    parser.add_argument(
        "--task2-dataset",
        type=str,
        default="eslab1234/smolvla_task2_hil_575_285k_v4_127ep_trimmed_merged",
        help="Task 2 clean branch dataset path or repo ID.",
    )
    parser.add_argument(
        "--job-name",
        type=str,
        default="smolvla_822base_branch247_samecolor_pcgrad_expertonly_lr2e6_15k",
        help="Job and run name for logging and Hub repo naming (default smolvla_822base_branch247_samecolor_pcgrad_expertonly_lr2e6_15k).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/train/smolvla_822base_branch247_samecolor_pcgrad_expertonly_lr2e6_15k",
        help="Output directory for checkpoints and training logs.",
    )

    # Training Hyperparameters
    parser.add_argument(
        "--microbatch-per-task",
        type=int,
        default=16,
        help="Batch size per task per step (default 16; total 32 samples per optimizer step).",
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=8,
        help="VRAM micro-batch chunk size for gradient accumulation (default 8, fits RTX 3090 comfortably).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=15000,
        help="Total training steps (default 15,000).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-6,
        help="Peak learning rate after warmup (default 2e-6).",
    )
    parser.add_argument(
        "--final-lr",
        type=float,
        default=1e-6,
        help="Final decayed learning rate at max steps (default 1e-6).",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=300,
        help="Number of linear warmup steps (default 300).",
    )
    parser.add_argument(
        "--save-freq",
        type=int,
        default=2500,
        help="Checkpoint save frequency in steps (default 2,500).",
    )
    parser.add_argument(
        "--log-freq",
        type=int,
        default=20,
        help="Console and metric logging frequency in steps (default 20).",
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=10.0,
        help="Gradient norm clipping value applied to final gradient (default 10.0).",
    )

    # PCGrad Options
    parser.add_argument(
        "--use-pcgrad",
        type=lambda x: str(x).lower() in ("true", "1", "yes"),
        default=True,
        help="Whether to apply PCGrad orthogonal projection on conflicting gradients (default True).",
    )
    parser.add_argument(
        "--paired-fm-randomness",
        type=lambda x: str(x).lower() in ("true", "1", "yes"),
        default=True,
        help="Whether to inject identical noise and time tensors into paired T1/T2 batches (default True).",
    )

    # System & Hardware
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on ('cuda' or 'cpu').",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--use-wandb",
        type=lambda x: str(x).lower() in ("true", "1", "yes"),
        default=True,
        help="Whether to enable Weights & Biases logging (CHANGE 18, default True).",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="lerobot-so101-grad",
        help="Weights & Biases project name (CHANGE 18, default 'lerobot-so101-grad').",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Weights & Biases entity/team name.",
    )
    parser.add_argument(
        "--push-to-hub",
        type=lambda x: str(x).lower() in ("true", "1", "yes"),
        default=True,
        help="Whether to automatically push final 15k model checkpoint to Hugging Face Hub (CHANGE 18, default True).",
    )
    parser.add_argument(
        "--hub-repo-id",
        type=str,
        default="eslab1234/smolvla_822base_branch247_samecolor_pcgrad_expertonly_lr2e6_15k",
        help="Hugging Face model repository ID for final model push (CHANGE 18).",
    )
    parser.add_argument(
        "--sampler-dryrun-steps",
        type=int,
        default=25,
        help="Number of steps for pre-training dataset/sampler dry-run (CHANGE 15, default 25).",
    )
    parser.add_argument(
        "--dryrun-only",
        type=lambda x: str(x).lower() in ("true", "1", "yes"),
        default=False,
        help="Run only CHANGE 14 verification and CHANGE 15 sampler dry-run, then exit (default False).",
    )

    return parser.parse_args()


def resolve_local_path(dataset_identifier: str) -> Path:
    cached = Path.home() / ".cache/huggingface/lerobot" / dataset_identifier
    if cached.exists():
        return cached
    pure_name = dataset_identifier.split("/")[-1]
    cached_pure = Path.home() / ".cache/huggingface/lerobot" / pure_name
    if cached_pure.exists():
        return cached_pure
    cached_eslab = Path.home() / ".cache/huggingface/lerobot/eslab1234" / pure_name
    if cached_eslab.exists():
        return cached_eslab
    return Path(dataset_identifier)


def load_model_exact_stats(policy_path_or_repo: str, logger: logging.Logger) -> Dict[str, Dict[str, torch.Tensor]]:
    logger.info(f"Resolving normalization statistics for {policy_path_or_repo}...")
    p = Path(policy_path_or_repo)
    safetensors_file = None

    if p.exists() and (p / "policy_preprocessor_step_5_normalizer_processor.safetensors").exists():
        safetensors_file = p / "policy_preprocessor_step_5_normalizer_processor.safetensors"
    else:
        hub_name = f"models--{policy_path_or_repo.replace('/', '--')}"
        cached_snapshots = Path.home() / ".cache/huggingface/hub" / hub_name / "snapshots"
        if cached_snapshots.exists():
            for snap in cached_snapshots.glob("*"):
                cand = snap / "policy_preprocessor_step_5_normalizer_processor.safetensors"
                if cand.exists():
                    safetensors_file = cand
                    break

    if safetensors_file is None:
        try:
            downloaded = hf_hub_download(policy_path_or_repo, "policy_preprocessor_step_5_normalizer_processor.safetensors")
            safetensors_file = Path(downloaded)
        except Exception:
            safetensors_file = None

    if safetensors_file and safetensors_file.exists():
        logger.info(f"  [Direct Checkpoint Fidelity] Loading exact stats from: {safetensors_file}")
        stats: Dict[str, Dict[str, torch.Tensor]] = {}
        with safe_open(safetensors_file, framework="pt") as f:
            for k in f.keys():
                parts = k.split(".")
                stat_name = parts[-1]
                feat_name = ".".join(parts[:-1])
                if feat_name not in stats:
                    stats[feat_name] = {}
                stats[feat_name][stat_name] = f.get_tensor(k)
        return stats

    raise FileNotFoundError(f"Could not resolve normalization statistics for {policy_path_or_repo}")


class SameColorBalancedSampler:
    """Manages same-color pairing and episode-balanced sample indexing.
    Guarantees:
      1. 5-step cycle: Each of the 5 colors is selected exactly once every 5 steps (exact 20% balance).
      2. Same-color pairing: In step s, both Task 1 and Task 2 batches draw exclusively from color C.
      3. Episode balancing: For the 16 samples per task, 16 distinct episodes are sampled uniformly
         without replacement from the task-color episode pool before repeating an episode.
    """

    def __init__(
        self,
        t1_parquet: Path,
        t2_parquet: Path,
        t1_ranges: Dict[str, Tuple[int, int]],
        t2_ranges: Dict[str, Tuple[int, int]],
        batch_size_per_task: int = 16,
        seed: int = 42,
    ):
        self.batch_size = batch_size_per_task
        self.rng = np.random.default_rng(seed)

        self.t1_pools = self._build_episode_frame_pools(t1_parquet, t1_ranges)
        self.t2_pools = self._build_episode_frame_pools(t2_parquet, t2_ranges)

        # Episode cycle queues per (task, color)
        self.t1_ep_queues: Dict[str, List[int]] = {c: [] for c in COLOR_NAMES}
        self.t2_ep_queues: Dict[str, List[int]] = {c: [] for c in COLOR_NAMES}

        # 5-step color cycle queue
        self.color_cycle: List[str] = []

    def _build_episode_frame_pools(
        self, parquet_path: Path, ranges: Dict[str, Tuple[int, int]]
    ) -> Dict[str, Dict[int, List[int]]]:
        ep_df = pd.read_parquet(parquet_path)
        pools: Dict[str, Dict[int, List[int]]] = {c: {} for c in ranges}

        for color, (start_ep, end_ep) in ranges.items():
            for ep in range(start_ep, end_ep + 1):
                if ep >= len(ep_df):
                    continue
                row = ep_df.iloc[ep]
                f_start = int(row["dataset_from_index"])
                f_end = int(row["dataset_to_index"])
                # Exclude first 5 frames (static dwell) and last 5 frames (post-release static dwell)
                s = f_start + 5
                e = max(s + 1, f_end - 5)
                pools[color][ep] = list(range(s, e))

        return pools

    def next_color(self) -> str:
        if not self.color_cycle:
            self.color_cycle = self.rng.permutation(COLOR_NAMES).tolist()
        return self.color_cycle.pop(0)

    def _sample_task_frames(
        self,
        task_pools: Dict[str, Dict[int, List[int]]],
        task_queues: Dict[str, List[int]],
        color: str,
    ) -> List[int]:
        ep_dict = task_pools[color]
        all_eps = list(ep_dict.keys())
        queue = task_queues[color]

        chosen_frames: List[int] = []
        for _ in range(self.batch_size):
            if not queue:
                queue.extend(self.rng.permutation(all_eps).tolist())
            ep = queue.pop(0)
            avail_frames = ep_dict[ep]
            frame_idx = self.rng.choice(avail_frames)
            chosen_frames.append(int(frame_idx))

        return chosen_frames

    def sample_step(self) -> Tuple[str, List[int], List[int]]:
        """Returns (color, t1_indices, t2_indices)."""
        color = self.next_color()
        idx_t1 = self._sample_task_frames(self.t1_pools, self.t1_ep_queues, color)
        idx_t2 = self._sample_task_frames(self.t2_pools, self.t2_ep_queues, color)
        return color, idx_t1, idx_t2


def verify_dataset_and_configuration(
    task1_dataset: str,
    task2_dataset: str,
    ds_t1: LeRobotDataset,
    ds_t2: LeRobotDataset,
    t1_ranges: Dict[str, Tuple[int, int]],
    t2_ranges: Dict[str, Tuple[int, int]],
    logger: logging.Logger,
) -> None:
    """[CHANGE 14] Strict dataset and configuration verification against task definitions."""
    t1_counts = {c: (t1_ranges[c][1] - t1_ranges[c][0] + 1) for c in COLOR_NAMES}
    t2_counts = {c: (t2_ranges[c][1] - t2_ranges[c][0] + 1) for c in COLOR_NAMES}

    logger.info("=" * 70)
    logger.info("[CHANGE 14] Dataset Loading & Configuration Verification")
    logger.info("=" * 70)
    logger.info(f"Task1 repo:\n{task1_dataset}\n")
    logger.info(f"Task1 episode count:\n{ds_t1.num_episodes}\n")
    logger.info(f"Task2 repo:\n{task2_dataset}\n")
    logger.info(f"Task2 episode count:\n{ds_t2.num_episodes}\n")
    logger.info("각 color별 episode count:\n")
    logger.info("Task1:")
    for c in ["Red", "Yellow", "Wood", "Green", "Blue"]:
        logger.info(f"{c} {t1_counts[c]}")
    logger.info("\nTask2:")
    for c in ["Red", "Yellow", "Wood", "Green", "Blue"]:
        logger.info(f"{c} {t2_counts[c]}")
    logger.info("\n그리고:\n")
    logger.info("normalization source:\n822 checkpoint preprocessor\n")
    logger.info("new merged dataset:\nNO\n")
    logger.info("new dataset stats:\nNO\n")
    logger.info("575 full rollout included:\nNO")
    logger.info("=" * 70)

    # 1. Total Episode Counts
    if ds_t1.num_episodes != 120:
        raise ValueError(f"Task1 episode count mismatch: expected 120, got {ds_t1.num_episodes}")
    if ds_t2.num_episodes != 127:
        raise ValueError(f"Task2 episode count mismatch: expected 127, got {ds_t2.num_episodes}")

    # 2. Per-color counts
    expected_t1 = {"Red": 40, "Yellow": 20, "Wood": 20, "Green": 20, "Blue": 20}
    expected_t2 = {"Red": 40, "Yellow": 20, "Wood": 25, "Green": 22, "Blue": 20}

    for c, exp in expected_t1.items():
        if t1_counts[c] != exp:
            raise ValueError(f"Task1 {c} episode count mismatch: expected {exp}, got {t1_counts[c]}")

    for c, exp in expected_t2.items():
        if t2_counts[c] != exp:
            raise ValueError(f"Task2 {c} episode count mismatch: expected {exp}, got {t2_counts[c]}")

    # 3. Forbidden dataset checks
    banned_keywords = ["575ep_merged", "multitask_5blocks_v3_575ep", "247ep", "247_merged"]
    for kw in banned_keywords:
        if kw in task1_dataset.lower() or kw in task2_dataset.lower():
            raise ValueError(f"Forbidden dataset keyword '{kw}' detected in dataset configuration!")

    logger.info("✅ [CHANGE 14] All dataset counts and configuration assertions PASSED.\n")


def run_dataset_sampler_dryrun(
    sampler: SameColorBalancedSampler,
    ds_t1: LeRobotDataset,
    ds_t2: LeRobotDataset,
    t1_ranges: Dict[str, Tuple[int, int]],
    t2_ranges: Dict[str, Tuple[int, int]],
    dryrun_steps: int,
    logger: logging.Logger,
) -> None:
    """[CHANGE 15] 20~50 step pre-training dataset / sampler dry-run.
    Validates same-color pairing, episode ranges, and distinct task instructions for each step.
    """
    logger.info("=" * 70)
    logger.info(f"[CHANGE 15] Running Dataset / Sampler Dry-Run ({dryrun_steps} steps)...")
    logger.info("=" * 70)

    for s in range(1, dryrun_steps + 1):
        color, idx_t1, idx_t2 = sampler.sample_step()

        # Sample 0 of current step
        s1 = ds_t1[idx_t1[0]]
        s2 = ds_t2[idx_t2[0]]

        ep1 = int(s1["episode_index"].item() if isinstance(s1["episode_index"], torch.Tensor) else s1["episode_index"])
        frame1 = int(s1["frame_index"].item() if isinstance(s1["frame_index"], torch.Tensor) else s1["frame_index"])
        sample_idx1 = idx_t1[0]
        prompt1 = s1.get("task", "")

        ep2 = int(s2["episode_index"].item() if isinstance(s2["episode_index"], torch.Tensor) else s2["episode_index"])
        frame2 = int(s2["frame_index"].item() if isinstance(s2["frame_index"], torch.Tensor) else s2["frame_index"])
        sample_idx2 = idx_t2[0]
        prompt2 = s2.get("task", "")

        logger.info(f"[Step {s:02d}] current color: {color}")
        logger.info(f"  Task1 episode id: {ep1} (frame index: {frame1}, dataset sample index: {sample_idx1})")
        logger.info(f"  Task1 prompt/task: {prompt1}")
        logger.info(f"  Task2 episode id: {ep2} (frame index: {frame2}, dataset sample index: {sample_idx2})")
        logger.info(f"  Task2 prompt/task: {prompt2}")

        # Verification 1: Color ranges for representative samples
        r1_min, r1_max = t1_ranges[color]
        r2_min, r2_max = t2_ranges[color]
        if not (r1_min <= ep1 <= r1_max):
            raise AssertionError(f"Step {s}: Task1 Ep {ep1} not in [{r1_min}, {r1_max}] for color {color}")
        if not (r2_min <= ep2 <= r2_max):
            raise AssertionError(f"Step {s}: Task2 Ep {ep2} not in [{r2_min}, {r2_max}] for color {color}")

        # Verification 2: All batch samples must strictly belong to current color
        for i_t1 in idx_t1:
            sm = ds_t1[i_t1]
            ep = int(sm["episode_index"].item() if isinstance(sm["episode_index"], torch.Tensor) else sm["episode_index"])
            if not (r1_min <= ep <= r1_max):
                raise AssertionError(f"Step {s}: T1 sample {i_t1} ep {ep} not in [{r1_min}, {r1_max}] for color {color}")

        for i_t2 in idx_t2:
            sm = ds_t2[i_t2]
            ep = int(sm["episode_index"].item() if isinstance(sm["episode_index"], torch.Tensor) else sm["episode_index"])
            if not (r2_min <= ep <= r2_max):
                raise AssertionError(f"Step {s}: T2 sample {i_t2} ep {ep} not in [{r2_min}, {r2_max}] for color {color}")

        # Verification 3: Task instructions distinction
        if "separately" not in prompt1.lower():
            raise AssertionError(f"Step {s}: T1 prompt missing expected target keyword 'separately': {prompt1}")
        if "stack" not in prompt2.lower():
            raise AssertionError(f"Step {s}: T2 prompt missing expected target keyword 'stack': {prompt2}")

    logger.info("=" * 70)
    logger.info(f"✅ [CHANGE 15] Sampler dry-run passed all {dryrun_steps} steps successfully.")
    logger.info("   - Task1 color == Task2 color confirmed across all steps.")
    logger.info("   - All sample episode IDs match color ranges exactly.")
    logger.info("   - Task1 and Task2 prompt texts verified distinct and accurate.")
    logger.info("=" * 70 + "\n")


def compute_pcgrad(
    g1: List[torch.Tensor],
    g2: List[torch.Tensor],
    eps: float = 1e-12,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], float, float, bool]:
    """Canonical Two-Task PCGrad Orthogonal Projection (Yu et al., NeurIPS 2020).
    Given two task gradient vectors g1 and g2:
      dot = g1 . g2
      If dot < 0:
        g1_pc = g1 - (dot / (||g2||^2 + eps)) * g2
        g2_pc = g2 - (dot / (||g1||^2 + eps)) * g1
      Else:
        g1_pc = g1
        g2_pc = g2
    Returns:
      (g1_pc, g2_pc, raw_dot, raw_cosine, conflict_flag)
    """
    dot = 0.0
    norm1_sq = 0.0
    norm2_sq = 0.0

    for p1, p2 in zip(g1, g2, strict=False):
        if p1 is not None and p2 is not None:
            p1_f = p1.detach().float()
            p2_f = p2.detach().float()
            dot += torch.sum(p1_f * p2_f).item()
            norm1_sq += torch.sum(p1_f * p1_f).item()
            norm2_sq += torch.sum(p2_f * p2_f).item()

    norm1 = math.sqrt(max(norm1_sq, 0.0))
    norm2 = math.sqrt(max(norm2_sq, 0.0))
    cosine = dot / (norm1 * norm2 + eps) if (norm1 > eps and norm2 > eps) else 0.0

    conflict = dot < 0.0

    if conflict:
        # Scale factors computed strictly using original, unprojected denominators
        scale1 = dot / (norm2_sq + eps)
        scale2 = dot / (norm1_sq + eps)

        g1_pc = [
            (p1 - scale1 * p2) if (p1 is not None and p2 is not None) else p1
            for p1, p2 in zip(g1, g2, strict=False)
        ]
        g2_pc = [
            (p2 - scale2 * p1) if (p1 is not None and p2 is not None) else p2
            for p1, p2 in zip(g1, g2, strict=False)
        ]
    else:
        g1_pc = list(g1)
        g2_pc = list(g2)

    return g1_pc, g2_pc, dot, cosine, conflict


def prepare_torch_batch(
    raw_samples: List[Dict[str, Any]],
    preprocessor: Any,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    batch = lerobot_collate_fn(raw_samples)
    for cam_key in ["observation.images.top", "observation.images.wrist"]:
        if cam_key in batch and batch[cam_key].dtype == torch.uint8:
            batch[cam_key] = batch[cam_key].to(dtype=torch.float32) / 255.0

    batch = preprocessor(batch)
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device=device)
    return batch


def compute_accumulated_task_gradient(
    policy: SmolVLAPolicy,
    trainable_params: List[torch.nn.Parameter],
    raw_samples: List[Dict[str, Any]],
    preprocessor: Any,
    noise_tensor: torch.Tensor,
    time_tensor: torch.Tensor,
    micro_batch_size: int,
    device: torch.device,
) -> Tuple[List[torch.Tensor], float]:
    """Computes exact mean Flow Matching gradient over raw_samples using micro-batching.
    Returns (gradient_list, mean_loss).
    """
    total_samples = len(raw_samples)
    accum_grads: Optional[List[torch.Tensor]] = None
    total_loss_val = 0.0

    for start_idx in range(0, total_samples, micro_batch_size):
        end_idx = min(start_idx + micro_batch_size, total_samples)
        mb_raw = raw_samples[start_idx:end_idx]
        mb_noise = noise_tensor[start_idx:end_idx]
        mb_time = time_tensor[start_idx:end_idx]
        mb_weight = (end_idx - start_idx) / float(total_samples)

        batch = prepare_torch_batch(mb_raw, preprocessor, device)
        policy.zero_grad(set_to_none=True)
        loss, _ = policy.forward(batch, noise=mb_noise, time=mb_time)
        scaled_loss = loss * mb_weight

        grads = torch.autograd.grad(scaled_loss, trainable_params, retain_graph=False, create_graph=False)
        total_loss_val += loss.item() * mb_weight

        if accum_grads is None:
            accum_grads = [g.detach().clone() if g is not None else torch.zeros_like(p) for g, p in zip(grads, trainable_params)]
        else:
            for i, g in enumerate(grads):
                if g is not None:
                    accum_grads[i].add_(g.detach())

        del batch, loss, scaled_loss, grads

    policy.zero_grad(set_to_none=True)
    assert accum_grads is not None
    return accum_grads, total_loss_val


def build_optimizer_and_scheduler(
    trainable_params: List[torch.nn.Parameter],
    peak_lr: float,
    final_lr: float,
    warmup_steps: int,
    total_steps: int,
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=peak_lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=1e-10,
    )

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        decayed_lr = final_lr + (peak_lr - final_lr) * cosine_decay
        return decayed_lr / peak_lr

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    return optimizer, scheduler


class PCGradWandBLogger:
    """[CHANGE 18] W&B logging interface matching LeRobot WandBLogger."""

    def __init__(
        self,
        project: str,
        entity: Optional[str],
        job_name: str,
        output_dir: Path,
        config: Dict[str, Any],
        logger: logging.Logger,
    ):
        self.project = project
        self.entity = entity
        self.job_name = job_name
        self.output_dir = output_dir
        self.logger = logger

        import wandb
        self._wandb = wandb
        os.environ["WANDB_SILENT"] = "True"
        self.run = wandb.init(
            project=project,
            entity=entity,
            name=job_name,
            dir=str(output_dir),
            config=config,
            resume=None,
        )
        logger.info(f"W&B initialized. Track run at: {wandb.run.get_url()}")

    def log_dict(self, d: Dict[str, Any], step: int):
        self._wandb.log(d, step=step)

    def log_policy(self, checkpoint_dir: Path):
        """Checkpoints the policy to wandb as a model artifact, matching WandBLogger.log_policy."""
        try:
            step_id = checkpoint_dir.name
            artifact_name = f"{self.job_name}-{step_id}".replace(":", "_").replace("/", "_")
            artifact = self._wandb.Artifact(artifact_name, type="model")
            pretrained_dir = checkpoint_dir / PRETRAINED_MODEL_DIR

            standard_model_file = pretrained_dir / "model.safetensors"
            config_file = pretrained_dir / "config.json"
            if standard_model_file.exists():
                artifact.add_file(str(standard_model_file))
            if config_file.exists():
                artifact.add_file(str(config_file))

            self._wandb.log_artifact(artifact)
            self.logger.info(f"Logged policy artifact '{artifact_name}' to WandB.")
        except Exception as e:
            self.logger.warning(f"Could not log policy artifact to WandB: {e}")


def save_compatible_checkpoint(
    out_dir: Path,
    total_steps: int,
    step: int,
    policy: SmolVLAPolicy,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    base_policy_path: str,
    logger: logging.Logger,
    wandb_logger: Optional[PCGradWandBLogger] = None,
) -> Path:
    """[CHANGE 18] Saves policy matching LeRobot standard checkpoint structure:
      outputs/train/${RUN_NAME}/checkpoints/{step:06d}/
        ├── pretrained_model/
        │   ├── config.json
        │   ├── model.safetensors
        │   ├── train_config.json
        │   ├── policy_preprocessor.json
        │   ├── policy_preprocessor_step_5_normalizer_processor.safetensors
        │   ├── policy_postprocessor.json
        │   └── policy_postprocessor_step_0_unnormalizer_processor.safetensors
        └── training_state/
            ├── optimizer_param_groups.json
            ├── optimizer_state.safetensors
            ├── rng_state.safetensors
            ├── scheduler_state.json
            └── training_step.json

      And creates/updates:
        checkpoints/last -> {step:06d}
    """
    checkpoint_dir = get_step_checkpoint_dir(out_dir, total_steps, step)
    pretrained_dir = checkpoint_dir / PRETRAINED_MODEL_DIR
    pretrained_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving standard LeRobot checkpoint at step {step} to {checkpoint_dir} ...")

    # 1. Save model weights and base config into pretrained_model
    policy.save_pretrained(pretrained_dir)

    # 2. Copy preprocessor and postprocessor artifacts from base policy cache
    base_dir = Path(base_policy_path)
    if not base_dir.exists():
        hub_name = f"models--{base_policy_path.replace('/', '--')}"
        snapshots = Path.home() / ".cache/huggingface/hub" / hub_name / "snapshots"
        if snapshots.exists():
            for snap in snapshots.glob("*"):
                if (snap / "policy_preprocessor.json").exists():
                    base_dir = snap
                    break

    artifacts_to_copy = [
        "policy_preprocessor.json",
        "policy_preprocessor_step_5_normalizer_processor.safetensors",
        "policy_postprocessor.json",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        "train_config.json",
    ]

    for fname in artifacts_to_copy:
        src = base_dir / fname
        dst = pretrained_dir / fname
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

    # 3. Save training state using LeRobot standard utility
    save_training_state(
        checkpoint_dir=checkpoint_dir,
        train_step=step,
        optimizer=optimizer,
        scheduler=scheduler,
    )

    # 4. Update checkpoints/last symlink
    update_last_checkpoint(checkpoint_dir)

    # 5. Log policy artifact to WandB if enabled
    if wandb_logger is not None:
        wandb_logger.log_policy(checkpoint_dir)

    logger.info(f"✅ Checkpoint {step} successfully saved ({pretrained_dir}) and 'checkpoints/last' symlink updated.")
    return checkpoint_dir


def push_final_checkpoint_to_hub(
    final_checkpoint_dir: Path,
    hub_repo_id: str,
    logger: logging.Logger,
) -> None:
    """[CHANGE 18] Automatically push final 15k model checkpoint to Hugging Face Hub."""
    logger.info("=" * 70)
    logger.info(f"🚀 [CHANGE 18] Pushing final model to Hugging Face Hub: {hub_repo_id}")
    logger.info("=" * 70)
    pretrained_dir = final_checkpoint_dir / PRETRAINED_MODEL_DIR
    if not pretrained_dir.exists():
        raise FileNotFoundError(f"Cannot find pretrained_model directory at {pretrained_dir}")

    try:
        api = HfApi()
        api.create_repo(repo_id=hub_repo_id, exist_ok=True, repo_type="model")

        commit_info = api.upload_folder(
            repo_id=hub_repo_id,
            folder_path=str(pretrained_dir),
            commit_message=f"Upload final SmolVLA same-color PCGrad checkpoint ({final_checkpoint_dir.name})",
            repo_type="model",
            allow_patterns=["*.safetensors", "*.json", "*.yaml", "*.md"],
            ignore_patterns=["*.tmp", "*.log"],
        )
        logger.info(f"🎉 Final model successfully pushed to Hugging Face Hub: {commit_info}\n")
    except Exception as e:
        logger.error(f"Failed to push final model to Hugging Face Hub: {e}")


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out_dir)

    logger.info("=" * 85)
    logger.info("🚀 SmolVLA Same-Color Paired PCGrad Expert-Only Fine-Tuning")
    logger.info("=" * 85)
    logger.info(f"Base Policy          : {args.policy_path}")
    logger.info(f"Task 1 Branch Dataset: {args.task1_dataset}")
    logger.info(f"Task 2 Branch Dataset: {args.task2_dataset}")
    logger.info(f"Output Directory     : {out_dir}")
    logger.info(f"Samples per step     : {args.microbatch_per_task} T1 + {args.microbatch_per_task} T2 = {args.microbatch_per_task * 2} total")
    logger.info(f"Micro-batch size     : {args.micro_batch_size} (gradient accumulation)")
    logger.info(f"Training Steps       : {args.steps}")
    logger.info(f"Learning Rate        : {args.lr} -> {args.final_lr} (warmup: {args.warmup_steps} steps)")
    logger.info(f"Apply PCGrad         : {args.use_pcgrad}")
    logger.info(f"Paired FM Noise/Time : {args.paired_fm_randomness}")
    logger.info(f"Save Frequency       : Every {args.save_freq} steps")
    logger.info(f"Device               : {args.device}")
    logger.info(f"Seed                 : {args.seed}")
    logger.info("=" * 85)

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # 1. Resolve Dataset Paths
    p1 = resolve_local_path(args.task1_dataset)
    p2 = resolve_local_path(args.task2_dataset)
    logger.info(f"Resolved Task 1 Path : {p1}")
    logger.info(f"Resolved Task 2 Path : {p2}")

    # 2. Setup Image Transforms from Base 822 train_config
    chunk_size = 50
    fps = 30.0
    delta_ts = {"action": [i / fps for i in range(chunk_size)]}

    tf_config = ImageTransformsConfig(
        enable=True,
        max_num_transforms=3,
        random_order=False,
    )
    image_transforms = ImageTransforms(tf_config)
    logger.info("Instantiated ImageTransforms (3-subset ColorJitter/Sharpness) matching 822 training.")

    # 3. Instantiate LeRobot Datasets
    logger.info("Loading Task 1 and Task 2 LeRobotDataset instances...")
    ds_t1 = LeRobotDataset(
        args.task1_dataset,
        root=p1,
        image_transforms=image_transforms,
        delta_timestamps=delta_ts,
        return_uint8=True,
    )
    ds_t2 = LeRobotDataset(
        args.task2_dataset,
        root=p2,
        image_transforms=image_transforms,
        delta_timestamps=delta_ts,
        return_uint8=True,
    )

    # 4. Instantiate Same-Color Balanced Sampler
    ep_p1 = p1 / "meta/episodes/chunk-000/file-000.parquet"
    ep_p2 = p2 / "meta/episodes/chunk-000/file-000.parquet"
    sampler = SameColorBalancedSampler(
        t1_parquet=ep_p1,
        t2_parquet=ep_p2,
        t1_ranges=TASK1_EPISODE_RANGES,
        t2_ranges=TASK2_EPISODE_RANGES,
        batch_size_per_task=args.microbatch_per_task,
        seed=args.seed,
    )
    logger.info("SameColorBalancedSampler initialized with 5-step color cycle and episode balancing.")

    # 4a. [CHANGE 14] Verify Dataset Loading and Configuration
    verify_dataset_and_configuration(
        task1_dataset=args.task1_dataset,
        task2_dataset=args.task2_dataset,
        ds_t1=ds_t1,
        ds_t2=ds_t2,
        t1_ranges=TASK1_EPISODE_RANGES,
        t2_ranges=TASK2_EPISODE_RANGES,
        logger=logger,
    )

    # 4b. [CHANGE 15] Dataset / Sampler Dry-Run
    if args.sampler_dryrun_steps > 0:
        run_dataset_sampler_dryrun(
            sampler=sampler,
            ds_t1=ds_t1,
            ds_t2=ds_t2,
            t1_ranges=TASK1_EPISODE_RANGES,
            t2_ranges=TASK2_EPISODE_RANGES,
            dryrun_steps=args.sampler_dryrun_steps,
            logger=logger,
        )

    if args.dryrun_only:
        logger.info("Dry-run only flag set (--dryrun-only true). All verification checks passed. Exiting successfully.")
        return

    # Reset sampler to ensure training begins from deterministic initial state
    sampler = SameColorBalancedSampler(
        t1_parquet=ep_p1,
        t2_parquet=ep_p2,
        t1_ranges=TASK1_EPISODE_RANGES,
        t2_ranges=TASK2_EPISODE_RANGES,
        batch_size_per_task=args.microbatch_per_task,
        seed=args.seed,
    )

    # 5. Load Base 822 Model & Normalizer
    logger.info(f"Loading Base 822 Model from: {args.policy_path} ...")
    policy = SmolVLAPolicy.from_pretrained(args.policy_path)
    policy.to(device)
    policy.train()

    # Trainable Parameter Identification (Expert-Only)
    trainable_params: List[torch.nn.Parameter] = []
    param_names: List[str] = []
    for name, param in policy.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
            param_names.append(name)

    num_trainable_tensors = len(trainable_params)
    num_trainable_params = sum(p.numel() for p in trainable_params)
    logger.info(f"Trainable Tensors: {num_trainable_tensors}")
    logger.info(f"Trainable Parameters: {num_trainable_params:,} ({num_trainable_params / 1e6:.2f}M)")
    prefix_summary = sorted(set(".".join(n.split(".")[:3]) for n in param_names))
    logger.info(f"Trainable Parameter Prefixes: {prefix_summary}")

    # Load exact 822 training normalization stats
    stats = load_model_exact_stats(args.policy_path, logger)
    preprocessor, _ = make_pre_post_processors(policy_cfg=policy.config, dataset_stats=stats)
    logger.info("Exact 822 checkpoint normalization preprocessor initialized.")

    # 6. Optimizer and Scheduler
    optimizer, scheduler = build_optimizer_and_scheduler(
        trainable_params=trainable_params,
        peak_lr=args.lr,
        final_lr=args.final_lr,
        warmup_steps=args.warmup_steps,
        total_steps=args.steps,
    )
    logger.info(f"AdamW optimizer and warmup-cosine scheduler built (warmup={args.warmup_steps}, steps={args.steps}).")

    # WandB setup (CHANGE 18)
    wandb_logger: Optional[PCGradWandBLogger] = None
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
            logger.warning(f"Could not initialize WandB: {e}. Proceeding without WandB.")
            wandb_logger = None

    # Metrics Tracking
    csv_log_path = out_dir / "training_metrics.csv"
    csv_fields = [
        "step", "color", "loss_total", "loss_t1", "loss_t2", "lr",
        "cosine", "dot", "conflict", "projection_applied",
        "g1_norm", "g2_norm", "final_grad_norm",
        "proj_ratio_t1", "proj_ratio_t2",
        "step_time_s", "gpu_mem_alloc_mb",
    ]
    csv_file = open(csv_log_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    csv_writer.writeheader()

    # Color-specific running conflict trackers
    color_conflicts: Dict[str, collections.deque] = {
        c: collections.deque(maxlen=500) for c in COLOR_NAMES
    }
    color_total_counts: Dict[str, int] = {c: 0 for c in COLOR_NAMES}
    color_total_conflicts: Dict[str, int] = {c: 0 for c in COLOR_NAMES}
    total_conflicts: int = 0
    last_saved_ckpt: Optional[Path] = None

    beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)

    logger.info("\n--- Starting Training Loop ---")
    start_time = time.time()

    for step in range(1, args.steps + 1):
        step_t0 = time.time()

        # 1. Sample Same-Color Balanced Batches
        color, idx_t1, idx_t2 = sampler.sample_step()
        raw_t1 = [ds_t1[i] for i in idx_t1]
        raw_t2 = [ds_t2[i] for i in idx_t2]

        # 2. Flow Matching Noise and Time Tensors
        if args.paired_fm_randomness:
            time_tensor = beta_dist.sample((args.microbatch_per_task,)).to(device=device, dtype=torch.float32) * 0.999 + 0.001
            noise_tensor = torch.randn((args.microbatch_per_task, chunk_size, 32), device=device, dtype=torch.float32)
            time_t1, time_t2 = time_tensor, time_tensor
            noise_t1, noise_t2 = noise_tensor, noise_tensor
        else:
            time_t1 = beta_dist.sample((args.microbatch_per_task,)).to(device=device, dtype=torch.float32) * 0.999 + 0.001
            time_t2 = beta_dist.sample((args.microbatch_per_task,)).to(device=device, dtype=torch.float32) * 0.999 + 0.001
            noise_t1 = torch.randn((args.microbatch_per_task, chunk_size, 32), device=device, dtype=torch.float32)
            noise_t2 = torch.randn((args.microbatch_per_task, chunk_size, 32), device=device, dtype=torch.float32)

        # 3. Compute Gradients for Task 1 and Task 2 Independently
        g1, loss1 = compute_accumulated_task_gradient(
            policy=policy,
            trainable_params=trainable_params,
            raw_samples=raw_t1,
            preprocessor=preprocessor,
            noise_tensor=noise_t1,
            time_tensor=time_t1,
            micro_batch_size=args.micro_batch_size,
            device=device,
        )

        g2, loss2 = compute_accumulated_task_gradient(
            policy=policy,
            trainable_params=trainable_params,
            raw_samples=raw_t2,
            preprocessor=preprocessor,
            noise_tensor=noise_t2,
            time_tensor=time_t2,
            micro_batch_size=args.micro_batch_size,
            device=device,
        )

        # 4. PCGrad Orthogonal Projection
        if args.use_pcgrad:
            g1_pc, g2_pc, raw_dot, raw_cos, is_conflict = compute_pcgrad(g1, g2)
            proj_applied = is_conflict
        else:
            g1_pc, g2_pc = g1, g2
            dot_calc = sum(torch.sum(p1.float() * p2.float()).item() for p1, p2 in zip(g1, g2, strict=False))
            n1 = math.sqrt(sum(torch.sum(p.float() ** 2).item() for p in g1))
            n2 = math.sqrt(sum(torch.sum(p.float() ** 2).item() for p in g2))
            raw_dot = dot_calc
            raw_cos = dot_calc / (n1 * n2 + 1e-12) if (n1 > 1e-12 and n2 > 1e-12) else 0.0
            is_conflict = raw_dot < 0.0
            proj_applied = False

        # 5. Combined Gradient Assignment (0.5 scaling matches batch-32 scale)
        g_final = [0.5 * (p1 + p2) for p1, p2 in zip(g1_pc, g2_pc, strict=False)]

        for param, grad_val in zip(trainable_params, g_final, strict=False):
            param.grad = grad_val.to(dtype=param.dtype)

        # 6. Gradient Norm Clipping & Optimizer Step
        final_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=args.grad_clip_norm).item()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        step_duration = time.time() - step_t0
        current_lr = scheduler.get_last_lr()[0]
        loss_total = 0.5 * (loss1 + loss2)

        # Norm computations for tracking
        g1_norm = math.sqrt(sum(torch.sum(p.float() ** 2).item() for p in g1))
        g2_norm = math.sqrt(sum(torch.sum(p.float() ** 2).item() for p in g2))
        g1_pc_norm = math.sqrt(sum(torch.sum(p.float() ** 2).item() for p in g1_pc))
        g2_pc_norm = math.sqrt(sum(torch.sum(p.float() ** 2).item() for p in g2_pc))
        ratio_t1 = g1_pc_norm / (g1_norm + 1e-12)
        ratio_t2 = g2_pc_norm / (g2_norm + 1e-12)

        # Conflict Rate Bookkeeping
        color_total_counts[color] += 1
        if is_conflict:
            color_total_conflicts[color] += 1
            total_conflicts += 1
        color_conflicts[color].append(1 if is_conflict else 0)

        rate_100 = float(np.mean(list(color_conflicts[color])[-20:])) if len(color_conflicts[color]) >= 20 else float(np.mean(list(color_conflicts[color])))
        rate_500 = float(np.mean(list(color_conflicts[color])))
        rate_total = color_total_conflicts[color] / float(color_total_counts[color])

        gpu_alloc_mb = torch.cuda.memory_allocated(device) / (1024 * 1024) if torch.cuda.is_available() else 0.0

        # Record to CSV
        log_row = {
            "step": step,
            "color": color,
            "loss_total": round(loss_total, 5),
            "loss_t1": round(loss1, 5),
            "loss_t2": round(loss2, 5),
            "lr": f"{current_lr:.3e}",
            "cosine": round(raw_cos, 4),
            "dot": round(raw_dot, 4),
            "conflict": int(is_conflict),
            "projection_applied": int(proj_applied),
            "g1_norm": round(g1_norm, 3),
            "g2_norm": round(g2_norm, 3),
            "final_grad_norm": round(final_norm, 3),
            "proj_ratio_t1": round(ratio_t1, 4),
            "proj_ratio_t2": round(ratio_t2, 4),
            "step_time_s": round(step_duration, 3),
            "gpu_mem_alloc_mb": round(gpu_alloc_mb, 1),
        }
        csv_writer.writerow(log_row)
        if step % 50 == 0:
            csv_file.flush()

        # WandB logging (CHANGE 18)
        if wandb_logger is not None:
            conflict_rate_val = (
                float(total_conflicts) / float(step) if step > 0 else 0.0
            )
            wb_dict = {
                "train/loss": loss_total,
                "train/loss_t1": loss1,
                "train/loss_t2": loss2,
                "train/lr": current_lr,
                "train/step_time_s": step_duration,
                "train/gpu_mem_alloc_mb": gpu_alloc_mb,
                "pcgrad/cosine": raw_cos,
                "pcgrad/dot": raw_dot,
                "pcgrad/conflict": 1.0 if is_conflict else 0.0,
                "pcgrad/conflict_rate": conflict_rate_val,
                "pcgrad/g1_norm": g1_norm,
                "pcgrad/g2_norm": g2_norm,
                "pcgrad/final_grad_norm": final_norm,
                "pcgrad/proj_ratio_t1": ratio_t1,
                "pcgrad/proj_ratio_t2": ratio_t2,
                f"pcgrad/{color.lower()}/cosine": raw_cos,
                f"pcgrad/{color.lower()}/conflict_rate": rate_total,
                f"train/{color.lower()}/loss_t1": loss1,
                f"train/{color.lower()}/loss_t2": loss2,
            }
            wandb_logger.log_dict(wb_dict, step=step)

        # Periodic Console Logging
        if step % args.log_freq == 0 or step == 1 or is_conflict:
            status_flag = "⚠️ CONFLICT (PCGrad Projected)" if proj_applied else "✅ ALIGNED"
            logger.info(
                f"Step {step:5d}/{args.steps} | Color: {color:<6} | Total Loss: {loss_total:.4f} (T1: {loss1:.4f}, T2: {loss2:.4f}) | "
                f"Cos: {raw_cos:+.3f} | {status_flag} | Final ||g||: {final_norm:.2f} | LR: {current_lr:.2e} | "
                f"Conflict Rate ({color}): {rate_total*100:.1f}% ({color_total_conflicts[color]}/{color_total_counts[color]}) | {step_duration:.2f}s"
            )

        # Checkpoint Saving (CHANGE 18)
        if step % args.save_freq == 0 or step == args.steps:
            last_saved_ckpt = save_compatible_checkpoint(
                out_dir=out_dir,
                total_steps=args.steps,
                step=step,
                policy=policy,
                optimizer=optimizer,
                scheduler=scheduler,
                base_policy_path=args.policy_path,
                logger=logger,
                wandb_logger=wandb_logger,
            )

    csv_file.close()
    total_elapsed = time.time() - start_time
    logger.info("=" * 85)
    logger.info(f"🎉 Training Complete! Total time: {total_elapsed / 3600:.2f} hours ({total_elapsed:.1f}s)")
    logger.info(f"Final Checkpoints saved under: {out_dir / 'checkpoints'}")
    logger.info("Summary of Cumulative Color Conflict Rates:")
    for c in COLOR_NAMES:
        tot = color_total_counts[c]
        conf = color_total_conflicts[c]
        pct = (conf / float(tot) * 100.0) if tot > 0 else 0.0
        logger.info(f"  {c:<6}: {pct:5.1f}% ({conf}/{tot} steps)")
    logger.info("=" * 85)

    # Final Hugging Face Hub Push (CHANGE 18)
    if args.push_to_hub and last_saved_ckpt is not None:
        push_final_checkpoint_to_hub(
            final_checkpoint_dir=last_saved_ckpt,
            hub_repo_id=args.hub_repo_id,
            logger=logger,
        )


if __name__ == "__main__":
    main()
