# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Sample weighting abstraction for training.

This module provides an abstract base class for sample weighting strategies (e.g., RA-BC)
that can be used during training without polluting the training script with
policy-specific code.

Example usage:
    # In training config
    sample_weighting:
        type: rabc
        progress_path: hf://datasets/my-dataset/sarm_progress.parquet
        head_mode: sparse
        kappa: 0.01

    # In training script
    sample_weighter = make_sample_weighter(cfg.sample_weighting, policy, device, dataset_root=cfg.dataset.root, dataset_repo_id=cfg.dataset.repo_id)
    ...
    weights, stats = sample_weighter.compute_batch_weights(batch)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from lerobot.policies.pretrained import PreTrainedPolicy


class SampleWeighter(ABC):
    """
    Implementations compute per-sample weights that can be used to weight
    the loss during training. This enables techniques like:
    - RA-BC (Reward-Aligned Behavior Cloning)
    - Importance sampling
    - Curriculum learning
    - Quality-based filtering
    """

    @abstractmethod
    def compute_batch_weights(self, batch: dict) -> tuple[torch.Tensor, dict]:
        """
        Compute per-sample weights for a training batch.

        Args:
            batch: Training batch dictionary containing at minimum an "index" key
                   with global frame indices.
        """

    @abstractmethod
    def get_stats(self) -> dict:
        """
        Get global statistics about the weighting strategy.
        """


@dataclass
class SampleWeightingConfig:
    """
    Configuration for sample weighting during training.

    This is a generic config that supports multiple weighting strategies.
    The `type` field determines which implementation to use, and `extra_params`
    contains additional type-specific parameters.

    Attributes:
        type: Weighting strategy type ("reward_weighted", "rabc", "uniform", etc.)
        progress_path: Path to precomputed progress values (for RABC)
        head_mode: Which model head to use for progress ("sparse" or "dense")
        kappa: Hard threshold for high-quality samples (RABC-specific)
        epsilon: Small constant for numerical stability
        temperature: Temperature parameter for reward-weighted exponential loss (Offline RL / AWR / AWAC)
        reward_map_path: Optional path to JSON file mapping episode index -> reward float
        frame_reward_path: Optional path to JSON or Parquet file for frame-level/chunk rewards
        chunk_size: Action chunk size over which frame rewards are averaged (default 50)
        gamma: Discount factor across chunk frames (1.0 = simple mean)
        mistake_window_frames: Number of frames before human intervention considered failure/mistake
        default_reward: Default reward score if not found in batch or map
        default_normal_reward: Default reward for normal autonomous execution (default 0.8)
        default_failure_reward: Default reward for mistake/failure segment (default 0.1)
        default_correction_reward: Default reward for human correction/recovery (default 1.0)
        min_weight: Minimum weight clamp threshold
        max_weight: Maximum weight clamp threshold
        extra_params: Additional type-specific parameters passed to the weighter
    """

    type: str = "rabc"
    progress_path: str | None = None
    head_mode: str = "sparse"
    kappa: float = 0.01
    epsilon: float = 1e-6
    temperature: float = 0.5
    reward_map_path: str | None = None
    frame_reward_path: str | None = None
    chunk_size: int = 50
    gamma: float = 1.0
    mistake_window_frames: int = 45
    default_reward: float = 1.0
    default_normal_reward: float = 0.8
    default_failure_reward: float = 0.1
    default_correction_reward: float = 1.0
    min_weight: float = 0.0
    max_weight: float = 100.0
    # Additional type-specific params can be added here or passed via extra_params
    extra_params: dict = field(default_factory=dict)


def make_sample_weighter(
    config: SampleWeightingConfig | None,
    policy: PreTrainedPolicy,
    device: torch.device,
    dataset_root: str | None = None,
    dataset_repo_id: str | None = None,
) -> SampleWeighter | None:
    """
    Factory function to create a SampleWeighter from config.

    This keeps policy-specific initialization logic out of the training script.

    Args:
        config: Sample weighting configuration, or None to disable weighting.
        policy: The policy being trained (used to extract chunk_size, etc.)
        device: Device to place weight tensors on.
        dataset_root: Local path to dataset root (for auto-detecting progress_path).
        dataset_repo_id: HuggingFace repo ID (for auto-detecting progress_path).
    """
    if config is None:
        return None

    if config.type in {"reward_weighted", "reward", "awac", "offline_rl", "rwfm"}:
        policy_chunk_size = getattr(getattr(policy, "config", None), "chunk_size", None)
        if policy_chunk_size is not None and config.chunk_size == 50:
            config.chunk_size = int(policy_chunk_size)
        return RewardSampleWeighter(
            config=config,
            device=device,
            dataset_root=dataset_root,
            dataset_repo_id=dataset_repo_id,
        )

    if config.type == "rabc":
        return _make_rabc_weighter(config, policy, device, dataset_root, dataset_repo_id)

    if config.type == "uniform":
        # No-op weighter that returns uniform weights
        return UniformWeighter(device=device)

    raise ValueError(
        f"Unknown sample weighting type: '{config.type}'. "
        f"Supported types: 'reward_weighted', 'rabc', 'uniform'"
    )


def _make_rabc_weighter(
    config: SampleWeightingConfig,
    policy: PreTrainedPolicy,
    device: torch.device,
    dataset_root: str | None = None,
    dataset_repo_id: str | None = None,
) -> SampleWeighter:
    """Create RABC weighter with policy-specific initialization.

    Args:
        config: Sample weighting configuration.
        policy: The policy being trained (used to extract chunk_size).
        device: Device to place weight tensors on.
        dataset_root: Local path to dataset root (for auto-detecting progress_path).
        dataset_repo_id: HuggingFace repo ID (for auto-detecting progress_path).
    """
    # Import here to avoid circular imports and keep RABC code in SARM module
    from lerobot.rewards.sarm.rabc import RABCWeights

    # Extract chunk_size from policy config
    chunk_size = getattr(policy.config, "chunk_size", None)
    if chunk_size is None:
        raise ValueError(
            "RABC sample weighting requires a policy with 'chunk_size' in its config. "
            "This is typically set for action-chunking policies like ACT, Diffusion, PI0, etc."
        )

    # Determine progress_path: use explicit config or auto-detect from dataset
    progress_path = config.progress_path
    if progress_path is None:
        if dataset_root:
            progress_path = str(Path(dataset_root) / "sarm_progress.parquet")
        elif dataset_repo_id:
            progress_path = f"hf://datasets/{dataset_repo_id}/sarm_progress.parquet"
        else:
            raise ValueError(
                "RABC sample weighting requires 'progress_path' to be set, "
                "or dataset_root/dataset_repo_id for auto-detection. "
                "Generate progress values using: "
                "python -m lerobot.rewards.sarm.compute_rabc_weights --help"
            )

    return RABCWeights(
        progress_path=progress_path,
        chunk_size=chunk_size,
        head_mode=config.head_mode,
        kappa=config.kappa,
        epsilon=config.epsilon,
        device=device,
        **config.extra_params,
    )


class UniformWeighter(SampleWeighter):
    """
    No-op sample weighter that returns uniform weights.

    Useful as a baseline or when you want to disable weighting without
    changing the training code structure.

    Note:
        Batch size is determined by looking for tensor values in the batch
        dictionary. The method checks common keys like "action", "index",
        and "observation.state" first, then falls back to scanning all values.
    """

    def __init__(self, device: torch.device):
        self.device = device

    def compute_batch_weights(self, batch: dict) -> tuple[torch.Tensor, dict]:
        """Return uniform weights (all ones)."""
        batch_size = self._determine_batch_size(batch)

        weights = torch.ones(batch_size, device=self.device)
        stats = {"mean_weight": 1.0, "type": "uniform"}
        return weights, stats

    def _determine_batch_size(self, batch: dict) -> int:
        """
        Determine batch size from the batch dictionary.

        Checks common keys first, then scans all values for tensors.

        Args:
            batch: Training batch dictionary.
        """
        if not batch:
            raise ValueError("Cannot determine batch size from empty batch")

        # Check common keys first
        for key in ["action", "index", "observation.state"]:
            if key in batch and isinstance(batch[key], torch.Tensor):
                return batch[key].shape[0]

        # Scan all values for any tensor
        for value in batch.values():
            if isinstance(value, torch.Tensor) and value.ndim >= 1:
                return value.shape[0]

        # Last resort: return 1 (this handles non-tensor batches)
        return 1

    def get_stats(self) -> dict:
        """Return empty stats for uniform weighting."""
        return {"type": "uniform"}


class RewardSampleWeighter(SampleWeighter):
    """
    Sample weighter for Offline RL / Advantage-Weighted Regression (AWR / AWAC / Reward-Weighted Flow-Matching).

    Computes per-sample weights based on episodic or per-frame chunk reward:
        weights = exp((reward - max_reward) / temperature)
    and normalizes them across the batch so that the sum equals batch_size.

    Supports:
    1. Frame-level chunk reward weighting (50-step window mean/discounted sum):
       - Explicitly from 'frame_reward_path' (JSON or Parquet with segment, dense, or intervention metadata).
       - Auto-detected from dataset_root ('meta/episode_interventions.json' or 'meta/episode_frame_rewards.json').
    2. Episode-level reward lookup from 'reward_map_path' using batch['episode_index'].
    3. Direct 'reward' field in batch dictionary (e.g. batch['reward']).
    4. Fallback default reward (default_reward, e.g. 1.0 for demonstration datasets).
    """

    def __init__(
        self,
        config: SampleWeightingConfig,
        device: torch.device,
        dataset_root: str | None = None,
        dataset_repo_id: str | None = None,
    ):
        self.config = config
        self.device = device
        self.temperature = max(config.temperature, 1e-4)
        self.epsilon = config.epsilon
        self.min_weight = config.min_weight
        self.max_weight = config.max_weight
        self.default_reward = config.default_reward
        self.chunk_size = max(1, config.chunk_size)
        self.gamma = max(1e-4, min(1.0, config.gamma))
        self.discount_weights = np.array(
            [self.gamma ** k for k in range(self.chunk_size + 100)], dtype=np.float32
        )

        self.reward_map: dict[int, float] = {}
        self.ep_frame_rewards: dict[int, np.ndarray] = {}

        # 1. Load episode-level reward map if specified
        if config.reward_map_path:
            reward_path = Path(config.reward_map_path)
            if reward_path.exists():
                with open(reward_path, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                    self.reward_map = {int(k): float(v) for k, v in raw_data.items()}
                logging.info(
                    "RewardSampleWeighter: loaded %d episode rewards from %s",
                    len(self.reward_map),
                    reward_path,
                )
            else:
                logging.warning(
                    "RewardSampleWeighter: reward map path %s not found; fallback to default_reward=%s",
                    reward_path,
                    self.default_reward,
                )

        # 2. Resolve frame_reward_path (explicit or auto-detect from dataset_root)
        frame_path = config.frame_reward_path
        if frame_path is None and dataset_root:
            root_p = Path(dataset_root)
            candidates = [
                root_p / "meta" / "episode_interventions.json",
                root_p / "meta" / "episode_frame_rewards.json",
                root_p / "meta" / "frame_rewards.parquet",
                root_p / "episode_interventions.json",
            ]
            for cand in candidates:
                if cand.exists():
                    frame_path = str(cand)
                    logging.info("RewardSampleWeighter: auto-detected frame rewards from %s", frame_path)
                    break

        # 3. Load frame-level rewards if resolved
        if frame_path:
            self._load_frame_rewards(frame_path)

    def _load_frame_rewards(self, path_str: str) -> None:
        p = Path(path_str)
        if not p.exists():
            logging.warning("RewardSampleWeighter: frame reward path %s not found", p)
            return

        if p.suffix == ".parquet":
            try:
                import pandas as pd

                df = pd.read_parquet(p)
                if "episode_index" in df.columns and "reward" in df.columns:
                    sort_cols = ["frame_index"] if "frame_index" in df.columns else []
                    for ep_idx, group in df.groupby("episode_index"):
                        if sort_cols:
                            group = group.sort_values(by=sort_cols)
                        self.ep_frame_rewards[int(ep_idx)] = group["reward"].to_numpy(dtype=np.float32)
                    logging.info(
                        "RewardSampleWeighter: loaded parquet frame rewards for %d episodes from %s",
                        len(self.ep_frame_rewards),
                        p,
                    )
                    return
            except Exception as e:
                logging.warning("RewardSampleWeighter: failed to read parquet %s: %s", p, e)

        # JSON format
        try:
            with open(p, "r", encoding="utf-8") as f:
                raw = json.load(f)

            for ep_str, val in raw.items():
                try:
                    ep_id = int(ep_str)
                except ValueError:
                    continue

                # Format A: Segment list [[start, end, reward], ...]
                if (
                    isinstance(val, list)
                    and len(val) > 0
                    and isinstance(val[0], (list, tuple))
                    and len(val[0]) >= 3
                ):
                    max_frame = max(int(seg[1]) for seg in val)
                    arr = np.full(max_frame, self.default_reward, dtype=np.float32)
                    for seg in val:
                        start_f = max(0, int(seg[0]))
                        end_f = min(max_frame, int(seg[1]))
                        r_val = float(seg[2])
                        arr[start_f:end_f] = r_val
                    self.ep_frame_rewards[ep_id] = arr

                # Format B: Dense float list [r0, r1, ...]
                elif isinstance(val, list) and len(val) > 0 and isinstance(val[0], (int, float)):
                    self.ep_frame_rewards[ep_id] = np.asarray(val, dtype=np.float32)

                # Format C: Intervention metadata dict {"total_frames": 580, "intervention_start_frame": 412, ...}
                elif isinstance(val, dict):
                    total_f = int(val.get("total_frames", 600))
                    interv_f = val.get("intervention_start_frame")
                    norm_r = float(val.get("normal_reward", self.config.default_normal_reward))
                    fail_r = float(val.get("failure_reward", self.config.default_failure_reward))
                    corr_r = float(val.get("correction_reward", self.config.default_correction_reward))
                    succ_r = float(val.get("success_reward", self.default_reward))

                    arr = np.full(total_f, norm_r, dtype=np.float32)
                    if interv_f is not None and interv_f > 0:
                        mistake_len = int(val.get("mistake_window", self.config.mistake_window_frames))
                        mistake_start = max(0, int(interv_f) - mistake_len)
                        arr[:mistake_start] = norm_r
                        arr[mistake_start:int(interv_f)] = fail_r
                        arr[int(interv_f):] = corr_r
                    else:
                        arr[:] = succ_r
                    self.ep_frame_rewards[ep_id] = arr

            logging.info(
                "RewardSampleWeighter: loaded frame rewards for %d episodes from %s",
                len(self.ep_frame_rewards),
                p,
            )
        except Exception as e:
            logging.warning("RewardSampleWeighter: failed to parse frame reward file %s: %s", p, e)

    def compute_batch_weights(self, batch: dict) -> tuple[torch.Tensor, dict]:
        batch_size = self._determine_batch_size(batch)

        # 1. Frame-level Chunk Reward weighting (if frame rewards and batch indices exist)
        if self.ep_frame_rewards and "episode_index" in batch and "frame_index" in batch:
            ep_indices = batch["episode_index"].cpu().view(-1).tolist()
            frame_indices = batch["frame_index"].cpu().view(-1).tolist()

            chunk_rewards = []
            for ep_idx, f_idx in zip(ep_indices, frame_indices):
                ep_int = int(ep_idx)
                f_int = int(f_idx)
                if ep_int in self.ep_frame_rewards:
                    arr = self.ep_frame_rewards[ep_int]
                    n = len(arr)
                    if n == 0:
                        chunk_rewards.append(self.default_reward)
                        continue
                    start = min(max(0, f_int), n - 1)
                    end = min(start + self.chunk_size, n)
                    slice_r = arr[start:end]
                    if len(slice_r) == 0:
                        chunk_rewards.append(self.default_reward)
                    elif self.gamma == 1.0 or len(slice_r) == 1:
                        chunk_rewards.append(float(slice_r.mean()))
                    else:
                        w = self.discount_weights[: len(slice_r)]
                        chunk_rewards.append(float(np.dot(slice_r, w) / w.sum()))
                elif ep_int in self.reward_map:
                    chunk_rewards.append(self.reward_map[ep_int])
                else:
                    chunk_rewards.append(self.default_reward)

            rewards = torch.tensor(chunk_rewards, device=self.device, dtype=torch.float32)
            weighter_type = "reward_weighted_chunk"

        # 2. Batch-provided reward tensor (e.g. batch['reward'])
        elif "reward" in batch and isinstance(batch["reward"], torch.Tensor):
            rewards = batch["reward"].to(self.device, dtype=torch.float32).view(-1)
            weighter_type = "reward_weighted_batch"

        # 3. Episode-level reward lookup from reward_map
        elif "episode_index" in batch and self.reward_map:
            ep_indices = batch["episode_index"].cpu().view(-1).tolist()
            rewards = torch.tensor(
                [self.reward_map.get(int(ep), self.default_reward) for ep in ep_indices],
                device=self.device,
                dtype=torch.float32,
            )
            weighter_type = "reward_weighted_episode"

        # 4. Fallback uniform default
        else:
            rewards = torch.full((batch_size,), self.default_reward, device=self.device, dtype=torch.float32)
            weighter_type = "reward_weighted_default"

        # Exponential reward weighting: w_i = exp((R_i - max(R)) / temperature)
        shifted_rewards = (rewards - rewards.max()) / self.temperature
        raw_weights = torch.exp(shifted_rewards)

        if self.min_weight > 0.0 or self.max_weight < float("inf"):
            raw_weights = torch.clamp(raw_weights, min=self.min_weight, max=self.max_weight)

        # Normalize weights so they sum to batch_size
        weight_sum = raw_weights.sum() + self.epsilon
        normalized_weights = raw_weights * (batch_size / weight_sum)

        stats = {
            "type": weighter_type,
            "mean_reward": rewards.mean().item(),
            "min_reward": rewards.min().item(),
            "max_reward": rewards.max().item(),
            "mean_weight": normalized_weights.mean().item(),
            "min_weight": normalized_weights.min().item(),
            "max_weight": normalized_weights.max().item(),
        }
        return normalized_weights, stats

    def _determine_batch_size(self, batch: dict) -> int:
        for key in ["action", "index", "episode_index", "frame_index", "observation.state"]:
            if key in batch and isinstance(batch[key], torch.Tensor):
                return batch[key].shape[0]
        for value in batch.values():
            if isinstance(value, torch.Tensor) and value.ndim >= 1:
                return value.shape[0]
        return 1

    def get_stats(self) -> dict:
        return {
            "type": "reward_weighted",
            "temperature": self.temperature,
            "chunk_size": self.chunk_size,
            "gamma": self.gamma,
            "num_frame_episodes": len(self.ep_frame_rewards),
            "num_episode_rewards": len(self.reward_map),
        }

