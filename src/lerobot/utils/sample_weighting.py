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
from pathlib import Path
from typing import TYPE_CHECKING

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
        default_reward: Default reward score if not found in batch or map
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
    default_reward: float = 1.0
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

    if config.type in {"reward_weighted", "reward", "awac", "offline_rl"}:
        return RewardSampleWeighter(config=config, device=device)

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

    Computes per-sample weights based on episodic or per-frame reward:
        weights = exp(reward / temperature)
    and normalizes them across the batch so that the sum equals batch_size.

    Supports:
    1. Direct 'reward' field in batch dictionary (e.g. batch['reward']).
    2. Episode-based reward lookup from a JSON file (reward_map_path) using batch['episode_index'].
    3. Fallback default reward (default_reward, e.g. 1.0 for demonstration datasets).
    """

    def __init__(
        self,
        config: SampleWeightingConfig,
        device: torch.device,
    ):
        self.config = config
        self.device = device
        self.temperature = max(config.temperature, 1e-4)
        self.epsilon = config.epsilon
        self.min_weight = config.min_weight
        self.max_weight = config.max_weight
        self.default_reward = config.default_reward
        self.reward_map: dict[int, float] = {}

        if config.reward_map_path:
            import json
            import logging
            reward_path = Path(config.reward_map_path)
            if reward_path.exists():
                with open(reward_path, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                    self.reward_map = {int(k): float(v) for k, v in raw_data.items()}
                logging.info("RewardSampleWeighter: loaded %d episode rewards from %s", len(self.reward_map), reward_path)
            else:
                logging.warning("RewardSampleWeighter: reward map path %s not found; fallback to default_reward=%s", reward_path, self.default_reward)

    def compute_batch_weights(self, batch: dict) -> tuple[torch.Tensor, dict]:
        batch_size = self._determine_batch_size(batch)

        if "reward" in batch and isinstance(batch["reward"], torch.Tensor):
            rewards = batch["reward"].to(self.device, dtype=torch.float32).view(-1)
        elif "episode_index" in batch and self.reward_map:
            ep_indices = batch["episode_index"].cpu().view(-1).tolist()
            rewards = torch.tensor(
                [self.reward_map.get(int(ep), self.default_reward) for ep in ep_indices],
                device=self.device,
                dtype=torch.float32,
            )
        else:
            rewards = torch.full((batch_size,), self.default_reward, device=self.device, dtype=torch.float32)

        # Exponential reward weighting (Advantage / Return weighted)
        # Shift rewards by max in batch for numerical stability in exp
        shifted_rewards = (rewards - rewards.max()) / self.temperature
        raw_weights = torch.exp(shifted_rewards)

        if self.min_weight > 0.0 or self.max_weight < float("inf"):
            raw_weights = torch.clamp(raw_weights, min=self.min_weight, max=self.max_weight)

        # Normalize weights so they sum to batch_size
        weight_sum = raw_weights.sum() + self.epsilon
        normalized_weights = raw_weights * (batch_size / weight_sum)

        stats = {
            "type": "reward_weighted",
            "mean_reward": rewards.mean().item(),
            "min_reward": rewards.min().item(),
            "max_reward": rewards.max().item(),
            "mean_weight": normalized_weights.mean().item(),
            "min_weight": normalized_weights.min().item(),
            "max_weight": normalized_weights.max().item(),
        }
        return normalized_weights, stats

    def _determine_batch_size(self, batch: dict) -> int:
        for key in ["action", "index", "episode_index", "observation.state"]:
            if key in batch and isinstance(batch[key], torch.Tensor):
                return batch[key].shape[0]
        for value in batch.values():
            if isinstance(value, torch.Tensor) and value.ndim >= 1:
                return value.shape[0]
        return 1

    def get_stats(self) -> dict:
        return {"type": "reward_weighted", "temperature": self.temperature}

