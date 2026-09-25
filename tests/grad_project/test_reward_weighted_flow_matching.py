#!/usr/bin/env python3
"""Unit tests for Reward-Weighted Flow Matching (RWFM) sample weighting."""

import json
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from lerobot.utils.sample_weighting import (
    RewardSampleWeighter,
    SampleWeighter,
    SampleWeightingConfig,
    make_sample_weighter,
)


def test_chunk_mean_formula_exact_prompt_example(tmp_path):
    """
    Test section 3.3 formula:
    Chunk size = 50.
    Frames 410~429 (20 frames): reward = 0.1 (mistake)
    Frames 430~459 (30 frames): reward = 1.0 (recovery)
    Expected chunk reward: (20 * 0.1 + 30 * 1.0) / 50 = 0.64
    """
    frame_rewards_file = tmp_path / "frame_rewards.json"
    data = {
        "0": [
            [0, 410, 0.8],
            [410, 430, 0.1],
            [430, 600, 1.0],
        ]
    }
    with open(frame_rewards_file, "w", encoding="utf-8") as f:
        json.dump(data, f)

    config = SampleWeightingConfig(
        type="reward_weighted",
        frame_reward_path=str(frame_rewards_file),
        chunk_size=50,
        gamma=1.0,
        temperature=0.5,
    )
    device = torch.device("cpu")
    weighter = RewardSampleWeighter(config=config, device=device)

    # Batch with 1 sample starting at frame 410 of episode 0
    batch = {
        "episode_index": torch.tensor([0]),
        "frame_index": torch.tensor([410]),
        "action": torch.randn(1, 50, 6),
    }

    weights, stats = weighter.compute_batch_weights(batch)

    assert stats["type"] == "reward_weighted_chunk"
    assert abs(stats["mean_reward"] - 0.64) < 1e-4
    # With batch_size=1, normalized weight should be 1.0
    assert abs(weights[0].item() - 1.0) < 1e-4


def test_discounted_chunk_reward(tmp_path):
    """Test gamma-discounted chunk reward formula."""
    frame_rewards_file = tmp_path / "frame_rewards.json"
    data = {
        "0": [
            [0, 10, 1.0],
            [10, 20, 0.0],
        ]
    }
    with open(frame_rewards_file, "w", encoding="utf-8") as f:
        json.dump(data, f)

    gamma = 0.95
    config = SampleWeightingConfig(
        type="reward_weighted",
        frame_reward_path=str(frame_rewards_file),
        chunk_size=20,
        gamma=gamma,
    )
    device = torch.device("cpu")
    weighter = RewardSampleWeighter(config=config, device=device)

    batch = {
        "episode_index": torch.tensor([0]),
        "frame_index": torch.tensor([0]),
    }

    _, stats = weighter.compute_batch_weights(batch)

    # Analytical discounted reward
    discount_weights = np.array([gamma ** k for k in range(20)])
    frame_vals = np.array([1.0] * 10 + [0.0] * 10)
    expected_r = float(np.dot(frame_vals, discount_weights) / discount_weights.sum())

    assert abs(stats["mean_reward"] - expected_r) < 1e-4


def test_exponential_loss_weighting_multi_sample(tmp_path):
    """Test exponential weighting: 1.0 vs 0.1 reward downweighting."""
    frame_rewards_file = tmp_path / "frame_rewards.json"
    data = {
        "0": [[0, 100, 1.0]],  # High reward
        "1": [[0, 100, 0.1]],  # Low reward (mistake)
    }
    with open(frame_rewards_file, "w", encoding="utf-8") as f:
        json.dump(data, f)

    config = SampleWeightingConfig(
        type="reward_weighted",
        frame_reward_path=str(frame_rewards_file),
        chunk_size=50,
        temperature=0.5,
    )
    device = torch.device("cpu")
    weighter = RewardSampleWeighter(config=config, device=device)

    batch = {
        "episode_index": torch.tensor([0, 1]),
        "frame_index": torch.tensor([10, 10]),
        "action": torch.randn(2, 50, 6),
    }

    weights, stats = weighter.compute_batch_weights(batch)

    assert weights.shape == (2,)
    # Sum of normalized weights must equal batch_size = 2
    assert abs(weights.sum().item() - 2.0) < 1e-4

    # Raw weight ratio: exp((1.0 - 1.0)/0.5) / exp((0.1 - 1.0)/0.5) = exp(0) / exp(-1.8) = exp(1.8) ~= 6.05
    ratio = weights[0].item() / weights[1].item()
    expected_ratio = np.exp(1.8)
    assert abs(ratio - expected_ratio) < 0.05
    assert weights[0].item() > weights[1].item()


def test_intervention_metadata_format_auto_parsing(tmp_path):
    """Test parsing episode_interventions.json directly."""
    interv_file = tmp_path / "episode_interventions.json"
    data = {
        "0": {
            "episode_index": 0,
            "total_frames": 200,
            "intervention_start_frame": 100,
            "mistake_window": 40,
            "normal_reward": 0.8,
            "failure_reward": 0.1,
            "correction_reward": 1.0,
        }
    }
    with open(interv_file, "w", encoding="utf-8") as f:
        json.dump(data, f)

    config = SampleWeightingConfig(
        type="reward_weighted",
        frame_reward_path=str(interv_file),
        chunk_size=20,
    )
    device = torch.device("cpu")
    weighter = RewardSampleWeighter(config=config, device=device)

    # Frame 0~59: normal (0.8)
    # Frame 60~99: failure (0.1)
    # Frame 100~199: correction (1.0)
    batch = {
        "episode_index": torch.tensor([0, 0, 0]),
        "frame_index": torch.tensor([10, 70, 120]),
    }
    _, stats = weighter.compute_batch_weights(batch)

    assert stats["min_reward"] <= 0.15
    assert stats["max_reward"] >= 0.95


def test_factory_auto_detect_from_dataset_root(tmp_path):
    """Test auto-detection of meta/episode_interventions.json."""
    meta_dir = tmp_path / "meta"
    meta_dir.mkdir(parents=True)
    interv_file = meta_dir / "episode_interventions.json"
    with open(interv_file, "w", encoding="utf-8") as f:
        json.dump({"0": {"total_frames": 100, "intervention_start_frame": 50}}, f)

    config = SampleWeightingConfig(type="reward_weighted")
    policy = Mock()
    policy.config = Mock()
    policy.config.chunk_size = 50
    device = torch.device("cpu")

    weighter = make_sample_weighter(config, policy, device, dataset_root=str(tmp_path))

    assert isinstance(weighter, RewardSampleWeighter)
    assert 0 in weighter.ep_frame_rewards
    assert len(weighter.ep_frame_rewards[0]) == 100


def test_rwfm_loss_backward_integration():
    """Simulate lerobot_train.py weighted Flow-MSE loss."""
    config = SampleWeightingConfig(type="reward_weighted", temperature=0.5)
    device = torch.device("cpu")
    weighter = RewardSampleWeighter(config=config, device=device)

    # Batch with per-sample rewards in batch['reward']
    batch = {
        "reward": torch.tensor([1.0, 0.1]),
        "action": torch.randn(2, 50, 6),
    }
    sample_weights, _ = weighter.compute_batch_weights(batch)

    # Simulated per_sample_loss from SmolVLA policy(batch, reduction="none")
    per_sample_loss = torch.tensor([0.05, 0.05], requires_grad=True)

    # Weighted loss formula from lerobot_train.py
    loss = (per_sample_loss * sample_weights).sum() / (sample_weights.sum() + 1e-6)
    loss.backward()

    # The high reward sample gradient should be ~6x larger than the low reward sample
    grad_ratio = per_sample_loss.grad[0].item() / per_sample_loss.grad[1].item()
    assert abs(grad_ratio - np.exp(1.8)) < 0.05
