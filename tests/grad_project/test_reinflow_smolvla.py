#!/usr/bin/env python3
"""Unit tests for SmolVLA ReinFlow online RL pipeline."""

import numpy as np
import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.grad_project.rl.buffer import ReinFlowRolloutBuffer
from lerobot.grad_project.rl.reinflow_smolvla import SmolVLAReinFlowPolicy
from lerobot.grad_project.rl.reward_evaluator import AutoRewardEvaluator
from lerobot.grad_project.rl.trainer import ReinFlowPPOTrainer
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE


@pytest.fixture
def dummy_policy():
    config = SmolVLAConfig(
        chunk_size=10,
        n_action_steps=10,
        num_steps=4,
        vlm_model_name="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        load_vlm_weights=False,  # fast initialization without downloading 500M weights
    )
    config.input_features = {
        f"{OBS_IMAGES}.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 64, 64)),
        f"{OBS_STATE}": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
    }
    config.output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
    }
    policy = SmolVLAReinFlowPolicy(config=config, default_rl_steps=4, default_sigma=0.05, enable_critic=True)
    return policy


def test_stochastic_sampling_and_eval(dummy_policy):
    device = torch.device("cpu")
    dummy_policy.to(device)

    batch = {
        f"{OBS_IMAGES}.top": torch.randn(2, 3, 64, 64, device=device),
        f"{OBS_STATE}": torch.randn(2, 6, device=device),
        f"{OBS_LANGUAGE_TOKENS}": torch.zeros((2, 10), dtype=torch.long, device=device),
        f"{OBS_LANGUAGE_ATTENTION_MASK}": torch.ones((2, 10), dtype=torch.bool, device=device),
    }

    # 1. Test Stochastic Sampling
    actions, log_prob, trajectory, extra_info = dummy_policy.sample_actions_stochastic(
        batch, num_steps=4, return_trajectory=True
    )

    assert actions.shape == (2, 10, 6)
    assert log_prob.shape == (2,)
    assert trajectory.shape == (2, 5, 10, 32)
    assert not torch.isnan(log_prob).any()
    assert extra_info["values"] is not None

    # 2. Test Trajectory Re-evaluation
    new_log_prob, new_val, entropy = dummy_policy.evaluate_trajectory_log_prob(
        batch, trajectory, num_steps=4
    )

    assert new_log_prob.shape == (2,)
    assert new_val.shape == (2, 1)
    assert entropy.shape == (2,)
    # Under same weights, log_prob should match exactly
    assert torch.allclose(log_prob, new_log_prob, atol=1e-4)


def test_buffer_and_ppo_step(dummy_policy):
    device = torch.device("cpu")
    dummy_policy.to(device)

    buffer = ReinFlowRolloutBuffer(capacity=50, device="cpu")
    evaluator = AutoRewardEvaluator()
    trainer = ReinFlowPPOTrainer(policy=dummy_policy, lr=1e-4, num_denoising_steps=4)

    # Collect 4 steps into buffer
    for _ in range(4):
        batch = {
            f"{OBS_IMAGES}.top": torch.randn(1, 3, 64, 64, device=device),
            f"{OBS_STATE}": torch.randn(1, 6, device=device),
            f"{OBS_LANGUAGE_TOKENS}": torch.zeros((1, 10), dtype=torch.long, device=device),
            f"{OBS_LANGUAGE_ATTENTION_MASK}": torch.ones((1, 10), dtype=torch.bool, device=device),
        }
        with torch.no_grad():
            actions, log_prob, trajectory, extra = dummy_policy.sample_actions_stochastic(batch, num_steps=4)
        reward, done, _ = evaluator.evaluate_step(batch, actions)
        buffer.add(
            obs=batch,
            action=actions,
            trajectory=trajectory,
            log_prob=log_prob,
            reward=reward,
            done=done,
            value=extra["values"].item(),
        )

    assert len(buffer) == 4
    for mini_batch in buffer.get_batches(batch_size=2):
        stats = trainer.train_step(mini_batch)
        assert "loss_total" in stats
        assert not np.isnan(stats["loss_total"])
