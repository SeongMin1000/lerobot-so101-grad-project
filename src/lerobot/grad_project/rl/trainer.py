#!/usr/bin/env python3
"""ReinFlow PPO Trainer for SmolVLA.

Implements Clipped Surrogate Objective, Value loss, Entropy regularization,
and SFT Reference KL constraint for flow matching action policies.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from .reinflow_smolvla import SmolVLAReinFlowPolicy


class ReinFlowPPOTrainer:
    """Trainer executing PPO updates over ReinFlow stochastic rollout batches."""

    def __init__(
        self,
        policy: SmolVLAReinFlowPolicy,
        ref_policy: Optional[SmolVLAReinFlowPolicy] = None,
        lr: float = 3e-5,
        critic_lr: float = 1e-4,
        clip_eps: float = 0.2,
        value_coeff: float = 0.5,
        entropy_coeff: float = 0.001,
        kl_coeff: float = 0.05,
        grad_clip_norm: float = 1.0,
        num_denoising_steps: int = 4,
    ):
        self.policy = policy
        self.ref_policy = ref_policy
        self.clip_eps = clip_eps
        self.value_coeff = value_coeff
        self.entropy_coeff = entropy_coeff
        self.kl_coeff = kl_coeff
        self.grad_clip_norm = grad_clip_norm
        self.num_denoising_steps = num_denoising_steps

        # Separate actor (action expert) & critic parameters
        actor_params = [
            p for n, p in self.policy.model.named_parameters() if p.requires_grad
        ]
        if self.policy.critic is not None:
            critic_params = list(self.policy.critic.parameters())
            self.optimizer = AdamW(
                [
                    {"params": actor_params, "lr": lr},
                    {"params": critic_params, "lr": critic_lr},
                ],
                weight_decay=1e-4,
            )
        else:
            self.optimizer = AdamW(actor_params, lr=lr, weight_decay=1e-4)

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Perform a single PPO mini-batch gradient step."""
        self.policy.train()
        self.optimizer.zero_grad()

        obs = batch["obs"]
        trajectories = batch["trajectories"]
        old_log_probs = batch["old_log_probs"]
        old_values = batch["old_values"]
        returns = batch["returns"]
        advantages = batch["advantages"]

        # 1. Re-evaluate log probabilities and values under current parameters
        new_log_probs, new_values, entropies = self.policy.evaluate_trajectory_log_prob(
            obs, trajectories, num_steps=self.num_denoising_steps
        )

        # 2. Probability ratio r_t(theta) = exp(log_pi_new - log_pi_old)
        log_ratio = new_log_probs - old_log_probs
        ratio = torch.exp(torch.clamp(log_ratio, min=-10.0, max=10.0))

        # 3. Clipped Surrogate Policy Loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # 4. Value Loss (Clipped Value Objective)
        value_loss = torch.tensor(0.0, device=policy_loss.device)
        if new_values is not None:
            val_clipped = old_values + torch.clamp(new_values - old_values, -self.clip_eps, self.clip_eps)
            val_loss1 = (new_values - returns) ** 2
            val_loss2 = (val_clipped - returns) ** 2
            value_loss = 0.5 * torch.max(val_loss1, val_loss2).mean()

        # 5. Entropy Regularization
        entropy_loss = -self.entropy_coeff * entropies.mean()

        # 6. SFT Reference KL Divergence Penalty (Optional Anchor)
        kl_loss = torch.tensor(0.0, device=policy_loss.device)
        if self.ref_policy is not None:
            with torch.no_grad():
                ref_log_probs, _, _ = self.ref_policy.evaluate_trajectory_log_prob(
                    obs, trajectories, num_steps=self.num_denoising_steps
                )
            kl_approx = 0.5 * ((new_log_probs - ref_log_probs) ** 2).mean()
            kl_loss = self.kl_coeff * kl_approx

        # Total Objective
        total_loss = policy_loss + self.value_coeff * value_loss + entropy_loss + kl_loss
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip_norm)
        self.optimizer.step()

        return {
            "loss_total": total_loss.item(),
            "loss_policy": policy_loss.item(),
            "loss_value": value_loss.item(),
            "loss_entropy": entropy_loss.item(),
            "loss_kl": kl_loss.item(),
            "approx_kl": 0.5 * ((new_log_probs - old_log_probs) ** 2).mean().item(),
            "mean_ratio": ratio.mean().item(),
        }
