#!/usr/bin/env python3
"""ReinFlow Rollout Buffer with Generalized Advantage Estimation (GAE).

Stores transitions of chunk-level stochastic flow trajectories for on-policy PPO updates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Generator, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor


@dataclass
class Transition:
    obs: Dict[str, Tensor]
    action: Tensor
    trajectory: Tensor
    log_prob: Tensor
    reward: float
    done: bool
    value: float


class ReinFlowRolloutBuffer:
    """Experience buffer storing chunk-level trajectories for ReinFlow PPO training."""

    def __init__(
        self,
        capacity: int = 1000,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: str = "cuda",
    ):
        self.capacity = capacity
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = torch.device(device)
        self.transitions: List[Transition] = []

    def __len__(self) -> int:
        return len(self.transitions)

    def add(
        self,
        obs: Dict[str, Tensor],
        action: Tensor,
        trajectory: Tensor,
        log_prob: Tensor,
        reward: float,
        done: bool,
        value: float = 0.0,
    ) -> None:
        # Ensure tensors are detached and on cpu or specified device
        obs_detached = {k: v.detach().clone() for k, v in obs.items() if isinstance(v, torch.Tensor)}
        traj_tensor = trajectory.detach().clone() if isinstance(trajectory, torch.Tensor) else torch.empty(0)
        log_prob_tensor = log_prob.detach().clone() if isinstance(log_prob, torch.Tensor) else torch.tensor(float(log_prob))
        action_tensor = action.detach().clone() if isinstance(action, torch.Tensor) else torch.tensor(action)
        
        trans = Transition(
            obs=obs_detached,
            action=action_tensor,
            trajectory=traj_tensor,
            log_prob=log_prob_tensor,
            reward=float(reward),
            done=bool(done),
            value=float(value),
        )
        self.transitions.append(trans)

    def clear(self) -> None:
        """Clear all stored transitions."""
        self.transitions.clear()

    def compute_returns_and_advantages(
        self, last_value: float = 0.0
    ) -> Tuple[Tensor, Tensor]:
        """Compute Generalized Advantage Estimation (GAE) and discounted returns."""
        n = len(self.transitions)
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)

        last_gae = 0.0
        for t in reversed(range(n)):
            if t == n - 1:
                next_val = last_value
                next_non_terminal = 1.0 - float(self.transitions[t].done)
            else:
                next_val = self.transitions[t + 1].value
                next_non_terminal = 1.0 - float(self.transitions[t].done)

            delta = self.transitions[t].reward + self.gamma * next_val * next_non_terminal - self.transitions[t].value
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
            returns[t] = advantages[t] + self.transitions[t].value

        adv_tensor = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        ret_tensor = torch.tensor(returns, dtype=torch.float32, device=self.device)

        # Standardize advantages
        if len(adv_tensor) > 1:
            adv_tensor = (adv_tensor - adv_tensor.mean()) / (adv_tensor.std() + 1e-8)

        return ret_tensor, adv_tensor

    def get_batches(
        self, batch_size: int = 16, shuffle: bool = True
    ) -> Generator[Dict[str, Any], None, None]:
        """Yield batched training dictionaries ready for PPO loss computation."""
        n = len(self.transitions)
        returns, advantages = self.compute_returns_and_advantages()

        indices = np.arange(n)
        if shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, n, batch_size):
            batch_idx = indices[start_idx : start_idx + batch_size]
            b_len = len(batch_idx)

            # Collate observations
            obs_keys = self.transitions[0].obs.keys()
            b_obs = {}
            for k in obs_keys:
                tensors = [self.transitions[i].obs[k] for i in batch_idx]
                b_obs[k] = torch.cat(tensors, dim=0).to(self.device)

            b_actions = torch.cat([self.transitions[i].action for i in batch_idx], dim=0).to(self.device)
            b_trajectories = torch.cat([self.transitions[i].trajectory for i in batch_idx], dim=0).to(self.device)
            b_old_log_probs = torch.stack([self.transitions[i].log_prob for i in batch_idx], dim=0).to(self.device)
            b_old_values = torch.tensor([self.transitions[i].value for i in batch_idx], dtype=torch.float32, device=self.device).unsqueeze(1)
            b_returns = returns[batch_idx].unsqueeze(1)
            b_advantages = advantages[batch_idx]

            yield {
                "obs": b_obs,
                "actions": b_actions,
                "trajectories": b_trajectories,
                "old_log_probs": b_old_log_probs,
                "old_values": b_old_values,
                "returns": b_returns,
                "advantages": b_advantages,
            }
