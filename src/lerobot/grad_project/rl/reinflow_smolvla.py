#!/usr/bin/env python3
"""SmolVLA ReinFlow Policy Wrapper.

Extends Hugging Face's SmolVLA with stochastic discrete-time Markov denoising
(ReinFlow, NeurIPS 2025) to enable tractable log-probability computation and
sample-efficient online policy gradient (PPO / GRPO) optimization.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Unpack

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import (
    ActionSelectKwargs,
    SmolVLAPolicy,
    make_att_2d_masks,
)
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE


class SmolVLACriticHead(nn.Module):
    """Value function head estimating state value V(s) for PPO actor-critic optimization."""

    def __init__(self, hidden_dim: int = 576, mlp_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, mlp_dim),
            nn.LayerNorm(mlp_dim),
            nn.SiLU(),
            nn.Linear(mlp_dim, mlp_dim),
            nn.SiLU(),
            nn.Linear(mlp_dim, 1),
        )

    def forward(self, prefix_feature: Tensor) -> Tensor:
        """prefix_feature: (B, hidden_dim) -> Returns V(s): (B, 1)"""
        return self.net(prefix_feature)


class SmolVLAReinFlowPolicy(SmolVLAPolicy):
    """SmolVLA with ReinFlow stochastic flow matching and tractable log-probabilities."""

    def __init__(
        self,
        config: SmolVLAConfig,
        default_rl_steps: int = 4,
        default_sigma: float = 0.05,
        enable_critic: bool = True,
        **kwargs,
    ):
        super().__init__(config, **kwargs)
        self.default_rl_steps = default_rl_steps
        self.default_sigma = default_sigma

        # Step-wise log-sigma parameter (learnable or fixed)
        # For K steps, we have K-1 transitions
        self.log_sigmas = nn.Parameter(
            torch.full((default_rl_steps,), math.log(default_sigma), dtype=torch.float32)
        )

        # Critic value head attached to expert hidden size
        expert_hidden_dim = self.model.vlm_with_expert.expert_hidden_size
        self.critic = SmolVLACriticHead(hidden_dim=expert_hidden_dim) if enable_critic else None

    def get_log_sigmas(self, num_steps: int, device: torch.device) -> Tensor:
        """Retrieve or interpolate step-wise noise sigma."""
        log_sigmas = getattr(self, "log_sigmas", None)
        if log_sigmas is not None and len(log_sigmas) == num_steps:
            return torch.clamp(log_sigmas.to(device), min=math.log(1e-4), max=math.log(0.5)).exp()
        default_sigma = getattr(self, "default_sigma", 0.05)
        return torch.full((num_steps,), float(default_sigma), device=device)

    def sample_actions_stochastic(
        self,
        batch: Dict[str, Tensor],
        num_steps: Optional[int] = None,
        custom_sigma: Optional[float] = None,
        return_trajectory: bool = True,
        sigma: Optional[float] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Tensor], Dict[str, Any]]:
        """Stochastic flow matching action generation with tractable log-probabilities.
        
        Args:
            batch: LeRobot observation batch dictionary.
            num_steps: Number of denoising integration steps (default 4).
            custom_sigma: Noise scale override (if None, uses self.log_sigmas).
            return_trajectory: Whether to return the intermediate x_k trajectory tensor.
            
        Returns:
            actions: Cleaned action chunk (B, chunk_size, action_dim).
            total_log_prob: Exact log-probability sum_k log p(x_{k+1}|x_k, s) (B,).
            trajectory: Intermediate latent trajectory (B, num_steps + 1, chunk_size, max_action_dim).
            extra_info: Value estimate V(s) and auxiliary stats.
        """
        custom_sigma = custom_sigma if custom_sigma is not None else sigma
        num_steps = num_steps or self.default_rl_steps
        batch = self._prepare_batch(batch)

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        bsize = state.shape[0]
        device = state.device
        actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)

        # 1. Compute VLM prefix embeddings and KV-cache
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        _, past_key_values = self.model.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            fill_kv_cache=True,
        )

        # Value estimate V(s) from prefix feature mean
        value_est = None
        if self.critic is not None:
            # Aggregate prefix tokens as state representation
            prefix_summary = prefix_embs.mean(dim=1)
            # Map through critic head
            # If hidden sizes differ, project or slice to expert_hidden_size
            critic_dim = self.model.vlm_with_expert.expert_hidden_size
            if prefix_summary.shape[-1] != critic_dim:
                prefix_summary = prefix_summary[:, :critic_dim]
            value_est = self.critic(prefix_summary)

        # 2. Stochastic Flow-SDE rollout
        dt = -1.0 / num_steps
        sigmas = self.get_log_sigmas(num_steps, device)

        x_0 = self.model.sample_noise(actions_shape, device)
        x_t = x_0

        log_probs: List[Tensor] = []
        trajectory_list: List[Tensor] = [x_0]

        for step in range(num_steps):
            time_val = 1.0 + step * dt
            time_tensor = torch.tensor(time_val, dtype=torch.float32, device=device).expand(bsize)

            # Predict velocity field v_t
            v_t = self.model.denoise_step(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=x_t,
                timestep=time_tensor,
            )

            # Mean drift for the next step
            mean_next = x_t + dt * v_t

            if step < num_steps - 1:
                sigma_k = custom_sigma if custom_sigma is not None else sigmas[step]
                eps = torch.randn_like(x_t)
                x_next = mean_next + sigma_k * eps

                sigma_val = torch.as_tensor(sigma_k, dtype=torch.float32, device=device)
                diff_sq = ((x_next - mean_next) / sigma_val) ** 2
                dim_elements = x_t.shape[1] * x_t.shape[2]
                step_log_prob = -0.5 * (
                    diff_sq.sum(dim=(1, 2)) + dim_elements * (math.log(2 * math.pi) + 2 * torch.log(sigma_val))
                )
            else:
                # Final step deterministic projection
                x_next = mean_next
                step_log_prob = torch.zeros(bsize, device=device)

            log_probs.append(step_log_prob)
            trajectory_list.append(x_next)
            x_t = x_next

        total_log_prob = torch.stack(log_probs, dim=0).sum(dim=0)
        trajectory = torch.stack(trajectory_list, dim=1) if return_trajectory else None

        # Clean final action chunk
        original_action_dim = self.config.action_feature.shape[0]
        final_actions = x_t[:, :, :original_action_dim]

        if self.config.adapt_to_pi_aloha:
            final_actions = self._pi_aloha_encode_actions(final_actions)

        extra_info = {
            "values": value_est,
            "sigmas": sigmas.detach().cpu().tolist(),
        }

        return final_actions, total_log_prob, trajectory, extra_info

    def evaluate_trajectory_log_prob(
        self,
        batch: Dict[str, Tensor],
        trajectory: Tensor,
        num_steps: Optional[int] = None,
        custom_sigma: Optional[float] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Tensor]:
        """Re-evaluate the log-probability of a recorded rollout trajectory under current parameters.
        
        Args:
            batch: Observation dictionary.
            trajectory: (B, num_steps + 1, chunk_size, max_action_dim).
            num_steps: Number of integration steps.
            custom_sigma: Noise scale override.
            
        Returns:
            total_log_prob: Recomputed log-probabilities (B,).
            values: Current value function estimates V(s) (B, 1).
            entropy: Approximate differential entropy of the stochastic transitions (B,).
        """
        num_steps = num_steps or (trajectory.shape[1] - 1)
        batch = self._prepare_batch(batch)

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        bsize = state.shape[0]
        device = state.device

        # 1. Compute VLM prefix embeddings and KV-cache
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        _, past_key_values = self.model.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            fill_kv_cache=True,
        )

        value_est = None
        if self.critic is not None:
            prefix_summary = prefix_embs.mean(dim=1)
            critic_dim = self.model.vlm_with_expert.expert_hidden_size
            if prefix_summary.shape[-1] != critic_dim:
                prefix_summary = prefix_summary[:, :critic_dim]
            value_est = self.critic(prefix_summary)

        dt = -1.0 / num_steps
        sigmas = self.get_log_sigmas(num_steps, device)

        log_probs: List[Tensor] = []
        entropies: List[Tensor] = []
        dim_elements = trajectory.shape[2] * trajectory.shape[3]

        for step in range(num_steps):
            time_val = 1.0 + step * dt
            time_tensor = torch.tensor(time_val, dtype=torch.float32, device=device).expand(bsize)

            x_k = trajectory[:, step]
            x_next_target = trajectory[:, step + 1]

            # Model prediction with current weights
            v_t = self.model.denoise_step(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=x_k,
                timestep=time_tensor,
            )
            mean_next = x_k + dt * v_t

            if step < num_steps - 1:
                sigma_k = custom_sigma if custom_sigma is not None else sigmas[step]
                sigma_val = torch.as_tensor(sigma_k, dtype=torch.float32, device=device)
                diff_sq = ((x_next_target - mean_next) / sigma_val) ** 2
                step_log_prob = -0.5 * (
                    diff_sq.sum(dim=(1, 2)) + dim_elements * (math.log(2 * math.pi) + 2 * torch.log(sigma_val))
                )
                step_entropy = 0.5 * dim_elements * (1.0 + math.log(2 * math.pi) + 2 * torch.log(sigma_val))
                step_entropy_t = torch.full((bsize,), step_entropy.item() if isinstance(step_entropy, torch.Tensor) else step_entropy, device=device)
            else:
                step_log_prob = torch.zeros(bsize, device=device)
                step_entropy_t = torch.zeros(bsize, device=device)

            log_probs.append(step_log_prob)
            entropies.append(step_entropy_t)

        total_log_prob = torch.stack(log_probs, dim=0).sum(dim=0)
        total_entropy = torch.stack(entropies, dim=0).sum(dim=0)

        return total_log_prob, value_est, total_entropy
