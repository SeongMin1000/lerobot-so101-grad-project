#!/usr/bin/env python3
"""Autonomous Real-World Reward Evaluator for SO-101 5-Block Manipulation.

Combines top-camera OpenCV occupancy verification, gripper grasp status,
and action smoothness/jerk penalties.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor


class AutoRewardEvaluator:
    """Evaluates task success and dense rewards on real robot hardware."""

    def __init__(
        self,
        target_verifier: Optional[Any] = None,
        success_reward: float = 10.0,
        step_penalty: float = -0.02,
        action_smoothness_coeff: float = 0.01,
        gripper_grasp_reward: float = 0.5,
    ):
        self.target_verifier = target_verifier
        self.success_reward = success_reward
        self.step_penalty = step_penalty
        self.action_smoothness_coeff = action_smoothness_coeff
        self.gripper_grasp_reward = gripper_grasp_reward

    def compute_action_smoothness_penalty(self, action_chunk: Tensor) -> float:
        """Compute jerk penalty on predicted action chunk: sum ||(a_{t+1} - a_t) - (a_t - a_{t-1})||^2."""
        if action_chunk.shape[1] < 3:
            return 0.0
        # Second difference along time dimension
        diff1 = action_chunk[:, 1:] - action_chunk[:, :-1]
        diff2 = diff1[:, 1:] - diff1[:, :-1]
        jerk = torch.mean(diff2 ** 2).item()
        return -float(self.action_smoothness_coeff * jerk)

    def evaluate_step(
        self,
        obs_dict: Dict[str, Any],
        action_chunk: Tensor,
        target_color: str = "red",
        top_camera_image: Optional[np.ndarray] = None,
        is_human_intervention: bool = False,
    ) -> Tuple[float, bool, Dict[str, float]]:
        """Calculate reward and termination flag for current step.
        
        Returns:
            reward: Total step reward.
            done: Whether episode terminated.
            info: Breakdown of reward components.
        """
        reward = self.step_penalty
        done = False
        info = {
            "step_penalty": self.step_penalty,
            "smoothness": 0.0,
            "target_success": 0.0,
            "gripper_bonus": 0.0,
        }

        # 1. Action smoothness penalty
        smooth_pen = self.compute_action_smoothness_penalty(action_chunk)
        reward += smooth_pen
        info["smoothness"] = smooth_pen

        # 2. Gripper status bonus (if gripper is closed near pick/place)
        if "observation.state" in obs_dict:
            state = obs_dict["observation.state"]
            if isinstance(state, Tensor):
                state_np = state.detach().cpu().numpy().flatten()
            else:
                state_np = np.asarray(state).flatten()

            # Gripper is joint 5 / last motor
            gripper_val = state_np[-1] if len(state_np) >= 6 else 0.0
            # If closed (typically low value or high torque depending on calibration)
            if gripper_val < 0.3:
                reward += self.gripper_grasp_reward
                info["gripper_bonus"] = self.gripper_grasp_reward

        # 3. Vision-based target verification
        if self.target_verifier is not None and top_camera_image is not None:
            try:
                # Check occupancy using OpenCV target verifier
                res = self.target_verifier.check_slot_occupancy(
                    top_camera_image, slot_label=target_color
                )
                if res.get("is_occupied", False):
                    reward += self.success_reward
                    done = True
                    info["target_success"] = self.success_reward
            except Exception:
                pass

        # 4. Human intervention penalty / terminal condition
        if is_human_intervention:
            done = True

        return reward, done, info
