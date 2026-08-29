"""ReinFlow Online Reinforcement Learning Module for SmolVLA on SO-101.

This module provides:
1. SmolVLAReinFlowPolicy: Stochastic Flow Matching with tractable log-probabilities.
2. ReinFlowRolloutBuffer: GAE buffer for action chunk transitions.
3. AutoRewardEvaluator: Autonomous vision + hardware reward calculator.
4. ReinFlowPPOTrainer: Flow-PPO loss with SFT-KL regularization & action smoothness.
"""

from .reinflow_smolvla import SmolVLAReinFlowPolicy, SmolVLACriticHead
from .buffer import ReinFlowRolloutBuffer
from .reward_evaluator import AutoRewardEvaluator
from .trainer import ReinFlowPPOTrainer

__all__ = [
    "SmolVLAReinFlowPolicy",
    "SmolVLACriticHead",
    "ReinFlowRolloutBuffer",
    "AutoRewardEvaluator",
    "ReinFlowPPOTrainer",
]
