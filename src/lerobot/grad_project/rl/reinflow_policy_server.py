#!/usr/bin/env python3
"""ReinFlow Online RL Asynchronous Policy Server for SmolVLA (SO-101 5-Block Task).

Integrates SmolVLAReinFlowPolicy (4-step Flow-SDE stochastic sampling),
ReinFlowRolloutBuffer (GAE advantage estimation), and ReinFlowPPOTrainer
into the Hugging Face LeRobot Async Policy Server.
"""

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import draccus
import grpc
import numpy as np
import torch
from concurrent import futures

from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.helpers import TimedObservation, get_logger
from lerobot.async_inference.policy_server import PolicyServer
from lerobot.grad_project.perception.opencv_target_verifier import (
    TargetOccupancyVerifier,
    load_or_default_config,
)
from lerobot.grad_project.rl.buffer import ReinFlowRolloutBuffer
from lerobot.grad_project.rl.reinflow_smolvla import SmolVLACriticHead, SmolVLAReinFlowPolicy
from lerobot.grad_project.rl.reward_evaluator import AutoRewardEvaluator
from lerobot.grad_project.rl.trainer import ReinFlowPPOTrainer
from lerobot.policies import get_policy_class, make_pre_post_processors
from lerobot.transport import services_pb2, services_pb2_grpc


@dataclass
class ReinFlowPolicyServerConfig(PolicyServerConfig):
    """Configuration for ReinFlow Online RL Policy Server."""
    sigma: float = field(default=0.05, metadata={"help": "Flow-SDE noise injection std"})
    rl_steps: int = field(default=4, metadata={"help": "Flow-SDE denoising steps (fast 4-step)"})
    actor_lr: float = field(default=3e-5, metadata={"help": "Actor learning rate"})
    critic_lr: float = field(default=1e-4, metadata={"help": "Critic learning rate"})
    buffer_capacity: int = field(default=1000, metadata={"help": "Rollout buffer capacity"})
    update_batch_size: int = field(default=16, metadata={"help": "Transitions before PPO update"})
    ppo_epochs: int = field(default=4, metadata={"help": "PPO optimization epochs"})
    save_interval_updates: int = field(default=5, metadata={"help": "Updates between checkpoint saves"})
    checkpoint_output_dir: str = field(default="outputs/reinflow_smolvla_live", metadata={"help": "Checkpoint dir"})


class ReinFlowPolicyServer(PolicyServer):
    """gRPC Async Policy Server with live ReinFlow Flow-PPO Online RL."""

    def __init__(self, config: ReinFlowPolicyServerConfig):
        super().__init__(config)
        self.rf_config = config
        self.rf_buffer: Optional[ReinFlowRolloutBuffer] = None
        self.rf_trainer: Optional[ReinFlowPPOTrainer] = None
        self.reward_evaluator: Optional[AutoRewardEvaluator] = None
        self.target_verifier: Optional[TargetOccupancyVerifier] = None

        self._rollout_lock = threading.Lock()
        self._training_thread: Optional[threading.Thread] = None
        self._total_updates = 0
        self._episode_step_count = 0
        self._last_step_time = time.time()

        # Initialize Target Verifier for live auto-reward if config exists
        try:
            tv_path = Path("project/config/target_verifier.json")
            if tv_path.exists():
                tv_cfg = load_or_default_config(tv_path)
                self.target_verifier = TargetOccupancyVerifier(
                    tv_cfg, config_path=tv_path, frame_color="bgr"
                )
                self.logger.info("[REINFLOW] TargetOccupancyVerifier loaded for live auto-reward.")
        except Exception as e:
            self.logger.warning(f"[REINFLOW] Could not load TargetOccupancyVerifier: {e}")

        self.reward_evaluator = AutoRewardEvaluator(target_verifier=self.target_verifier)

    def InitPolicySpecs(self, policy_specs, context):  # noqa: N802
        """Initialize and wrap policy with SmolVLAReinFlowPolicy."""
        self.device = policy_specs.device
        self.policy_type = policy_specs.policy_type
        self.lerobot_features = policy_specs.lerobot_features
        self.actions_per_chunk = policy_specs.actions_per_chunk
        self._logged_policy_input_stats = False

        self.logger.info(f"[REINFLOW] Loading SmolVLA model from: {policy_specs.pretrained_name_or_path}")
        start = time.perf_counter()

        # Load policy and transmute to SmolVLAReinFlowPolicy
        policy_class = get_policy_class(self.policy_type)
        base_policy = policy_class.from_pretrained(policy_specs.pretrained_name_or_path)
        base_policy.__class__ = SmolVLAReinFlowPolicy
        base_policy.default_rl_steps = self.rf_config.rl_steps
        hidden_dim = getattr(getattr(base_policy.model, "vlm_with_expert", None), "expert_hidden_size", 576)
        base_policy.critic_head = SmolVLACriticHead(hidden_dim=hidden_dim).to(self.device)
        self.policy = base_policy
        self.policy.to(self.device)

        # Initialize ReinFlow Buffer and PPO Trainer
        self.rf_buffer = ReinFlowRolloutBuffer(
            capacity=self.rf_config.buffer_capacity,
            device=str(self.device),
        )
        self.rf_trainer = ReinFlowPPOTrainer(
            policy=self.policy,
            lr=self.rf_config.actor_lr,
            critic_lr=self.rf_config.critic_lr,
            num_denoising_steps=self.rf_config.rl_steps,
        )

        # Pre/Post processors
        device_override = {"device": self.device}
        preprocessor_overrides = {"device_processor": device_override}
        if policy_specs.rename_map:
            preprocessor_overrides["rename_observations_processor"] = {"rename_map": policy_specs.rename_map}

        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=policy_specs.pretrained_name_or_path,
            preprocessor_overrides=preprocessor_overrides,
            postprocessor_overrides={"device_processor": device_override},
        )

        end = time.perf_counter()
        self.logger.info(
            f"[REINFLOW] Policy ready on {self.device} (Fast {self.rf_config.rl_steps}-Step Flow-SDE Denoising, sigma={self.rf_config.sigma}) in {end - start:.2f}s"
        )
        return services_pb2.Empty()

    def _get_action_chunk(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        """Stochastic 4-step action sampling + live rollout buffer tracking."""
        with self._rollout_lock:
            # 1. Stochastic ReinFlow Action Sampling
            with torch.no_grad():
                if isinstance(self.policy, SmolVLAReinFlowPolicy):
                    actions, log_prob, trajectory, extra_info = self.policy.sample_actions_stochastic(
                        observation,
                        num_steps=self.rf_config.rl_steps,
                        sigma=self.rf_config.sigma,
                        return_trajectory=True,
                    )
                    value_est = extra_info["values"].item() if extra_info.get("values") is not None else 0.0
                else:
                    actions = self.policy.predict_action_chunk(observation)
                    log_prob = torch.tensor(0.0)
                    trajectory = []
                    value_est = 0.0

            # 2. Extract Top Camera Image for Auto-Reward if available
            top_img_bgr = None
            for key in ("observation.images.top", "observation.image.top", "observation.images.cam_top"):
                if key in observation:
                    raw_t = observation[key]
                    if isinstance(raw_t, torch.Tensor) and raw_t.ndim >= 3:
                        # (B, C, H, W) or (C, H, W)
                        img_np = raw_t[0].detach().cpu().numpy() if raw_t.ndim == 4 else raw_t.detach().cpu().numpy()
                        if img_np.shape[0] == 3:  # (3, H, W)
                            img_np = np.transpose(img_np, (1, 2, 0))
                        if img_np.max() <= 1.0:
                            img_np = (img_np * 255).astype(np.uint8)
                        top_img_bgr = img_np
                        break

            # 3. Calculate Real-Time Reward
            reward, done, info = self.reward_evaluator.evaluate_step(
                obs_dict=observation,
                action_chunk=actions,
                top_camera_image=top_img_bgr,
            )

            # 4. Store Transition into ReinFlow Rollout Buffer
            if self.rf_buffer is not None:
                self.rf_buffer.add(
                    obs=observation,
                    action=actions,
                    trajectory=trajectory,
                    log_prob=log_prob,
                    reward=reward,
                    done=done,
                    value=value_est,
                )

            self._episode_step_count += 1
            log_prob_val = log_prob.item() if isinstance(log_prob, torch.Tensor) else float(log_prob)
            buf_len = len(self.rf_buffer) if self.rf_buffer is not None else 0
            
            # Print live ReinFlow step progress
            self.logger.info(
                f"[REINFLOW #{self._episode_step_count:03d}] "
                f"LogProb: {log_prob_val:7.1f} | "
                f"Reward: {reward:+6.3f} | "
                f"V(s): {value_est:5.2f} | "
                f"Buffer: {buf_len:02d}/{self.rf_config.update_batch_size:02d}"
            )

            # 5. Trigger Online PPO Policy Update when buffer threshold reached
            if (
                self.rf_buffer is not None
                and len(self.rf_buffer) >= self.rf_config.update_batch_size
                and (self._training_thread is None or not self._training_thread.is_alive())
            ):
                self._trigger_background_ppo_update()

            if actions.ndim != 3:
                actions = actions.unsqueeze(0)

            return actions[:, : self.actions_per_chunk, :]

    def _trigger_background_ppo_update(self):
        """Asynchronously runs PPO update without blocking real-time robot streaming."""
        self._training_thread = threading.Thread(target=self._run_ppo_update, daemon=True)
        self._training_thread.start()

    def _run_ppo_update(self):
        """Executes PPO update on collected rollout transitions."""
        try:
            self.logger.info("\n" + "=" * 65)
            self.logger.info(">>> [REINFLOW ONLINE RL] Starting Background PPO Optimization...")
            self.logger.info("=" * 65)
            update_start = time.perf_counter()

            # Compute Generalized Advantage Estimation (GAE)
            self.rf_buffer.compute_advantages(gamma=0.99, gae_lambda=0.95)

            # Run PPO Mini-batch Optimization
            metrics = self.rf_trainer.train_step(
                self.rf_buffer,
                batch_size=min(8, len(self.rf_buffer)),
                ppo_epochs=self.rf_config.ppo_epochs,
            )

            self.rf_buffer.clear()
            self._total_updates += 1
            elapsed = time.perf_counter() - update_start

            save_dir = Path(self.rf_config.checkpoint_output_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            latest_path = save_dir / "reinflow_latest.pt"
            torch.save(self.policy.state_dict(), str(latest_path))

            self.logger.info(
                f"[REINFLOW PPO UPDATE #{self._total_updates} FINISHED in {elapsed:.2f}s]\n"
                f"  • Policy Loss:    {metrics.get('policy_loss', 0.0):.4f}\n"
                f"  • Value Loss:     {metrics.get('value_loss', 0.0):.4f}\n"
                f"  • KL Divergence:  {metrics.get('kl_divergence', 0.0):.4f}\n"
                f"  • Checkpoint:     {latest_path}\n"
                + "=" * 65 + "\n"
            )

            # Save numbered checkpoint periodically
            if self._total_updates % self.rf_config.save_interval_updates == 0:
                save_path = save_dir / f"reinflow_update_{self._total_updates:04d}.pt"
                torch.save(self.policy.state_dict(), str(save_path))
                self.logger.info(f"[REINFLOW] Periodic checkpoint saved: {save_path}")

        except Exception as e:
            self.logger.error(f"[REINFLOW RL] PPO Update Error: {e}", exc_info=True)


@draccus.wrap()
def main(config: ReinFlowPolicyServerConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    server = ReinFlowPolicyServer(config)
    
    import grpc
    from concurrent import futures
    from lerobot.transport import services_pb2_grpc
    
    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(server, grpc_server)
    
    bind_addr = f"{config.host}:{config.port}"
    grpc_server.add_insecure_port(bind_addr)
    server.logger.info(f"[REINFLOW] Starting Server on {bind_addr}...")
    grpc_server.start()
    
    try:
        grpc_server.wait_for_termination()
    except KeyboardInterrupt:
        server.logger.info("[REINFLOW] Shutting down Policy Server...")
        grpc_server.stop(0)


if __name__ == "__main__":
    main()
