#!/usr/bin/env python3
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Dedicated interactive async inference client for SO-101 (SmolVLA / ACT).
# - Pure inference (NO dataset recording, NO video encoding overhead).
# - Loads model ONCE onto GPU server via gRPC.
# - Interactive keyboard controls:
#     [SPACE] / [R] : Stop inference -> Return arm to OBSERVE pose -> Ready for next run.
#     [SPACE] / [R] : Start inference from OBSERVE pose.
#     [Q] / [Ctrl+C]: Stop and exit cleanly.

import json
import logging
import os
import select
import sys
import termios
import threading
import time
import tty
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat
from queue import Queue
from typing import Any

import draccus
import torch

from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.helpers import visualize_action_queue_size
from lerobot.async_inference.robot_client import RobotClient
from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    bi_so_follower,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    so_follower,
)
from lerobot.utils.import_utils import register_third_party_plugins


class TerminalKeyReader:
    """Non-blocking key reader that works in a foreground local or SSH TTY."""

    def __init__(self) -> None:
        self._fd: int | None = None
        self._original_attributes: list[Any] | None = None

    def __enter__(self) -> "TerminalKeyReader":
        if not sys.stdin.isatty():
            return self
        try:
            self._fd = sys.stdin.fileno()
            self._original_attributes = termios.tcgetattr(self._fd)
            tty.setcbreak(self._fd)
            termios.tcflush(self._fd, termios.TCIFLUSH)
        except Exception:
            self._fd = None
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._fd is not None and self._original_attributes is not None:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._original_attributes)
                termios.tcflush(self._fd, termios.TCIFLUSH)
            except Exception:
                pass
        self._fd = None
        self._original_attributes = None

    def poll_keys(self) -> list[str]:
        if self._fd is None:
            return []
        keys = []
        try:
            while select.select([self._fd], [], [], 0.0)[0]:
                chunk = os.read(self._fd, 64)
                if not chunk:
                    break
                for b in chunk:
                    if b == ord(" "):
                        keys.append("space")
                    elif b in (ord("r"), ord("R")):
                        keys.append("r")
                    elif b in (ord("q"), ord("Q"), 3):  # 3: Ctrl+C
                        keys.append("quit")
        except Exception:
            pass
        return keys


@dataclass
class InteractiveInferenceConfig(RobotClientConfig):
    """Configuration for interactive async inference with observe-pose reset."""

    runtime_config: str = "project/config/runtime.json"
    observe_duration_s: float = 2.5
    auto_start: bool = False


class InteractiveInferenceClient(RobotClient):
    """Async Robot Client with interactive Spacebar pause, observe return, and resume."""

    def __init__(self, config: InteractiveInferenceConfig):
        super().__init__(config)
        self.interactive_cfg = config
        self.is_running_inference = config.auto_start
        self._observe_target = self._load_observe_target(config.runtime_config)
        self._observe_lock = threading.Lock()

    def _load_observe_target(self, runtime_config_path: str) -> dict[str, float]:
        path = Path(runtime_config_path).expanduser()
        if not path.exists():
            repo_root = Path(__file__).resolve().parents[4]
            alt = repo_root / "project" / "config" / "runtime.json"
            if alt.exists():
                path = alt

        if not path.exists():
            logging.warning("runtime.json not found at %s. Returning empty target.", path)
            return {}

        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            observe_pose = data.get("poses", {}).get("observe", {})
            return {k: float(v) for k, v in observe_pose.items()}
        except Exception as e:
            logging.warning("Failed to load observe pose from %s: %s", path, e)
            return {}

    def move_to_observe(self, duration_s: float | None = None) -> bool:
        """Smoothly returns follower arm to observe pose."""
        duration_s = duration_s or self.interactive_cfg.observe_duration_s
        if not self._observe_target:
            print("⚠️  [WARN] observe_target is empty; cannot return to observe pose.", flush=True)
            return False

        with self._observe_lock:
            try:
                obs = self.robot.get_observation()
                cur = {k: float(obs[k]) for k in self.robot.action_features if k in obs}
                keys = [k for k in self.robot.action_features if k in cur and k in self._observe_target]
                if not keys:
                    return False

                steps = max(2, int(duration_s * self.config.fps))
                # Send current pose first to relieve any mechanical pressure
                self.robot.send_action({k: cur[k] for k in keys})

                for i in range(1, steps + 1):
                    a = i / steps
                    target_action = {k: (1.0 - a) * cur[k] + a * self._observe_target[k] for k in keys}
                    self.robot.send_action(target_action)
                    time.sleep(1.0 / self.config.fps)

                return True
            except Exception as e:
                self.logger.error(f"Error moving to observe pose: {e}")
                return False

    def toggle_observe_inference(self):
        """Toggle between active inference and returning to observe pose."""
        if self.is_running_inference:
            # STOP INFERENCE -> RETURN TO OBSERVE POSE
            self.is_running_inference = False
            with self.action_queue_lock:
                self.action_queue = Queue()

            print("\n" + "=" * 72, flush=True)
            print("🔄 [STOPPING INFERENCE] Returning follower arm to OBSERVE pose...", flush=True)
            self.move_to_observe()
            print("=" * 72, flush=True)
            print("⏸️  [PAUSED AT OBSERVE POSE] Follower arm is ready at OBSERVE pose.")
            print("   👉 Model is ALREADY LOADED in GPU memory (no reload needed!).")
            print("   👉 Reset or rearrange the 5 blocks on the workspace now.")
            print("   👉 Press [SPACE] or [R] to START new inference rollout!")
            print("   👉 Press [Q] to quit cleanly.")
            print("=" * 72 + "\n", flush=True)

        else:
            # START INFERENCE FROM OBSERVE POSE
            with self.action_queue_lock:
                self.action_queue = Queue()
            with self.latest_action_lock:
                self.latest_action = -1
            self.must_go.set()
            self.is_running_inference = True

            print("\n" + "=" * 72, flush=True)
            print("🚀 [INFERENCE RUNNING] SmolVLA autonomous inference started from OBSERVE pose!")
            print("   👉 Press [SPACE] or [R] anytime to STOP & return to OBSERVE pose.")
            print("   👉 Press [Q] to quit.")
            print("=" * 72 + "\n", flush=True)

    def _aggregate_action_queues(self, incoming_actions, aggregate_fn=None):
        if not self.is_running_inference:
            # Discard server actions while paused / at observe pose
            return
        super()._aggregate_action_queues(incoming_actions, aggregate_fn)

    def control_loop(self, task: str, verbose: bool = False):
        """Combined function for executing actions and streaming observations with keyboard controls."""
        self.start_barrier.wait()
        self.logger.info("Interactive control loop starting")

        # Initial move to observe pose
        print("\n[INIT] Moving follower arm smoothly to OBSERVE pose...", flush=True)
        self.move_to_observe()

        if self.is_running_inference:
            print("\n" + "=" * 72, flush=True)
            print("🚀 [INFERENCE RUNNING] SmolVLA autonomous inference started!")
            print("   👉 Press [SPACE] or [R] to STOP & return to OBSERVE pose.")
            print("   👉 Press [Q] to quit.")
            print("=" * 72 + "\n", flush=True)
        else:
            print("\n" + "=" * 72, flush=True)
            print("⏸️  [READY AT OBSERVE POSE] Model loaded & connected.")
            print("   👉 Verify 5 blocks on workspace.")
            print("   👉 Press [SPACE] or [R] to START inference!")
            print("   👉 Press [Q] to quit.")
            print("=" * 72 + "\n", flush=True)

        _performed_action = None
        _captured_observation = None

        with TerminalKeyReader() as key_reader:
            while self.running:
                control_loop_start = time.perf_counter()

                if key_reader is not None:
                    for key in key_reader.poll_keys():
                        if key in ("space", "r"):
                            self.toggle_observe_inference()
                        elif key == "quit":
                            print("\n🛑 [STOP] Quitting interactive inference session...", flush=True)
                            self.shutdown_event.set()
                            break

                if not self.running:
                    break

                if self.is_running_inference:
                    if self.actions_available():
                        _performed_action = self.control_loop_action(verbose)

                    if self._ready_to_send_observation():
                        _captured_observation = self.control_loop_observation(task, verbose)
                else:
                    time.sleep(self.config.environment_dt)

                time.sleep(max(0, self.config.environment_dt - (time.perf_counter() - control_loop_start)))

        return _captured_observation, _performed_action


def main():
    register_third_party_plugins()
    cfg = draccus.parse(InteractiveInferenceConfig)
    logging.info(pformat(cfg.to_dict()))

    client = InteractiveInferenceClient(cfg)

    if client.start():
        client.logger.info("Starting action receiver thread...")
        action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)
        action_receiver_thread.start()

        try:
            client.control_loop(task=cfg.task)
        finally:
            client.stop()
            action_receiver_thread.join()
            if cfg.debug_visualize_queue_size:
                visualize_action_queue_size(client.action_queue_size)
            client.logger.info("Client stopped")


if __name__ == "__main__":
    main()
