#!/usr/bin/env python3
"""
Hybrid OpenCV + FSM/script + Diffusion Policy client for SO-101.

This module is intentionally separate from hybrid_cv_act_client.py so the
existing ACT client remains untouched. It uses the same gRPC transport, but
only accepts policy_type=diffusion.

High-level loop:
  observe pose -> top OpenCV detect -> nearest pregrasp script move
  -> Diffusion grasp window -> script move to target slot
  -> open gripper -> repeat

Start diffusion_policy_server on the GPU PC first, then run this on the
local robot PC.
"""


import json
import logging
import math
import pickle  # nosec - same trust boundary as LeRobot async client
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat
from queue import Empty, Queue
from typing import Any

import cv2
import draccus
import grpc

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so_follower,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    so_follower,
)
from lerobot.transport import services_pb2, services_pb2_grpc  # type: ignore
from lerobot.transport.utils import grpc_channel_options
from lerobot.utils.import_utils import register_third_party_plugins

from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.helpers import (
    FPSTracker,
    RemotePolicyConfig,
    TimedObservation,
    get_logger,
    map_robot_keys_to_lerobot_features,
    visualize_action_queue_size,
)
from lerobot.grad_project.perception.opencv_block_detector import DetectionResult, TopBlockDetector
from lerobot.grad_project.paths import detector_calib_path, runtime_config_path


@dataclass
class HybridDiffusionRobotClientConfig(RobotClientConfig):
    # Hybrid-specific config files.
    runtime_config: str = str(runtime_config_path())
    detector_calib: str = str(detector_calib_path())

    # Camera observation key from robot.get_observation(). You said this is top.
    top_key: str = "top"

    # Start with max_blocks=1. After one-block MVP succeeds, set to 5.
    max_blocks: int = 1

    # Diffusion Policy is active only during this grasp window.
    diffusion_grasp_seconds: float = 30.0

    # Current trained model uses n_action_steps=8.
    # Change this value only when using a Diffusion model trained with
    # a different n_action_steps value.
    expected_n_action_steps: int = 8

    # Script motion durations.
    observe_duration_s: float = 1.2
    pregrasp_duration_s: float = 1.5
    drop_duration_s: float = 1.5
    post_place_wait_s: float = 0.2
    settle_s: float = 0.25

    # Pose names in runtime_config["poses"].
    observe_pose_name: str = "observe"
    fallback_drop_pose_name: str = "drop_center"

    # If true, do OpenCV and pose selection but do not move motors or run Diffusion.
    dry_run: bool = False

    # Save detector overlays for debugging.
    debug_save_dir: str = ""

    # Verbose LeRobot async logs during Diffusion windows.
    verbose: bool = False

    def __post_init__(self) -> None:
        """Validate common async options and Diffusion-specific constraints."""
        super().__post_init__()

        # This client must never accidentally load ACT, SmolVLA, etc.
        if self.policy_type != "diffusion":
            raise ValueError(
                "hybrid_cv_diffusion_client는 "
                "--policy_type=diffusion만 허용합니다. "
                f"입력값: {self.policy_type!r}"
            )

        if self.diffusion_grasp_seconds <= 0:
            raise ValueError(
                "diffusion_grasp_seconds must be positive, "
                f"got {self.diffusion_grasp_seconds}"
            )

        if self.expected_n_action_steps <= 0:
            raise ValueError(
                "expected_n_action_steps must be positive, "
                f"got {self.expected_n_action_steps}"
            )

        # Current model:
        # grasp_C1_red_diffusion_h32_a8_ddim10_50k
        # has n_action_steps=8, so the client must request 8 actions.
        if self.actions_per_chunk != self.expected_n_action_steps:
            raise ValueError(
                "actions_per_chunk must match the Diffusion model's "
                "n_action_steps. "
                f"actions_per_chunk={self.actions_per_chunk}, "
                f"expected_n_action_steps={self.expected_n_action_steps}"
            )

        if self.aggregate_fn_name != "latest_only":
            logging.warning(
                "Diffusion async 추론에서는 최신 관측으로 생성된 action을 "
                "우선하도록 --aggregate_fn_name=latest_only 사용을 권장합니다. "
                "현재값=%s",
                self.aggregate_fn_name,
            )


class HybridCVDiffusionClient:
    prefix = "hybrid_cv_diffusion_client"
    logger = get_logger(prefix)

    def __init__(self, config: HybridDiffusionRobotClientConfig):
        self.config = config
        self.robot: Robot = make_robot_from_config(config.robot)
        self.robot.connect()

        self.runtime_config_path = runtime_config_path(config.runtime_config)
        self.detector_calib_path = detector_calib_path(config.detector_calib)
        self.logger.info("Using runtime config: %s", self.runtime_config_path)
        self.logger.info("Using detector calibration: %s", self.detector_calib_path)

        self.runtime = self._load_runtime(self.runtime_config_path)
        self.detector = TopBlockDetector.load(self.detector_calib_path, frame_color="rgb")

        self.debug_save_dir = Path(config.debug_save_dir) if config.debug_save_dir else None
        if self.debug_save_dir:
            self.debug_save_dir.mkdir(parents=True, exist_ok=True)

        self.placed_count = 0
        self.detect_count = 0

        lerobot_features = map_robot_keys_to_lerobot_features(self.robot)
        self.server_address = config.server_address
        self.policy_config = RemotePolicyConfig(
            config.policy_type,
            config.pretrained_name_or_path,
            lerobot_features,
            config.actions_per_chunk,
            config.policy_device,
        )

        self.channel = grpc.insecure_channel(
            self.server_address,
            grpc_channel_options(initial_backoff=f"{config.environment_dt:.4f}s"),
        )
        self.stub = services_pb2_grpc.AsyncInferenceStub(self.channel)
        self.logger.info("Initializing hybrid Diffusion client to connect to server at %s", self.server_address)

        self.shutdown_event = threading.Event()
        self.latest_action_lock = threading.Lock()
        self.latest_action = -1
        self.action_chunk_size = 1
        self._chunk_size_threshold = config.chunk_size_threshold
        self.action_queue: Queue = Queue()
        self.action_queue_lock = threading.Lock()
        self.action_queue_size: list[int] = []
        self.start_barrier = threading.Barrier(2)
        self.fps_tracker = FPSTracker(target_fps=self.config.fps)
        self.must_go = threading.Event()
        self.must_go.set()

        self.logger.info("Robot connected and hybrid Diffusion client ready")

    @property
    def running(self) -> bool:
        return not self.shutdown_event.is_set()

    def _load_runtime(self, path: str | Path) -> dict[str, Any]:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"runtime_config not found: {path}. Create it with hybrid_save_pose.py first."
            )
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("poses", {})
        data.setdefault("drop_slots", [])
        data.setdefault("pregrasp_points", [])
        data.setdefault("gripper", {})
        return data

    def start(self) -> bool:
        try:
            start_time = time.perf_counter()
            self.stub.Ready(services_pb2.Empty())
            self.logger.debug("Connected to policy server in %.4fs", time.perf_counter() - start_time)

            policy_config_bytes = pickle.dumps(self.policy_config)
            policy_setup = services_pb2.PolicySetup(data=policy_config_bytes)
            self.logger.info(
                "Sending policy instructions: type=%s path=%s device=%s actions_per_chunk=%s",
                self.policy_config.policy_type,
                self.policy_config.pretrained_name_or_path,
                self.policy_config.device,
                self.policy_config.actions_per_chunk,
            )
            self.stub.SendPolicyInstructions(policy_setup)
            self.shutdown_event.clear()
            return True
        except grpc.RpcError as e:
            self.logger.error("Failed to connect to policy server: %s", e)
            return False

    def stop(self) -> None:
        self.shutdown_event.set()
        try:
            self.clear_action_queue()
        except Exception:
            pass
        try:
            self.robot.disconnect()
            self.logger.debug("Robot disconnected")
        finally:
            self.channel.close()
            self.logger.debug("Client stopped, channel closed")

    # ---------------------------------------------------------------------
    # Queue / async Diffusion methods, adapted from robot_client.py
    # ---------------------------------------------------------------------
    def clear_action_queue(self) -> None:
        with self.action_queue_lock:
            self.action_queue = Queue()
            self.action_queue_size.append(0)
        self.must_go.set()
        self.logger.debug("Action queue cleared")

    def send_observation(self, obs: TimedObservation) -> bool:
        if not self.running:
            raise RuntimeError("Client not running. Call start() first.")
        from lerobot.transport.utils import send_bytes_in_chunks

        observation_bytes = pickle.dumps(obs)
        try:
            observation_iterator = send_bytes_in_chunks(
                observation_bytes,
                services_pb2.Observation,
                log_prefix="[CLIENT] Observation",
                silent=True,
            )
            _ = self.stub.SendObservations(observation_iterator)
            return True
        except grpc.RpcError as e:
            self.logger.error("Error sending observation #%s: %s", obs.get_timestep(), e)
            return False

    def receive_actions(self) -> None:
        self.start_barrier.wait()
        self.logger.info("Action receiving thread starting")
        while self.running:
            try:
                actions_chunk = self.stub.GetActions(services_pb2.Empty())
                if len(actions_chunk.data) == 0:
                    continue
                timed_actions = pickle.loads(actions_chunk.data)  # nosec
                if not timed_actions:
                    continue

                client_device = self.config.client_device
                if client_device != "cpu":
                    for timed_action in timed_actions:
                        if timed_action.get_action().device.type != client_device:
                            timed_action.action = timed_action.get_action().to(client_device)

                self.action_chunk_size = max(self.action_chunk_size, len(timed_actions))
                self._aggregate_action_queues(timed_actions)
                self.must_go.set()
            except grpc.RpcError as e:
                if self.running:
                    self.logger.error("Error receiving actions: %s", e)
            except Exception as e:
                if self.running:
                    self.logger.error("Unexpected receive_actions error: %s", e)

    def _aggregate_action_queues(self, incoming_actions: list[Any]) -> None:
        aggregate_fn = self.config.aggregate_fn
        future_action_queue: Queue = Queue()
        with self.action_queue_lock:
            internal_queue = list(self.action_queue.queue)
        current_action_queue = {a.get_timestep(): a.get_action() for a in internal_queue}

        for new_action in incoming_actions:
            with self.latest_action_lock:
                latest_action = self.latest_action
            if new_action.get_timestep() <= latest_action:
                continue
            if new_action.get_timestep() not in current_action_queue:
                future_action_queue.put(new_action)
            else:
                from lerobot.async_inference.helpers import TimedAction

                future_action_queue.put(
                    TimedAction(
                        timestamp=new_action.get_timestamp(),
                        timestep=new_action.get_timestep(),
                        action=aggregate_fn(current_action_queue[new_action.get_timestep()], new_action.get_action()),
                    )
                )
        with self.action_queue_lock:
            self.action_queue = future_action_queue

    def actions_available(self) -> bool:
        with self.action_queue_lock:
            return not self.action_queue.empty()

    def _ready_to_send_observation(self) -> bool:
        with self.action_queue_lock:
            qsize = self.action_queue.qsize()
        return qsize / max(1, self.action_chunk_size) <= self._chunk_size_threshold

    def _action_tensor_to_action_dict(self, action_tensor: Any) -> dict[str, float]:
        return {key: float(action_tensor[i].item()) for i, key in enumerate(self.robot.action_features)}

    def control_loop_action(self, verbose: bool = False) -> dict[str, Any] | None:
        with self.action_queue_lock:
            self.action_queue_size.append(self.action_queue.qsize())
            try:
                timed_action = self.action_queue.get_nowait()
            except Empty:
                return None

        action_dict = self._action_tensor_to_action_dict(timed_action.get_action())
        performed = self.robot.send_action(action_dict)
        with self.latest_action_lock:
            self.latest_action = timed_action.get_timestep()
        if verbose:
            self.logger.info("Diffusion action #%s performed", timed_action.get_timestep())
        return performed

    def control_loop_observation(self, task: str, verbose: bool = False) -> dict[str, Any]:
        raw_observation = self.robot.get_observation()
        raw_observation["task"] = task

        with self.latest_action_lock:
            latest_action = self.latest_action

        observation = TimedObservation(
            timestamp=time.time(),
            observation=raw_observation,
            timestep=max(latest_action, 0),
        )
        with self.action_queue_lock:
            observation.must_go = self.must_go.is_set() and self.action_queue.empty()
            current_queue_size = self.action_queue.qsize()

        self.send_observation(observation)
        if observation.must_go:
            self.must_go.clear()

        if verbose:
            fps_metrics = self.fps_tracker.calculate_fps_metrics(observation.get_timestamp())
            self.logger.info(
                "Obs #%s | Avg FPS %.2f / Target %.2f | queue=%s must_go=%s",
                observation.get_timestep(),
                fps_metrics["avg_fps"],
                fps_metrics["target_fps"],
                current_queue_size,
                observation.must_go,
            )
        return raw_observation

    def run_diffusion_grasp_window(self, seconds: float) -> None:
        if self.config.dry_run:
            self.logger.info("[DRY RUN] would run Diffusion Policy for %.2fs", seconds)
            return
        self.logger.info("Running Diffusion grasp window for %.2fs", seconds)
        self.clear_action_queue()
        start = time.perf_counter()
        # First observation immediately so the server can start inference.
        self.control_loop_observation(self.config.task, self.config.verbose)
        while self.running and time.perf_counter() - start < seconds:
            loop_start = time.perf_counter()
            if self.actions_available():
                self.control_loop_action(self.config.verbose)
            if self._ready_to_send_observation():
                self.control_loop_observation(self.config.task, self.config.verbose)
            time.sleep(max(0.0, self.config.environment_dt - (time.perf_counter() - loop_start)))
        self.clear_action_queue()
        self.logger.info("Diffusion grasp window done")

    # ---------------------------------------------------------------------
    # Script/FSM methods
    # ---------------------------------------------------------------------
    def get_current_pose(self) -> dict[str, float]:
        obs = self.robot.get_observation()
        return {key: float(obs[key]) for key in self.robot.action_features if key in obs}

    def pose(self, name: str) -> dict[str, float]:
        poses = self.runtime.get("poses", {})
        if name not in poses:
            raise KeyError(f"Pose '{name}' not found in runtime_config. Available: {list(poses)}")
        return {k: float(v) for k, v in poses[name].items()}

    def send_script_action(self, action: dict[str, float]) -> None:
        if self.config.dry_run:
            self.logger.info("[DRY RUN] script action: %s", action)
            return
        self.robot.send_action(action)

    def move_to_pose(self, target_pose: dict[str, float], duration_s: float, name: str = "") -> None:
        current = self.get_current_pose()
        keys = [k for k in self.robot.action_features if k in current and k in target_pose]
        if not keys:
            raise RuntimeError("No overlapping action keys between current pose and target pose")
        steps = max(2, int(duration_s * self.config.fps))
        self.logger.info("Script move to %s over %.2fs (%d steps)", name or "pose", duration_s, steps)
        for i in range(1, steps + 1):
            alpha = i / steps
            action = {k: (1.0 - alpha) * current[k] + alpha * float(target_pose[k]) for k in keys}
            # If target pose lacks some action keys, hold current values.
            for k in self.robot.action_features:
                if k not in action and k in current:
                    action[k] = current[k]
            self.send_script_action(action)
            time.sleep(self.config.environment_dt)
        time.sleep(self.config.settle_s)

    def open_gripper(self) -> None:
        open_value = self.runtime.get("gripper", {}).get("open")
        if open_value is None:
            # Fallback: use observe pose gripper value if present.
            open_value = self.pose(self.config.observe_pose_name).get("gripper.pos")
        if open_value is None:
            raise KeyError("No gripper.open in runtime_config and observe pose has no gripper.pos")
        current = self.get_current_pose()
        current["gripper.pos"] = float(open_value)
        self.logger.info("Opening gripper to %.3f", float(open_value))
        self.move_to_pose(current, duration_s=0.35, name="open_gripper")

    def detect_top(self) -> DetectionResult:
        raw = self.robot.get_observation()
        if self.config.top_key not in raw:
            raise KeyError(f"top_key '{self.config.top_key}' not in observation keys={list(raw)}")
        result = self.detector.detect(raw[self.config.top_key])
        self.logger.info(
            "CV detect: blocks=%d outside=%d target_count=%d chosen=%s",
            len(result.blocks),
            len(result.outside_blocks),
            result.target_count,
            None if result.chosen is None else f"{result.chosen.color}@({result.chosen.cx:.1f},{result.chosen.cy:.1f})",
        )
        if self.debug_save_dir:
            overlay = self.detector.draw(raw[self.config.top_key], result)
            out = self.debug_save_dir / f"detect_{self.detect_count:04d}.jpg"
            cv2.imwrite(str(out), overlay)
            self.detect_count += 1
            self.logger.info("Saved detector overlay %s", out)
        return result

    def nearest_pregrasp_pose(self, u: float, v: float) -> tuple[str, dict[str, float]]:
        points = self.runtime.get("pregrasp_points", [])
        if not points:
            raise RuntimeError("runtime_config has no pregrasp_points. Save grid points first.")
        best = None
        best_dist = float("inf")
        for p in points:
            du = float(p["u"]) - float(u)
            dv = float(p["v"]) - float(v)
            dist = math.sqrt(du * du + dv * dv)
            if dist < best_dist:
                best = p
                best_dist = dist
        assert best is not None
        label = str(best.get("label", "unknown"))
        pose = {k: float(val) for k, val in best["pose"].items()}
        self.logger.info(
            "Nearest pregrasp: label=%s calib_pixel=(%.1f,%.1f) query=(%.1f,%.1f) dist=%.1fpx",
            label,
            float(best["u"]),
            float(best["v"]),
            u,
            v,
            best_dist,
        )
        return label, pose

    def drop_pose_for_count(self, placed_count: int) -> tuple[str, dict[str, float]]:
        slots = self.runtime.get("drop_slots", [])
        if slots:
            name = slots[min(placed_count, len(slots) - 1)]
        else:
            name = self.config.fallback_drop_pose_name
        return name, self.pose(name)

    def pick_and_place_one(self, index: int) -> bool:
        self.logger.info("========== BLOCK %d / %d ==========" , index + 1, self.config.max_blocks)

        self.clear_action_queue()
        self.move_to_pose(self.pose(self.config.observe_pose_name), self.config.observe_duration_s, self.config.observe_pose_name)
        result = self.detect_top()
        if result.chosen is None:
            self.logger.warning("No outside block chosen. Stop loop.")
            return False

        chosen = result.chosen
        _, pregrasp_pose = self.nearest_pregrasp_pose(chosen.cx, chosen.cy)
        # Force gripper open before approaching if open value is known.
        open_value = self.runtime.get("gripper", {}).get("open")
        if open_value is not None:
            pregrasp_pose["gripper.pos"] = float(open_value)

        self.move_to_pose(pregrasp_pose, self.config.pregrasp_duration_s, "pregrasp")
        self.run_diffusion_grasp_window(self.config.diffusion_grasp_seconds)

        drop_name, drop_pose = self.drop_pose_for_count(self.placed_count)

        # drop pose는 팔 위치만 사용한다.
        # gripper.pos까지 따라가면 target으로 이동하는 도중 집게가 열릴 수 있음.
        # Diffusion이 끝난 직후의 gripper 상태, 즉 블록을 잡고 있는 상태를 유지한 채 이동한다.
        drop_pose = dict(drop_pose)
        for k in list(drop_pose.keys()):
            if "gripper" in k:
                drop_pose.pop(k)

        self.move_to_pose(drop_pose, self.config.drop_duration_s, drop_name)
        self.open_gripper()
        time.sleep(self.config.post_place_wait_s)
        self.placed_count += 1
        self.move_to_pose(self.pose(self.config.observe_pose_name), self.config.observe_duration_s, self.config.observe_pose_name)
        return True

    def hybrid_loop(self) -> None:
        self.start_barrier.wait()
        self.logger.info("Hybrid FSM loop starting")
        try:
            for i in range(self.config.max_blocks):
                ok = self.pick_and_place_one(i)
                if not ok:
                    break
            self.logger.info("Hybrid loop finished. placed_count=%d", self.placed_count)
        finally:
            self.clear_action_queue()


@draccus.wrap()
def hybrid_diffusion_client(cfg: HybridDiffusionRobotClientConfig) -> None:
    logging.info(pformat(asdict(cfg)))
    client = HybridCVDiffusionClient(cfg)
    if client.start():
        action_receiver_thread = threading.Thread(target=client.receive_actions, daemon=True)
        action_receiver_thread.start()
        try:
            client.hybrid_loop()
        finally:
            client.stop()
            action_receiver_thread.join(timeout=2.0)
            if cfg.debug_visualize_queue_size:
                visualize_action_queue_size(client.action_queue_size)
            client.logger.info("Hybrid Diffusion client stopped")


if __name__ == "__main__":
    register_third_party_plugins()
    hybrid_diffusion_client()
