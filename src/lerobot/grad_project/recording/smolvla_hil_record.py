#!/usr/bin/env python3
#
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Remote-policy Human-in-the-Loop recorder for the graduation-project SO-101.

This recorder is deliberately *corrections only*:

* SmolVLA runs on the existing remote async policy server.
* The robot PC executes policy chunks but does not save autonomous actions.
* Space pauses the policy and freezes the follower at its measured position.
* The actuated SO-101 leader is moved to the follower pose without recording.
* Enter/C starts a human recovery/correction window.
* Right arrow saves only that expert window as one normal LeRobot episode.
* Space can then hand control back to the policy in the same physical rollout.

Keeping autonomous mistakes out of the dataset is intentional.  The normal
``lerobot-train`` behavioral-cloning path does not filter actions by an
``intervention`` flag, so saving failed policy actions would teach the new
checkpoint to reproduce them.  Every saved episode here has the same feature
schema as the existing top/wrist/belly demonstration datasets and can be
merged with them directly.
"""

from __future__ import annotations

from collections import deque
import contextlib
import enum
import json
import logging
import math
import os
from pathlib import Path

# Provide a generous default timeout (3.0s) for OpenCV camera frame reading to tolerate USB hub contention
os.environ.setdefault("CAMERA_MAX_AGE_MS", "3000")
import select
import sys
import termios
import threading
import time
import tty
from dataclasses import asdict, dataclass, is_dataclass
from pprint import pformat
from queue import Queue
from typing import Any

import cv2
import grpc
import numpy as np

from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.helpers import TimedAction
from lerobot.async_inference.robot_client import RobotClient
from lerobot.common.control_utils import sanity_check_dataset_robot_compatibility
from lerobot.configs import parser
from lerobot.datasets import LeRobotDataset, VideoEncodingManager
from lerobot.grad_project.control.hybrid_goto_both_pose import load_runtime, move_both
from lerobot.grad_project.paths import detector_calib_path, lerobot_root, runtime_config_path
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.transport import services_pb2
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

from .smolvla_record_observe_return import (
    RecordConfig as ObserveRecordConfig,
    _discard_current_episode,
    _goto_observe,
    _make_dataset_features,
    _pending_frame_count,
    _validate_observe_pose,
)


HIL_RECORDER_BUILD = "2026-09-12-remote-smolvla-correction-only-v4"


@dataclass
class HILRecordConfig(ObserveRecordConfig):
    """Observe-return recorder options plus remote-policy HIL controls."""

    server_address: str = "localhost:8080"
    policy_type: str = "smolvla"
    pretrained_name_or_path: str = ""
    policy_device: str = "cuda"
    client_device: str = "cpu"
    actions_per_chunk: int = 30
    chunk_size_threshold: float = 0.6
    aggregate_fn_name: str = "latest_only"
    server_rpc_timeout_s: float = 3.0

    record_mode: str = "corrections_only"  # "corrections_only" | "full_on_intervention"

    leader_handover_duration_s: float = 1.2
    leader_handover_fps: int = 30
    paused_poll_hz: int = 100

    # YOLO & Kinematics Auto-Recovery (Z key)
    use_yolo_recovery: bool = True
    use_yolo_detection: bool = True
    yolo_model_path: str = "project/models/yolo_block_detector/best.pt"
    detector_calib: str = str(detector_calib_path(must_exist=False))
    grasp_calibration: str = "project/config/grasp_pixel_to_robot_record.json"
    top_key: str = "top"
    pan_bias_direction: str = "left"
    pan_bias_near_deg: float = 1.0
    pan_bias_far_deg: float = 3.5
    macro_goto_duration_s: float = 2.0
    macro_return_duration_s: float = 3.0
    hover_z_offset_m: float = 0.08
    local_retract_duration_s: float = 1.0
    local_retract_ratio: float = 0.4
    local_retract_lock_pan: bool = True
    pre_intervention_seconds: float = 1.5

    debug_observation_dir: str | None = None
    debug_observation_limit: int = 1
    debug_motor_trace_dir: str | None = None
    debug_motor_trace_limit: int = 300

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.server_address.strip():
            raise ValueError("--server_address must not be empty")
        if not self.pretrained_name_or_path.strip():
            raise ValueError("--pretrained_name_or_path must name the deployed SmolVLA checkpoint")
        if self.record_mode not in {"full_on_intervention", "corrections_only"}:
            raise ValueError("--record_mode must be 'full_on_intervention' or 'corrections_only'")
        if self.actions_per_chunk < 2:
            raise ValueError("HIL resume requires --actions_per_chunk >= 2")
        if not 0 <= self.chunk_size_threshold <= 1:
            raise ValueError("--chunk_size_threshold must be in [0, 1]")
        if self.server_rpc_timeout_s <= 0:
            raise ValueError("--server_rpc_timeout_s must be greater than 0")
        if self.leader_handover_duration_s <= 0:
            raise ValueError("--leader_handover_duration_s must be greater than 0")
        if self.leader_handover_fps <= 0:
            raise ValueError("--leader_handover_fps must be greater than 0")
        if self.paused_poll_hz <= 0:
            raise ValueError("--paused_poll_hz must be greater than 0")
        if self.dataset.episode_time_s <= 0:
            raise ValueError("--dataset.episode_time_s is the correction timeout and must be positive")



class HILPhase(str, enum.Enum):
    AUTONOMOUS = "autonomous"
    INTERVENTION_PAUSED = "intervention_paused"
    DIRECT_RECORDING = "direct_recording"
    LOCAL_RECORDING = "local_recording"
    OBSERVE_RECORDING = "observe_recording"
    STACK_RECORDING = "stack_recording"
    REVIEW_PAUSED = "review_paused"

    # Backward compatibility aliases
    PAUSED = "intervention_paused"
    CORRECTING = "local_recording"


HIL_RECORDING_PHASES = {
    HILPhase.DIRECT_RECORDING,
    HILPhase.LOCAL_RECORDING,
    HILPhase.OBSERVE_RECORDING,
    HILPhase.STACK_RECORDING,
}


class HILCommand(str, enum.Enum):
    PAUSE_INTERVENTION = "pause_intervention"
    START_DIRECT_CORRECTION = "start_direct_correction"
    START_LOCAL_CORRECTION = "start_local_correction"
    START_STACK_HOVER_CORRECTION = "start_stack_hover_correction"
    START_OBSERVE_CORRECTION = "start_observe_correction"
    FREEZE_CORRECTION = "freeze_correction"
    DISCARD_FROZEN_CORRECTION = "discard_frozen_correction"
    SAVE_FROZEN_CORRECTION = "save_frozen_correction"
    RESUME_AUTONOMOUS_VIA_OBSERVE = "resume_autonomous_via_observe"
    NEXT_TRIAL = "next_trial"
    STOP = "stop"

    # Backward compatibility aliases
    TOGGLE_POLICY = "pause_intervention"
    START_CORRECTION = "freeze_correction"
    SAVE_CORRECTION = "save_frozen_correction"
    DISCARD_CORRECTION = "discard_frozen_correction"
    RESET_TO_HOVER = "start_observe_correction"
    LOCAL_RETRACT = "start_local_correction"


class TrialOutcome(str, enum.Enum):
    NEXT = "next"
    TARGET_REACHED = "target_reached"
    STOP = "stop"


def decode_hil_key_bytes(data: bytes) -> list[HILCommand]:
    """Decode complete terminal key bytes into HIL commands.

    Exposed as a pure helper so the safety-critical key mapping can be tested
    without opening serial devices or cameras.
    """

    commands: list[HILCommand] = []
    index = 0
    while index < len(data):
        remaining = data[index:]
        if remaining.startswith(b"\x1b[C"):
            commands.append(HILCommand.SAVE_FROZEN_CORRECTION)
            index += 3
        elif remaining.startswith(b"\x1b[D"):
            commands.append(HILCommand.DISCARD_FROZEN_CORRECTION)
            index += 3
        elif remaining.startswith((b"\x1b[A", b"\x1b[B")):
            index += 3
        else:
            byte = remaining[:1]
            index += 1
            if byte == b" ":
                commands.append(HILCommand.PAUSE_INTERVENTION)
            elif byte in {b"\r", b"\n"}:
                commands.append(HILCommand.START_DIRECT_CORRECTION)
            elif byte in {b"l", b"L", b"x", b"X"}:
                commands.append(HILCommand.START_LOCAL_CORRECTION)
            elif byte in {b"m", b"M"}:
                commands.append(HILCommand.START_STACK_HOVER_CORRECTION)
            elif byte in {b"o", b"O", b"z", b"Z"}:
                commands.append(HILCommand.START_OBSERVE_CORRECTION)
            elif byte in {b"c", b"C"}:
                commands.append(HILCommand.FREEZE_CORRECTION)
            elif byte in {b"r", b"R"}:
                commands.append(HILCommand.RESUME_AUTONOMOUS_VIA_OBSERVE)
            elif byte in {b"n", b"N"}:
                commands.append(HILCommand.NEXT_TRIAL)
            elif byte in {b"q", b"Q", b"\x03", b"\x1b"}:
                commands.append(HILCommand.STOP)
    return commands


class TerminalKeyReader:
    """Non-blocking key reader that works in a foreground local or SSH TTY."""

    def __init__(self) -> None:
        self._fd: int | None = None
        self._original_attributes: list[Any] | None = None
        self._buffer = b""
        self._escape_started_at: float | None = None

    def __enter__(self) -> "TerminalKeyReader":
        if not sys.stdin.isatty():
            raise RuntimeError(
                "HIL controls require an interactive foreground terminal. "
                "Do not pipe stdin or run the robot-side recorder in the background."
            )
        self._fd = sys.stdin.fileno()
        self._original_attributes = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        termios.tcflush(self._fd, termios.TCIFLUSH)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._fd is not None and self._original_attributes is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._original_attributes)
            termios.tcflush(self._fd, termios.TCIFLUSH)
        self._fd = None
        self._original_attributes = None
        self._buffer = b""
        self._escape_started_at = None

    def poll(self) -> list[HILCommand]:
        if self._fd is None:
            raise RuntimeError("TerminalKeyReader must be entered before poll()")

        while select.select([self._fd], [], [], 0.0)[0]:
            chunk = os.read(self._fd, 64)
            if not chunk:
                break
            self._buffer += chunk

        if not self._buffer:
            return []

        # Arrow keys begin with ESC.  Preserve a split prefix briefly instead
        # of misreading it as a standalone stop command.
        if self._buffer in {b"\x1b", b"\x1b["}:
            now = time.monotonic()
            if self._escape_started_at is None:
                self._escape_started_at = now
                return []
            if now - self._escape_started_at < 0.05:
                return []

        data = self._buffer
        self._buffer = b""
        self._escape_started_at = None
        return decode_hil_key_bytes(data)


class HILAsyncRobotClient(RobotClient):
    """RobotClient with pause/resume gating for safe human handovers."""

    def __init__(self, config: RobotClientConfig, *, rpc_timeout_s: float = 3.0):
        super().__init__(config)
        self._accept_policy_actions = threading.Event()
        self._minimum_chunk_timestamp = float("inf")
        self._policy_gate_lock = threading.Lock()
        self._rpc_timeout_s = rpc_timeout_s

    @staticmethod
    def action_chunk_is_fresh(incoming_actions: list[TimedAction], minimum_timestamp: float) -> bool:
        """Reject an entire pre-resume chunk, including its future-timestamped tail."""

        return bool(incoming_actions) and incoming_actions[0].get_timestamp() >= minimum_timestamp

    def _clear_policy_action_queue(self) -> None:
        with self.action_queue_lock:
            self.action_queue = Queue()
        self.must_go.set()

    def _aggregate_action_queues(self, incoming_actions, aggregate_fn=None):
        # Serialize this check and insertion with pause/resume.  Without the
        # gate, a receiver thread could pass the freshness check immediately
        # before pause, stall, and insert the old chunk after resume.
        with self._policy_gate_lock:
            if not self._accept_policy_actions.is_set():
                return
            if not self.action_chunk_is_fresh(incoming_actions, self._minimum_chunk_timestamp):
                self.logger.debug("Discarding a stale action chunk received across an HIL handover")
                return
            super()._aggregate_action_queues(incoming_actions, aggregate_fn)

    def _flush_remote_policy_queue(self) -> None:
        try:
            self.stub.Ready(services_pb2.Empty(), timeout=self._rpc_timeout_s)
        except grpc.RpcError as exc:
            raise RuntimeError(
                f"Failed to flush remote policy queue at {self.server_address}: {exc}"
            ) from exc

    def pause_policy_control(self) -> None:
        """Stop accepting chunks and flush local/remote pre-intervention work."""

        with self._policy_gate_lock:
            self._accept_policy_actions.clear()
            self._clear_policy_action_queue()
            self._flush_remote_policy_queue()
            self._clear_policy_action_queue()

    def resume_policy_control(self) -> None:
        """Resume from a fresh observation and reject every pre-resume chunk."""

        with self._policy_gate_lock:
            self._accept_policy_actions.clear()
            self._clear_policy_action_queue()
            self._flush_remote_policy_queue()
            self._minimum_chunk_timestamp = time.time()
            self._clear_policy_action_queue()
            self._accept_policy_actions.set()


def _hold_follower_at_measured_position(robot: Any) -> dict[str, float]:
    """Immediately replace the last policy goal with the measured SO-101 pose."""

    if hasattr(robot, "bus") and robot.bus is not None:
        try:
            present = robot.bus.sync_read("Present_Position")
            if isinstance(present, dict) and present:
                robot.bus.sync_write("Goal_Position", present)
                if hasattr(robot, "_last_goal_pos"):
                    robot._last_goal_pos = present.copy()
                if hasattr(robot, "_tracking_error_counts"):
                    robot._tracking_error_counts = dict.fromkeys(present, 0)
                if hasattr(robot, "_last_action_diagnostics"):
                    robot._last_action_diagnostics = {
                        "event": "hil_hold",
                        "requested_goal_pos": present.copy(),
                        "sent_goal_pos": present.copy(),
                        "previous_goal_pos": None,
                        "present_pos": present.copy(),
                        "tracking_error": dict.fromkeys(present, 0.0),
                    }
                return {f"{motor}.pos": float(value) for motor, value in present.items()}
        except Exception:
            pass

    observation = robot.get_observation()
    hold_action = {
        key: float(observation[key])
        for key in robot.action_features
        if key.endswith(".pos") and key in observation
    }
    if not hold_action:
        raise RuntimeError("No measured joint positions are available for the HIL hold")
    res = robot.send_action(hold_action)
    return res if isinstance(res, dict) and res else hold_action



def _enable_leader_torque_safe(teleop: Any) -> None:
    """Staggered torque enable with retries to prevent simultaneous inrush voltage sag on USB power."""
    if hasattr(teleop, "bus"):
        for name in teleop.bus.motors:
            try:
                teleop.bus.write("Torque_Enable", name, 1, num_retry=3)
                time.sleep(0.01)  # 10ms stagger prevents simultaneous inrush current spike
            except Exception as e:
                logger.warning(f"Could not enable torque on leader {name}: {e}")
    elif hasattr(teleop, "enable_torque"):
        try:
            teleop.enable_torque()
        except Exception as e:
            logger.warning(f"teleop.enable_torque failed: {e}")


def _enable_leader_hold_at_current_pose(teleop: Any) -> dict[str, float]:
    """Set current position as the goal before enabling torque, avoiding a jump."""

    current = {key: float(value) for key, value in teleop.get_action().items() if key.endswith(".pos")}
    if hasattr(teleop, "send_feedback"):
        try:
            teleop.send_feedback(current)
        except Exception as e:
            logger.debug(f"teleop.send_feedback failed: {e}")

    _enable_leader_torque_safe(teleop)
    return current


def _disable_leader_torque(teleop: Any) -> None:
    if hasattr(teleop, "disable_torque"):
        try:
            teleop.disable_torque()
        except Exception:
            pass


def _flush_terminal_input() -> None:
    try:
        if sys.stdin.isatty():
            termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    except Exception:
        pass


def _align_leader_to_follower(
    teleop: Any,
    follower_pose: dict[str, float],
    *,
    duration_s: float,
    fps: int,
) -> None:
    """Safely drive the leader arm to the frozen follower pose, then relax torque for human teleop."""

    print(f"🤖 [ALIGNING LEADER] 리더암을 팔로워암 자세로 동기화 이동 중 ({duration_s:.1f}s)...")
    current = _enable_leader_hold_at_current_pose(teleop)
    overlap = sorted(set(current) & set(follower_pose))
    if not overlap:
        return

    steps = max(2, int(duration_s * fps))
    for step in range(1, steps + 1):
        alpha = step / steps
        feedback = current.copy()
        for key in overlap:
            feedback[key] = (1 - alpha) * current[key] + alpha * follower_pose[key]
        try:
            teleop.send_feedback(feedback)
        except Exception as e:
            logger.warning(f"teleop.send_feedback failed during alignment: {e}")
            break
        precise_sleep(1.0 / fps)

    # Handover: Leader arm is now at follower pose.
    # Keep holding follower pose with torque enabled until operator presses Enter!
    # This prevents the leader arm from sagging/drooping under gravity before teleoperation starts.
    try:
        teleop.send_feedback({k: follower_pose[k] for k in overlap})
    except Exception as e:
        logger.debug(f"teleop.send_feedback final hold failed: {e}")
    print("✅ [LEADER ALIGNED] 리더암 동기화 완료! 현재 자세를 토크로 단단히 유지 중입니다.")
    print("   👉 리더암을 가볍게 잡은 후 Enter를 누르면 토크가 즉시 풀리며 매끄럽게 텔레오퍼레이션이 시작됩니다.")


def _create_or_resume_dataset(
    *,
    cfg: HILRecordConfig,
    robot: Any,
    dataset_features: dict,
) -> LeRobotDataset:
    if cfg.resume:
        num_cameras = len(robot.cameras) if hasattr(robot, "cameras") else 0
        dataset = LeRobotDataset.resume(
            cfg.dataset.repo_id,
            root=cfg.dataset.root,
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
            camera_encoder=cfg.dataset.camera_encoder,
            encoder_threads=cfg.dataset.encoder_threads,
            streaming_encoding=cfg.dataset.streaming_encoding,
            encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
            image_writer_processes=(cfg.dataset.num_image_writer_processes if num_cameras > 0 else 0),
            image_writer_threads=(
                cfg.dataset.num_image_writer_threads_per_camera * num_cameras if num_cameras > 0 else 0
            ),
        )
        sanity_check_dataset_robot_compatibility(
            dataset,
            robot,
            cfg.dataset.fps,
            dataset_features,
        )
        return dataset

    repo_name = cfg.dataset.repo_id.split("/", 1)[-1]
    if repo_name.startswith("eval_"):
        raise ValueError("HIL training datasets must not use the reserved 'eval_' prefix")

    cfg.dataset.stamp_repo_id()
    dataset = LeRobotDataset.create(
        cfg.dataset.repo_id,
        cfg.dataset.fps,
        root=cfg.dataset.root,
        robot_type=robot.name,
        features=dataset_features,
        use_videos=cfg.dataset.video,
        image_writer_processes=cfg.dataset.num_image_writer_processes,
        image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
        batch_encoding_size=cfg.dataset.video_encoding_batch_size,
        camera_encoder=cfg.dataset.camera_encoder,
        encoder_threads=cfg.dataset.encoder_threads,
        streaming_encoding=cfg.dataset.streaming_encoding,
        encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
    )
    print(f"[HIL DATASET] actual repo_id={dataset.repo_id}")
    return dataset


def _wait_for_trial_ready(
    *,
    trial_index: int,
    saved_corrections: int,
    cfg: HILRecordConfig,
    dataset: LeRobotDataset,
) -> bool:
    print()
    print("#" * 78)
    print(
        f"[HIL READY] physical trial {trial_index + 1} | "
        f"saved corrections {saved_corrections}/{cfg.dataset.num_episodes}"
    )
    print(f"[DATASET] {dataset.repo_id} | next episode {dataset.num_episodes}")
    print(f"[TASK] {cfg.dataset.single_task}")
    print("1) 평가와 같은 조건으로 5개 블록을 무작위 배치")
    print("2) 특히 현재 모델의 실패 위치/각도(B·G 정면 등)를 이번 trial에 포함")
    print("3) 작업공간과 비상정지 준비 후 ENTER -> autonomous 시작")
    print("   q + ENTER -> 새 trial을 시작하지 않고 종료")
    print("#" * 78)
    while True:
        answer = input("준비 완료: ENTER / 종료: q + ENTER > ").strip().lower()
        if answer == "":
            return True
        if answer in {"q", "quit", "exit"}:
            return False
        print("ENTER만 누르거나 q를 입력하세요.")


def _print_active_controls(record_mode: str = "corrections_only") -> None:
    print()
    print(f"[HIL CONTROLS - {record_mode.upper()} - terminal focus required]")
    print("  SPACE       자율추론 일시정지 / 현 위치에서 재개 (Pause/Resume 토글)")
    print("  R           OBSERVE 복귀 (미녹화) + 처음부터 추론 재시작 (Trial Reset)")
    print("  ENTER       현 위치 즉시 텔레오퍼레이션 시작 (Direct Teleop, 후퇴 없음) / 녹화 중엔 완료(Freeze)")
    print("  L           Smart Hover Rise (타깃 블록 상공 Hover 복귀 후 리더암 인계 -> 조작 녹화)")
    print("  M           Stack Hover (목표 구역 상공으로 이동 후 리더암 인계 -> 적재 조작 녹화)")
    print("  O           Observe Correction 시작 (Observe 복귀 -> YOLO 탐지 -> Taught Hover -> 조작 녹화)")
    print("  C           Correction 완료 및 녹화 즉시 동결(freeze) -> 리뷰 대기")
    print("  ←           현재 동결된 교정 에피소드 폐기 (인덱스 유지)")
    print("  →           현재 동결된 교정 에피소드 저장 (1개 독립 LeRobot episode)")
    print("  N           현재 trial 완료 및 다음 블록 배치 준비")
    print("  Q or ESC    즉시 종료; 미저장 데이터 폐기 후 안전 셧다운")
    print()


class HardcodedHoverResolver:
    """Resolves target block hover joint angles using taught RBF model and calibration (NO numerical IK)."""

    def __init__(self, cfg: HILRecordConfig):
        self.cfg = cfg
        self.root = lerobot_root()
        self.runtime = load_runtime(cfg.runtime_config)
        self.pregrasp_points = self.runtime.get("pregrasp_points", [])

        # 1. Load YOLO detector
        self.yolo_detector = None
        yolo_path = (
            self.root / cfg.yolo_model_path
            if not Path(cfg.yolo_model_path).is_absolute()
            else Path(cfg.yolo_model_path)
        )
        if cfg.use_yolo_detection and yolo_path.is_file():
            try:
                from lerobot.grad_project.perception.yolo_block_detector import YoloBlockDetector

                det_cfg_path = detector_calib_path(cfg.detector_calib, must_exist=False)
                self.yolo_detector = YoloBlockDetector.load(
                    str(yolo_path),
                    det_cfg_path if det_cfg_path.is_file() else None,
                    frame_color="rgb",
                )
                logging.info("YOLO Block Detector loaded successfully from %s", yolo_path)
            except Exception as e:
                logging.warning("Could not initialize YOLO detector: %s", e)

        # 2. Load Grasp homography
        self.grasp_homography = None
        grasp_calib_file = (
            self.root / cfg.grasp_calibration
            if not Path(cfg.grasp_calibration).is_absolute()
            else Path(cfg.grasp_calibration)
        )
        if grasp_calib_file.is_file():
            try:
                g_data = json.loads(grasp_calib_file.read_text())
                self.grasp_homography = np.array(g_data["homography_pixel_to_robot_xy_m"], dtype=np.float64)
                logging.info("Loaded taught grasp homography from %s", grasp_calib_file)
            except Exception as e:
                logging.warning("Could not load grasp homography: %s", e)

        # 3. Load Taught RBF Hover Joint Model
        self.hover_joint_model = None
        model_path = self.root / "project/config/hover_joint_model_record.json"
        if model_path.is_file():
            try:
                self.hover_joint_model = json.loads(model_path.read_text())
                logging.info("Loaded taught hover joint model from %s", model_path)
            except Exception as e:
                logging.warning("Failed to load hover joint model: %s", e)

    def pixel_to_robot_xy(self, px: float, py: float) -> np.ndarray | None:
        if self.grasp_homography is None:
            return None
        point = np.array([[[px, py]]], dtype=np.float64)
        x, y = cv2.perspectiveTransform(point, self.grasp_homography)[0, 0]
        return np.array([float(x), float(y)])

    def resolve_hover_pose(
        self,
        target_xy: np.ndarray | None,
        color: str,
        current_joint_deg: dict[str, float],
    ) -> dict[str, float]:
        if target_xy is not None and self.hover_joint_model is not None:
            try:
                names = self.hover_joint_model.get(
                    "joint_names",
                    ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
                )
                weights = np.array(self.hover_joint_model["weights"], dtype=np.float64)
                xy = np.array([[target_xy[0], target_xy[1]]], dtype=np.float64)

                m_type = self.hover_joint_model.get("type", "polynomial")
                if m_type == "rbf_multiquadric":
                    centers = np.array(self.hover_joint_model["centers_xy"], dtype=np.float64)
                    eps = float(self.hover_joint_model.get("eps", 0.08))
                    dists = np.linalg.norm(xy[:, None, :] - centers[None, :, :], axis=-1)
                    phi = np.sqrt(dists**2 + eps**2)
                    pred_joints = phi @ weights
                else:
                    deg = self.hover_joint_model.get("degree", 2)
                    x, y = xy[:, 0], xy[:, 1]
                    cols = [np.ones_like(x)]
                    for d in range(1, deg + 1):
                        for i in range(d + 1):
                            cols.append((x ** (d - i)) * (y**i))
                    feat = np.column_stack(cols)
                    pred_joints = feat @ weights

                pose_hover = {f"{n}.pos": float(pred_joints[0, i]) for i, n in enumerate(names)}

                # Distance-adaptive shoulder_pan direction compensation
                radius = float(np.hypot(target_xy[0], target_xy[1]))
                dist_norm = float(np.clip((radius - 0.12) / 0.25, 0.0, 1.0))
                mag = float(
                    self.cfg.pan_bias_near_deg
                    + (self.cfg.pan_bias_far_deg - self.cfg.pan_bias_near_deg) * dist_norm
                )
                dir_lower = str(self.cfg.pan_bias_direction).strip().lower()
                if dir_lower in {"right", "r", "우", "우측"}:
                    pan_bias = +mag
                    dir_tag = f"RIGHT +{mag:.2f}°"
                elif dir_lower in {"left", "l", "좌", "좌측"}:
                    pan_bias = -mag
                    dir_tag = f"LEFT -{mag:.2f}°"
                else:
                    pan_bias = 0.0
                    dir_tag = "OFF"

                if "shoulder_pan.pos" in pose_hover:
                    pose_hover["shoulder_pan.pos"] += pan_bias

                pose_hover["gripper.pos"] = 45.0
                print(
                    f"🎯 [TAUGHT RBF HOVER] RBF model evaluated for {color.upper()} at "
                    f"R={radius*100:.1f}cm (pan_bias={dir_tag}) - NO IK"
                )
                return pose_hover
            except Exception as e:
                print(f"[TAUGHT MODEL WARN] RBF evaluation failed: {e}. Falling back to taught pregrasp points.")

        # Hardcoded fallback: taught pregrasp points from runtime.json
        if self.pregrasp_points:
            for p in self.pregrasp_points:
                if p.get("color", "").lower() == color.lower() or p.get("label", "").lower() == color.lower():
                    pose = {k: float(v) for k, v in p["pose"].items()}
                    pose["gripper.pos"] = 45.0
                    print(f"📍 [HARDCODED PREGRASP] Selected taught pregrasp point '{p.get('label')}' for {color.upper()}")
                    return pose

            if target_xy is not None and self.grasp_homography is not None:
                best_point = None
                best_dist = float("inf")
                for p in self.pregrasp_points:
                    u, v = p.get("u"), p.get("v")
                    if u is not None and v is not None:
                        pt_xy = self.pixel_to_robot_xy(u, v)
                        if pt_xy is not None:
                            d = float(np.hypot(pt_xy[0] - target_xy[0], pt_xy[1] - target_xy[1]))
                            if d < best_dist:
                                best_dist = d
                                best_point = p
                if best_point is not None:
                    pose = {k: float(v) for k, v in best_point["pose"].items()}
                    pose["gripper.pos"] = 45.0
                    print(
                        f"📍 [HARDCODED PREGRASP] Selected closest taught pregrasp '{best_point.get('label')}' "
                        f"(dist={best_dist*100:.1f}cm)"
                    )
                    return pose

            # If nothing matched, select pregrasp closest to current arm pan angle
            cur_pan = current_joint_deg.get("shoulder_pan.pos", 0.0)
            best_p = min(
                self.pregrasp_points,
                key=lambda p: abs(float(p["pose"].get("shoulder_pan.pos", 0.0)) - cur_pan),
            )
            pose = {k: float(v) for k, v in best_p["pose"].items()}
            pose["gripper.pos"] = 45.0
            print(f"📍 [HARDCODED PREGRASP] Selected closest pan pregrasp '{best_p.get('label')}'")
            return pose

        fallback = dict(current_joint_deg)
        fallback["shoulder_lift.pos"] = min(-20.0, current_joint_deg.get("shoulder_lift.pos", -20.0) - 20.0)
        fallback["gripper.pos"] = 45.0
        return fallback

    def resolve_stack_hover_pose(self, cur_gripper: float = 0.0) -> dict[str, float]:
        """Returns taught stack_hover pose from runtime.json, keeping the current gripper angle."""
        poses = self.runtime.get("poses", {})
        if "stack_hover" in poses:
            hover_pose = {k: float(v) for k, v in poses["stack_hover"].items()}
            hover_pose["gripper.pos"] = cur_gripper
            logging.info("🎯 [STACK HOVER] Using exact taught runtime.json 'stack_hover' pose")
            return hover_pose

        if "drop_center" in poses:
            hover_pose = {k: float(v) for k, v in poses["drop_center"].items()}
            hover_pose["gripper.pos"] = cur_gripper
            logging.info("🎯 [STACK HOVER FALLBACK] 'stack_hover' not found; fallback to 'drop_center'")
            return hover_pose

        raise KeyError(
            f"Neither 'stack_hover' nor 'drop_center' found in runtime config. Available poses: {list(poses.keys())}"
        )


class HILSession:
    """One connected HIL session spanning any number of physical trials."""

    def __init__(
        self,
        *,
        cfg: HILRecordConfig,
        client: HILAsyncRobotClient,
        teleop: Any,
        dataset: LeRobotDataset,
        teleop_action_processor: RobotProcessorPipeline,
        robot_action_processor: RobotProcessorPipeline,
        robot_observation_processor: RobotProcessorPipeline,
        display_compressed_images: bool,
        observe_target: dict[str, float] | None = None,
    ) -> None:
        self.cfg = cfg
        self.client = client
        self.robot = client.robot
        self.teleop = teleop
        self.dataset = dataset
        self.teleop_action_processor = teleop_action_processor
        self.robot_action_processor = robot_action_processor
        self.robot_observation_processor = robot_observation_processor
        self.display_compressed_images = display_compressed_images
        self.observe_target = observe_target or {}

        self.phase = HILPhase.INTERVENTION_PAUSED
        self.saved_corrections = 0
        self.physical_trial = 0
        self._trial_intervention_count = 0
        self.has_intervened = False
        self.intervention_start_frame: int | None = None
        self.is_trial_recording = False
        self._correction_started_at: float | None = None
        self._expected_episode_index: int | None = None
        self._current_recovery_type: str | None = None
        self._current_target_block: str | None = None
        self._recorded_correction_frames = 0

        self.hover_resolver = HardcodedHoverResolver(self.cfg)
        self._cached_block_coords: dict[str, np.ndarray] = {}
        print("[HIL INIT] HardcodedHoverResolver initialized (NO numerical IK)")

        pre_sec = float(getattr(self.cfg, "pre_intervention_seconds", 1.5))
        self.pre_buffer_maxlen = max(0, int(round(self.cfg.dataset.fps * pre_sec)))
        self.pre_intervention_buffer: deque[dict] = deque(maxlen=self.pre_buffer_maxlen)
        if self.cfg.record_mode == "corrections_only":
            print("🚫 [PRE-BUFFER BYPASSED] Correction-Only 모드: 실패 구간 버퍼링 비활성화 (L/O 시점부터만 기록)")
        elif self.pre_buffer_maxlen > 0:
            print(f"📼 [PRE-BUFFER INIT] 실패 직전 {pre_sec:.1f}초 ({self.pre_buffer_maxlen}프레임) 롤링 프리버퍼 활성화")

    def _get_pending_frame_count(self) -> int:
        real_count = _pending_frame_count(self.dataset)
        if real_count > 0:
            return real_count
        return self._recorded_correction_frames

    def _flush_pre_intervention_buffer(self) -> int:
        """Flushes rolling pre-intervention frames (only used in full_on_intervention mode)."""
        if self.cfg.record_mode != "full_on_intervention":
            return 0
        flushed = 0
        while self.pre_intervention_buffer:
            frame = self.pre_intervention_buffer.popleft()
            self.dataset.add_frame(frame)
            flushed += 1
        if flushed > 0:
            self.intervention_start_frame = flushed
        return flushed

    def _pause_and_align(self) -> None:
        print("\n[INTERVENE] Flushing policy chunks and freezing follower...")
        follower_pose = _hold_follower_at_measured_position(self.robot)
        self.client.pause_policy_control()
        follower_pose = _hold_follower_at_measured_position(self.robot)
        print("[HANDOVER] 손을 리더암에서 떼세요. leader -> frozen follower pose")
        _align_leader_to_follower(
            self.teleop,
            follower_pose,
            duration_s=self.cfg.leader_handover_duration_s,
            fps=self.cfg.leader_handover_fps,
        )
        self.phase = HILPhase.INTERVENTION_PAUSED
        print("=" * 78)
        print("[INTERVENTION PAUSED] Follower 고정 및 Leader 동기화 완료 (프레임 미기록)")
        print("  👉 SPACE: 현 위치에서 자율추론 이어서 재개")
        print("  👉 R: OBSERVE 복귀 (미녹화) + 처음부터 자율추론 재시작 (Trial Reset)")
        print("  👉 Enter: 현 위치 즉시 텔레오퍼레이션 개입 시작 (Direct Teleop, 후퇴 없음)")
        print("  👉 L: Smart Hover Rise (타깃 블록 상공 Hover 복귀 후 리더암 인계 -> 조작 녹화)")
        print("  👉 M: Stack Hover (목표 구역 상공 이동 후 적재 교정 시작)")
        print("  👉 O: Observe Correction (Observe 복귀 -> YOLO 탐지 -> Taught Hover -> 조작 녹화)")
        print("=" * 78)

    def _get_top_image(self, obs: dict[str, Any] | None) -> Any | None:
        """Robustly retrieve top camera image supporting both policy ('camera1') and raw ('top') keys."""
        if not obs:
            return None
        candidates = [
            self.cfg.top_key,
            "camera1",
            "top",
            "cam_top",
            f"observation.images.{self.cfg.top_key}",
            "observation.images.camera1",
            "observation.images.top",
        ]
        for key in candidates:
            if key in obs and obs[key] is not None:
                return obs[key]
        return None

    def _update_cached_block_coords(self, obs: dict[str, Any] | None = None) -> None:
        """Cache block robot (x, y) coordinates from top camera image while at observe pose."""
        if self.hover_resolver is None or self.hover_resolver.yolo_detector is None:
            return
        if obs is None:
            try:
                obs = self.robot.get_observation()
            except Exception as e:
                logging.warning("Failed to get observation for YOLO cache: %s", e)
                return
        top_img = self._get_top_image(obs)
        if top_img is None:
            return
        try:
            res = self.hover_resolver.yolo_detector.detect(top_img)
            if res and res.blocks:
                for blk in res.blocks:
                    xy = self.hover_resolver.pixel_to_robot_xy(blk.cx, blk.cy)
                    if xy is not None:
                        self._cached_block_coords[blk.color.lower()] = xy
                if self._cached_block_coords:
                    print(f"🎯 [YOLO CACHE] 블록 좌표 캐싱 완료 ({len(self._cached_block_coords)}개): {list(self._cached_block_coords.keys())}")
        except Exception as e:
            logging.warning("Failed to cache block coordinates from observe frame: %s", e)

    def _pause_without_leader_motion(self) -> None:
        _hold_follower_at_measured_position(self.robot)
        self.client.pause_policy_control()
        _hold_follower_at_measured_position(self.robot)
        self.phase = HILPhase.INTERVENTION_PAUSED

    def _resume_autonomous(self) -> None:
        _disable_leader_torque(self.teleop)
        self.client.resume_policy_control()
        self.phase = HILPhase.AUTONOMOUS
        print("[AUTONOMOUS] fresh observation부터 policy 자율추론 시작")

    def _start_direct_correction(self) -> None:
        if self.phase != HILPhase.INTERVENTION_PAUSED:
            print(f"[IGNORED] INTERVENTION_PAUSED 상태에서만 Enter를 시작할 수 있습니다. (현재 상태: {self.phase.value})")
            return

        self._expected_episode_index = self.dataset.num_episodes
        self._correction_started_at = time.perf_counter()
        self._current_recovery_type = "direct"
        self._recorded_correction_frames = 0
        self._trial_intervention_count += 1
        self.has_intervened = True

        _flush_terminal_input()
        _disable_leader_torque(self.teleop)
        self.phase = HILPhase.DIRECT_RECORDING

        print("=" * 78)
        print("🎮 [ENTER: DIRECT TELEOP CORRECTION] 현 위치 즉시 텔레오퍼레이션 시작! (리더암 토크 해제됨)")
        print("   🔴 [녹화 중!] 리더암을 잡고 즉시 조작을 수행하세요. (후퇴 없이 현 위치에서 즉시 추종)")
        print("   👉 조작 완료 후 Enter 또는 C를 누르면 녹화가 즉시 동결(freeze)됩니다.")
        print("=" * 78)

    def _start_local_correction(self) -> None:
        if self.phase != HILPhase.INTERVENTION_PAUSED:
            print(f"[IGNORED] INTERVENTION_PAUSED 상태에서만 L을 시작할 수 있습니다. (현재 상태: {self.phase.value})")
            return

        self._expected_episode_index = self.dataset.num_episodes
        self._correction_started_at = time.perf_counter()
        self._current_recovery_type = "local"
        self._recorded_correction_frames = 0
        self._trial_intervention_count += 1
        self.has_intervened = True

        duration = float(getattr(self.cfg, "local_retract_duration_s", 1.0))

        obs = self.robot.get_observation()
        cur_robot = {
            k: float(obs[k])
            for k in self.robot.action_features
            if k.endswith(".pos") and k in obs
        }

        cur_pan = cur_robot.get("shoulder_pan.pos", 0.0)

        # 1. If observe cache is empty, attempt live YOLO from current top camera view
        if not self._cached_block_coords:
            top_img = self._get_top_image(obs)
            if top_img is not None and self.hover_resolver is not None and self.hover_resolver.yolo_detector is not None:
                try:
                    res = self.hover_resolver.yolo_detector.detect(top_img)
                    if res and res.blocks:
                        for blk in res.blocks:
                            xy = self.hover_resolver.pixel_to_robot_xy(blk.cx, blk.cy)
                            if xy is not None:
                                self._cached_block_coords[blk.color.lower()] = xy
                except Exception as e:
                    logging.warning("Live YOLO detection failed during local correction: %s", e)

        target_color = None
        target_xy = None
        hover_target = None

        # 2. Select the candidate block whose hover pose is closest to current arm position (shoulder_pan)
        if self._cached_block_coords and self.hover_resolver is not None:
            best_diff = float("inf")
            for c, xy in self._cached_block_coords.items():
                candidate_hover = self.hover_resolver.resolve_hover_pose(xy, c, cur_robot)
                diff = abs(candidate_hover.get("shoulder_pan.pos", 0.0) - cur_pan)
                if diff < best_diff:
                    best_diff = diff
                    target_color = c
                    target_xy = xy
                    hover_target = candidate_hover
            print(f"🎯 [AUTO TARGET SELECTION] 현재 실패 위치와 가장 가까운 타깃 블록: {target_color.upper()} (각도 편차: {best_diff:.1f}°)")

        if hover_target is None:
            # Fallback: lift vertically at current pan angle without jumping to table center
            if self.hover_resolver is not None:
                hover_target = self.hover_resolver.resolve_hover_pose(None, "red", cur_robot)
            else:
                hover_target = dict(cur_robot)
                hover_target["shoulder_lift.pos"] = min(-20.0, cur_robot.get("shoulder_lift.pos", -20.0) - 20.0)
                hover_target["gripper.pos"] = 45.0
            target_color = "current"

        self._current_target_block = target_color

        print("\n" + "=" * 78)
        print(f"⚡ [L: SMART HOVER RISE] {target_color.upper()} 블록 상공 Hover 자세로 부드럽게 복귀 중 ({duration:.1f}s)...")
        print("=" * 78)

        self._record_joint_trajectory(
            target=hover_target,
            duration_s=duration,
            fps=self.cfg.dataset.fps,
            smooth=True,
            record=True,
        )

        _flush_terminal_input()
        _disable_leader_torque(self.teleop)
        self.phase = HILPhase.LOCAL_RECORDING

        print("=" * 78)
        print(f"🎯 [HOVER ARRIVED] {target_color.upper()} 블록 상공 도착 완료! (그리퍼 45° 개방, 리더암 토크 해제됨)")
        print("   🔴 [녹화 중!] 리더암을 잡고 수직 하강하여 정상 파지 및 조작을 완료하세요.")
        print("   👉 조작 완료 후 Enter 또는 C를 누르면 녹화가 즉시 동결(freeze)됩니다.")
        print("=" * 78)

    def resolve_stack_hover_pose(self, cur_gripper: float = 0.0) -> dict[str, float]:
        """Resolves target stack hover pose from runtime config while preserving cur_gripper."""
        if self.hover_resolver is not None:
            try:
                return self.hover_resolver.resolve_stack_hover_pose(cur_gripper=cur_gripper)
            except Exception as e:
                logging.warning("Hover resolver failed to resolve stack_hover: %s", e)

        try:
            runtime = load_runtime(self.cfg.runtime_config)
            poses = runtime.get("poses", {})
            if "stack_hover" in poses:
                hover_pose = {k: float(v) for k, v in poses["stack_hover"].items()}
                hover_pose["gripper.pos"] = cur_gripper
                return hover_pose
            if "drop_center" in poses:
                hover_pose = {k: float(v) for k, v in poses["drop_center"].items()}
                hover_pose["gripper.pos"] = cur_gripper
                return hover_pose
        except Exception as e:
            logging.warning("Could not load runtime config for stack_hover: %s", e)

        # Fallback to standard SO-101 taught stack_hover with cur_gripper
        return {
            "shoulder_pan.pos": 1.0,
            "shoulder_lift.pos": -14.813186813186814,
            "elbow_flex.pos": -19.736263736263737,
            "wrist_flex.pos": 80.08791208791209,
            "wrist_roll.pos": -23.384615384615383,
            "gripper.pos": cur_gripper,
        }

    def _start_stack_hover_correction(self) -> None:
        if self.phase not in {HILPhase.INTERVENTION_PAUSED, HILPhase.AUTONOMOUS}:
            print(f"[IGNORED] INTERVENTION_PAUSED 또는 AUTONOMOUS 상태에서만 M을 시작할 수 있습니다. (현재 상태: {self.phase.value})")
            return

        if self.phase is HILPhase.AUTONOMOUS:
            self._pause_without_leader_motion()

        self._expected_episode_index = self.dataset.num_episodes
        self._correction_started_at = time.perf_counter()
        self._current_recovery_type = "stack_hover"
        self._current_target_block = "stack_target"
        self._recorded_correction_frames = 0
        self._trial_intervention_count += 1
        self.has_intervened = True

        duration = float(getattr(self.cfg, "macro_goto_duration_s", 2.0))

        obs = self.robot.get_observation()
        cur_gripper = float(obs.get("gripper.pos", 0.0)) if obs else 0.0

        # 블록 파지 여부 판별 (27.0° 이하이면 파지 상태로 간주하여 0.0° 완전 닫힘 토크 명령 인가)
        if cur_gripper <= 27.0:
            target_gripper = 0.0
            grip_msg = f"🔒 블록 파지 감지 (현재 각도: {cur_gripper:.1f}° <= 27.0°) -> 완전 닫힘 명령(0.0°)으로 꽉 쥐고 이동 (블록 낙하 방지)"
        else:
            target_gripper = cur_gripper
            grip_msg = f"🔓 그리퍼 개방 상태 유지 ({cur_gripper:.1f}° > 27.0°)"

        stack_hover_target = self.resolve_stack_hover_pose(cur_gripper=target_gripper)

        print("\n" + "=" * 78)
        print(f"⚡ [M: STACK HOVER] 목표 구역 상공(stack_hover)으로 부드럽게 이동 중 ({duration:.1f}s)...")
        print(f"   {grip_msg}")
        print("=" * 78)

        self._record_joint_trajectory(
            target=stack_hover_target,
            duration_s=duration,
            fps=self.cfg.dataset.fps,
            smooth=True,
            record=True,
        )

        _flush_terminal_input()
        _disable_leader_torque(self.teleop)
        self.phase = HILPhase.STACK_RECORDING

        print("=" * 78)
        print("🎯 [STACK HOVER ARRIVED] 목표 구역 상공 도착 완료! (리더암 토크 해제됨)")
        print("   🔴 [녹화 중!] 리더암을 잡고 타워 상단에 정밀 적재 후 그리퍼를 개방하세요.")
        print("   👉 적재 완료 후 Enter 또는 C를 누르면 녹화가 즉시 동결(freeze)됩니다.")
        print("=" * 78)

    def _start_observe_correction(self) -> None:
        if self.phase != HILPhase.INTERVENTION_PAUSED:
            print(f"[IGNORED] INTERVENTION_PAUSED 상태에서만 O를 시작할 수 있습니다. (현재 상태: {self.phase.value})")
            return

        if not self.observe_target:
            print("[WARN] observe_target이 설정되지 않아 Observe Correction을 진행할 수 없습니다.")
            return

        self._expected_episode_index = self.dataset.num_episodes
        self._correction_started_at = time.perf_counter()
        self._current_recovery_type = "observe"
        self._recorded_correction_frames = 0
        self._trial_intervention_count += 1
        self.has_intervened = True

        # Record failure arm pose before moving to observe pose
        pre_obs = self.robot.get_observation()
        fail_pan = float(pre_obs.get("shoulder_pan.pos", 0.0)) if pre_obs else 0.0

        print("\n" + "=" * 78)
        print("🚨 [O: OBSERVE CORRECTION] Observe 복귀 및 탐색 시작!")
        print(f"🤖 [STEP 1: RECOVERY] 실패 위치 -> OBSERVE POSE 복귀 궤적 녹화 중 ({self.cfg.macro_return_duration_s:.1f}s)...")
        print("=" * 78)

        self._record_joint_trajectory(
            target=self.observe_target,
            duration_s=self.cfg.macro_return_duration_s,
            fps=self.cfg.dataset.fps,
            smooth=True,
            record=True,
        )
        precise_sleep(0.3)

        obs = self.robot.get_observation()
        self._update_cached_block_coords(obs)
        cur_robot = {
            k: float(obs[k])
            for k in self.robot.action_features
            if k.endswith(".pos") and k in obs
        }

        priority_order = ["red", "yellow", "wood", "green", "blue"]
        priority_map = {c: idx for idx, c in enumerate(priority_order)}
        task_lower = str(self.cfg.dataset.single_task).lower()
        task_colors = [c for c in priority_order if c in task_lower]
        single_target_color = task_colors[0] if len(task_colors) == 1 else None

        target_color = single_target_color or "red"
        target_block = None
        target_xy = None

        top_img = self._get_top_image(obs)
        if top_img is not None and self.hover_resolver is not None and self.hover_resolver.yolo_detector is not None:
            try:
                res = self.hover_resolver.yolo_detector.detect(top_img)
                if res and res.blocks:
                    # 1. 구역 안(배치 완료)과 구역 밖(미완료) 블록 분리
                    outside_blocks = [b for b in res.blocks if not getattr(b, "in_target", False)]
                    inside_blocks = [b for b in res.blocks if getattr(b, "in_target", False)]

                    out_colors = [b.color.lower() for b in outside_blocks]
                    in_colors = [b.color.lower() for b in inside_blocks]
                    print(f"🔍 [YOLO DETECT] 구역 안(배치완료): {in_colors} | 구역 밖(대기중): {out_colors}")

                    # 2. 단일 타깃 지정이 있는 경우
                    if single_target_color:
                        matched = [b for b in outside_blocks if b.color.lower() == single_target_color]
                        if not matched:
                            matched = [b for b in res.blocks if b.color.lower() == single_target_color]
                        if matched:
                            target_block = matched[0]

                    # 3. 다중 블록(5개 블록 시퀀스): 구역 밖에 있는 블록 중 기본 우선순위(빨 -> 노 -> 나 -> 초 -> 파)가 가장 높은 블록 선택
                    if target_block is None:
                        if outside_blocks:
                            sorted_cands = sorted(
                                outside_blocks,
                                key=lambda b: (priority_map.get(b.color.lower(), 999), -b.cy, b.cx),
                            )
                            target_block = sorted_cands[0]
                            print(f"🎯 [AUTO TARGET] 구역 밖 최고 우선순위 블록 선택: {target_block.color.upper()} (구역 밖 후보: {out_colors})")
                        else:
                            print("⚠️ [AUTO TARGET] 구역 밖 블록이 없음 (모두 구역 안 배치됨). 전체 검출 블록 중 우선순위 선택")
                            sorted_cands = sorted(
                                res.blocks,
                                key=lambda b: (priority_map.get(b.color.lower(), 999), -b.cy, b.cx),
                            )
                            target_block = sorted_cands[0]

                    if target_block is not None:
                        target_color = target_block.color.lower()
                        target_xy = self.hover_resolver.pixel_to_robot_xy(target_block.cx, target_block.cy)
                        print(f"🎯 [YOLO TARGET] {target_color.upper()} 블록 위치 (cx={target_block.cx:.1f}, cy={target_block.cy:.1f}) -> robot_xy={target_xy}")
                else:
                    print("[YOLO] 테이블 위에 블록이 검출되지 않았습니다.")
            except Exception as e:
                print(f"[YOLO ERROR] 검출 중 오류 발생: {e}")

        self._current_target_block = target_color

        if self.hover_resolver is not None:
            hover_target = self.hover_resolver.resolve_hover_pose(target_xy, target_color, cur_robot)
        else:
            hover_target = dict(cur_robot)
            hover_target["gripper.pos"] = 45.0

        print(f"🤖 [STEP 2: RECOVERY] OBSERVE -> {target_color.upper()} 블록 상공 Hover 비행 궤적 녹화 중 ({self.cfg.macro_goto_duration_s:.1f}s)...")
        self._record_joint_trajectory(
            target=hover_target,
            duration_s=self.cfg.macro_goto_duration_s,
            fps=self.cfg.dataset.fps,
            smooth=True,
            record=True,
        )

        _flush_terminal_input()
        _disable_leader_torque(self.teleop)
        self.phase = HILPhase.OBSERVE_RECORDING

        print("=" * 78)
        print(f"🎯 [OBSERVE CORRECTION RECORDING] {target_color.upper()} 블록 상공 Hover 도착 완료! (리더암 토크 해제됨)")
        print("   🔴 [녹화 중!] 리더암을 잡고 블록으로 하강하여 정상 파지 및 조작을 수행하세요.")
        print("   👉 조작 완료 후 C를 누르면 녹화가 즉시 동결(freeze)됩니다.")
        print("=" * 78)

    def _freeze_correction(self) -> None:
        if self.phase not in HIL_RECORDING_PHASES:
            print(f"[IGNORED] 교정 녹화 중(Enter, L, M 또는 O 시작)에만 Enter/C로 완료할 수 있습니다. (현재 상태: {self.phase.value})")
            return

        _enable_leader_hold_at_current_pose(self.teleop)
        _hold_follower_at_measured_position(self.robot)
        self.phase = HILPhase.REVIEW_PAUSED
        frame_count = self._get_pending_frame_count()
        print("\n" + "=" * 78)
        print(f"⏸️ [REVIEW PAUSED] 교정 녹화 완료 및 동결 (총 {frame_count} 프레임)")
        print("   👉 → (오른쪽 화살표): 이 교정 에피소드 저장")
        print("   👉 ← (왼쪽 화살표): 이 교정 에피소드 폐기")
        print("=" * 78)

    def _discard_frozen_correction(self) -> None:
        if self.phase != HILPhase.REVIEW_PAUSED:
            print(f"[IGNORED] 동결된 리뷰 상태(REVIEW_PAUSED)에서만 ←(폐기)가 가능합니다. (현재 상태: {self.phase.value})")
            return
        self._discard_pending_correction("correction discarded by user with left arrow")
        _enable_leader_hold_at_current_pose(self.teleop)
        _hold_follower_at_measured_position(self.robot)
        self.phase = HILPhase.INTERVENTION_PAUSED
        print("\n" + "=" * 78)
        print("🗑️ [DISCARDED] 현재 교정 에피소드가 폐기되었습니다.")
        print("   👉 SPACE: 현 위치에서 자율추론 이어서 재개")
        print("   👉 R: OBSERVE 복귀 (미녹화) + 처음부터 자율추론 재시작 (Trial Reset)")
        print("   👉 Enter: 다시 Direct Teleoperation 시작")
        print("   👉 L: 다시 Local Correction 시작")
        print("   👉 M: 다시 Stack Hover Correction 시작")
        print("   👉 O: 다시 Observe Correction 시작")
        print("=" * 78)

    def _save_frozen_correction(self) -> TrialOutcome | None:
        if self.phase != HILPhase.REVIEW_PAUSED:
            print(f"[IGNORED] 동결된 리뷰 상태(REVIEW_PAUSED)에서만 →(저장)이 가능합니다. (현재 상태: {self.phase.value})")
            return None
        saved = self._save_correction_if_any()
        _enable_leader_hold_at_current_pose(self.teleop)
        _hold_follower_at_measured_position(self.robot)
        self.phase = HILPhase.INTERVENTION_PAUSED
        if not saved:
            print("\n❌ [SAVE FAILED] 저장할 교정 프레임이 없거나 이미 처리되었습니다.")
            return None
        print("\n" + "=" * 78)
        print(f"💾 [SAVED] 교정 에피소드가 저장되었습니다! (누적: {self.saved_corrections}/{self.cfg.dataset.num_episodes})")
        print("   👉 SPACE: 현 위치에서 자율추론 이어서 재개 (RESUME)")
        print("   👉 R: OBSERVE로 복귀 후 처음부터 다시 추론 (RESET)")
        print("   👉 Enter: 현 위치 즉시 Direct Teleoperation 시작")
        print("   👉 L: 다시 Local Correction 시작")
        print("   👉 M: 다시 Stack Hover Correction 시작")
        print("   👉 O: 다시 Observe Correction 시작")
        print("   👉 N: 현재 Trial 완료 및 다음 블록 준비")
        print("   👉 Q/ESC: 즉시 세션 종료")
        print("=" * 78)
        if saved and self.saved_corrections >= self.cfg.dataset.num_episodes:
            return TrialOutcome.TARGET_REACHED
        return None

    def _resume_autonomous_from_current(self) -> None:
        if self.phase != HILPhase.INTERVENTION_PAUSED:
            print(f"[IGNORED] INTERVENTION_PAUSED 상태에서만 현 위치 재개가 가능합니다. (현재: {self.phase.value})")
            return
        _flush_terminal_input()
        _disable_leader_torque(self.teleop)
        self.client.resume_policy_control()
        self.phase = HILPhase.AUTONOMOUS
        print("\n🚀 [AUTONOMOUS RESUMED] 현 위치에서 SmolVLA 자율추론 이어서 재개 (SPACE)!")

    def _resume_autonomous_via_observe(self) -> None:
        if self.phase in HIL_RECORDING_PHASES:
            print("[IGNORED] 먼저 Enter 또는 C를 눌러 교정을 완료/동결하세요.")
            return
        if self.phase == HILPhase.REVIEW_PAUSED:
            print("[IGNORED] 먼저 →(저장) 또는 ←(폐기)를 선택하세요.")
            return

        if self.phase == HILPhase.AUTONOMOUS:
            print("\n[RESET] 자율추론 일시 중단 후 OBSERVE 복귀 및 처음부터 리셋...")
            _hold_follower_at_measured_position(self.robot)
            self.client.pause_policy_control()

        print("\n🔄 [RESET TRIAL] Follower Arm OBSERVE 복귀 (미녹화) 및 처음부터 자율추론 재시작 (R)...")
        if self.observe_target:
            self._record_joint_trajectory(
                target=self.observe_target,
                duration_s=self.cfg.macro_return_duration_s,
                fps=self.cfg.dataset.fps,
                smooth=True,
                record=False,
            )
            self._update_cached_block_coords()
        _flush_terminal_input()
        _disable_leader_torque(self.teleop)
        self.client.resume_policy_control()
        self.phase = HILPhase.AUTONOMOUS
        print("🚀 [AUTONOMOUS RESTARTED] OBSERVE 자세에서 SmolVLA 자율추론 처음부터 시작!")

    def _save_correction_if_any(self) -> bool:
        """Saves any recorded correction frames as one episode."""
        if self._expected_episode_index is None:
            return False
        frame_count = self._get_pending_frame_count()
        if frame_count <= 0:
            self._discard_pending_correction("Empty correction")
            return False

        expected = self._expected_episode_index
        self.dataset.save_episode()
        self._recorded_correction_frames = 0
        if self.dataset.num_episodes != expected + 1:
            raise RuntimeError(
                f"HIL save verification failed: expected {expected + 1} episodes, "
                f"found {self.dataset.num_episodes}"
            )
        self.saved_corrections += 1
        task_phase = "stack" if self._current_recovery_type == "stack_hover" else "grasp"
        self._record_episode_intervention_meta(
            episode_index=expected,
            parent_rollout_id=self.physical_trial,
            intervention_index=self._trial_intervention_count,
            recovery_type=self._current_recovery_type or "unknown",
            task_phase=task_phase,
            target_block=self._current_target_block or "unknown",
            total_frames=frame_count,
        )
        self._expected_episode_index = None
        self._correction_started_at = None
        self._current_recovery_type = None
        print(
            f"\n🎉 [SAVED] 교정 에피소드 {expected} 저장 완료! "
            f"({frame_count} 프레임, 누적 {self.saved_corrections}/{self.cfg.dataset.num_episodes})"
        )
        print("   👉 SPACE: 현 위치에서 자율추론 이어서 재개")
        print("   👉 R: OBSERVE 복귀 (미녹화) + 처음부터 자율추론 재시작 (Trial Reset)")
        print("   👉 L: 추가 Local Correction / M: Stack Hover / O: 추가 Observe Correction")
        print("   👉 N: 현재 블록 시퀀스 종료 및 다음 trial 준비")
        return True

    def _record_joint_trajectory(
        self,
        target: dict[str, float],
        duration_s: float,
        fps: int,
        smooth: bool = True,
        record: bool = True,
    ) -> None:
        """Move follower and leader smoothly to target. Records frames if record=True."""
        obs = self.robot.get_observation()
        cur_robot = {k: float(obs[k]) for k in self.robot.action_features if k in obs and k.endswith(".pos")}
        cur_teleop = {k: float(v) for k, v in self.teleop.get_action().items() if k.endswith(".pos")}

        overlap = [k for k in self.robot.action_features if k in cur_robot and k in target]
        teleop_overlap = [k for k in self.teleop.action_features if k in cur_teleop and k in target]

        steps = max(2, int(duration_s * fps))
        dt = 1.0 / fps

        _enable_leader_hold_at_current_pose(self.teleop)

        for i in range(1, steps + 1):
            start_t = time.perf_counter()
            if smooth:
                alpha = 0.5 * (1.0 - math.cos(math.pi * (i / steps)))
            else:
                alpha = i / steps

            robot_cmd = {}
            for k in overlap:
                robot_cmd[k] = (1.0 - alpha) * cur_robot[k] + alpha * target[k]

            teleop_cmd = {}
            for k in teleop_overlap:
                teleop_cmd[k] = (1.0 - alpha) * cur_teleop[k] + alpha * target[k]

            self.robot.send_action(robot_cmd)
            self.teleop.send_feedback(teleop_cmd)

            if record:
                step_obs = self.robot.get_observation()
                proc_obs = self.robot_observation_processor(step_obs)
                obs_frame = build_dataset_frame(self.dataset.features, proc_obs, prefix=OBS_STR)

                proc_act = self.teleop_action_processor((robot_cmd, step_obs))
                act_frame = build_dataset_frame(self.dataset.features, proc_act, prefix=ACTION)

                self.dataset.add_frame({**obs_frame, **act_frame, "task": self.cfg.dataset.single_task})
                self._recorded_correction_frames += 1

                if self.cfg.display_data:
                    log_rerun_data(
                        observation=proc_obs,
                        action=proc_act,
                        compress_images=self.display_compressed_images,
                    )

            elapsed = time.perf_counter() - start_t
            precise_sleep(max(0.0, dt - elapsed))

        settle_steps = max(2, int(0.8 * fps))
        for _ in range(settle_steps):
            step_obs = self.robot.get_observation()
            cur_pos = {k: float(step_obs[k]) for k in overlap if k in step_obs}

            def _joint_err(k: str) -> float:
                # 블록 파지 시(목표 5° 이하, 실제 각도 30° 이하) 물리적 저항으로 0°에 도달 불가하므로 수렴 검사에서 제외
                if "gripper" in k and target.get(k, 0.0) <= 5.0 and cur_pos.get(k, 0.0) <= 30.0:
                    return 0.0
                return abs(cur_pos[k] - target[k])

            max_err = max((_joint_err(k) for k in overlap if k in cur_pos), default=0.0)
            if max_err < 2.0:
                break

            start_t = time.perf_counter()
            self.robot.send_action(target)
            self.teleop.send_feedback(target)

            if record:
                proc_obs = self.robot_observation_processor(step_obs)
                obs_frame = build_dataset_frame(self.dataset.features, proc_obs, prefix=OBS_STR)
                proc_act = self.teleop_action_processor((target, step_obs))
                act_frame = build_dataset_frame(self.dataset.features, proc_act, prefix=ACTION)
                self.dataset.add_frame({**obs_frame, **act_frame, "task": self.cfg.dataset.single_task})
                self._recorded_correction_frames += 1

                if self.cfg.display_data:
                    log_rerun_data(
                        observation=proc_obs,
                        action=proc_act,
                        compress_images=self.display_compressed_images,
                    )

            elapsed = time.perf_counter() - start_t
            precise_sleep(max(0.0, dt - elapsed))

    def _discard_pending_correction(self, reason: str) -> None:
        if self._expected_episode_index is None:
            return
        if self.dataset.has_pending_frames() or self._get_pending_frame_count() > 0:
            _discard_current_episode(
                dataset=self.dataset,
                expected_episode_index=self._expected_episode_index,
                reason=reason,
            )
        self._recorded_correction_frames = 0
        self._expected_episode_index = None
        self._correction_started_at = None
        self._current_recovery_type = None
        self.has_intervened = False
        self.intervention_start_frame = None
        self.pre_intervention_buffer.clear()
        self.is_trial_recording = False

    def _finish_trial(self, *, save: bool) -> TrialOutcome:
        _enable_leader_hold_at_current_pose(self.teleop)
        _hold_follower_at_measured_position(self.robot)

        if not save:
            self._discard_pending_correction("trial finished by user")
            self.phase = HILPhase.INTERVENTION_PAUSED
            print("[PAUSED] trial ended. Preparing next trial...")
            return TrialOutcome.NEXT

        if self._expected_episode_index is not None:
            self._save_correction_if_any()
        self.has_intervened = False
        self.is_trial_recording = False
        self.phase = HILPhase.INTERVENTION_PAUSED
        print(
            f"[TRIAL FINISHED] trial 종료. 다음 블록을 준비합니다. "
            f"(누적 저장 교정: {self.saved_corrections}/{self.cfg.dataset.num_episodes})"
        )
        if self.saved_corrections >= self.cfg.dataset.num_episodes:
            return TrialOutcome.TARGET_REACHED
        return TrialOutcome.NEXT

    def _record_episode_intervention_meta(
        self,
        *,
        episode_index: int,
        parent_rollout_id: int,
        intervention_index: int,
        recovery_type: str,
        task_phase: str = "grasp",
        target_block: str = "red",
        total_frames: int,
    ) -> None:
        """Saves episode metadata for RWFM and HIL analysis."""
        try:
            meta_dir = Path(self.dataset.root) / "meta"
            meta_dir.mkdir(parents=True, exist_ok=True)
            meta_path = meta_dir / "episode_interventions.json"

            existing = {}
            if meta_path.exists():
                with open(meta_path, "r", encoding="utf-8") as f:
                    try:
                        existing = json.load(f)
                    except Exception:
                        existing = {}

            existing[str(episode_index)] = {
                "episode_index": episode_index,
                "parent_rollout_id": parent_rollout_id,
                "intervention_index": intervention_index,
                "recovery_type": recovery_type,
                "task_phase": task_phase,
                "target_block": target_block,
                "total_frames": total_frames,
                "intervention_start_frame": 0,
                "timestamp": time.time(),
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2)
            print(
                f"📝 [RWFM METADATA] 에피소드 {episode_index}: "
                f"rollout={parent_rollout_id}, intv={intervention_index}, "
                f"type={recovery_type}, target={target_block}, "
                f"frames={total_frames} -> {meta_path}"
            )
        except Exception as e:
            print(f"[METADATA WARN] Failed to save intervention metadata: {e}")

    def _autonomous_tick(self) -> None:
        performed_action = None
        if self.client.actions_available():
            performed_action = self.client.control_loop_action()
        if self.client._ready_to_send_observation():
            self.client.control_loop_observation(self.cfg.dataset.single_task)

        if performed_action is not None:
            should_record_full = self.cfg.record_mode == "full_on_intervention" and self.is_trial_recording

            if should_record_full or self.cfg.display_data:
                observation = self.robot.get_observation()
                processed_observation = self.robot_observation_processor(observation)
                observation_frame = build_dataset_frame(
                    self.dataset.features,
                    processed_observation,
                    prefix=OBS_STR,
                )
                action_frame = build_dataset_frame(
                    self.dataset.features,
                    performed_action,
                    prefix=ACTION,
                )
                frame_data = {
                    **observation_frame,
                    **action_frame,
                    "task": self.cfg.dataset.single_task,
                }

                if should_record_full:
                    self.dataset.add_frame(frame_data)

                if self.cfg.display_data:
                    log_rerun_data(
                        observation=processed_observation,
                        action=performed_action,
                        compress_images=self.display_compressed_images,
                    )

    def _correction_tick(self) -> None:
        try:
            observation = self.robot.get_observation()
        except TimeoutError as e:
            logger.warning(f"Camera frame delay in correction tick ({e}); attempting immediate retry...")
            precise_sleep(0.03)
            observation = self.robot.get_observation()
        processed_observation = self.robot_observation_processor(observation)
        observation_frame = build_dataset_frame(
            self.dataset.features,
            processed_observation,
            prefix=OBS_STR,
        )

        teleop_action = self.teleop.get_action()
        processed_teleop_action = self.teleop_action_processor((teleop_action, observation))
        robot_action = self.robot_action_processor((processed_teleop_action, observation))
        self.robot.send_action(robot_action)

        action_frame = build_dataset_frame(
            self.dataset.features,
            processed_teleop_action,
            prefix=ACTION,
        )
        self.dataset.add_frame(
            {
                **observation_frame,
                **action_frame,
                "task": self.cfg.dataset.single_task,
            }
        )
        self._recorded_correction_frames += 1

        if self.cfg.display_data:
            log_rerun_data(
                observation=processed_observation,
                action=processed_teleop_action,
                compress_images=self.display_compressed_images,
            )

    def _handle_command(self, command: HILCommand) -> TrialOutcome | None:
        if command is HILCommand.STOP:
            if self.phase in HIL_RECORDING_PHASES or self.phase is HILPhase.REVIEW_PAUSED:
                self._discard_pending_correction("HIL stop requested")
            self._pause_without_leader_motion()
            _disable_leader_torque(self.teleop)
            print("[STOP] Automatic observe return skipped for safety")
            return TrialOutcome.STOP

        if command is HILCommand.PAUSE_INTERVENTION:
            if self.phase is HILPhase.AUTONOMOUS:
                self._pause_and_align()
            elif self.phase is HILPhase.INTERVENTION_PAUSED:
                self._resume_autonomous_from_current()
            elif self.phase in HIL_RECORDING_PHASES:
                print("[HINT] 교정을 완료하려면 Enter 또는 C를 누르세요.")
            elif self.phase is HILPhase.REVIEW_PAUSED:
                print("[HINT] →로 저장하거나 ←로 폐기하세요.")
            return None

        if command is HILCommand.START_DIRECT_CORRECTION:
            if self.phase in HIL_RECORDING_PHASES:
                # Toggle: Enter while recording freezes the correction
                self._freeze_correction()
            elif self.phase is HILPhase.INTERVENTION_PAUSED:
                self._start_direct_correction()
            elif self.phase is HILPhase.AUTONOMOUS:
                self._pause_without_leader_motion()
                self._start_direct_correction()
            elif self.phase is HILPhase.REVIEW_PAUSED:
                print("[IGNORED] 먼저 →(저장) 또는 ←(폐기)를 선택하세요.")
            return None

        if command is HILCommand.START_LOCAL_CORRECTION:
            if self.phase is HILPhase.INTERVENTION_PAUSED:
                self._start_local_correction()
            elif self.phase is HILPhase.AUTONOMOUS:
                print("[IGNORED] 먼저 SPACE로 자율주행을 정지하세요.")
            elif self.phase is HILPhase.REVIEW_PAUSED:
                print("[IGNORED] 먼저 →(저장) 또는 ←(폐기)를 선택하세요.")
            return None

        if command is HILCommand.START_STACK_HOVER_CORRECTION:
            if self.phase is HILPhase.INTERVENTION_PAUSED:
                self._start_stack_hover_correction()
            elif self.phase is HILPhase.AUTONOMOUS:
                self._pause_without_leader_motion()
                self._start_stack_hover_correction()
            elif self.phase in HIL_RECORDING_PHASES:
                print("[HINT] 이미 교정 녹화 중입니다. C 또는 Enter로 녹화를 완료하세요.")
            elif self.phase is HILPhase.REVIEW_PAUSED:
                print("[IGNORED] 먼저 →(저장) 또는 ←(폐기)를 선택하세요.")
            return None

        if command is HILCommand.START_OBSERVE_CORRECTION:
            if self.phase is HILPhase.INTERVENTION_PAUSED:
                self._start_observe_correction()
            elif self.phase is HILPhase.AUTONOMOUS:
                print("[IGNORED] 먼저 SPACE로 자율주행을 정지하세요.")
            elif self.phase is HILPhase.REVIEW_PAUSED:
                print("[IGNORED] 먼저 →(저장) 또는 ←(폐기)를 선택하세요.")
            return None

        if command is HILCommand.FREEZE_CORRECTION:
            if self.phase in HIL_RECORDING_PHASES:
                self._freeze_correction()
            else:
                print("[IGNORED] 교정 녹화 중(Enter, L, M 또는 O 시작)에만 C/Enter로 완료할 수 있습니다.")
            return None

        if command is HILCommand.DISCARD_FROZEN_CORRECTION:
            if self.phase is HILPhase.REVIEW_PAUSED:
                self._discard_frozen_correction()
            elif self.phase in HIL_RECORDING_PHASES:
                self._discard_pending_correction("correction discarded by user with left arrow")
                _enable_leader_hold_at_current_pose(self.teleop)
                _hold_follower_at_measured_position(self.robot)
                self.phase = HILPhase.INTERVENTION_PAUSED
                print("[DISCARDED] 현재 교정 프레임이 폐기되었습니다.")
            return None

        if command is HILCommand.SAVE_FROZEN_CORRECTION:
            if self.phase is HILPhase.REVIEW_PAUSED:
                return self._save_frozen_correction()
            elif self.phase in HIL_RECORDING_PHASES:
                print("[HINT] 먼저 Enter 또는 C를 눌러 교정을 완료한 뒤 →를 누르세요.")
            return None

        if command is HILCommand.RESUME_AUTONOMOUS_VIA_OBSERVE:
            self._resume_autonomous_via_observe()
            return None

        if command is HILCommand.NEXT_TRIAL:
            if self.phase in HIL_RECORDING_PHASES or self.phase is HILPhase.REVIEW_PAUSED:
                print("[HINT] 먼저 현재 교정을 저장(→)하거나 폐기(←)하세요.")
                return None
            return self._finish_trial(save=False)

        raise RuntimeError(f"Unhandled HIL command: {command}")

    def run_trial(self, physical_trial: int = 0) -> TrialOutcome:
        self.physical_trial = physical_trial
        self._trial_intervention_count = 0
        self._current_recovery_type = None
        self._current_target_block = None
        self.has_intervened = False
        self.intervention_start_frame = None
        self.pre_intervention_buffer.clear()
        self._expected_episode_index = None
        if self.cfg.record_mode == "full_on_intervention":
            self.is_trial_recording = True
        else:
            self.is_trial_recording = False

        # Cache block coordinates while arm is at observe pose before trial begins
        self._update_cached_block_coords()

        self._resume_autonomous()
        _print_active_controls(self.cfg.record_mode)

        with TerminalKeyReader() as keys:
            while True:
                loop_started_at = time.perf_counter()

                for command in keys.poll():
                    outcome = self._handle_command(command)
                    if outcome is not None:
                        return outcome

                if self.phase in HIL_RECORDING_PHASES:
                    if self._correction_started_at is not None:
                        elapsed = time.perf_counter() - self._correction_started_at
                        if elapsed >= self.cfg.dataset.episode_time_s:
                            print(
                                f"\n⚠️ [TIMEOUT] 교정 제한 시간({self.cfg.dataset.episode_time_s:.0f}s) 초과! "
                                "녹화를 동결합니다."
                            )
                            self._freeze_correction()
                        else:
                            self._correction_tick()
                    else:
                        self._correction_tick()
                elif self.phase is HILPhase.AUTONOMOUS:
                    self._autonomous_tick()

                target_hz = (
                    self.cfg.paused_poll_hz
                    if self.phase in {HILPhase.INTERVENTION_PAUSED, HILPhase.REVIEW_PAUSED}
                    else self.cfg.dataset.fps
                )
                precise_sleep(max(0.0, 1.0 / target_hz - (time.perf_counter() - loop_started_at)))

    def discard_if_pending(self, reason: str) -> None:
        self._discard_pending_correction(reason)


def _build_client_config(cfg: HILRecordConfig) -> RobotClientConfig:
    return RobotClientConfig(
        policy_type=cfg.policy_type,
        pretrained_name_or_path=cfg.pretrained_name_or_path,
        robot=cfg.robot,
        actions_per_chunk=cfg.actions_per_chunk,
        task=cfg.dataset.single_task,
        server_address=cfg.server_address,
        policy_device=cfg.policy_device,
        client_device=cfg.client_device,
        chunk_size_threshold=cfg.chunk_size_threshold,
        fps=cfg.dataset.fps,
        aggregate_fn_name=cfg.aggregate_fn_name,
        debug_visualize_queue_size=False,
        debug_observation_dir=cfg.debug_observation_dir,
        debug_observation_limit=cfg.debug_observation_limit,
        debug_motor_trace_dir=cfg.debug_motor_trace_dir,
        debug_motor_trace_limit=cfg.debug_motor_trace_limit,
    )


def record_hil(
    cfg: HILRecordConfig,
    teleop_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ]
    | None = None,
    robot_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ]
    | None = None,
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation]
    | None = None,
) -> LeRobotDataset | None:
    """Collect remote-policy recovery/correction windows as normal LeRobot episodes."""

    init_logging()
    print(f"[HIL RECORDER BUILD] {HIL_RECORDER_BUILD}")
    cfg.runtime_config = str(runtime_config_path(cfg.runtime_config))
    logging.info("Using runtime config: %s", cfg.runtime_config)
    logging.info(pformat(asdict(cfg)))

    if not sys.stdin.isatty():
        raise RuntimeError("The robot-side HIL recorder must run in an interactive foreground terminal")
    if cfg.dataset.reset_time_s != 0:
        logging.warning("--dataset.reset_time_s is ignored; use 0 for this HIL recorder")

    runtime = load_runtime(cfg.runtime_config)

    if cfg.display_data:
        init_rerun(session_name="smolvla_hil_recording", ip=cfg.display_ip, port=cfg.display_port)
    display_compressed_images = (
        True
        if cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None
        else cfg.display_compressed_images
    )

    client: HILAsyncRobotClient | None = None
    teleop = None
    dataset: LeRobotDataset | None = None
    receiver_thread: threading.Thread | None = None
    session: HILSession | None = None

    try:
        client = HILAsyncRobotClient(
            _build_client_config(cfg),
            rpc_timeout_s=cfg.server_rpc_timeout_s,
        )
        robot = client.robot
        teleop = make_teleoperator_from_config(cfg.teleop)
        teleop.connect()

        if (
            teleop_action_processor is None
            or robot_action_processor is None
            or robot_observation_processor is None
        ):
            default_teleop, default_robot, default_observation = make_default_processors()
            teleop_action_processor = teleop_action_processor or default_teleop
            robot_action_processor = robot_action_processor or default_robot
            robot_observation_processor = robot_observation_processor or default_observation

        dataset_features = _make_dataset_features(
            robot=robot,
            teleop_action_processor=teleop_action_processor,
            robot_observation_processor=robot_observation_processor,
            use_videos=cfg.dataset.video,
        )
        observe_target = _validate_observe_pose(
            runtime,
            cfg.observe_pose_name,
            robot.action_features,
        )

        if not client.start():
            raise RuntimeError(f"Could not initialize the remote policy at {cfg.server_address}")

        dataset = _create_or_resume_dataset(
            cfg=cfg,
            robot=robot,
            dataset_features=dataset_features,
        )

        client.pause_policy_control()
        receiver_thread = threading.Thread(
            target=client.receive_actions,
            name="hil-action-receiver",
            daemon=True,
        )
        receiver_thread.start()
        client.start_barrier.wait(timeout=10)

        session = HILSession(
            cfg=cfg,
            client=client,
            teleop=teleop,
            dataset=dataset,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            display_compressed_images=display_compressed_images,
            observe_target=observe_target,
        )

        physical_trial = 0
        with VideoEncodingManager(dataset):
            try:
                while session.saved_corrections < cfg.dataset.num_episodes:
                    _goto_observe(
                        robot=robot,
                        teleop=teleop,
                        target=observe_target,
                        cfg=cfg,
                        reason=f"prepare HIL physical trial {physical_trial + 1}",
                    )
                    ready = _wait_for_trial_ready(
                        trial_index=physical_trial,
                        saved_corrections=session.saved_corrections,
                        cfg=cfg,
                        dataset=dataset,
                    )
                    if not ready:
                        break

                    outcome = session.run_trial(physical_trial=physical_trial)
                    if outcome is TrialOutcome.NEXT:
                        physical_trial += 1
                        continue
                    if outcome in {TrialOutcome.TARGET_REACHED, TrialOutcome.STOP}:
                        break
            finally:
                # Clear an interrupted correction before VideoEncodingManager
                # finalizes parquet/video metadata.
                session.discard_if_pending("HIL capture loop ended")

    except KeyboardInterrupt:
        print("\n[CTRL+C] Stopping HIL session; unsaved correction will be discarded")
    finally:
        if session is not None:
            with contextlib.suppress(Exception):
                session.discard_if_pending("HIL recorder shutdown")

        if client is not None and client.running:
            with contextlib.suppress(Exception):
                _hold_follower_at_measured_position(client.robot)
            with contextlib.suppress(Exception):
                client.pause_policy_control()
            with contextlib.suppress(Exception):
                _hold_follower_at_measured_position(client.robot)
            client.stop()

        if receiver_thread is not None:
            receiver_thread.join(timeout=5)

        if teleop is not None and teleop.is_connected:
            teleop.disconnect()

        if dataset is not None:
            dataset.finalize()
            print(
                f"[HIL COMPLETE] repo_id={dataset.repo_id} | "
                f"total episodes={dataset.num_episodes}"
            )
            if cfg.dataset.push_to_hub and dataset.num_episodes > 0:
                dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)
                print(f"[HUB] uploaded {dataset.repo_id}")

        log_say("Exiting HIL recorder", cfg.play_sounds)

    return dataset


def _make_hil_cli_entrypoint():
    if not is_dataclass(HILRecordConfig):
        raise TypeError("HILRecordConfig must remain a dataclass")
    record_hil.__annotations__["cfg"] = HILRecordConfig
    return parser.wrap()(record_hil)


record_hil = _make_hil_cli_entrypoint()


def main() -> None:
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass
    register_third_party_plugins()
    record_hil()


if __name__ == "__main__":
    main()
