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

import contextlib
import enum
import logging
import os
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

import grpc

from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.helpers import TimedAction
from lerobot.async_inference.robot_client import RobotClient
from lerobot.common.control_utils import sanity_check_dataset_robot_compatibility
from lerobot.configs import parser
from lerobot.datasets import LeRobotDataset, VideoEncodingManager
from lerobot.grad_project.control.hybrid_goto_both_pose import load_runtime
from lerobot.grad_project.paths import runtime_config_path
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


HIL_RECORDER_BUILD = "2026-09-02-remote-smolvla-full-trial-hil-v2"


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

    record_mode: str = "full_on_intervention"  # "full_on_intervention" | "corrections_only"

    leader_handover_duration_s: float = 1.2
    leader_handover_fps: int = 30
    paused_poll_hz: int = 100

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
    PAUSED = "paused"
    CORRECTING = "correcting"


class HILCommand(str, enum.Enum):
    TOGGLE_POLICY = "toggle_policy"
    START_CORRECTION = "start_correction"
    SAVE_CORRECTION = "save_correction"
    DISCARD_CORRECTION = "discard_correction"
    NEXT_TRIAL = "next_trial"
    STOP = "stop"


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
            commands.append(HILCommand.SAVE_CORRECTION)
            index += 3
        elif remaining.startswith(b"\x1b[D"):
            commands.append(HILCommand.DISCARD_CORRECTION)
            index += 3
        elif remaining.startswith((b"\x1b[A", b"\x1b[B")):
            index += 3
        else:
            byte = remaining[:1]
            index += 1
            if byte == b" ":
                commands.append(HILCommand.TOGGLE_POLICY)
            elif byte in {b"\r", b"\n", b"c", b"C"}:
                commands.append(HILCommand.START_CORRECTION)
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

    if hasattr(robot, "bus"):
        present = robot.bus.sync_read("Present_Position")
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

    observation = robot.get_observation()
    hold_action = {
        key: float(observation[key])
        for key in robot.action_features
        if key.endswith(".pos") and key in observation
    }
    if not hold_action:
        raise RuntimeError("No measured joint positions are available for the HIL hold")
    return robot.send_action(hold_action)


def _enable_leader_hold_at_current_pose(teleop: Any) -> dict[str, float]:
    """Set current position as the goal before enabling torque, avoiding a jump."""

    if not all(hasattr(teleop, name) for name in ("get_action", "send_feedback", "enable_torque")):
        raise RuntimeError("HIL requires an actuated leader with get_action/send_feedback/enable_torque")
    current = {key: float(value) for key, value in teleop.get_action().items() if key.endswith(".pos")}
    teleop.send_feedback(current)
    teleop.enable_torque()
    return current


def _disable_leader_torque(teleop: Any) -> None:
    if not hasattr(teleop, "disable_torque"):
        raise RuntimeError("HIL requires an actuated leader with disable_torque")
    teleop.disable_torque()


def _align_leader_to_follower(
    teleop: Any,
    follower_pose: dict[str, float],
    *,
    duration_s: float,
    fps: int,
) -> None:
    """Safely drive only the leader to the frozen follower pose."""

    current = _enable_leader_hold_at_current_pose(teleop)
    overlap = sorted(set(current) & set(follower_pose))
    if not overlap:
        raise RuntimeError("Leader and follower have no overlapping position keys")

    steps = max(2, int(duration_s * fps))
    for step in range(1, steps + 1):
        alpha = step / steps
        feedback = current.copy()
        for key in overlap:
            feedback[key] = (1 - alpha) * current[key] + alpha * follower_pose[key]
        teleop.send_feedback(feedback)
        precise_sleep(1.0 / fps)


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


def _print_active_controls(record_mode: str = "full_on_intervention") -> None:
    print()
    print(f"[HIL CONTROLS - {record_mode.upper()} - terminal focus required]")
    print("  SPACE       autonomous pause/resume (일시정지 후 리더암 자동 정렬 / 재개)")
    print("  ENTER or C  paused 상태에서 human correction 시작 / 종료 (토크 해제)")
    print("  → (또는 N)   trial 완료 (사람 개입 있었으면 전체 시퀀스 저장, 없었으면 자동 폐기)")
    print("  ←           현재 trial 전체 즉시 폐기 및 재시도")
    print("  Q or ESC    즉시 종료; 미저장 데이터 폐기")
    print()


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

        self.phase = HILPhase.PAUSED
        self.saved_corrections = 0
        self.has_intervened = False
        self.is_trial_recording = False
        self._correction_started_at: float | None = None
        self._expected_episode_index: int | None = None

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
        self.phase = HILPhase.PAUSED
        print("[PAUSED] follower hold / leader aligned+locked. ENTER 또는 C로 correction 시작")

    def _pause_without_leader_motion(self) -> None:
        _hold_follower_at_measured_position(self.robot)
        self.client.pause_policy_control()
        _hold_follower_at_measured_position(self.robot)
        self.phase = HILPhase.PAUSED

    def _resume_autonomous(self) -> None:
        _disable_leader_torque(self.teleop)
        self.client.resume_policy_control()
        self.phase = HILPhase.AUTONOMOUS
        print("[AUTONOMOUS] fresh observation부터 policy 재개")

    def _start_correction(self) -> None:
        if self.cfg.record_mode == "corrections_only":
            if self.dataset.has_pending_frames() or _pending_frame_count(self.dataset) != 0:
                raise RuntimeError("Cannot start correction: an unexpected dataset frame buffer is not empty")
            self._expected_episode_index = self.dataset.num_episodes
            self._correction_started_at = time.perf_counter()

        self.has_intervened = True
        _disable_leader_torque(self.teleop)
        self.phase = HILPhase.CORRECTING
        print(
            "[CORRECTING + RECORDING] recovery/correction 조작 중... "
            "SPACE=자율 주행 재개 / →=trial 완료 / ←=폐기"
        )

    def _discard_pending_correction(self, reason: str) -> None:
        if self._expected_episode_index is None:
            return
        if self.dataset.has_pending_frames() or _pending_frame_count(self.dataset) > 0:
            _discard_current_episode(
                dataset=self.dataset,
                expected_episode_index=self._expected_episode_index,
                reason=reason,
            )
        self._expected_episode_index = None
        self._correction_started_at = None
        self.has_intervened = False
        self.is_trial_recording = False

    def _finish_trial(self, *, save: bool) -> TrialOutcome:
        if self._expected_episode_index is None:
            return TrialOutcome.NEXT

        _enable_leader_hold_at_current_pose(self.teleop)
        _hold_follower_at_measured_position(self.robot)

        if not save:
            self._discard_pending_correction("trial manually discarded by user with left arrow")
            self.phase = HILPhase.PAUSED
            print("[PAUSED] trial discarded. Preparing next trial...")
            return TrialOutcome.NEXT

        if self.cfg.record_mode == "full_on_intervention" and not self.has_intervened:
            self._discard_pending_correction("trial completed without intervention (autonomous success)")
            self.phase = HILPhase.PAUSED
            print("✨ [DISCARDED] 사람이 개입하지 않고 자율 주행으로 성공한 에피소드이므로 저장하지 않고 폐기합니다.")
            return TrialOutcome.NEXT

        frame_count = _pending_frame_count(self.dataset)
        if frame_count <= 0:
            self._discard_pending_correction("trial finished before any frame recorded")
            self.phase = HILPhase.PAUSED
            print("[NOT SAVED] empty trial.")
            return TrialOutcome.NEXT

        expected = self._expected_episode_index
        self.dataset.save_episode()
        if self.dataset.num_episodes != expected + 1:
            raise RuntimeError(
                f"HIL save verification failed: expected {expected + 1} episodes, "
                f"found {self.dataset.num_episodes}"
            )
        self.saved_corrections += 1
        self._expected_episode_index = None
        self._correction_started_at = None
        self.has_intervened = False
        self.is_trial_recording = False
        self.phase = HILPhase.PAUSED
        print(
            f"🎉 [HIL SAVED] 사람이 개입해 교정한 풀 시퀀스 에피소드 {self.dataset.num_episodes - 1} 저장 완료! | "
            f"총 프레임수={frame_count} | 진행도: {self.saved_corrections}/{self.cfg.dataset.num_episodes}"
        )
        if self.saved_corrections >= self.cfg.dataset.num_episodes:
            return TrialOutcome.TARGET_REACHED
        return TrialOutcome.NEXT

    def _autonomous_tick(self) -> None:
        performed_action = None
        if self.client.actions_available():
            performed_action = self.client.control_loop_action()
        if self.client._ready_to_send_observation():
            self.client.control_loop_observation(self.cfg.dataset.single_task)

        if self.cfg.record_mode == "full_on_intervention" and self.is_trial_recording and performed_action is not None:
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
            self.dataset.add_frame(
                {
                    **observation_frame,
                    **action_frame,
                    "task": self.cfg.dataset.single_task,
                }
            )
            if self.cfg.display_data:
                log_rerun_data(
                    observation=processed_observation,
                    action=performed_action,
                    compress_images=self.display_compressed_images,
                )

    def _correction_tick(self) -> None:
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

        if self.cfg.display_data:
            log_rerun_data(
                observation=processed_observation,
                action=processed_teleop_action,
                compress_images=self.display_compressed_images,
            )

    def _handle_command(self, command: HILCommand) -> TrialOutcome | None:
        if command is HILCommand.STOP:
            if self.phase is HILPhase.CORRECTING or self.is_trial_recording:
                self._discard_pending_correction("HIL stop requested")
            self._pause_without_leader_motion()
            print("[STOP] Automatic observe return skipped for safety")
            return TrialOutcome.STOP

        if command is HILCommand.NEXT_TRIAL or command is HILCommand.SAVE_CORRECTION:
            return self._finish_trial(save=True)

        if command is HILCommand.DISCARD_CORRECTION:
            return self._finish_trial(save=False)

        if command is HILCommand.TOGGLE_POLICY:
            if self.phase is HILPhase.AUTONOMOUS:
                self._pause_and_align()
            elif self.phase is HILPhase.PAUSED:
                self._resume_autonomous()
            elif self.phase is HILPhase.CORRECTING:
                _enable_leader_hold_at_current_pose(self.teleop)
                _hold_follower_at_measured_position(self.robot)
                self._resume_autonomous()
            return None

        if command is HILCommand.START_CORRECTION:
            if self.phase is HILPhase.PAUSED:
                self._start_correction()
            elif self.phase is HILPhase.CORRECTING:
                _enable_leader_hold_at_current_pose(self.teleop)
                _hold_follower_at_measured_position(self.robot)
                self.phase = HILPhase.PAUSED
                print("[PAUSED] correction pause. SPACE=자율주행 재개 / →=trial 완료 / ←=폐기")
            elif self.phase is HILPhase.AUTONOMOUS:
                print("[IGNORED] 먼저 SPACE로 policy를 정지하세요")
            return None

        raise RuntimeError(f"Unhandled HIL command: {command}")

    def run_trial(self) -> TrialOutcome:
        self.has_intervened = False
        self._expected_episode_index = self.dataset.num_episodes
        if self.cfg.record_mode == "full_on_intervention":
            self.is_trial_recording = True
        else:
            self.is_trial_recording = False

        self._resume_autonomous()
        _print_active_controls(self.cfg.record_mode)

        with TerminalKeyReader() as keys:
            while True:
                loop_started_at = time.perf_counter()

                for command in keys.poll():
                    outcome = self._handle_command(command)
                    if outcome is not None:
                        return outcome

                if self.phase is HILPhase.CORRECTING:
                    if self.cfg.record_mode == "corrections_only":
                        assert self._correction_started_at is not None
                        elapsed = time.perf_counter() - self._correction_started_at
                        if elapsed >= self.cfg.dataset.episode_time_s:
                            _enable_leader_hold_at_current_pose(self.teleop)
                            _hold_follower_at_measured_position(self.robot)
                            self._discard_pending_correction("HIL correction time limit reached")
                            self.phase = HILPhase.PAUSED
                            print("[TIMEOUT] correction discarded; follower/leader hold in PAUSED")
                        else:
                            self._correction_tick()
                    else:
                        self._correction_tick()
                elif self.phase is HILPhase.AUTONOMOUS:
                    self._autonomous_tick()

                target_hz = self.cfg.paused_poll_hz if self.phase is HILPhase.PAUSED else self.cfg.dataset.fps
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

                    outcome = session.run_trial()
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
    register_third_party_plugins()
    record_hil()


if __name__ == "__main__":
    main()
