#!/usr/bin/env python3
#
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""SmolVLA end-to-end task recorder with a safe ``observe`` return.

This module keeps the normal ``lerobot-record`` data path and keyboard controls,
but replaces the timed reset loop with a deterministic reset:

1. Move follower and leader to ``poses.observe`` in ``project/config/runtime.json``.
2. Wait for the operator to arrange the scene and press Enter.
3. Record the full commanded task by teleoperation: pick, transport, place or
   stack, release the block, and briefly hold the released endpoint.
4. Right arrow: explicitly save only after the task is complete and the
   gripper is open. Reaching the time limit never saves an episode.
5. Immediately after a successful save, return both arms to ``observe`` outside
   the dataset, then show the preparation prompt for the next episode.
6. Left arrow: cancel streaming encoding, clear every buffered frame and
   temporary camera file, return both arms to ``observe``, show the normal
   preparation instructions again, and wait for Enter before recording the
   same dataset episode index.
7. Escape: discard the current partial episode and stop without automatic
   movement.

This is a separate recorder module. It does not replace or modify the existing
``hybrid_record_color_sequence.py`` module. It also does not run OpenCV, choose
a color, or move to a pregrasp pose. Record one task/color per command by
setting ``--dataset.single_task``.

Install this file at:

    ~/lerobot/src/lerobot/grad_project/recording/smolvla_record_observe_return.py

Run it with:

    python -m lerobot.grad_project.recording.smolvla_record_observe_return ...
"""

import logging
import select
import sys
import termios
import time
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pprint import pformat
from typing import Any

from lerobot.grad_project.control.hybrid_goto_both_pose import (
    find_target_pose,
    load_runtime,
    move_both,
)
from lerobot.grad_project.paths import runtime_config_path
from lerobot.common.control_utils import (
    init_keyboard_listener,
    is_headless,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.configs import parser
from lerobot.datasets import (
    LeRobotDataset,
    VideoEncodingManager,
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.robots import make_robot_from_config
from lerobot.scripts.lerobot_record import RecordConfig as LeRobotRecordConfig
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data


RECORDER_BUILD = "2026-08-04-unified-record-start-log-v11"


@dataclass
class RecordConfig(LeRobotRecordConfig):
    """Normal LeRobot record options plus named-pose reset options."""

    runtime_config: str = "project/config/runtime.json"
    observe_pose_name: str = "observe"
    observe_duration_s: float = 3.0
    observe_fps: int = 30
    observe_settle_s: float = 0.2
    auto_goto_observe: bool = True
    wait_enter_before_episode: bool = True
    teleop_before_episode: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.observe_duration_s <= 0:
            raise ValueError("observe_duration_s must be greater than 0")
        if self.observe_fps <= 0:
            raise ValueError("observe_fps must be greater than 0")
        if self.observe_settle_s < 0:
            raise ValueError("observe_settle_s must be non-negative")
        if self.teleop_before_episode and not self.wait_enter_before_episode:
            raise ValueError(
                "--teleop_before_episode=true requires "
                "--wait_enter_before_episode=true"
            )
        if not self.dataset.single_task.strip():
            raise ValueError(
                "--dataset.single_task must contain the exact task instruction "
                "for every episode in this recording run."
            )


def _validate_observe_pose(
    runtime: dict[str, Any],
    pose_name: str,
    robot_action_features: dict[str, type],
) -> dict[str, float]:
    """Load the named pose and reject an incomplete SO-101 pose."""

    target = find_target_pose(runtime, pose_name, "")
    action_keys = {key for key in robot_action_features if key.endswith(".pos")}
    target_keys = {key for key in target if key.endswith(".pos")}

    missing = sorted(action_keys - target_keys)
    if missing:
        raise ValueError(
            f"poses.{pose_name} is missing robot joints: {missing}. "
            "Re-save the pose before recording."
        )

    return target


def _goto_observe(
    *,
    robot: Any,
    teleop: Any,
    target: dict[str, float],
    cfg: RecordConfig,
    reason: str,
) -> None:
    """Move both arms without adding any frame to the dataset."""

    if not cfg.auto_goto_observe:
        return

    print()
    print("=" * 78)
    print(f"[AUTO RETURN] {reason}")
    print(
        f"follower + leader -> {cfg.observe_pose_name} "
        f"({cfg.observe_duration_s:.1f}s)"
    )
    print(
        "자동복귀 중에는 로봇팔과 리더팔을 잡거나 "
        "작업공간에 손을 넣지 마세요."
    )
    print("=" * 78)

    move_both(
        robot=robot,
        teleop=teleop,
        target=target,
        duration_s=cfg.observe_duration_s,
        fps=cfg.observe_fps,
    )
    if cfg.observe_settle_s:
        time.sleep(cfg.observe_settle_s)

    print(f"[OK] follower + leader reached {cfg.observe_pose_name}")


def _release_leader_torque(teleop: Any) -> None:
    """Unlock the leader after the operator confirms that recording may start."""

    if hasattr(teleop, "disable_torque"):
        teleop.disable_torque()
    elif hasattr(teleop, "bus"):
        teleop.bus.disable_torque()
    else:
        raise RuntimeError(
            "The configured teleoperator cannot release leader torque. "
            "This recorder requires a leader arm with disable_torque support."
        )
    print("[UNLOCK] leader torque OFF -> teleoperation recording starts")


def _wait_for_episode_ready(
    *,
    dataset: LeRobotDataset,
    session_episode_index: int,
    cfg: RecordConfig,
) -> bool:
    """Wait for scene reset. Return False when the operator requests quit."""

    if not cfg.wait_enter_before_episode:
        return True

    print()
    print("#" * 78)
    print(
        f"[READY] session {session_episode_index + 1}/{cfg.dataset.num_episodes} "
        f"| dataset episode {dataset.num_episodes}"
    )
    print(f"[TASK] {cfg.dataset.single_task}")
    print()
    if cfg.teleop_before_episode:
        print(
            "1) 준비용 teleoperation으로 follower를 원하는 시작 자세로 이동"
        )
        print(
            "2) 목표 구역과 5개 블록을 이번 episode 시작 상태로 재배치"
        )
        print("3) 카메라 시야, 블록 겹침, 작업공간 충돌 위험 확인")
        print("4) 원하는 실패 시작 자세에서 ENTER를 눌러 녹화 시작")
        print("   (ENTER 전의 관측과 동작은 dataset에 저장되지 않음)")
    else:
        print("1) follower와 leader가 observe에 있는지 확인")
        print("2) 목표 구역과 5개 블록을 이번 episode 시작 상태로 재배치")
        print("3) 카메라 시야, 블록 겹침, 작업공간 충돌 위험 확인")
    print()
    print("이번 episode의 끝 상태:")
    print("  observe -> 블록 잡기 -> 들어 올리기 -> 명령에 맞게 place/stack")
    print("  블록을 완전히 내려놓고 gripper를 연 채 0.5~1초 정지")
    print()
    print("촬영 중 키:")
    print("  →  place/stack 완료 및 gripper open 후 episode 종료·저장")
    print(
        "  ←  현재 episode 완전 폐기 -> observe 복귀 -> "
        "ENTER 후 같은 번호로 재촬영"
    )
    print("  Esc  현재 미완성 episode를 폐기하고 전체 종료")
    print("  제한시간 도달  자동 폐기 후 같은 번호로 즉시 재촬영 (저장 안 함)")
    print("#" * 78)

    if cfg.teleop_before_episode:
        return True

    while True:
        try:
            answer = input("준비 완료: ENTER / 전체 종료: q + ENTER > ").strip().lower()
        except EOFError as exc:
            raise RuntimeError(
                "Enter confirmation needs an interactive terminal. "
                "Run this module directly from the Ubuntu terminal."
            ) from exc

        if answer == "":
            return True
        if answer in {"q", "quit", "exit"}:
            return False
        print("ENTER만 누르거나 q를 입력하세요.")


def _clear_transient_key_events(events: dict[str, bool]) -> None:
    """Ignore arrow presses made before the episode actually starts."""

    events["exit_early"] = False
    events["rerecord_episode"] = False


def _flush_terminal_input() -> None:
    """Discard arrow-key escape bytes left in the terminal input queue.

    ``pynput`` delivers the arrow-key event to LeRobot, while the terminal may
    also enqueue bytes such as ``ESC [ D`` for a later ``input()`` call. Those
    bytes must not leak into the next READY prompt.
    """

    if not sys.stdin.isatty():
        return
    try:
        termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    except (OSError, termios.error):
        logging.debug("Could not flush terminal input", exc_info=True)


def _teleop_until_enter_without_recording(
    *,
    robot: Any,
    teleop: Any,
    events: dict[str, bool],
    fps: int,
    teleop_action_processor: RobotProcessorPipeline,
    robot_action_processor: RobotProcessorPipeline,
) -> bool:
    """Teleoperate to the desired start pose without adding dataset frames."""

    if not sys.stdin.isatty():
        raise RuntimeError(
            "--teleop_before_episode=true requires an interactive terminal."
        )

    _flush_terminal_input()
    _clear_transient_key_events(events)
    _release_leader_torque(teleop)

    print()
    print(
        "[PRE-RECORD TELEOP] follower를 원하는 episode 시작 자세로 이동하세요."
    )
    print("[NOT RECORDING] 지금 움직임은 dataset에 저장되지 않습니다.")
    print("준비 완료: ENTER / 전체 종료: q + ENTER > ", end="", flush=True)

    control_interval_s = 1.0 / fps
    while True:
        if events["stop_recording"]:
            print("\n[ESC] 준비용 teleoperation을 종료합니다.")
            return False

        readable, _, _ = select.select([sys.stdin], [], [], 0.0)
        if readable:
            line = sys.stdin.readline()
            if line == "":
                raise RuntimeError(
                    "Interactive terminal input closed during pre-record teleoperation."
                )

            answer = line.strip().lower()
            if answer == "":
                # The outer recording loop announces the real dataset start.
                # Keeping that announcement in one place makes observe-pose
                # and pre-record teleoperation modes use exactly the same log.
                return True
            if answer in {"q", "quit", "exit"}:
                return False

            print("ENTER만 누르거나 q를 입력하세요.")
            print(
                "준비 완료: ENTER / 전체 종료: q + ENTER > ",
                end="",
                flush=True,
            )

        start_loop_t = time.perf_counter()
        observation = robot.get_observation()
        teleop_action = teleop.get_action()
        processed_teleop_action = teleop_action_processor(
            (teleop_action, observation)
        )
        robot_action = robot_action_processor(
            (processed_teleop_action, observation)
        )
        robot.send_action(robot_action)

        elapsed_loop_s = time.perf_counter() - start_loop_t
        precise_sleep(max(control_interval_s - elapsed_loop_s, 0.0))


def _announce_recording_start(
    *,
    dataset: LeRobotDataset,
    cfg: RecordConfig,
) -> None:
    """Emit one identical start log for every recording-start mode."""

    log_say(
        f"[RECORD START] Recording episode {dataset.num_episodes}",
        cfg.play_sounds,
    )


class EpisodeOutcome(str, Enum):
    """The only four ways an active recording attempt may finish."""

    SAVE = "save"
    RERECORD = "rerecord"
    STOP = "stop"
    TIMEOUT = "timeout"


def _pending_frame_count(dataset: LeRobotDataset) -> int:
    """Return the number of frames in the unsaved episode buffer."""

    writer = getattr(dataset, "writer", None)
    episode_buffer = getattr(writer, "episode_buffer", None)
    if not isinstance(episode_buffer, dict):
        return 0
    return int(episode_buffer.get("size", 0))


def _discard_current_episode(
    *,
    dataset: LeRobotDataset,
    expected_episode_index: int,
    reason: str,
) -> int:
    """Discard one attempt and verify that no episode was committed.

    ``LeRobotDataset.clear_episode_buffer()`` cancels an active streaming
    encoder. In non-streaming mode, this LeRobot revision only removes
    directories for image features, not all video-frame directories. Calling
    ``cleanup_interrupted_episode()`` as well prevents leftover PNG frames from
    a longer failed attempt from leaking into a shorter retry.
    """

    before_count = dataset.num_episodes
    if before_count != expected_episode_index:
        raise RuntimeError(
            "Refusing to discard an ambiguous episode: "
            f"expected dataset episode count {expected_episode_index}, "
            f"but found {before_count}."
        )

    discarded_frames = _pending_frame_count(dataset)
    dataset.clear_episode_buffer(delete_images=True)

    writer = getattr(dataset, "writer", None)
    cleanup = getattr(writer, "cleanup_interrupted_episode", None)
    if callable(cleanup):
        cleanup(expected_episode_index)

    after_count = dataset.num_episodes
    if after_count != before_count:
        raise RuntimeError(
            "Discard safety check failed: dataset episode count changed from "
            f"{before_count} to {after_count}."
        )
    if dataset.has_pending_frames() or _pending_frame_count(dataset) != 0:
        raise RuntimeError(
            "Discard safety check failed: frames remain in the episode buffer."
        )

    print(
        f"[DISCARDED] {reason} | removed {discarded_frames} buffered frames "
        f"| dataset episodes unchanged: {after_count}"
    )
    return discarded_frames


def _consume_episode_outcome(
    events: dict[str, bool],
) -> EpisodeOutcome | None:
    """Consume the shared keyboard flags by safety priority."""

    # Esc always wins. A nearly simultaneous left/right press must discard,
    # never save, so RERECORD has priority over SAVE.
    if events["stop_recording"]:
        events["exit_early"] = False
        events["rerecord_episode"] = False
        return EpisodeOutcome.STOP
    if events["rerecord_episode"]:
        events["rerecord_episode"] = False
        events["exit_early"] = False
        return EpisodeOutcome.RERECORD
    if events["exit_early"]:
        events["exit_early"] = False
        return EpisodeOutcome.SAVE
    return None


def _record_episode_until_command(
    *,
    robot: Any,
    teleop: Any,
    events: dict[str, bool],
    fps: int,
    teleop_action_processor: RobotProcessorPipeline,
    robot_action_processor: RobotProcessorPipeline,
    robot_observation_processor: RobotProcessorPipeline,
    dataset: LeRobotDataset,
    control_time_s: float,
    single_task: str,
    display_data: bool,
    display_compressed_images: bool,
) -> EpisodeOutcome:
    """Record frames and return an explicit decision instead of mutable flags.

    Only a captured right-arrow event returns ``SAVE``. A timeout returns
    ``TIMEOUT`` and is discarded by the caller. This prevents a failed key
    listener or a missed left-arrow press from silently saving bad data.
    """

    if dataset.fps != fps:
        raise ValueError(
            "The dataset fps should equal the requested fps "
            f"({dataset.fps} != {fps})."
        )

    control_interval_s = 1.0 / fps
    start_episode_t = time.perf_counter()

    while True:
        outcome = _consume_episode_outcome(events)
        if outcome is not None:
            return outcome

        elapsed_episode_s = time.perf_counter() - start_episode_t
        if elapsed_episode_s >= control_time_s:
            return EpisodeOutcome.TIMEOUT

        start_loop_t = time.perf_counter()

        observation = robot.get_observation()
        processed_observation = robot_observation_processor(observation)
        observation_frame = build_dataset_frame(
            dataset.features,
            processed_observation,
            prefix=OBS_STR,
        )

        teleop_action = teleop.get_action()
        processed_teleop_action = teleop_action_processor(
            (teleop_action, observation)
        )
        robot_action = robot_action_processor(
            (processed_teleop_action, observation)
        )
        robot.send_action(robot_action)

        action_frame = build_dataset_frame(
            dataset.features,
            processed_teleop_action,
            prefix=ACTION,
        )
        dataset.add_frame(
            {
                **observation_frame,
                **action_frame,
                "task": single_task,
            }
        )

        if display_data:
            log_rerun_data(
                observation=processed_observation,
                action=processed_teleop_action,
                compress_images=display_compressed_images,
            )

        elapsed_loop_s = time.perf_counter() - start_loop_t
        sleep_time_s = control_interval_s - elapsed_loop_s
        if sleep_time_s < 0:
            logging.warning(
                "Record loop is running slower (%.1f Hz) than the target FPS "
                "(%d Hz). Dataset frames may be dropped and control may be "
                "unstable.",
                1.0 / elapsed_loop_s,
                fps,
            )
        precise_sleep(max(sleep_time_s, 0.0))


def _make_dataset_features(
    *,
    robot: Any,
    teleop_action_processor: RobotProcessorPipeline,
    robot_observation_processor: RobotProcessorPipeline,
    use_videos: bool,
) -> dict:
    return combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=use_videos,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(
                observation=robot.observation_features
            ),
            use_videos=use_videos,
        ),
    )


# LeRobot's parser.wrap() inspects the first parameter annotation at call time
# and passes it directly to draccus.  Keep this function undecorated here; the
# concrete RecordConfig class is bound to the annotation and parser.wrap() is
# applied explicitly near the bottom of the module.  That order also works if
# annotations are ever postponed and would otherwise become "RecordConfig".
def record(
    cfg: RecordConfig,
    teleop_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ]
    | None = None,
    robot_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ]
    | None = None,
    robot_observation_processor: RobotProcessorPipeline[
        RobotObservation, RobotObservation
    ]
    | None = None,
) -> LeRobotDataset:
    """Record episodes and perform every observe return outside the dataset."""

    init_logging()
    print(f"[RECORDER BUILD] {RECORDER_BUILD}")
    cfg.runtime_config = str(runtime_config_path(cfg.runtime_config))
    logging.info("Using runtime config: %s", cfg.runtime_config)
    logging.info(pformat(asdict(cfg)))

    if cfg.wait_enter_before_episode and not sys.stdin.isatty():
        raise RuntimeError(
            "--wait_enter_before_episode=true requires an interactive terminal."
        )
    if cfg.dataset.reset_time_s != 0:
        logging.warning(
            "--dataset.reset_time_s=%s is ignored by this recorder. "
            "Scene reset is performed at the Enter prompt; use "
            "--dataset.reset_time_s=0 for a clear configuration.",
            cfg.dataset.reset_time_s,
        )

    runtime = load_runtime(cfg.runtime_config)
    # Validate the configured name before opening any serial port.
    find_target_pose(runtime, cfg.observe_pose_name, "")

    if cfg.display_data:
        init_rerun(
            session_name="recording",
            ip=cfg.display_ip,
            port=cfg.display_port,
        )
    display_compressed_images = (
        True
        if (
            cfg.display_data
            and cfg.display_ip is not None
            and cfg.display_port is not None
        )
        else cfg.display_compressed_images
    )

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop)

    if (
        teleop_action_processor is None
        or robot_action_processor is None
        or robot_observation_processor is None
    ):
        default_teleop, default_robot, default_observation = (
            make_default_processors()
        )
        teleop_action_processor = teleop_action_processor or default_teleop
        robot_action_processor = robot_action_processor or default_robot
        robot_observation_processor = (
            robot_observation_processor or default_observation
        )

    dataset_features = _make_dataset_features(
        robot=robot,
        teleop_action_processor=teleop_action_processor,
        robot_observation_processor=robot_observation_processor,
        use_videos=cfg.dataset.video,
    )

    dataset = None
    listener = None

    try:
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
                image_writer_processes=(
                    cfg.dataset.num_image_writer_processes
                    if num_cameras > 0
                    else 0
                ),
                image_writer_threads=(
                    cfg.dataset.num_image_writer_threads_per_camera * num_cameras
                    if num_cameras > 0
                    else 0
                ),
            )
            sanity_check_dataset_robot_compatibility(
                dataset,
                robot,
                cfg.dataset.fps,
                dataset_features,
            )
        else:
            repo_name = cfg.dataset.repo_id.split("/", 1)[-1]
            if repo_name.startswith("eval_"):
                raise ValueError(
                    "Dataset names starting with 'eval_' are reserved for "
                    "policy evaluation. Use a normal recording dataset name."
                )

            cfg.dataset.stamp_repo_id()
            dataset = LeRobotDataset.create(
                cfg.dataset.repo_id,
                cfg.dataset.fps,
                root=cfg.dataset.root,
                robot_type=robot.name,
                features=dataset_features,
                use_videos=cfg.dataset.video,
                image_writer_processes=cfg.dataset.num_image_writer_processes,
                image_writer_threads=(
                    cfg.dataset.num_image_writer_threads_per_camera
                    * len(robot.cameras)
                ),
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                camera_encoder=cfg.dataset.camera_encoder,
                encoder_threads=cfg.dataset.encoder_threads,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
            )

        observe_target = _validate_observe_pose(
            runtime,
            cfg.observe_pose_name,
            robot.action_features,
        )

        robot.connect()
        teleop.connect()

        _goto_observe(
            robot=robot,
            teleop=teleop,
            target=observe_target,
            cfg=cfg,
            reason="initial recording pose",
        )

        listener, events = init_keyboard_listener()
        if listener is None:
            raise RuntimeError(
                "Keyboard listener is unavailable, so save/discard keys cannot "
                "be trusted. No episode was saved. Run from the Ubuntu desktop "
                "session with pynput/X11 access; do not record through a "
                "headless terminal."
            )

        if not cfg.dataset.streaming_encoding:
            logging.info(
                "Streaming encoding is disabled. With three cameras, consider "
                "--dataset.streaming_encoding=true and "
                "--dataset.encoder_threads=2 if saving becomes a bottleneck."
            )

        with VideoEncodingManager(dataset):
            recorded_episodes = 0
            retry_immediately = False

            while (
                recorded_episodes < cfg.dataset.num_episodes
                and not events["stop_recording"]
            ):
                if hasattr(listener, "is_alive") and not listener.is_alive():
                    raise RuntimeError(
                        "Keyboard listener stopped unexpectedly. The pending "
                        "episode will be discarded instead of saved."
                    )

                if retry_immediately:
                    print(
                        "[RERECORD] Starting the same dataset episode "
                        f"{dataset.num_episodes} immediately."
                    )
                else:
                    ready = _wait_for_episode_ready(
                        dataset=dataset,
                        session_episode_index=recorded_episodes,
                        cfg=cfg,
                    )
                    if ready and cfg.teleop_before_episode:
                        ready = _teleop_until_enter_without_recording(
                            robot=robot,
                            teleop=teleop,
                            events=events,
                            fps=cfg.dataset.fps,
                            teleop_action_processor=teleop_action_processor,
                            robot_action_processor=robot_action_processor,
                        )
                    if not ready:
                        print("[STOP] No new episode was started.")
                        break

                retry_immediately = False
                _flush_terminal_input()
                _clear_transient_key_events(events)
                if not cfg.teleop_before_episode:
                    _release_leader_torque(teleop)

                _announce_recording_start(
                    dataset=dataset,
                    cfg=cfg,
                )
                expected_episode_index = dataset.num_episodes
                try:
                    outcome = _record_episode_until_command(
                        robot=robot,
                        teleop=teleop,
                        events=events,
                        fps=cfg.dataset.fps,
                        teleop_action_processor=teleop_action_processor,
                        robot_action_processor=robot_action_processor,
                        robot_observation_processor=robot_observation_processor,
                        dataset=dataset,
                        control_time_s=cfg.dataset.episode_time_s,
                        single_task=cfg.dataset.single_task,
                        display_data=cfg.display_data,
                        display_compressed_images=display_compressed_images,
                    )
                except BaseException:
                    if (
                        dataset.num_episodes == expected_episode_index
                        and (
                            dataset.has_pending_frames()
                            or _pending_frame_count(dataset) > 0
                        )
                    ):
                        try:
                            _discard_current_episode(
                                dataset=dataset,
                                expected_episode_index=expected_episode_index,
                                reason="recording error",
                            )
                        except Exception:
                            logging.exception(
                                "Failed to clean the interrupted episode buffer"
                            )
                    raise

                print(
                    f"[DECISION] {outcome.value.upper()} "
                    f"| buffered frames: {_pending_frame_count(dataset)} "
                    f"| dataset episodes before decision: "
                    f"{expected_episode_index}"
                )

                if outcome is EpisodeOutcome.STOP:
                    _discard_current_episode(
                        dataset=dataset,
                        expected_episode_index=expected_episode_index,
                        reason="Esc pressed",
                    )
                    print(
                        "[ESC] Current partial episode discarded. "
                        "Automatic movement is skipped for safety."
                    )
                    break

                if outcome is EpisodeOutcome.RERECORD:
                    log_say("Re-record episode", cfg.play_sounds)
                    _discard_current_episode(
                        dataset=dataset,
                        expected_episode_index=expected_episode_index,
                        reason="left arrow pressed",
                    )
                    _flush_terminal_input()

                    # A discarded attempt never advances either counter. Return
                    # both arms to the configured observe pose outside the
                    # dataset, then let the next outer-loop iteration show the
                    # full READY instructions and wait for Enter.
                    _clear_transient_key_events(events)
                    _goto_observe(
                        robot=robot,
                        teleop=teleop,
                        target=observe_target,
                        cfg=cfg,
                        reason=(
                            "left arrow discarded the current attempt; "
                            "prepare the same episode again"
                        ),
                    )
                    retry_immediately = False
                    print(
                        "[RERECORD READY] Returned to observe. "
                        f"Dataset episode {expected_episode_index} is still "
                        "unsaved; arrange the scene and press Enter at the "
                        "READY prompt."
                    )
                    continue

                if outcome is EpisodeOutcome.TIMEOUT:
                    log_say("Re-record episode", cfg.play_sounds)
                    _discard_current_episode(
                        dataset=dataset,
                        expected_episode_index=expected_episode_index,
                        reason="episode time limit reached without right arrow",
                    )
                    _flush_terminal_input()
                    retry_immediately = True
                    print(
                        "[RERECORD] Time limit reached. No episode was saved; "
                        f"dataset episode {expected_episode_index} will restart "
                        "immediately."
                    )
                    continue

                if outcome is not EpisodeOutcome.SAVE:
                    raise RuntimeError(f"Unhandled episode outcome: {outcome}")
                if _pending_frame_count(dataset) <= 0:
                    _discard_current_episode(
                        dataset=dataset,
                        expected_episode_index=expected_episode_index,
                        reason="right arrow pressed before any frame was recorded",
                    )
                    print(
                        "[RETRY] Empty episode was not saved. "
                        "The same episode will restart immediately."
                    )
                    _flush_terminal_input()
                    retry_immediately = True
                    continue

                dataset.save_episode()
                if dataset.num_episodes != expected_episode_index + 1:
                    raise RuntimeError(
                        "Save verification failed: expected dataset episode "
                        f"count {expected_episode_index + 1}, "
                        f"found {dataset.num_episodes}."
                    )
                recorded_episodes += 1
                print(
                    f"[SAVED] dataset episode {dataset.num_episodes - 1} "
                    f"| session {recorded_episodes}/{cfg.dataset.num_episodes}"
                )

                _flush_terminal_input()
                _clear_transient_key_events(events)

                # No dataset object is passed to move_both(), so these frames and
                # actions are deliberately excluded from the saved episode.
                _goto_observe(
                    robot=robot,
                    teleop=teleop,
                    target=observe_target,
                    cfg=cfg,
                    reason=(
                        "end-to-end episode saved with the block released; "
                        "prepare the next start pose"
                    ),
                )
    finally:
        log_say("Stop recording", cfg.play_sounds, blocking=True)

        if dataset:
            dataset.finalize()

        if robot.is_connected:
            robot.disconnect()
        if teleop.is_connected:
            teleop.disconnect()

        if not is_headless() and listener:
            listener.stop()

        if cfg.dataset.push_to_hub:
            if dataset and dataset.num_episodes > 0:
                dataset.push_to_hub(
                    tags=cfg.dataset.tags,
                    private=cfg.dataset.private,
                )
            else:
                logging.warning("No episodes saved — skipping push to hub")

        log_say("Exiting", cfg.play_sounds)

    return dataset


def _make_record_cli_entrypoint():
    """Bind the real dataclass before LeRobot builds the CLI wrapper."""

    if not is_dataclass(RecordConfig):
        raise TypeError(
            "RecordConfig must remain decorated with @dataclass for draccus."
        )

    # Do this *before* parser.wrap() captures the function.  LeRobot's wrapper
    # later uses inspect.getfullargspec(fn).annotations["cfg"] as
    # draccus.parse(config_class=...), so it must be the class object rather
    # than a string or typing object.
    record.__annotations__["cfg"] = RecordConfig
    return parser.wrap()(record)


record = _make_record_cli_entrypoint()


def main() -> None:
    register_third_party_plugins()
    record()


if __name__ == "__main__":
    main()
