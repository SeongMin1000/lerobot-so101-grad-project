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

"""
Records fixed-color-sequence pick-and-place episodes via teleoperation.
tool — no policy inference.  For deploying trained policies, use
``lerobot-rollout`` instead.

Requires: pip install 'lerobot[core_scripts]'  (includes dataset + hardware + viz extras)

Example:

```shell
lerobot-record \\
    --robot.type=so100_follower \\
    --robot.port=/dev/tty.usbmodem58760431541 \\
    --robot.cameras="{laptop: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \\
    --robot.id=black \\
    --teleop.type=so100_leader \\
    --teleop.port=/dev/tty.usbmodem58760431551 \\
    --teleop.id=blue \\
    --dataset.repo_id=<my_username>/<my_dataset_name> \\
    --dataset.num_episodes=2 \\
    --dataset.single_task="Grab the cube" \\
    --dataset.streaming_encoding=true \\
    --dataset.encoder_threads=2 \\
    --display_data=true
```

Example recording with bimanual so100:
```shell
lerobot-record \\
  --robot.type=bi_so_follower \\
  --robot.left_arm_config.port=/dev/tty.usbmodem5A460822851 \\
  --robot.right_arm_config.port=/dev/tty.usbmodem5A460814411 \\
  --robot.id=bimanual_follower \\
  --robot.left_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 1, "width": 640, "height": 480, "fps": 30},
    top: {"type": "opencv", "index_or_path": 3, "width": 640, "height": 480, "fps": 30},
  }' --robot.right_arm_config.cameras='{
    wrist: {"type": "opencv", "index_or_path": 2, "width": 640, "height": 480, "fps": 30},
    front: {"type": "opencv", "index_or_path": 4, "width": 640, "height": 480, "fps": 30},
  }' \\
  --teleop.type=bi_so_leader \\
  --teleop.left_arm_config.port=/dev/tty.usbmodem5A460852721 \\
  --teleop.right_arm_config.port=/dev/tty.usbmodem5A460819811 \\
  --teleop.id=bimanual_leader \\
  --display_data=true \\
  --dataset.repo_id=${HF_USER}/bimanual-so-handover-cube \\
  --dataset.num_episodes=25 \\
  --dataset.single_task="Grab and handover the red cube to the other arm" \\
  --dataset.streaming_encoding=true \\
  --dataset.encoder_threads=2
```

Example recording with custom video encoding parameters:
```shell
lerobot-record \\
    --robot.type=so100_follower \\
    --robot.port=/dev/tty.usbmodem58760431541 \\
    --robot.cameras="{laptop: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \\
    --robot.id=black \\
    --teleop.type=so100_leader \\
    --teleop.port=/dev/tty.usbmodem58760431551 \\
    --teleop.id=blue \\
    --dataset.repo_id=<my_username>/<my_dataset_name> \\
    --dataset.num_episodes=2 \\
    --dataset.single_task="Grab the cube" \\
    --dataset.streaming_encoding=true \\
    --dataset.encoder_threads=2 \\
    --dataset.camera_encoder.vcodec=h264 \\
    --dataset.camera_encoder.preset=fast \\
    --dataset.camera_encoder.extra_options={"tune": "film", "profile:v": "high", "bf": 2} \\
    --display_data=true
```
"""

import json
import logging
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat

import cv2

from lerobot.async_inference.opencv_block_detector import DetectionResult, TopBlockDetector
from lerobot.cameras import CameraConfig  # noqa: F401
from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.reachy2_camera import Reachy2CameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq import ZMQCameraConfig  # noqa: F401
from lerobot.common.control_utils import (
    init_keyboard_listener,
    is_headless,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.configs import parser
from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.datasets import (
    LeRobotDataset,
    VideoEncodingManager,
    aggregate_pipeline_dataset_features,
    create_initial_features,
    safe_stop_image_writer,
)
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
)
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_openarm_follower,
    bi_rebot_b601_follower,
    bi_so_follower,
    earthrover_mini_plus,
    hope_jr,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    openarm_follower,
    reachy2,
    rebot_b601_follower,
    so_follower,
    unitree_g1 as unitree_g1_robot,
)
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_openarm_leader,
    bi_openarm_mini,
    bi_rebot_102_leader,
    bi_so_leader,
    homunculus,
    koch_leader,
    make_teleoperator_from_config,
    omx_leader,
    openarm_leader,
    openarm_mini,
    reachy2_teleoperator,
    rebot_102_leader,
    so_leader,
    unitree_g1,
)
from lerobot.teleoperators.keyboard import KeyboardTeleop
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import (
    init_logging,
    log_say,
)
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
from lerobot.async_inference.hybrid_paths import detector_calib_path, runtime_config_path


@dataclass
class RecordConfig:
    robot: RobotConfig
    dataset: DatasetRecordConfig
    # Teleoperator to control the robot (required)
    teleop: TeleoperatorConfig | None = None
    # Display all cameras on screen
    display_data: bool = False
    # Display data on a remote Rerun server
    display_ip: str | None = None
    # Port of the remote Rerun server
    display_port: int | None = None
    # Whether to  display compressed images in Rerun
    display_compressed_images: bool = False
    # Use vocal synthesis to read events.
    play_sounds: bool = True
    # Resume recording on an existing dataset.
    resume: bool = False

    # Hybrid color-sequence dataset options.
    runtime_config: str = str(runtime_config_path())
    detector_calib: str = str(detector_calib_path())
    top_key: str = "top"

    # One scene is recorded in this fixed order. Keep target slots fixed too.
    color_sequence: str = "red,wood,yellow,green,blue"
    target_slot_sequence: str = "S1,S2,S3,S4,S5"

    observe_pose_name: str = "observe"
    observe_duration_s: float = 2.5
    goto_duration_s: float = 3.0
    auto_goto_before_each_episode: bool = True
    wait_enter_before_episode: bool = True
    wait_scene_reset: bool = True

    # ------------------------------------------------------------------
    # OpenCV 안정화 옵션
    # ------------------------------------------------------------------
    # 한 프레임만 보고 결정하면 그림자/반사 때문에 색상이나 IN/OUT 판정이
    # 흔들릴 수 있으므로 여러 프레임에 대해 다수결을 사용한다.
    detect_samples: int = 7
    detect_min_hits: int = 4
    detect_sample_interval_s: float = 0.08

    # 매 episode 시작 전에 현재 장면이 아래 순서를 정확히 만족하는지 검사한다.
    # 예) red, wood를 이미 넣은 상태라면:
    #     IN  = {red, wood}
    #     OUT = {yellow, green, blue}
    # 이 검사를 통과해야만 다음 yellow episode를 시작한다.
    verify_scene_state: bool = True
    state_samples: int = 9
    state_min_hits: int = 5
    state_sample_interval_s: float = 0.08

    # OpenCV 디버그 이미지를 저장할 폴더.
    debug_save_dir: str = ""

    def __post_init__(self):
        if self.teleop is None:
            raise ValueError(
                "A teleoperator is required for recording. "
                "Use --teleop.type=... to specify one. "
                "For policy-based deployment, use lerobot-rollout instead."
            )


@dataclass
class SceneState:
    """여러 top-camera 프레임을 종합한 색상별 IN/OUT 상태.

    정확히 한 개의 블록만 사용하는 프로젝트이므로 한 색상이 같은 프레임에서
    여러 번 검출되면 그 프레임은 해당 색상에 대해 모호한 것으로 처리한다.
    """

    inside_colors: set[str]
    outside_colors: set[str]
    ambiguous_colors: set[str]
    missing_colors: set[str]
    votes: dict[str, dict[str, int]]
    last_frame: object | None = None
    last_result: DetectionResult | None = None


def _hybrid_load_runtime(runtime_config: str) -> dict:
    with open(runtime_config, encoding="utf-8") as f:
        runtime = json.load(f)
    runtime.setdefault("poses", {})
    runtime.setdefault("pregrasp_points", [])
    runtime.setdefault("gripper", {})
    return runtime


def _hybrid_load_pregrasp_pose(runtime: dict, label: str) -> dict[str, float]:
    for item in runtime.get("pregrasp_points", []):
        if item.get("label") == label:
            pose = item.get("pose")
            if pose is None:
                raise ValueError(f"pregrasp '{label}' has no pose field")
            result = {k: float(v) for k, v in pose.items() if k.endswith(".pos")}
            open_value = runtime.get("gripper", {}).get("open")
            if open_value is not None:
                result["gripper.pos"] = float(open_value)
            return result

    labels = [p.get("label") for p in runtime.get("pregrasp_points", [])]
    raise KeyError(f"pregrasp_label '{label}' not found. Available: {labels}")


def _hybrid_load_named_pose(runtime: dict, pose_name: str) -> dict[str, float]:
    poses = runtime.get("poses", {})
    if pose_name not in poses:
        raise KeyError(f"pose '{pose_name}' not found. Available: {list(poses)}")
    return {k: float(v) for k, v in poses[pose_name].items() if k.endswith(".pos")}


def _hybrid_get_leader(teleop):
    if teleop is None:
        return None
    if isinstance(teleop, list):
        for t in teleop:
            if hasattr(t, "send_feedback"):
                return t
        return None
    return teleop


def _hybrid_goto_pose(
    robot: Robot,
    teleop,
    target_pose: dict[str, float],
    label: str,
    duration_s: float,
    fps: int,
) -> None:
    target_pose = {k: v for k, v in target_pose.items() if k in robot.action_features}
    if not target_pose:
        raise ValueError(f"No valid robot action keys found for '{label}'")

    print()
    print("=" * 70)
    print(f"[AUTO GOTO] {label}")
    print("=" * 70)

    obs = robot.get_observation()
    robot_start = {k: float(obs[k]) for k in target_pose if k in obs}

    leader = _hybrid_get_leader(teleop)
    leader_start = {}
    if leader is not None and hasattr(leader, "get_action"):
        try:
            act = leader.get_action()
            leader_start = {k: float(act[k]) for k in target_pose if k in act}
        except Exception as e:
            print(f"[WARN] leader start read failed: {e}")

    if leader is not None and hasattr(leader, "enable_torque"):
        try:
            leader.enable_torque()
            print("[LOCK] leader torque ON")
        except Exception as e:
            print(f"[WARN] leader enable_torque failed: {e}")

    steps = max(1, int(duration_s * fps))
    dt = 1.0 / fps
    for i in range(1, steps + 1):
        a = i / steps
        robot_cmd = {
            k: robot_start.get(k, target) + (target - robot_start.get(k, target)) * a
            for k, target in target_pose.items()
        }
        robot.send_action(robot_cmd)

        if leader is not None and hasattr(leader, "send_feedback"):
            leader_cmd = {
                k: leader_start.get(k, target) + (target - leader_start.get(k, target)) * a
                for k, target in target_pose.items()
            }
            leader.send_feedback(leader_cmd)
        precise_sleep(dt)

    print(f"[OK] reached {label}")


def _hybrid_nearest_pregrasp(runtime: dict, u: float, v: float) -> tuple[str, dict[str, float], float]:
    points = runtime.get("pregrasp_points", [])
    if not points:
        raise RuntimeError("runtime_config has no pregrasp_points")

    best = None
    best_dist = float("inf")
    for point in points:
        du = float(point["u"]) - float(u)
        dv = float(point["v"]) - float(v)
        dist = math.hypot(du, dv)
        if dist < best_dist:
            best = point
            best_dist = dist

    assert best is not None
    label = str(best.get("label", "unknown"))
    return label, _hybrid_load_pregrasp_pose(runtime, label), best_dist


def _hybrid_detect_scene_state(
    robot: Robot,
    detector: TopBlockDetector,
    top_key: str,
    color_order: list[str],
    samples: int,
    min_hits: int,
    sample_interval_s: float,
    debug_save_dir: str,
    episode_index: int,
) -> SceneState:
    """여러 프레임을 읽어 각 색상이 target 안/밖 어디에 있는지 판정한다.

    판정 규칙
    ---------
    * 한 프레임에서 해당 색상이 정확히 하나 검출된 경우에만 IN 또는 OUT 표를 준다.
    * 같은 색상이 두 개 이상 검출되면 오검출 가능성이 있으므로 ambiguous로 센다.
    * `min_hits` 이상 같은 상태로 검출되어야 최종 IN/OUT으로 확정한다.
    * IN과 OUT 표가 비슷하게 나뉘면 ambiguous로 처리하여 촬영을 중단한다.
    """

    votes: dict[str, dict[str, int]] = {
        color: {"in": 0, "out": 0, "ambiguous": 0, "missing": 0}
        for color in color_order
    }

    last_frame = None
    last_result = None

    for sample_index in range(max(1, samples)):
        observation = robot.get_observation()
        if top_key not in observation:
            raise KeyError(
                f"top_key '{top_key}' not found. observation keys={list(observation)}"
            )

        frame = observation[top_key]
        result = detector.detect(frame)

        # 한 색상당 실제 블록은 하나뿐이다. 따라서 한 프레임에서 같은 색상이
        # 2개 이상 잡히면 그 프레임은 해당 색상에 대해 신뢰하지 않는다.
        for color in color_order:
            matches = [d for d in result.blocks if d.color == color]

            if len(matches) == 0:
                votes[color]["missing"] += 1
            elif len(matches) > 1:
                votes[color]["ambiguous"] += 1
            elif matches[0].in_target:
                votes[color]["in"] += 1
            else:
                votes[color]["out"] += 1

        last_frame = frame
        last_result = result

        if sample_index + 1 < samples:
            time.sleep(max(0.0, sample_interval_s))

    inside_colors: set[str] = set()
    outside_colors: set[str] = set()
    ambiguous_colors: set[str] = set()
    missing_colors: set[str] = set()

    for color in color_order:
        in_hits = votes[color]["in"]
        out_hits = votes[color]["out"]

        # 한쪽 상태가 최소 검출 횟수를 넘고 반대쪽보다 확실히 많아야 확정한다.
        if in_hits >= min_hits and in_hits > out_hits:
            inside_colors.add(color)
        elif out_hits >= min_hits and out_hits > in_hits:
            outside_colors.add(color)
        else:
            # 검출 자체가 거의 안 된 색상과, IN/OUT이 흔들리는 색상을 구분한다.
            if in_hits + out_hits == 0:
                missing_colors.add(color)
            else:
                ambiguous_colors.add(color)

    state = SceneState(
        inside_colors=inside_colors,
        outside_colors=outside_colors,
        ambiguous_colors=ambiguous_colors,
        missing_colors=missing_colors,
        votes=votes,
        last_frame=last_frame,
        last_result=last_result,
    )

    # 상태 확인용 이미지를 매 episode마다 남긴다.
    if debug_save_dir and last_frame is not None and last_result is not None:
        out_dir = Path(debug_save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        overlay = detector.draw(last_frame, last_result)
        out_path = out_dir / f"episode_{episode_index:04d}_scene_state.jpg"
        cv2.imwrite(str(out_path), overlay)
        print(f"[DEBUG] saved {out_path}")

    return state


def _ordered_inside_prefix_length(
    color_order: list[str],
    inside_colors: set[str],
) -> int:
    """고정 색상 순서에서 앞에서부터 연속으로 완료된 개수를 반환한다.

    예:
      order = [red, wood, yellow, green, blue]
      IN    = {red, wood}
      반환값 = 2  -> 다음 색상은 yellow

    red가 없는데 wood가 먼저 IN인 것처럼 순서가 깨졌다면 RuntimeError를 낸다.
    """

    prefix_len = 0
    for color in color_order:
        if color in inside_colors:
            prefix_len += 1
        else:
            break

    # 아직 완료되지 않은 색상 이후에 다른 색상이 미리 target 안에 있으면
    # 촬영 순서가 깨진 것이므로 자동 진행하면 안 된다.
    unexpected_later_inside = set(color_order[prefix_len:]) & inside_colors
    if unexpected_later_inside:
        raise RuntimeError(
            "target 안 색상 순서가 깨졌습니다. "
            f"expected prefix={color_order[:prefix_len]}, "
            f"unexpected inside={sorted(unexpected_later_inside)}"
        )

    return prefix_len


def _verify_scene_state_for_episode(
    state: SceneState,
    color_order: list[str],
    dataset_step_index: int,
) -> tuple[str, set[str], set[str]]:
    """물리 장면과 저장된 dataset 진행 단계가 정확히 일치하는지 확인한다.

    dataset_step_index가 2라면 이미 red, wood 두 episode가 저장된 상태이므로
    실제 화면에서도 red와 wood만 IN, 나머지는 OUT이어야 한다.
    """

    if state.missing_colors or state.ambiguous_colors:
        raise RuntimeError(
            "색상 상태를 안정적으로 판정하지 못했습니다. "
            f"missing={sorted(state.missing_colors)}, "
            f"ambiguous={sorted(state.ambiguous_colors)}"
        )

    expected_inside = set(color_order[:dataset_step_index])
    expected_outside = set(color_order[dataset_step_index:])

    # OpenCV가 실제 화면에서 확인한 연속 완료 개수.
    physical_step_index = _ordered_inside_prefix_length(
        color_order=color_order,
        inside_colors=state.inside_colors,
    )

    if physical_step_index != dataset_step_index:
        raise RuntimeError(
            "dataset 진행 단계와 실제 블록 배치가 다릅니다. "
            f"dataset_step={dataset_step_index}, physical_step={physical_step_index}. "
            "재촬영 중이라면 현재 색 블록을 다시 target 밖으로 꺼내세요."
        )

    if state.inside_colors != expected_inside:
        raise RuntimeError(
            "target 안 색상이 예상과 다릅니다. "
            f"expected IN={sorted(expected_inside)}, "
            f"detected IN={sorted(state.inside_colors)}"
        )

    if state.outside_colors != expected_outside:
        raise RuntimeError(
            "target 밖 색상이 예상과 다릅니다. "
            f"expected OUT={sorted(expected_outside)}, "
            f"detected OUT={sorted(state.outside_colors)}"
        )

    if dataset_step_index >= len(color_order):
        raise RuntimeError("현재 scene의 모든 색상 episode가 이미 완료되었습니다.")

    next_color = color_order[physical_step_index]
    return next_color, expected_inside, expected_outside


def _print_scene_state(state: SceneState) -> None:
    """터미널에서 바로 원인을 찾을 수 있도록 색상별 투표 결과를 출력한다."""

    print("[SCENE STATE]")
    print("  IN       :", sorted(state.inside_colors))
    print("  OUT      :", sorted(state.outside_colors))
    print("  MISSING  :", sorted(state.missing_colors))
    print("  AMBIGUOUS:", sorted(state.ambiguous_colors))
    print("  votes:")
    for color, count in state.votes.items():
        print(
            f"    {color:>6}: "
            f"IN={count['in']} OUT={count['out']} "
            f"ambiguous={count['ambiguous']} missing={count['missing']}"
        )


def _hybrid_detect_color_pregrasp(
    robot: Robot,
    detector: TopBlockDetector,
    runtime: dict,
    top_key: str,
    target_color: str,
    samples: int,
    min_hits: int,
    sample_interval_s: float,
    debug_save_dir: str,
    episode_index: int,
) -> tuple[str, dict[str, float], float, float]:
    hits = []
    last_frame = None
    last_result = None

    for sample_index in range(max(1, samples)):
        obs = robot.get_observation()
        if top_key not in obs:
            raise KeyError(f"top_key '{top_key}' not found. observation keys={list(obs)}")

        frame = obs[top_key]
        result = detector.detect(frame)
        matches = [d for d in result.outside_blocks if d.color == target_color]

        # A scene contains one block per color. Multiple matches mean detector ambiguity.
        if len(matches) == 1:
            hits.append(matches[0])
        elif len(matches) > 1:
            print(
                f"[WARN] sample {sample_index}: multiple OUT detections for {target_color}: "
                + ", ".join(f"({d.cx:.1f},{d.cy:.1f},area={d.area:.0f})" for d in matches)
            )

        last_frame = frame
        last_result = result
        if sample_index + 1 < samples:
            time.sleep(max(0.0, sample_interval_s))

    if len(hits) < min_hits:
        raise RuntimeError(
            f"'{target_color}' detection unstable: hits={len(hits)}/{samples}, required={min_hits}. "
            "Check HSV/workspace or block placement."
        )

    u = float(statistics.median(d.cx for d in hits))
    v = float(statistics.median(d.cy for d in hits))
    label, pose, dist = _hybrid_nearest_pregrasp(runtime, u, v)

    print(
        f"[DETECT] color={target_color} pixel=({u:.1f},{v:.1f}) "
        f"-> pregrasp={label}, calib_dist={dist:.1f}px, hits={len(hits)}/{samples}"
    )

    if debug_save_dir and last_frame is not None and last_result is not None:
        out_dir = Path(debug_save_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        representative = min(hits, key=lambda d: math.hypot(d.cx - u, d.cy - v))
        selected_result = DetectionResult(
            blocks=last_result.blocks,
            outside_blocks=last_result.outside_blocks,
            target_count=last_result.target_count,
            chosen=representative,
        )
        overlay = detector.draw(last_frame, selected_result)
        out_path = out_dir / f"episode_{episode_index:04d}_{target_color}_{label}.jpg"
        cv2.imwrite(str(out_path), overlay)
        print(f"[DEBUG] saved {out_path}")

    return label, pose, u, v


def _hybrid_goto_pregrasp(
    robot: Robot,
    teleop,
    runtime: dict,
    label: str,
    duration_s: float,
    fps: int,
) -> None:
    _hybrid_goto_pose(
        robot=robot,
        teleop=teleop,
        target_pose=_hybrid_load_pregrasp_pose(runtime, label),
        label=f"{label} pregrasp",
        duration_s=duration_s,
        fps=fps,
    )


def _hybrid_goto_named_pose(
    robot: Robot,
    teleop,
    runtime: dict,
    pose_name: str,
    duration_s: float,
    fps: int,
) -> None:
    _hybrid_goto_pose(
        robot=robot,
        teleop=teleop,
        target_pose=_hybrid_load_named_pose(runtime, pose_name),
        label=pose_name,
        duration_s=duration_s,
        fps=fps,
    )


def _hybrid_release_leader_torque(teleop) -> None:
    leader = _hybrid_get_leader(teleop)

    if leader is None:
        return

    if hasattr(leader, "disable_torque"):
        try:
            leader.disable_torque()
            print("[UNLOCK] ENTER pressed -> leader torque OFF -> recording starts")
        except Exception as e:
            print(f"[WARN] leader disable_torque failed: {e}")



""" --------------- record_loop() data flow --------------------------
       [ Robot ]
           V
     [ robot.get_observation() ] ---> raw_obs
           V
     [ robot_observation_processor ] ---> processed_obs
           V
     [ Teleoperator ]
     |
     |  [teleop.get_action] -> raw_action
     |          |
     |          V
     | [teleop_action_processor]
     |          |
     '---> processed_teleop_action
                               V
                  [ robot_action_processor ] --> robot_action_to_send
                               V
                    [ robot.send_action() ] -- (Robot Executes)
                               V
                    ( Save to Dataset )
                               V
                  ( Rerun Log / Loop Wait )
"""


@safe_stop_image_writer
def record_loop(
    robot: Robot,
    events: dict,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],  # runs after teleop
    robot_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],  # runs before robot
    robot_observation_processor: RobotProcessorPipeline[
        RobotObservation, RobotObservation
    ],  # runs after robot
    dataset: LeRobotDataset | None = None,
    teleop: Teleoperator | list[Teleoperator] | None = None,
    control_time_s: int | None = None,
    single_task: str | None = None,
    display_data: bool = False,
    display_compressed_images: bool = False,
):
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps}).")

    teleop_arm = teleop_keyboard = None
    if isinstance(teleop, list):
        teleop_keyboard = next((t for t in teleop if isinstance(t, KeyboardTeleop)), None)
        teleop_arm = next(
            (
                t
                for t in teleop
                if isinstance(
                    t,
                    (
                        so_leader.SO100Leader
                        | so_leader.SO101Leader
                        | koch_leader.KochLeader
                        | omx_leader.OmxLeader
                    ),
                )
            ),
            None,
        )

        if not (teleop_arm and teleop_keyboard and len(teleop) == 2 and robot.name == "lekiwi_client"):
            raise ValueError(
                "For multi-teleop, the list must contain exactly one KeyboardTeleop and one arm teleoperator. Currently only supported for LeKiwi robot."
            )

    control_interval = 1 / fps

    no_action_count = 0
    timestamp = 0
    start_episode_t = time.perf_counter()
    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        # Get robot observation
        obs = robot.get_observation()

        # Applies a pipeline to the raw robot observation, default is IdentityProcessor
        obs_processed = robot_observation_processor(obs)

        if dataset is not None:
            observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)

        # Get action from teleop
        if isinstance(teleop, Teleoperator):
            act = teleop.get_action()
            if robot.name == "unitree_g1":
                teleop.send_feedback(obs)

            # Applies a pipeline to the raw teleop action, default is IdentityProcessor
            act_processed_teleop = teleop_action_processor((act, obs))
            action_values = act_processed_teleop
            robot_action_to_send = robot_action_processor((act_processed_teleop, obs))

        elif isinstance(teleop, list):
            arm_action = teleop_arm.get_action()
            arm_action = {f"arm_{k}": v for k, v in arm_action.items()}
            keyboard_action = teleop_keyboard.get_action()
            base_action = robot._from_keyboard_to_base_action(keyboard_action)
            act = {**arm_action, **base_action} if len(base_action) > 0 else arm_action
            act_processed_teleop = teleop_action_processor((act, obs))
            action_values = act_processed_teleop
            robot_action_to_send = robot_action_processor((act_processed_teleop, obs))
        else:
            no_action_count += 1
            if no_action_count == 1 or no_action_count % 10 == 0:
                logging.warning(
                    "No teleoperator provided, skipping action generation. "
                    "This is likely to happen when resetting the environment without a teleop device. "
                    "The robot won't be at its rest position at the start of the next episode."
                )
            continue

        # Send action to robot
        # Action can eventually be clipped using `max_relative_target`,
        # so action actually sent is saved in the dataset. action = postprocessor.process(action)
        # TODO(steven, pepijn, adil): we should use a pipeline step to clip the action, so the sent action is the action that we input to the robot.
        _sent_action = robot.send_action(robot_action_to_send)

        # Write to dataset
        if dataset is not None:
            action_frame = build_dataset_frame(dataset.features, action_values, prefix=ACTION)
            frame = {**observation_frame, **action_frame, "task": single_task}
            dataset.add_frame(frame)

        if display_data:
            log_rerun_data(
                observation=obs_processed, action=action_values, compress_images=display_compressed_images
            )

        dt_s = time.perf_counter() - start_loop_t

        sleep_time_s: float = control_interval - dt_s
        if sleep_time_s < 0:
            logging.warning(
                f"Record loop is running slower ({1 / dt_s:.1f} Hz) than the target FPS ({fps} Hz). Dataset frames might be dropped and robot control might be unstable. Common causes are: 1) Camera FPS not keeping up 2) Policy inference taking too long 3) CPU starvation"
            )

        precise_sleep(max(sleep_time_s, 0.0))

        timestamp = time.perf_counter() - start_episode_t


@parser.wrap()
def record(
    cfg: RecordConfig,
    teleop_action_processor: RobotProcessorPipeline | None = None,
    robot_action_processor: RobotProcessorPipeline | None = None,
    robot_observation_processor: RobotProcessorPipeline | None = None,
) -> LeRobotDataset:
    init_logging()
    cfg.runtime_config = str(runtime_config_path(cfg.runtime_config))
    cfg.detector_calib = str(detector_calib_path(cfg.detector_calib))
    logging.info("Using runtime config: %s", cfg.runtime_config)
    logging.info("Using detector calibration: %s", cfg.detector_calib)
    logging.info(pformat(asdict(cfg)))
    if cfg.display_data:
        init_rerun(session_name="recording", ip=cfg.display_ip, port=cfg.display_port)
    display_compressed_images = (
        True
        if (cfg.display_data and cfg.display_ip is not None and cfg.display_port is not None)
        else cfg.display_compressed_images
    )

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop) if cfg.teleop is not None else None

    # Fall back to identity pipelines when the caller doesn't supply processors.
    if (
        teleop_action_processor is None
        or robot_action_processor is None
        or robot_observation_processor is None
    ):
        _t, _r, _o = make_default_processors()
        teleop_action_processor = teleop_action_processor or _t
        robot_action_processor = robot_action_processor or _r
        robot_observation_processor = robot_observation_processor or _o

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(
                action=robot.action_features
            ),  # TODO(steven, pepijn): in future this should be come from teleop or policy
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=cfg.dataset.video,
        ),
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
                image_writer_processes=cfg.dataset.num_image_writer_processes if num_cameras > 0 else 0,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * num_cameras
                if num_cameras > 0
                else 0,
            )
            sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, dataset_features)
        else:
            # Reject eval_ prefix — for policy evaluation use lerobot-rollout
            repo_name = cfg.dataset.repo_id.split("/", 1)[-1]
            if repo_name.startswith("eval_"):
                raise ValueError(
                    "Dataset names starting with 'eval_' are reserved for policy evaluation. "
                    "lerobot-record is for data collection only. Use lerobot-rollout for policy deployment."
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
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                camera_encoder=cfg.dataset.camera_encoder,
                encoder_threads=cfg.dataset.encoder_threads,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
            )

        robot.connect()
        if teleop is not None:
            teleop.connect()

        listener, events = init_keyboard_listener()

        if not cfg.dataset.streaming_encoding:
            logging.info(
                "Streaming encoding is disabled. If you have capable hardware, consider enabling it for way faster episode saving. --dataset.streaming_encoding=true --dataset.encoder_threads=2 # --dataset.camera_encoder.vcodec=auto. More info in the documentation: https://huggingface.co/docs/lerobot/streaming_video_encoding"
            )

        runtime = _hybrid_load_runtime(cfg.runtime_config)
        detector = TopBlockDetector.load(cfg.detector_calib, frame_color="rgb")
        color_order = [c.strip() for c in cfg.color_sequence.split(",") if c.strip()]
        slot_order = [s.strip() for s in cfg.target_slot_sequence.split(",") if s.strip()]
        if not color_order:
            raise ValueError("color_sequence is empty")
        if len(color_order) != len(slot_order):
            raise ValueError(
                f"color_sequence and target_slot_sequence lengths differ: {color_order} vs {slot_order}"
            )
        if cfg.dataset.num_episodes % len(color_order) != 0:
            logging.warning(
                "dataset.num_episodes=%d is not a multiple of scene length=%d; final scene will be partial",
                cfg.dataset.num_episodes,
                len(color_order),
            )

        with VideoEncodingManager(dataset):
            recorded_episodes = 0

            # Start every scene and every detection from the same camera-clear pose.
            _hybrid_goto_named_pose(
                robot=robot,
                teleop=teleop,
                runtime=runtime,
                pose_name=cfg.observe_pose_name,
                duration_s=cfg.observe_duration_s,
                fps=cfg.dataset.fps,
            )

            while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
                # dataset에 실제로 저장된 episode 수를 기준으로 현재 scene 단계 계산.
                # 0=red, 1=wood, 2=yellow, 3=green, 4=blue
                sequence_index = dataset.num_episodes % len(color_order)
                scene_index = dataset.num_episodes // len(color_order)

                if sequence_index == 0 and cfg.wait_scene_reset:
                    print("\n" + "#" * 78)
                    print(f"[SCENE {scene_index}] 새 랜덤 배치 준비")
                    print("1) target 안의 이전 블록을 모두 제거")
                    print("2) red/wood/yellow/green/blue 5개를 target 밖에 랜덤 배치")
                    print("3) 블록 겹침 여부와 top camera 시야 확인")
                    print("준비가 끝나면 ENTER, 종료는 Ctrl+C")
                    print("#" * 78)
                    input()

                # --------------------------------------------------------------
                # 1) OpenCV로 현재 target IN/OUT 상태 확인
                # --------------------------------------------------------------
                # 예: red와 wood episode가 이미 저장되어 sequence_index==2라면
                # 반드시 red/wood는 IN, yellow/green/blue는 OUT이어야 한다.
                while True:
                    try:
                        if cfg.verify_scene_state:
                            state = _hybrid_detect_scene_state(
                                robot=robot,
                                detector=detector,
                                top_key=cfg.top_key,
                                color_order=color_order,
                                samples=cfg.state_samples,
                                min_hits=cfg.state_min_hits,
                                sample_interval_s=cfg.state_sample_interval_s,
                                debug_save_dir=cfg.debug_save_dir,
                                episode_index=dataset.num_episodes,
                            )
                            _print_scene_state(state)

                            target_color, expected_inside, expected_outside = (
                                _verify_scene_state_for_episode(
                                    state=state,
                                    color_order=color_order,
                                    dataset_step_index=sequence_index,
                                )
                            )
                        else:
                            # 상태 검사를 끈 경우에만 dataset 순서만 사용한다.
                            target_color = color_order[sequence_index]
                            expected_inside = set(color_order[:sequence_index])
                            expected_outside = set(color_order[sequence_index:])

                        target_slot = slot_order[sequence_index]
                        print("[SCENE CHECK OK]")
                        print("  expected IN :", sorted(expected_inside))
                        print("  expected OUT:", sorted(expected_outside))
                        print("  NEXT        :", target_color)
                        break
                    except RuntimeError as e:
                        print(f"[SCENE CHECK FAIL] {e}")
                        answer = input(
                            "배치/HSV/target polygon 확인 후 ENTER=다시 검사 / q=종료: "
                        ).strip().lower()
                        if answer == "q":
                            raise KeyboardInterrupt

                print("\n" + "=" * 78)
                print(
                    f"[PLAN] scene={scene_index} episode={dataset.num_episodes} "
                    f"step={sequence_index + 1}/{len(color_order)}: {target_color} -> {target_slot}"
                )
                print("=" * 78)

                # --------------------------------------------------------------
                # 2) 이번 차례의 색상은 반드시 target 밖에서 하나만 검출
                # --------------------------------------------------------------
                # 다른 색으로 임의 대체하지 않고, 검출이 불안정하면 멈춘다.
                while True:
                    try:
                        pregrasp_label, _pose, u, v = _hybrid_detect_color_pregrasp(
                            robot=robot,
                            detector=detector,
                            runtime=runtime,
                            top_key=cfg.top_key,
                            target_color=target_color,
                            samples=cfg.detect_samples,
                            min_hits=cfg.detect_min_hits,
                            sample_interval_s=cfg.detect_sample_interval_s,
                            debug_save_dir=cfg.debug_save_dir,
                            episode_index=dataset.num_episodes,
                        )
                        break
                    except RuntimeError as e:
                        print(f"[DETECT FAIL] {e}")
                        answer = input("ENTER=같은 색 다시 탐지 / q=종료: ").strip().lower()
                        if answer == "q":
                            raise KeyboardInterrupt

                if cfg.auto_goto_before_each_episode:
                    _hybrid_goto_pregrasp(
                        robot=robot,
                        teleop=teleop,
                        runtime=runtime,
                        label=pregrasp_label,
                        duration_s=cfg.goto_duration_s,
                        fps=cfg.dataset.fps,
                    )

                if cfg.wait_enter_before_episode:
                    print("\n" + "=" * 78)
                    print(f"[READY] {target_color} @ ({u:.1f}, {v:.1f}) / {pregrasp_label} -> {target_slot}")
                    print(f"이번 episode는 반드시 {target_color}를 집어서 {target_slot}에 놓는다.")
                    print("놓고 gripper를 연 뒤 살짝 들어 올린 상태에서 오른쪽 방향키로 episode 종료")
                    print("ENTER를 누르면 leader torque를 풀고 촬영 시작")
                    print("=" * 78)
                    input()
                    _hybrid_release_leader_torque(teleop)

                log_say(f"Recording {target_color} to {target_slot}, episode {dataset.num_episodes}", cfg.play_sounds)
                record_loop(
                    robot=robot,
                    events=events,
                    fps=cfg.dataset.fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=teleop,
                    dataset=dataset,
                    control_time_s=cfg.dataset.episode_time_s,
                    single_task=cfg.dataset.single_task,
                    display_data=cfg.display_data,
                    display_compressed_images=display_compressed_images,
                )

                if events["rerecord_episode"]:
                    # 재촬영을 누른 시점에는 현재 색 블록이 이미 target 안에 있을 수 있다.
                    # 다음 loop의 scene-state 검사 전에 그 블록을 원래 OUT 위치로 되돌려야 한다.
                    log_say("Re-record episode", cfg.play_sounds)
                    print(
                        f"[RERECORD] {target_color} 블록을 target 밖으로 되돌린 뒤 "
                        "다음 상태 검사를 진행하세요."
                    )
                    events["rerecord_episode"] = False
                    events["exit_early"] = False
                    dataset.clear_episode_buffer()
                    _hybrid_goto_named_pose(
                        robot=robot,
                        teleop=teleop,
                        runtime=runtime,
                        pose_name=cfg.observe_pose_name,
                        duration_s=cfg.observe_duration_s,
                        fps=cfg.dataset.fps,
                    )
                    continue

                dataset.save_episode()
                recorded_episodes += 1

                # This return is script-only and is intentionally not stored in the episode.
                _hybrid_goto_named_pose(
                    robot=robot,
                    teleop=teleop,
                    runtime=runtime,
                    pose_name=cfg.observe_pose_name,
                    duration_s=cfg.observe_duration_s,
                    fps=cfg.dataset.fps,
                )
    finally:
        log_say("Stop recording", cfg.play_sounds, blocking=True)

        if dataset:
            dataset.finalize()

        if robot.is_connected:
            robot.disconnect()
        if teleop and teleop.is_connected:
            teleop.disconnect()

        if not is_headless() and listener:
            listener.stop()

        if cfg.dataset.push_to_hub:
            if dataset and dataset.num_episodes > 0:
                dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)
            else:
                logging.warning("No episodes saved — skipping push to hub")

        log_say("Exiting", cfg.play_sounds)
    return dataset


def main():
    register_third_party_plugins()
    record()


if __name__ == "__main__":
    main()
