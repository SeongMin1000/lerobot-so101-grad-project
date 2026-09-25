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

"""Diagnose regressions previously seen in the graduation-project runtime.

The command is deliberately read-only with respect to robot hardware: it never
opens a motor or camera device. It combines executable code-path checks with
the most recent camera and motor flight-recorder artifacts.
"""

import argparse
import inspect
import json
import logging
import os
import pickle  # nosec: only used for an in-process round-trip self-check
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image, ImageDraw

from lerobot.async_inference.helpers import (
    TimedObservation,
    prepare_image,
    raw_observation_to_observation,
    robot_observation_image_to_chw,
)
from lerobot.async_inference.policy_server import PolicyServer
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad
from lerobot.robots.so_follower import SOFollower
from lerobot.robots.utils import ensure_synchronized_goal_position
from lerobot.utils.constants import OBS_IMAGES, OBS_STATE


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CLIENT_CAPTURES = REPO_ROOT / "var/debug/client_camera_inputs"
DEFAULT_SERVER_CAPTURES = REPO_ROOT / "var/debug/server_camera_inputs"
DEFAULT_MOTOR_TRACES = REPO_ROOT / "var/debug/motor_traces"
DEFAULT_REPORT_ROOT = REPO_ROOT / "var/diagnostics"

CAMERA_ALIASES = {
    "top": "top",
    "camera1": "top",
    "wrist": "wrist",
    "camera2": "wrist",
    "belly": "belly",
    "camera3": "belly",
}


class Status(StrEnum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    SKIP = "SKIP"


@dataclass
class CheckResult:
    check_id: str
    group: str
    status: Status
    summary: str
    details: dict[str, Any] = field(default_factory=dict)


def _result(
    check_id: str,
    group: str,
    status: Status,
    summary: str,
    **details: Any,
) -> CheckResult:
    return CheckResult(check_id, group, status, summary, details)


@contextmanager
def _silence_library_logs():
    previous_level = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        yield
    finally:
        logging.disable(previous_level)


def check_camera_code_path() -> CheckResult:
    """Exercise serialization, channel order and one-time image normalization."""
    height, width = 48, 64
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[..., 0] = np.arange(width, dtype=np.uint8)[None, :] * 3
    image[..., 1] = np.arange(height, dtype=np.uint8)[:, None] * 4
    image[..., 2] = 211

    timed = TimedObservation(timestamp=1.0, timestep=0, observation={"top": image})
    transported = pickle.loads(pickle.dumps(timed)).get_observation()["top"]  # nosec
    transport_exact = np.array_equal(image, transported)

    chw = robot_observation_image_to_chw(torch.from_numpy(transported.copy()))
    normalized = prepare_image(chw)
    expected = torch.from_numpy(np.moveaxis(image, -1, 0)).float() / 255.0
    uint8_correct = torch.equal(normalized, expected)

    normalized_again = prepare_image(normalized.clone())
    float_not_divided_twice = torch.equal(normalized_again, normalized)
    channel_means = normalized.mean(dim=(1, 2)).tolist()
    channel_order_preserved = channel_means[2] > channel_means[0] > channel_means[1]

    passed = transport_exact and uint8_correct and float_not_divided_twice and channel_order_preserved
    status = Status.PASS if passed else Status.FAIL
    summary = (
        "카메라 직렬화·RGB 채널·[0,1] 단일 정규화 경로가 정상입니다."
        if passed
        else "카메라 전달/정규화 코드 경로가 이전 정상 동작과 다릅니다."
    )
    return _result(
        "camera.code_path",
        "camera",
        status,
        summary,
        transport_exact=transport_exact,
        uint8_to_unit_range=uint8_correct,
        float_not_divided_twice=float_not_divided_twice,
        channel_order_preserved=channel_order_preserved,
        output_shape=list(normalized.shape),
        output_min=float(normalized.min()),
        output_max=float(normalized.max()),
        channel_mean_rgb=channel_means,
    )


def _call_server_resize_route(policy_type: str) -> bool:
    seen: list[bool] = []

    def fake_prepare(*_args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        seen.append(kwargs["resize_images"])
        return {f"{OBS_IMAGES}.camera1": torch.zeros(1, 3, 480, 640)}

    server = object.__new__(PolicyServer)
    server.config = SimpleNamespace(environment_dt=1 / 30, debug_observation_dir=None)
    server.policy_type = policy_type
    server.lerobot_features = {}
    server.actions_per_chunk = 1
    server.policy = SimpleNamespace(config=SimpleNamespace(image_features={}))
    server.preprocessor = lambda observation: observation
    server.postprocessor = lambda action: action
    server.last_processed_obs = None
    server._logged_policy_input_stats = True
    server._debug_capture_ids = set()
    server._get_action_chunk = lambda _observation: torch.zeros(1, 1, 1)

    timed = TimedObservation(timestamp=1.0, timestep=0, observation={})
    with (
        _silence_library_logs(),
        patch("lerobot.async_inference.policy_server.raw_observation_to_observation", fake_prepare),
    ):
        server._predict_action_chunk(timed)
    return seen == [policy_type != "smolvla"]


def check_smolvla_preprocessing_code_path() -> CheckResult:
    """Prove that async keeps 640x480 and SmolVLA alone makes 512x512."""
    raw_image = np.full((480, 640, 3), 127, dtype=np.uint8)
    raw_observation = {
        "shoulder_pan.pos": 0.0,
        "top": raw_image,
    }
    lerobot_features = {
        OBS_STATE: {"dtype": "float32", "shape": [1], "names": ["shoulder_pan.pos"]},
        f"{OBS_IMAGES}.top": {
            "dtype": "image",
            "shape": [480, 640, 3],
            "names": ["height", "width", "channels"],
        },
    }
    policy_image_features = {
        f"{OBS_IMAGES}.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256))
    }

    observation = raw_observation_to_observation(
        raw_observation,
        lerobot_features,
        policy_image_features,
        resize_images=False,
    )
    helper_image = observation[f"{OBS_IMAGES}.top"]
    helper_preserved = tuple(helper_image.shape) == (1, 3, 480, 640)

    padded = resize_with_pad(helper_image, 512, 512, pad_value=0)
    model_shape_correct = tuple(padded.shape) == (1, 3, 512, 512)
    top_padding_correct = bool(torch.count_nonzero(padded[:, :, :128, :]) == 0)
    content_height_correct = bool(torch.count_nonzero(padded[:, :, 128:, :]) > 0)

    smolvla_route_correct = _call_server_resize_route("smolvla")
    other_policy_route_correct = _call_server_resize_route("act")
    passed = all(
        (
            helper_preserved,
            model_shape_correct,
            top_padding_correct,
            content_height_correct,
            smolvla_route_correct,
            other_policy_route_correct,
        )
    )
    return _result(
        "smolvla.preprocessing",
        "smolvla",
        Status.PASS if passed else Status.FAIL,
        (
            "SmolVLA는 async 256px 축소 없이 640x480을 받고 모델에서 한 번만 512x512로 처리합니다."
            if passed
            else "SmolVLA 이중 resize 방지 경로가 이전 정상 동작과 다릅니다."
        ),
        helper_shape=list(helper_image.shape),
        model_shape=list(padded.shape),
        expected_top_padding_px=128,
        top_padding_correct=top_padding_correct,
        smolvla_resize_images_false=smolvla_route_correct,
        other_policy_resize_images_true=other_policy_route_correct,
    )


def check_motor_code_path() -> CheckResult:
    """Exercise proportional vector limiting and verify watchdog integration."""
    goals = {
        "shoulder_lift": (50.0, 0.0),
        "elbow_flex": (-40.0, 0.0),
        "wrist_flex": (10.0, 0.0),
    }
    with _silence_library_logs():
        safe = ensure_synchronized_goal_position(goals, 2.0)
    expected = {"shoulder_lift": 2.0, "elbow_flex": -1.6, "wrist_flex": 0.4}
    coordinated = all(abs(safe[key] - value) < 1e-6 for key, value in expected.items())

    source = inspect.getsource(SOFollower.send_action)
    synchronized_integrated = "ensure_synchronized_goal_position" in source
    watchdog_integrated = "_stop_on_tracking_error" in source and "previous_goal_pos" in source
    passed = coordinated and synchronized_integrated and watchdog_integrated
    return _result(
        "motor.code_path",
        "motor",
        Status.PASS if passed else Status.FAIL,
        (
            "전체 관절 목표가 같은 비율로 제한되고 모터 추종 watchdog이 연결되어 있습니다."
            if passed
            else "동기화 clamp 또는 모터 추종 watchdog 코드가 이전 정상 동작과 다릅니다."
        ),
        input_goals={key: list(value) for key, value in goals.items()},
        safe_goals=safe,
        expected_goals=expected,
        synchronized_limiter_integrated=synchronized_integrated,
        tracking_watchdog_integrated=watchdog_integrated,
    )


def _capture_directories(root: Path) -> list[Path]:
    if not root.exists():
        return []
    if (root / "metadata.json").is_file() or (root / "raw/metadata.json").is_file():
        return [root]
    return sorted(
        (
            path
            for path in root.iterdir()
            if path.is_dir()
            and ((path / "metadata.json").is_file() or (path / "raw/metadata.json").is_file())
        ),
        key=lambda path: (path.stat().st_mtime_ns, path.name),
    )


def _select_capture_pair(client_root: Path, server_root: Path) -> tuple[Path | None, Path | None, bool]:
    clients = _capture_directories(client_root)
    servers = _capture_directories(server_root)
    clients_by_name = {path.name: path for path in clients}
    servers_by_name = {path.name: path for path in servers}
    common_names = sorted(set(clients_by_name) & set(servers_by_name))
    if common_names:
        name = common_names[-1]
        return clients_by_name[name], servers_by_name[name], True
    return (clients[-1] if clients else None, servers[-1] if servers else None, False)


def _canonical_camera(path: Path) -> str | None:
    return CAMERA_ALIASES.get(path.stem)


def _load_stage_images(capture: Path | None, stage: str | None) -> dict[str, tuple[Path, np.ndarray]]:
    if capture is None:
        return {}
    stage_dir = capture if stage is None else capture / stage
    if not stage_dir.is_dir():
        return {}

    images: dict[str, tuple[Path, np.ndarray]] = {}
    for path in sorted(stage_dir.glob("*.png")):
        camera = _canonical_camera(path)
        if camera is not None:
            images[camera] = (path, np.asarray(Image.open(path).convert("RGB")))
    return images


def _image_metrics(image: np.ndarray) -> dict[str, Any]:
    rgb = image.astype(np.float32)
    white = np.all(rgb >= 250, axis=2)
    black = np.all(rgb <= 5, axis=2)
    purple = (rgb[..., 0] > rgb[..., 1] + 35) & (rgb[..., 2] > rgb[..., 1] + 35)
    return {
        "shape": list(image.shape),
        "min": int(image.min()),
        "max": int(image.max()),
        "global_std": float(image.std()),
        "channel_mean_rgb": [float(value) for value in rgb.mean(axis=(0, 1))],
        "channel_std_rgb": [float(value) for value in rgb.std(axis=(0, 1))],
        "white_fraction": float(white.mean()),
        "black_fraction": float(black.mean()),
        "purple_fraction": float(purple.mean()),
    }


def check_camera_artifacts(
    client_capture: Path | None,
    server_capture: Path | None,
    pair_matched: bool,
) -> list[CheckResult]:
    results: list[CheckResult] = []
    stages = {
        "client": _load_stage_images(client_capture, None),
        "server_raw": _load_stage_images(server_capture, "raw"),
        "server_helper": _load_stage_images(server_capture, "helper"),
        "server_policy": _load_stage_images(server_capture, "policy"),
    }

    if not any(stages.values()):
        return [
            _result(
                "camera.artifacts",
                "camera",
                Status.SKIP,
                "분석할 카메라 캡처가 없습니다. 코드 경로 자체 점검만 수행했습니다.",
            )
        ]

    for stage, images in stages.items():
        for camera, (path, image) in images.items():
            metrics = _image_metrics(image)
            if metrics["global_std"] < 2 or metrics["white_fraction"] > 0.95:
                status = Status.FAIL
                summary = f"{stage}/{camera} 영상이 거의 단색 또는 흰 화면입니다."
            elif metrics["purple_fraction"] > 0.75:
                status = Status.FAIL
                summary = f"{stage}/{camera} 영상 대부분이 비정상적인 보라색입니다."
            elif (
                metrics["global_std"] < 8
                or metrics["white_fraction"] > 0.80
                or metrics["purple_fraction"] > 0.45
            ):
                status = Status.WARN
                summary = f"{stage}/{camera} 영상 통계가 치우쳐 있어 직접 확인이 필요합니다."
            else:
                status = Status.PASS
                summary = f"{stage}/{camera} 영상에 흰 화면·전면 보라색 징후가 없습니다."
            results.append(
                _result(
                    f"camera.frame.{stage}.{camera}",
                    "camera",
                    status,
                    summary,
                    path=str(path),
                    **metrics,
                )
            )

    client_images = stages["client"]
    raw_images = stages["server_raw"]
    if client_images and raw_images:
        shared = sorted(set(client_images) & set(raw_images))
        for camera in shared:
            client_path, client = client_images[camera]
            server_path, server = raw_images[camera]
            same_shape = client.shape == server.shape
            exact = same_shape and np.array_equal(client, server)
            details: dict[str, Any] = {
                "capture_id_matched": pair_matched,
                "client_path": str(client_path),
                "server_path": str(server_path),
                "client_shape": list(client.shape),
                "server_shape": list(server.shape),
                "exact_pixel_match": exact,
            }
            if same_shape:
                difference = np.abs(client.astype(np.int16) - server.astype(np.int16))
                details.update(
                    mean_absolute_error=float(difference.mean()),
                    max_absolute_error=int(difference.max()),
                    different_pixel_fraction=float(np.any(difference != 0, axis=2).mean()),
                )
            if not pair_matched:
                status = Status.WARN
                summary = f"{camera}: 캡처 ID가 달라 전송 무결성을 확정할 수 없습니다."
            elif exact:
                status = Status.PASS
                summary = f"{camera}: 로봇 PC 전송 직전과 GPU 수신 직후 픽셀이 완전히 같습니다."
            else:
                status = Status.FAIL
                summary = f"{camera}: 로봇 PC와 GPU raw 이미지가 달라졌습니다."
            results.append(_result(f"camera.transport.{camera}", "camera", status, summary, **details))
    else:
        results.append(
            _result(
                "camera.transport",
                "camera",
                Status.SKIP,
                "클라이언트와 GPU raw 캡처가 모두 있어야 네트워크 전송 픽셀을 비교할 수 있습니다.",
                client_capture=None if client_capture is None else str(client_capture),
                server_capture=None if server_capture is None else str(server_capture),
            )
        )

    helper_images = stages["server_helper"]
    if raw_images and helper_images:
        shared = sorted(set(raw_images) & set(helper_images))
        for camera in shared:
            raw_path, raw = raw_images[camera]
            helper_path, helper = helper_images[camera]
            same_geometry = raw.shape[:2] == helper.shape[:2]
            intermediate_256 = helper.shape[:2] == (256, 256)
            if same_geometry and not intermediate_256:
                status = Status.PASS
                summary = f"{camera}: async helper가 원본 {raw.shape[1]}x{raw.shape[0]} 해상도를 유지했습니다."
            else:
                status = Status.FAIL
                summary = f"{camera}: SmolVLA 이전에 영상 해상도가 변경됐습니다."
            results.append(
                _result(
                    f"smolvla.artifact_geometry.{camera}",
                    "smolvla",
                    status,
                    summary,
                    raw_path=str(raw_path),
                    helper_path=str(helper_path),
                    raw_hw=list(raw.shape[:2]),
                    helper_hw=list(helper.shape[:2]),
                    intermediate_256_detected=intermediate_256,
                )
            )
    else:
        results.append(
            _result(
                "smolvla.artifact_geometry",
                "smolvla",
                Status.SKIP,
                "GPU raw/helper 단계 캡처가 없어 실제 실행 프레임의 중간 resize 여부는 건너뜁니다.",
            )
        )
    return results


def _latest_motor_trace(path: Path) -> Path | None:
    if path.is_file():
        return path
    if not path.is_dir():
        return None
    traces = sorted(path.glob("trace_*.json"), key=lambda item: (item.stat().st_mtime_ns, item.name))
    return traces[-1] if traces else None


def check_motor_trace(path: Path, threshold: float, grace_steps: int) -> CheckResult:
    trace_path = _latest_motor_trace(path)
    if trace_path is None:
        return _result(
            "motor.trace",
            "motor",
            Status.SKIP,
            "분석할 모터 command/feedback trace가 없습니다. 코드 경로 자체 점검만 수행했습니다.",
        )

    entries = json.loads(trace_path.read_text(encoding="utf-8"))
    errors_by_motor: dict[str, list[float]] = {}
    max_consecutive: dict[str, int] = {}
    consecutive: dict[str, int] = {}
    aborts: list[dict[str, Any]] = []
    coordination_spreads: list[float] = []

    for entry in entries:
        robot = entry.get("robot") or {}
        if robot.get("event") == "tracking_abort" or entry.get("error"):
            aborts.append(entry)

        tracking_errors = robot.get("tracking_error") or {}
        observed_motors = set(tracking_errors)
        for motor in set(errors_by_motor) | observed_motors:
            value = float(tracking_errors.get(motor, 0.0))
            errors_by_motor.setdefault(motor, []).append(value)
            if motor == "gripper":
                continue
            consecutive[motor] = consecutive.get(motor, 0) + 1 if value > threshold else 0
            max_consecutive[motor] = max(max_consecutive.get(motor, 0), consecutive[motor])

        requested = robot.get("requested_goal_pos")
        sent = robot.get("sent_goal_pos")
        previous = robot.get("previous_goal_pos")
        if not isinstance(requested, dict) or not isinstance(sent, dict) or not isinstance(previous, dict):
            continue
        scales = []
        for motor in set(requested) & set(sent) & set(previous):
            desired_delta = float(requested[motor]) - float(previous[motor])
            if abs(desired_delta) <= 1e-6:
                continue
            sent_delta = float(sent[motor]) - float(previous[motor])
            scales.append(sent_delta / desired_delta)
        if len(scales) >= 2:
            coordination_spreads.append(max(scales) - min(scales))

    joint_summary = {
        motor: {
            "samples": len(values),
            "max_tracking_error": max(values, default=0.0),
            "p95_tracking_error": float(np.percentile(values, 95)) if values else 0.0,
            "max_consecutive_over_threshold": max_consecutive.get(motor, 0),
        }
        for motor, values in sorted(errors_by_motor.items())
    }
    stalled = {
        motor: count
        for motor, count in max_consecutive.items()
        if motor != "gripper" and count >= grace_steps
    }
    max_coordination_spread = max(coordination_spreads, default=0.0)
    coordination_failed = max_coordination_spread > 1e-3

    if aborts or stalled or coordination_failed:
        status = Status.FAIL
        summary = "실제 trace에서 관절 추종 정지 또는 비동기 관절 명령 징후가 발견됐습니다."
    elif not errors_by_motor:
        status = Status.WARN
        summary = "trace는 있지만 follower의 현재값 진단 필드가 없어 하드웨어 추종을 확정할 수 없습니다."
    else:
        status = Status.PASS
        summary = "최근 trace에서 어깨 정지·관절 비동기 진행 징후가 발견되지 않았습니다."

    return _result(
        "motor.trace",
        "motor",
        status,
        summary,
        path=str(trace_path),
        entries=len(entries),
        threshold=threshold,
        grace_steps=grace_steps,
        tracking_aborts=len(aborts),
        stalled_joints=stalled,
        max_coordination_scale_spread=max_coordination_spread,
        joints=joint_summary,
    )


def check_running_policy_server() -> CheckResult:
    """Warn when the running GPU server predates the source currently on disk."""
    source_path = Path(inspect.getsourcefile(PolicyServer) or "")
    matches: list[dict[str, Any]] = []
    proc = Path("/proc")
    if proc.is_dir():
        for pid_dir in proc.iterdir():
            if not pid_dir.name.isdigit() or int(pid_dir.name) == os.getpid():
                continue
            try:
                command = (pid_dir / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
                if "lerobot.async_inference.policy_server" not in command:
                    continue
                process_started = pid_dir.stat().st_ctime
                process_cwd = Path(os.readlink(pid_dir / "cwd")).resolve()
                matches.append(
                    {
                        "pid": int(pid_dir.name),
                        "command": command.strip(),
                        "cwd": str(process_cwd),
                        "process_started_epoch": process_started,
                        "source_mtime_epoch": source_path.stat().st_mtime,
                        "source_newer_than_process": source_path.stat().st_mtime > process_started + 1,
                        "different_worktree": process_cwd != REPO_ROOT,
                    }
                )
            except (FileNotFoundError, PermissionError, ProcessLookupError):
                continue

    if not matches:
        return _result(
            "server.process_freshness",
            "server",
            Status.SKIP,
            "이 PC에서 실행 중인 Policy server를 찾지 못했습니다.",
        )
    stale = [
        process
        for process in matches
        if process["source_newer_than_process"] or process["different_worktree"]
    ]
    return _result(
        "server.process_freshness",
        "server",
        Status.WARN if stale else Status.PASS,
        (
            "Policy server가 오래된 소스 또는 다른 작업 폴더에서 실행 중입니다. 재시작이 필요합니다."
            if stale
            else "실행 중인 Policy server는 현재 소스보다 나중에 시작됐습니다."
        ),
        processes=matches,
    )


def _make_contact_sheet(
    output_path: Path,
    client_capture: Path | None,
    server_capture: Path | None,
) -> Path | None:
    stages = [
        ("client", _load_stage_images(client_capture, None)),
        ("server raw", _load_stage_images(server_capture, "raw")),
        ("server helper", _load_stage_images(server_capture, "helper")),
        ("server policy", _load_stage_images(server_capture, "policy")),
    ]
    if not any(images for _, images in stages):
        return None

    cell_width, cell_height, label_height = 320, 240, 28
    sheet = Image.new("RGB", (cell_width * 3, (cell_height + label_height) * 4), "white")
    draw = ImageDraw.Draw(sheet)
    for row, (stage, images) in enumerate(stages):
        for column, camera in enumerate(("top", "wrist", "belly")):
            x = column * cell_width
            y = row * (cell_height + label_height)
            draw.text((x + 5, y + 5), f"{stage} / {camera}", fill="black")
            if camera not in images:
                draw.text((x + 5, y + label_height + 5), "missing", fill="red")
                continue
            image = Image.fromarray(images[camera][1])
            image.thumbnail((cell_width, cell_height))
            sheet.paste(image, (x, y + label_height))
    sheet.save(output_path)
    return output_path


def _status_counts(results: list[CheckResult]) -> dict[str, int]:
    return {status.value: sum(result.status == status for result in results) for status in Status}


def _write_reports(
    output_dir: Path,
    results: list[CheckResult],
    client_capture: Path | None,
    server_capture: Path | None,
) -> tuple[Path, Path, Path | None]:
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = _status_counts(results)
    overall = "FAIL" if counts[Status.FAIL] else "WARN" if counts[Status.WARN] else "PASS"
    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "overall": overall,
        "counts": counts,
        "results": [asdict(result) for result in results],
    }
    json_path = output_dir / "report.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# SmolVLA 런타임 회귀 진단",
        "",
        f"- 전체 판정: **{payload['overall']}**",
        f"- PASS {counts['PASS']} / WARN {counts['WARN']} / FAIL {counts['FAIL']} / SKIP {counts['SKIP']}",
        "",
        "| 영역 | 검사 | 판정 | 결과 |",
        "|---|---|---:|---|",
    ]
    for result in results:
        lines.append(f"| {result.group} | `{result.check_id}` | **{result.status}** | {result.summary} |")
    lines.extend(["", "## 상세 정보", ""])
    for result in results:
        lines.extend(
            [
                f"### {result.check_id} — {result.status}",
                "",
                result.summary,
                "",
                "```json",
                json.dumps(result.details, indent=2, ensure_ascii=False),
                "```",
                "",
            ]
        )
    markdown_path = output_dir / "report.md"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")

    contact_sheet = _make_contact_sheet(output_dir / "camera_comparison.png", client_capture, server_capture)
    return json_path, markdown_path, contact_sheet


def run_diagnostics(
    *,
    client_capture_root: Path = DEFAULT_CLIENT_CAPTURES,
    server_capture_root: Path = DEFAULT_SERVER_CAPTURES,
    motor_trace_path: Path = DEFAULT_MOTOR_TRACES,
    output_dir: Path | None = None,
    tracking_error_threshold: float = 5.0,
    tracking_error_grace_steps: int = 2,
) -> tuple[list[CheckResult], Path]:
    client_capture, server_capture, pair_matched = _select_capture_pair(
        client_capture_root, server_capture_root
    )
    results = [
        check_camera_code_path(),
        check_motor_code_path(),
        check_smolvla_preprocessing_code_path(),
        check_running_policy_server(),
    ]
    results.extend(check_camera_artifacts(client_capture, server_capture, pair_matched))
    results.append(
        check_motor_trace(motor_trace_path, tracking_error_threshold, tracking_error_grace_steps)
    )

    if output_dir is None:
        output_dir = DEFAULT_REPORT_ROOT / f"regression_{time.strftime('%Y%m%d_%H%M%S')}"
    _write_reports(output_dir, results, client_capture, server_capture)
    return results, output_dir


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "과거 세 회귀(카메라 손상, 관절 비동기/정지, SmolVLA 이중 resize)를 "
            "하드웨어를 움직이지 않고 코드와 최근 flight-recorder 자료로 검사합니다."
        )
    )
    parser.add_argument("--client-captures", type=Path, default=DEFAULT_CLIENT_CAPTURES)
    parser.add_argument("--server-captures", type=Path, default=DEFAULT_SERVER_CAPTURES)
    parser.add_argument("--motor-traces", type=Path, default=DEFAULT_MOTOR_TRACES)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--tracking-error-threshold", type=float, default=5.0)
    parser.add_argument("--tracking-error-grace-steps", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    results, output_dir = run_diagnostics(
        client_capture_root=args.client_captures,
        server_capture_root=args.server_captures,
        motor_trace_path=args.motor_traces,
        output_dir=args.output_dir,
        tracking_error_threshold=args.tracking_error_threshold,
        tracking_error_grace_steps=args.tracking_error_grace_steps,
    )

    print("\nSmolVLA 런타임 회귀 진단")
    print("=" * 72)
    for result in results:
        print(f"[{result.status:4}] {result.check_id:42} {result.summary}")
    counts = _status_counts(results)
    print("-" * 72)
    print(
        f"PASS={counts['PASS']} WARN={counts['WARN']} "
        f"FAIL={counts['FAIL']} SKIP={counts['SKIP']}"
    )
    print(f"Report: {output_dir / 'report.md'}")
    print(f"JSON:   {output_dir / 'report.json'}")
    if (output_dir / "camera_comparison.png").is_file():
        print(f"Images: {output_dir / 'camera_comparison.png'}")
    return 1 if counts[Status.FAIL] else 0


if __name__ == "__main__":
    raise SystemExit(main())
