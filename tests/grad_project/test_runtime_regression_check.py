import json

import numpy as np
import torch

from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.debug_capture import save_image_mapping
from lerobot.async_inference.helpers import TimedObservation
from lerobot.async_inference.policy_server import PolicyServer
from lerobot.grad_project.tools.runtime_regression_check import (
    Status,
    check_camera_artifacts,
    check_camera_code_path,
    check_motor_code_path,
    check_motor_trace,
    check_smolvla_preprocessing_code_path,
)


def _camera_image() -> np.ndarray:
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    image[..., 0] = np.arange(64, dtype=np.uint8)[None, :] * 3
    image[..., 1] = np.arange(48, dtype=np.uint8)[:, None] * 4
    image[..., 2] = 180
    return image


def test_code_path_regressions_pass():
    assert check_camera_code_path().status == Status.PASS
    assert check_motor_code_path().status == Status.PASS
    assert check_smolvla_preprocessing_code_path().status == Status.PASS


def test_matched_camera_artifacts_verify_transport_and_geometry(tmp_path):
    capture_id = "capture_matched"
    client = tmp_path / "client" / capture_id
    server = tmp_path / "server" / capture_id
    image = _camera_image()

    save_image_mapping({"top": image}, client)
    save_image_mapping({"top": image}, server, stage="raw")
    save_image_mapping(
        {"observation.images.camera1": np.moveaxis(image.astype(np.float32) / 255.0, -1, 0)[None]},
        server,
        stage="helper",
    )

    results = check_camera_artifacts(client, server, pair_matched=True)
    by_id = {result.check_id: result for result in results}
    assert by_id["camera.transport.top"].status == Status.PASS
    assert by_id["smolvla.artifact_geometry.top"].status == Status.PASS


def test_policy_server_saves_matched_raw_and_helper_stages(tmp_path):
    server = PolicyServer(
        PolicyServerConfig(debug_observation_dir=str(tmp_path), debug_observation_limit=1)
    )
    timed = TimedObservation(
        timestamp=1.0,
        timestep=0,
        observation={"top": _camera_image()},
        debug_capture_id="capture_shared",
    )

    server._capture_debug_observation(timed, timed.observation, "raw")
    server._capture_debug_observation(
        timed,
        {"observation.images.camera1": torch.zeros(1, 3, 48, 64)},
        "helper",
    )

    assert (tmp_path / "capture_shared/raw/top.png").is_file()
    assert (tmp_path / "capture_shared/helper/camera1.png").is_file()


def test_white_camera_artifact_fails(tmp_path):
    capture = tmp_path / "capture_white"
    save_image_mapping({"top": np.full((48, 64, 3), 255, dtype=np.uint8)}, capture)

    results = check_camera_artifacts(capture, None, pair_matched=False)
    by_id = {result.check_id: result for result in results}
    assert by_id["camera.frame.client.top"].status == Status.FAIL


def test_motor_trace_reports_tracking_abort(tmp_path):
    trace_path = tmp_path / "trace_abort.json"
    entries = []
    for index in range(2):
        entries.append(
            {
                "timestep": index,
                "robot": {
                    "event": "tracking_abort" if index == 1 else "command",
                    "requested_goal_pos": {"shoulder_lift": 20.0, "elbow_flex": -10.0},
                    "sent_goal_pos": {"shoulder_lift": 2.0, "elbow_flex": -1.0},
                    "previous_goal_pos": {"shoulder_lift": 0.0, "elbow_flex": 0.0},
                    "present_pos": {"shoulder_lift": 0.0, "elbow_flex": -1.0},
                    "tracking_error": {"shoulder_lift": 6.0, "elbow_flex": 0.0},
                },
            }
        )
    trace_path.write_text(json.dumps(entries), encoding="utf-8")

    result = check_motor_trace(trace_path, threshold=5.0, grace_steps=2)
    assert result.status == Status.FAIL
    assert result.details["stalled_joints"] == {"shoulder_lift": 2}
    assert result.details["tracking_aborts"] == 1
