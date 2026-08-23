#!/usr/bin/env python3
"""Collect table(cm) <-> robot base frame(m) correspondence points with the
real arm, then fit+save the affine transform `pick_and_place_yolo.py` needs.

Workflow (do this physically, with the real robot, before running the YOLO
pick-and-place script):

  1) Teleoperate (or hand-jog with torque off, then re-enable) the FOLLOWER
     arm so its gripper tip touches a known point on the table -- easiest
     are the 4 target-zone corners (they're already marked, see
     project/config/detector.json's target_polygon) plus the center.
  2) With the arm held still at that point, run this script with
     --label and either --table_x_cm/--table_y_cm or --corner=tl|tr|br|bl
     (tl=(0,0), tr=(20,0), br=(20,10), bl=(0,10), center=(10,5), matching
     pixel_to_table.py's world frame: origin at target zone top-left,
     +x along the 20cm edge, +y along the 10cm edge).
  3) Repeat for >=3 points (4 corners + center recommended, 5 total).
  4) Run once more with --fit=true to fit the affine transform from every
     saved point and write project/config/table_robot_calibration.json.

Example:
  python -m lerobot.grad_project.tools.save_table_calibration_point \\
    --robot.type=so101_follower --robot.port=/dev/so101_follower \\
    --robot.id=follower --robot.disable_torque_on_disconnect=false \\
    --label=tl --corner=tl

  python -m lerobot.grad_project.tools.save_table_calibration_point --fit=true
"""

import logging
from dataclasses import dataclass
from pprint import pformat

import draccus
import numpy as np

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.grad_project.config_io import save_json_atomic
from lerobot.grad_project.control.table_ik import DEFAULT_URDF_PATH, load_kinematics
from lerobot.grad_project.control.table_frame_calibration import (
    DEFAULT_CALIBRATION_PATH,
    CalibrationPoint,
    fit_table_to_robot_affine,
    save_calibration,
)
from lerobot.grad_project.paths import lerobot_root
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    bi_so_follower,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    so_follower,
)
from lerobot.utils.import_utils import register_third_party_plugins

JOINT_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

CORNERS = {
    "tl": (0.0, 0.0),
    "tr": (20.0, 0.0),
    "br": (20.0, 10.0),
    "bl": (0.0, 10.0),
    "center": (10.0, 5.0),
}

DEFAULT_POINTS_PATH = "project/config/table_calibration_points.json"


@dataclass
class SaveTableCalibConfig:
    robot: RobotConfig | None = None

    label: str = ""
    corner: str = ""  # one of CORNERS, overrides table_x_cm/table_y_cm if set
    table_x_cm: float | None = None
    table_y_cm: float | None = None

    points_file: str = DEFAULT_POINTS_PATH
    urdf_path: str = DEFAULT_URDF_PATH
    calibration_out: str = DEFAULT_CALIBRATION_PATH

    # Set true to skip robot connection and just fit+save from saved points.
    fit: bool = False


def _points_path(cfg: SaveTableCalibConfig):
    path = lerobot_root() / cfg.points_file
    return path


def _load_points(cfg: SaveTableCalibConfig) -> list[dict]:
    path = _points_path(cfg)
    if not path.is_file():
        return []
    import json

    with path.open("r", encoding="utf-8") as f:
        return json.load(f).get("points", [])


def _save_points(cfg: SaveTableCalibConfig, points: list[dict]) -> None:
    save_json_atomic(_points_path(cfg), {"points": points})


@draccus.wrap()
def main(cfg: SaveTableCalibConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    logging.info("Config:\n%s", pformat(cfg))

    if cfg.fit:
        raw_points = _load_points(cfg)
        if len(raw_points) < 3:
            raise SystemExit(
                f"Only {len(raw_points)} saved calibration points in {_points_path(cfg)}, "
                "need >=3. Save more points first (see this script's docstring)."
            )
        points = [
            CalibrationPoint(
                label=p["label"],
                table_x_cm=p["table_x_cm"],
                table_y_cm=p["table_y_cm"],
                robot_xyz_m=tuple(p["robot_xyz_m"]),
            )
            for p in raw_points
        ]
        m = fit_table_to_robot_affine(points)
        out = save_calibration(m, points, cfg.calibration_out)
        print(f"[OK] fit affine transform from {len(points)} points, saved to {out}")
        print(m)
        return

    if cfg.robot is None:
        raise SystemExit("--robot.type=... is required unless --fit=true")
    if not cfg.label:
        raise SystemExit("--label is required, e.g. --label=tl")

    if cfg.corner:
        if cfg.corner not in CORNERS:
            raise SystemExit(f"--corner must be one of {list(CORNERS)}, got {cfg.corner!r}")
        table_x_cm, table_y_cm = CORNERS[cfg.corner]
    elif cfg.table_x_cm is not None and cfg.table_y_cm is not None:
        table_x_cm, table_y_cm = cfg.table_x_cm, cfg.table_y_cm
    else:
        raise SystemExit("Provide --corner=tl|tr|br|bl|center or both --table_x_cm/--table_y_cm")

    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    try:
        obs = robot.get_observation()
        current_joint_deg = np.array([float(obs[f"{name}.pos"]) for name in JOINT_ORDER])

        kin = load_kinematics(cfg.urdf_path)
        ee_pose = kin.forward_kinematics(current_joint_deg)
        robot_xyz_m = tuple(float(v) for v in ee_pose[:3, 3])

        points = _load_points(cfg)
        points = [p for p in points if p["label"] != cfg.label]
        points.append(
            {
                "label": cfg.label,
                "table_x_cm": table_x_cm,
                "table_y_cm": table_y_cm,
                "robot_xyz_m": list(robot_xyz_m),
            }
        )
        _save_points(cfg, points)
        print(
            f"[OK] saved point '{cfg.label}': table=({table_x_cm:.1f},{table_y_cm:.1f})cm "
            f"-> robot={tuple(round(v, 4) for v in robot_xyz_m)}m "
            f"({len(points)} points total in {_points_path(cfg)})"
        )
        if len(points) < 3:
            print(f"Need >=3 points before --fit=true works ({len(points)}/3 so far).")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    register_third_party_plugins()
    main()
