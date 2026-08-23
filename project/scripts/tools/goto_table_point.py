#!/usr/bin/env python3
"""Hover the gripper over one table-frame (cm) point and photograph the result.

Cross-checks the camera->table->robot chain: command a table coordinate, then
look at where the gripper actually lands in the camera image relative to the
labelled target-zone corners. Used to catch frame-orientation mistakes (e.g.
the camera facing the robot, which mirrors the table frame by 180 degrees).

Motion is deliberately slow -- shoulder_lift only tracks ~6 deg/s when lifting
the folded arm against gravity, and a faster ramp trips the tracking watchdog.

Usage: goto_table_point.py <x_cm> <y_cm> [z_hover_m] [out.jpg]
"""

import json
import sys
import time

import cv2
import numpy as np

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.grad_project.control.table_frame_calibration import apply_table_to_robot, load_calibration
from lerobot.grad_project.control.table_ik import load_kinematics, solve_to_position
from lerobot.grad_project.paths import lerobot_root
from lerobot.robots import make_robot_from_config, so_follower  # noqa: F401
from lerobot.robots.so_follower import SO101FollowerConfig

JOINT_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
RAMP_STEPS = 300
FPS = 30.0


def _rot_z(angle: float) -> np.ndarray:
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    return np.array([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]])


def grasp_orientation(kin, runtime, target_xyz, ref_label="A2"):
    point = next(p for p in runtime["pregrasp_points"] if p["label"] == ref_label)
    joints = np.array([point["pose"][f"{n}.pos"] for n in JOINT_ORDER])
    pose = kin.forward_kinematics(joints)
    ref_azimuth = np.arctan2(pose[1, 3], pose[0, 3])
    delta = np.arctan2(target_xyz[1], target_xyz[0]) - ref_azimuth
    return _rot_z(delta) @ pose[:3, :3]


def main() -> None:
    x_cm, y_cm = float(sys.argv[1]), float(sys.argv[2])
    z_hover = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
    out_path = sys.argv[4] if len(sys.argv) > 4 else "var/dryrun_debug/goto_check.jpg"

    root = lerobot_root()
    kin = load_kinematics()
    runtime = json.loads((root / "project/config/runtime.json").read_text())
    target = apply_table_to_robot(x_cm, y_cm, load_calibration()) + np.array([0.0, 0.0, z_hover])
    orientation = grasp_orientation(kin, runtime, target)

    config = SO101FollowerConfig(
        port="/dev/so101_follower",
        id="follower",
        disable_torque_on_disconnect=False,
        max_relative_target=15.0,
        max_tracking_error=35.0,
        cameras={"top": OpenCVCameraConfig(index_or_path="/dev/cam_top", width=640, height=480, fps=30)},
    )
    robot = make_robot_from_config(config)
    robot.connect()
    try:
        obs = robot.get_observation()
        current = np.array([float(obs[f"{n}.pos"]) for n in JOINT_ORDER])
        solved, err = solve_to_position(kin, current, target, keep_orientation=orientation, max_iters=60)
        print(f"table=({x_cm},{y_cm})cm -> robot={np.round(target, 4)}  IK err={err * 1000:.2f}mm")

        for step in range(1, RAMP_STEPS + 1):
            alpha = step / RAMP_STEPS
            robot.send_action(
                {f"{n}.pos": float((1 - alpha) * current[i] + alpha * solved[i]) for i, n in enumerate(JOINT_ORDER)}
            )
            time.sleep(1.0 / FPS)
        time.sleep(1.5)

        final = robot.get_observation()
        reached = kin.forward_kinematics(np.array([float(final[f"{n}.pos"]) for n in JOINT_ORDER]))[:3, 3]
        print(f"reached={np.round(reached, 4)}  (off by {np.linalg.norm(reached - target) * 1000:.1f}mm)")

        frame = cv2.cvtColor(final["top"], cv2.COLOR_RGB2BGR)
        polygon = np.array(
            json.loads((root / "project/config/detector.json").read_text())["target_polygon"], dtype=np.int32
        )
        cv2.polylines(frame, [polygon], True, (0, 0, 255), 2)
        for (px, py), label in zip(polygon, ["TL(0,0)", "TR(20,0)", "BR(20,10)", "BL(0,10)"], strict=True):
            cv2.circle(frame, (int(px), int(py)), 7, (0, 255, 255), -1)
            cv2.putText(
                frame, label, (int(px) - 30, int(py) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2
            )
        cv2.imwrite(str(root / out_path), frame)
        print(f"saved {out_path}")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
