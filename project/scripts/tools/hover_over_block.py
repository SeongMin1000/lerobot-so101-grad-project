#!/usr/bin/env python3
"""Park the open gripper at the grasp pose for one block and stop there.

Tuning the grasp offsets through full pick-and-place cycles is slow -- each
wrong guess costs a minute and disturbs the board. This runs exactly the same
detection, calibration, orientation and IK path as the real FSM, then holds
still at the grasp point so the alignment can just be looked at.

Usage: hover_over_block.py <color> [dx_m] [dy_m] [axial_m]
  dx/dy override the configured grasp offsets for this run only (robot frame,
  +x away from the operator, +y to the operator's left). axial_m pushes the
  grasp point further along the jaw axis, which is the direction the gripper
  actually advances into the block.
"""

import json
import sys
import time

import cv2
import numpy as np

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.grad_project.control.table_ik import load_kinematics, solve_to_position
from lerobot.grad_project.paths import lerobot_root
from lerobot.grad_project.perception.yolo_block_detector import YoloBlockDetector
from lerobot.robots import make_robot_from_config, so_follower  # noqa: F401
from lerobot.robots.so_follower import SO101FollowerConfig

JOINT_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
FPS = 30.0
MAX_JOINT_SPEED_DEG_S = 25.0
HOVER_M = 0.06
APPROACH_OPEN = 45.0


def _rot_z(angle: float) -> np.ndarray:
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    return np.array([[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]])


def _ramp(robot, kin, current, solved, gripper):
    travel = float(np.max(np.abs(solved[:5] - current[:5])))
    steps = max(2, int(max(travel / MAX_JOINT_SPEED_DEG_S, 0.8) * FPS))
    for step in range(1, steps + 1):
        alpha = step / steps
        action = {f"{n}.pos": float((1 - alpha) * current[i] + alpha * solved[i]) for i, n in enumerate(JOINT_ORDER)}
        action["gripper.pos"] = float(gripper)
        robot.send_action(action)
        time.sleep(1.0 / FPS)
    time.sleep(0.6)


def main() -> None:
    color = sys.argv[1]
    dx = float(sys.argv[2]) if len(sys.argv) > 2 else None
    dy = float(sys.argv[3]) if len(sys.argv) > 3 else None
    axial = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
    jaw_open = float(sys.argv[5]) if len(sys.argv) > 5 else APPROACH_OPEN

    root = lerobot_root()
    kin = load_kinematics()
    runtime = json.loads((root / "project/config/runtime.json").read_text())
    grasp = json.loads((root / "project/config/grasp_pixel_to_robot.json").read_text())
    homography = np.array(grasp["homography_pixel_to_robot_xy_m"])
    z_plane = np.array(grasp["grasp_z_plane_abc"])

    ref = next(p for p in runtime["pregrasp_points"] if p["label"] == "A2")
    ref_pose = kin.forward_kinematics(np.array([ref["pose"][f"{n}.pos"] for n in JOINT_ORDER]))
    ref_azimuth = float(np.arctan2(ref_pose[1, 3], ref_pose[0, 3]))

    detector = YoloBlockDetector.load(
        str(root / "project/models/yolo_block_detector/best.pt"),
        root / "project/config/detector.json",
        frame_color="rgb",
    )
    robot = make_robot_from_config(
        SO101FollowerConfig(
            port="/dev/so101_follower",
            id="follower",
            disable_torque_on_disconnect=False,
            max_relative_target=15.0,
            max_tracking_error=150.0,
            cameras={"top": OpenCVCameraConfig(index_or_path="/dev/cam_top", width=640, height=480, fps=30)},
        )
    )
    robot.connect()
    for motor in JOINT_ORDER[:-1]:
        robot.bus.write("P_Coefficient", motor, 32)
    try:
        observation = robot.get_observation()
        current = np.array([float(observation[f"{n}.pos"]) for n in JOINT_ORDER])
        _ramp(robot, kin, current, np.array([runtime["poses"]["observe"][f"{n}.pos"] for n in JOINT_ORDER]), APPROACH_OPEN)

        observation = robot.get_observation()
        result = detector.detect(observation["top"])
        block = next((b for b in result.blocks if b.color == color), None)
        if block is None:
            raise SystemExit(f"'{color}' not detected. Saw: {[b.color for b in result.blocks]}")

        point = np.array([[[block.cx, block.cy]]], dtype=np.float64)
        x, y = cv2.perspectiveTransform(point, homography)[0, 0]
        target = np.array([float(x), float(y), float(z_plane @ np.array([x, y, 1.0]))])
        target += np.array([dx or 0.0, dy or 0.0, 0.0])

        orientation = _rot_z(float(np.arctan2(target[1], target[0])) - ref_azimuth) @ ref_pose[:3, :3]
        axis = orientation[:, 2] / np.linalg.norm(orientation[:, 2])
        target = target + axis * axial
        print(f"{color}: pixel=({block.cx:.0f},{block.cy:.0f}) -> target={np.round(target, 4)}")

        for name, goal in (("hover", target + np.array([0.0, 0.0, HOVER_M])), ("grasp", target)):
            current = np.array([float(robot.get_observation()[f"{n}.pos"]) for n in JOINT_ORDER])
            best = min(
                (solve_to_position(kin, seed, goal, keep_orientation=orientation, max_iters=60) for seed in (current, np.zeros(6))),
                key=lambda r: r[1],
            )
            solved, err = best
            _ramp(robot, kin, current, solved, jaw_open)

            reached = kin.forward_kinematics(
                np.array([float(robot.get_observation()[f"{n}.pos"]) for n in JOINT_ORDER])
            )[:3, 3]
            gap = goal - reached
            print(f"  {name}: ik_err={err * 1000:.1f}mm  settled {np.linalg.norm(gap) * 1000:.1f}mm short")
            if np.linalg.norm(gap) > 0.006:
                current = np.array([float(robot.get_observation()[f"{n}.pos"]) for n in JOINT_ORDER])
                fixed, _ = solve_to_position(kin, current, goal + gap, keep_orientation=orientation, max_iters=60)
                _ramp(robot, kin, current, fixed, jaw_open)
                reached = kin.forward_kinematics(
                    np.array([float(robot.get_observation()[f"{n}.pos"]) for n in JOINT_ORDER])
                )[:3, 3]
                print(f"  {name}: after correction {np.linalg.norm(goal - reached) * 1000:.1f}mm short")

        print("\nHolding at the grasp pose. Look at how the jaws line up with the block.")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
