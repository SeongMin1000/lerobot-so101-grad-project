#!/usr/bin/env python3
"""Measure how far the arm settles BELOW a commanded height, by reach.

The grasp height has been guessed from "settled Nmm short" log lines, but
those are 3D distances -- using them as if they were all vertical over-lifted
every grasp and the jaws started catching the top of the block. Halving the
guess put the jaws back into the table. Neither is a measurement.

This commands a series of points in clear air and reads back where the gripper
actually ended up, so the vertical component is measured directly. Nothing
approaches the table: every point is `HEIGHT_M` above it.

Usage: measure_vertical_sag.py
"""

import json
import time

import numpy as np

from lerobot.grad_project.control.table_ik import load_kinematics, solve_to_position
from lerobot.grad_project.control.grasp_pitch_model import GraspPitchModel, orientation_from_tilt
from lerobot.grad_project.paths import lerobot_root
from lerobot.robots import make_robot_from_config, so_follower  # noqa: F401
from lerobot.robots.so_follower import SO101FollowerConfig

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
HEIGHT_M = 0.12  # well clear of the table
REACHES = (0.18, 0.24, 0.30, 0.36, 0.41)
FPS = 30.0
MOVE_S = 2.0


def main() -> None:
    root = lerobot_root()
    kin = load_kinematics()
    model = GraspPitchModel.load()
    floor = _floor_plane(root)

    robot = make_robot_from_config(
        SO101FollowerConfig(
            port="/dev/so101_follower", id="follower",
            disable_torque_on_disconnect=False, max_relative_target=15.0,
        )
    )
    robot.connect(calibrate=False)
    results = []
    try:
        for reach in REACHES:
            target = np.array([reach * 0.97, reach * 0.24, 0.0])
            target[2] = float(floor @ np.array([target[0], target[1], 1.0])) + HEIGHT_M
            tilt = model.tilt_for(reach) if model else 10.0
            orientation = orientation_from_tilt(target, tilt)

            current = np.array([float(robot.get_observation()[f"{n}.pos"]) for n in JOINTS])
            solved, err = solve_to_position(kin, current, target, keep_orientation=orientation)
            if err > 0.01:
                print(f"  reach {reach * 100:.0f}cm: IK off by {err * 1000:.0f}mm, skipping")
                continue

            pose = {f"{n}.pos": float(v) for n, v in zip(JOINTS, solved, strict=True)}
            pose["gripper.pos"] = float(current[-1])
            steps = int(MOVE_S * FPS)
            start = {f"{n}.pos": float(v) for n, v in zip(JOINTS, current, strict=True)}
            for i in range(1, steps + 1):
                a = i / steps
                robot.send_action({k: (1 - a) * start[k] + a * pose[k] for k in pose})
                time.sleep(1 / FPS)
            time.sleep(0.8)

            reached = np.array([float(robot.get_observation()[f"{n}.pos"]) for n in JOINTS])
            actual = kin.forward_kinematics(reached)[:3, 3]
            drop = (target[2] - actual[2]) * 1000
            lateral = float(np.linalg.norm((target - actual)[:2])) * 1000
            results.append((reach, drop, lateral))
            print(f"  reach {reach * 100:.0f}cm : 아래로 {drop:+6.1f}mm, 옆으로 {lateral:5.1f}mm")
    finally:
        robot.disconnect()

    if results:
        r = np.array(results)
        print("\n수직 처짐만 뽑은 결과:")
        print(f"  최소 {r[:, 1].min():.1f}mm  최대 {r[:, 1].max():.1f}mm")
        slope, intercept = np.polyfit(r[:, 0], r[:, 1], 1)
        print(f"  근사식: {intercept:.1f}mm + {slope:.1f}mm/m x 거리")
        print(f"  -> droop_base_m={max(intercept, 0) / 1000:.4f}  droop_per_reach_m={slope / 1000:.4f}")


def _floor_plane(root):
    points = json.loads((root / "project/config/table_calibration_points.json").read_text())["points"]
    xyz = np.array([p["robot_xyz_m"] for p in points])
    design = np.column_stack([xyz[:, :2], np.ones(len(xyz))])
    plane, *_ = np.linalg.lstsq(design, xyz[:, 2], rcond=None)
    return plane


if __name__ == "__main__":
    main()
