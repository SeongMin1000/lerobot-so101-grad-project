#!/usr/bin/env python3
"""Move the gripper smoothly to a specified position (default: 50.0, the midpoint)
while keeping all arm joints at their current positions.

Usage:
    python project/scripts/tools/set_gripper_pos.py [target_pos] [duration_sec]
"""

import sys
import time
import numpy as np

from lerobot.robots import make_robot_from_config, so_follower  # noqa: F401
from lerobot.robots.so_follower import SO101FollowerConfig

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def main():
    target_pos = float(sys.argv[1]) if len(sys.argv) > 1 else 50.0
    duration = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5

    robot = make_robot_from_config(
        SO101FollowerConfig(port="/dev/so101_follower", id="follower", disable_torque_on_disconnect=False)
    )
    robot.connect(calibrate=False)

    try:
        current_obs = robot.get_observation()
        pose = {f"{n}.pos": float(current_obs[f"{n}.pos"]) for n in JOINTS}
        start_gripper = pose["gripper.pos"]
        print(f"Current gripper position: {start_gripper:.2f}")
        print(f"Moving gripper to target: {target_pos:.2f} (duration: {duration:.1f}s)...")

        hz = 30
        steps = max(10, int(duration * hz))
        trajectory = np.linspace(start_gripper, target_pos, steps)

        for val in trajectory:
            pose["gripper.pos"] = float(val)
            robot.send_action(dict(pose))
            time.sleep(1.0 / hz)

        time.sleep(0.5)
        settled = [float(robot.get_observation()["gripper.pos"]) for _ in range(5)]
        avg_pos = sum(settled) / len(settled)
        print(f"Gripper moved successfully. Final settled position: {avg_pos:.2f}")

    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
