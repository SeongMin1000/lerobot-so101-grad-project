#!/usr/bin/env python3
"""Close the gripper under motor power with nothing in it, and report where it stops.

The grasp check assumes an empty close runs the jaws all the way shut, so any
larger reading means a block is in the way. That assumption is only as good as
the empty-close value actually is -- and closing by hand does not measure it,
because a hand pushes harder than the servo's torque cap. If a stiff or
repaired jaw stalls the motor early, an empty close reads like a full grasp and
every attempt is reported as a success.

Arm joints are held where they are; only the gripper moves.

Usage: test_gripper_close.py [close_value]
"""

import sys
import time

import numpy as np

from lerobot.robots import make_robot_from_config, so_follower  # noqa: F401
from lerobot.robots.so_follower import SO101FollowerConfig

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def main() -> None:
    close_to = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
    robot = make_robot_from_config(
        SO101FollowerConfig(port="/dev/so101_follower", id="follower", disable_torque_on_disconnect=False)
    )
    robot.connect(calibrate=False)
    try:
        pose = {f"{n}.pos": float(robot.get_observation()[f"{n}.pos"]) for n in JOINTS}
        print(f"start gripper = {pose['gripper.pos']:.2f}")

        # Open first, so the close starts from a known width.
        for value in np.linspace(pose["gripper.pos"], 45.0, 20):
            pose["gripper.pos"] = float(value)
            robot.send_action(dict(pose))
            time.sleep(1 / 30)
        time.sleep(0.5)
        print(f"opened to  = {robot.get_observation()['gripper.pos']:.2f}")

        for value in np.linspace(45.0, close_to, 30):
            pose["gripper.pos"] = float(value)
            robot.send_action(dict(pose))
            time.sleep(1 / 30)
        time.sleep(0.8)

        settled = [float(robot.get_observation()["gripper.pos"]) for _ in range(5)]
    finally:
        robot.disconnect()

    print(f"commanded  = {close_to:.2f}")
    print(f"stopped at = {np.mean(settled):.2f}  (min {min(settled):.2f}, max {max(settled):.2f})")
    print()
    print("If this is well above the grasp threshold with EMPTY jaws, every grasp")
    print("reads as a success and the threshold has to move above this value.")


if __name__ == "__main__":
    main()
