#!/usr/bin/env python3
"""Print the gripper's current opening. Reads only -- sends no motion at all.

The grasp check calls the jaws "holding a block" when they fail to close past
a threshold. That only works if the empty-and-fully-closed reading sits below
it, which a repaired or reshimmed jaw can break: if the jaws now bottom out
early, every grasp reads as a success. Comparing the empty reading against the
block-held reading is what tells you where the threshold belongs.

Usage: read_gripper.py [samples]
"""

import sys
import time

from lerobot.robots import make_robot_from_config, so_follower  # noqa: F401
from lerobot.robots.so_follower import SO101FollowerConfig


def main() -> None:
    samples = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    robot = make_robot_from_config(
        SO101FollowerConfig(port="/dev/so101_follower", id="follower", disable_torque_on_disconnect=False)
    )
    robot.connect(calibrate=False)
    try:
        values = []
        for _ in range(samples):
            values.append(float(robot.get_observation()["gripper.pos"]))
            time.sleep(0.1)
    finally:
        robot.disconnect()

    for value in values:
        print(f"  gripper.pos = {value:6.2f}")
    print(f"\naverage {sum(values) / len(values):.2f}  (min {min(values):.2f}, max {max(values):.2f})")


if __name__ == "__main__":
    main()
