#!/usr/bin/env python3
"""
Safe Live Pose Teaching Utility for SO-101.
Never drops the follower arm:
1. Connects leader and follower in one session.
2. Allows real-time teleoperation to position the arm.
3. On pressing [ENTER], saves the current joint angles to runtime.json (e.g. stack_hover).
4. Smoothly moves back to observe pose (3.0s S-curve) and safely holds torque on exit.
"""

import json
import logging
import select
import sys
import termios
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.grad_project.control.hybrid_goto_both_pose import find_target_pose, load_runtime
from lerobot.grad_project.paths import runtime_config_path
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    bi_so_follower,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    so_follower,
)
from lerobot.teleoperators import (  # noqa: F401
    TeleoperatorConfig,
    bi_so_leader,
    koch_leader,
    make_teleoperator_from_config,
    omx_leader,
    so_leader,
)

JOINT_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


@dataclass
class SafeTeachConfig:
    robot: RobotConfig
    teleop: TeleoperatorConfig
    runtime_config: str = str(runtime_config_path())
    pose_name: str = "stack_hover"
    observe_pose_name: str = "observe"
    fps: int = 30
    return_duration_s: float = 3.0


def _check_key_press() -> bool:
    dr, _, _ = select.select([sys.stdin], [], [], 0.0)
    if dr:
        ch = sys.stdin.read(1)
        if ch in ('\n', '\r', ' '):
            return True
    return False


@parser.wrap()
def main(cfg: SafeTeachConfig):
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    
    # Force disable_torque_on_disconnect = false so torque is NEVER dropped
    cfg.robot.disable_torque_on_disconnect = False

    print("=" * 70)
    print(f"🛡️  [SAFE LIVE POSE TEACHER] Target Pose: '{cfg.pose_name}'")
    print("=" * 70)
    print("Connecting robot follower and teleop leader...")

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop)

    robot.connect()
    teleop.connect()

    # Disable leader torque for smooth human manipulation
    if hasattr(teleop, "set_torque"):
        try:
            teleop.set_torque(False)
        except Exception:
            pass

    print("\n✅ Robot connected and torque active on Follower.")
    print("👉 Move the leader arm to the desired 'stack_hover' position.")
    print("👉 Press [ENTER] in this terminal when positioned.\n")

    # Set terminal to non-blocking raw mode for key check
    old_term = termios.tcgetattr(sys.stdin.fileno())
    new_term = termios.tcgetattr(sys.stdin.fileno())
    new_term[3] = new_term[3] & ~(termios.ICANON | termios.ECHO)

    saved_pose: dict[str, float] | None = None
    dt = 1.0 / cfg.fps

    try:
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, new_term)

        while True:
            t_start = time.perf_counter()

            # Read teleop action and send to follower
            act = teleop.get_action()
            robot.send_action(act)

            # Check Enter keypress
            if _check_key_press():
                obs = robot.get_observation()
                saved_pose = {k: float(v) for k, v in obs.items() if ".pos" in k}
                break

            elapsed = time.perf_counter() - t_start
            time.sleep(max(0.0, dt - elapsed))

    finally:
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSANOW, old_term)

    if saved_pose is None:
        print("\n[CANCELLED] No pose saved.")
        return

    # Save to runtime.json
    runtime_path = Path(cfg.runtime_config).expanduser()
    runtime_data = json.loads(runtime_path.read_text(encoding="utf-8")) if runtime_path.exists() else {"poses": {}}
    if "poses" not in runtime_data:
        runtime_data["poses"] = {}

    runtime_data["poses"][cfg.pose_name] = saved_pose
    runtime_path.write_text(json.dumps(runtime_data, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"🎉 Successfully SAVED '{cfg.pose_name}' to {runtime_path}:")
    for k in JOINT_ORDER:
        val = saved_pose.get(f"{k}.pos", 0.0)
        print(f"   {k:<16}: {val:+.2f}°")
    print("=" * 70)

    # Smoothly return follower to observe pose before disconnect
    obs_pose = find_target_pose(runtime_data, cfg.observe_pose_name, "observe")
    if obs_pose:
        print(f"\n🤖 Moving safely to '{cfg.observe_pose_name}' pose ({cfg.return_duration_s:.1f}s)...")
        start_pose = saved_pose
        steps = int(cfg.return_duration_s * cfg.fps)

        for i in range(1, steps + 1):
            t_step = time.perf_counter()
            s = 0.5 * (1.0 - np.cos(np.pi * (i / steps)))
            cmd = {}
            for k in JOINT_ORDER:
                pk = f"{k}.pos"
                if pk in obs_pose and pk in start_pose:
                    cmd[pk] = (1.0 - s) * start_pose[pk] + s * obs_pose[pk]

            robot.send_action(cmd)
            time.sleep(max(0.0, dt - (time.perf_counter() - t_step)))

        print("✅ Reached observe pose safely. Disconnecting with torque locked.")

    # Disconnect safely
    robot.disconnect()
    teleop.disconnect()
    print("🏁 Safe teaching finished.\n")


if __name__ == "__main__":
    main()
