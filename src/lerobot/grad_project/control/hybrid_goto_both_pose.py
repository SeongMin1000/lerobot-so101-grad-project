#!/usr/bin/env python3

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat
from typing import Any

import draccus

from lerobot.grad_project.paths import runtime_config_path

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
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
from lerobot.utils.import_utils import register_third_party_plugins


POSE_MOVER_BUILD = "2026-07-28-no-follower-retorque-v1"


@dataclass
class GotoBothPoseConfig:
    robot: RobotConfig
    teleop: TeleoperatorConfig

    runtime_config: str = str(runtime_config_path())

    # poses 안의 이름: observe, drop_center, slot1 등
    pose_name: str = ""

    # pregrasp_points 안의 label: A1, A2, B1 등
    pregrasp_label: str = ""

    duration_s: float = 3.0
    fps: int = 30
    settle_s: float = 0.2

    # True면 이동 후 포트는 닫지만 torque는 끄지 않음
    keep_torque_on_disconnect: bool = True


def load_runtime(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return json.load(open(path, encoding="utf-8"))


def find_target_pose(
    runtime: dict[str, Any],
    pose_name: str,
    pregrasp_label: str,
) -> dict[str, float]:
    if pose_name:
        poses = runtime.get("poses", {})
        if pose_name not in poses:
            raise KeyError(
                f"pose_name '{pose_name}' not found. Available: {list(poses)}"
            )
        return {k: float(v) for k, v in poses[pose_name].items()}

    if pregrasp_label:
        for p in runtime.get("pregrasp_points", []):
            if p.get("label") == pregrasp_label:
                return {k: float(v) for k, v in p["pose"].items()}
        labels = [p.get("label") for p in runtime.get("pregrasp_points", [])]
        raise KeyError(
            f"pregrasp_label '{pregrasp_label}' not found. Available: {labels}"
        )

    raise ValueError("Use either --pose_name=... or --pregrasp_label=...")


def robot_current_pose(robot: Any) -> dict[str, float]:
    obs = robot.get_observation()
    return {k: float(obs[k]) for k in robot.action_features if k in obs}


def teleop_current_pose(teleop: Any) -> dict[str, float]:
    act = teleop.get_action()
    return {k: float(v) for k, v in act.items() if k.endswith(".pos")}


def move_both(
    robot: Any,
    teleop: Any,
    target: dict[str, float],
    duration_s: float,
    fps: int,
) -> None:
    robot_cur = robot_current_pose(robot)
    teleop_cur = teleop_current_pose(teleop)

    robot_keys = [
        k for k in robot.action_features if k in robot_cur and k in target
    ]
    teleop_keys = [
        k for k in teleop.action_features if k in teleop_cur and k in target
    ]

    if not robot_keys:
        raise RuntimeError("No robot keys overlap with target pose")
    if not teleop_keys:
        raise RuntimeError("No teleop keys overlap with target pose")

    steps = max(2, int(duration_s * fps))
    print(f"[MOVE] follower + leader moving over {duration_s:.2f}s, steps={steps}")

    # SO follower는 robot.connect()의 configure()가 끝날 때 이미 torque ON이다.
    # 여기서 전체 모터의 torque를 다시 enable하면 gripper(ID 6)에 남아 있던
    # overload 상태가 enable_torque() 응답으로 보고되며 복귀가 중단될 수 있다.
    # 따라서 follower에는 현재 위치를 새 goal로 한 번 보내 압력을 풀고,
    # enable_torque()를 중복 호출하지 않는다.
    robot.send_action({k: robot_cur[k] for k in robot_keys})

    # SO leader는 teleoperation 중 torque OFF 상태다. torque를 켜기 전에
    # 현재 위치를 goal로 먼저 기록해야 갑자기 이전 goal로 튀지 않는다.
    teleop.send_feedback({k: teleop_cur[k] for k in teleop_keys})
    if hasattr(teleop, "enable_torque"):
        teleop.enable_torque()
    elif hasattr(teleop, "bus"):
        teleop.bus.enable_torque()
    else:
        raise RuntimeError("Teleoperator does not provide a torque-enable method")

    for i in range(1, steps + 1):
        a = i / steps

        robot_action = {}
        for k in robot_keys:
            robot_action[k] = (1 - a) * robot_cur[k] + a * target[k]

        teleop_feedback = {}
        for k in teleop_keys:
            teleop_feedback[k] = (1 - a) * teleop_cur[k] + a * target[k]

        robot.send_action(robot_action)
        teleop.send_feedback(teleop_feedback)

        time.sleep(1.0 / fps)


def disconnect_keep_torque(device: Any, keep_torque: bool) -> None:
    # SO follower/leader 둘 다 내부에 bus가 있음.
    # keep_torque=True면 torque를 끄지 않고 포트만 닫는다.
    if keep_torque and hasattr(device, "bus"):
        try:
            device.bus.disconnect(disable_torque=False)
            return
        except TypeError:
            pass
    device.disconnect()


@draccus.wrap()
def main(cfg: GotoBothPoseConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    logging.info("Config:\n%s", pformat(cfg))
    logging.info("Pose mover build: %s", POSE_MOVER_BUILD)

    runtime_path = runtime_config_path(cfg.runtime_config)
    print(f"[CONFIG] runtime_config={runtime_path}")
    runtime = load_runtime(runtime_path)
    target = find_target_pose(runtime, cfg.pose_name, cfg.pregrasp_label)

    print("[TARGET]")
    print(json.dumps(target, indent=2, ensure_ascii=False))

    teleop = make_teleoperator_from_config(cfg.teleop)
    robot = make_robot_from_config(cfg.robot)

    teleop.connect()
    robot.connect()

    try:
        move_both(robot, teleop, target, cfg.duration_s, cfg.fps)
        time.sleep(cfg.settle_s)
        print("[OK] follower + leader moved to target pose")
    finally:
        disconnect_keep_torque(robot, cfg.keep_torque_on_disconnect)
        disconnect_keep_torque(teleop, cfg.keep_torque_on_disconnect)


if __name__ == "__main__":
    register_third_party_plugins()
    main()
