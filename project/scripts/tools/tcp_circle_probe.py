#!/usr/bin/env python3
"""Measure the gripper's TCP offset once, by geometry, instead of guessing constants.

WHY: IK positions `gripper_frame_link`. The point the jaws actually close on sits
some fixed distance away from that frame, and the URDF does not model it (the
frame hangs off the gripper body, unconnected to the finger joints -- see the
issues log). Every "move it 5mm this way" constant we have added was really an
attempt to guess that one vector by eye, and each guess fixed one spot on the
table while breaking another, because a robot-frame constant does not turn with
the wrist while the real offset does.

THE MEASUREMENT: command the SAME robot XYZ over and over, changing ONLY the
wrist angle. The offset rides the tool, so it rotates with the wrist -- and the
place the gripper physically lands traces a CIRCLE. Fit that circle and you have
the answer directly:

    radius  = how far the jaw point sits from gripper_frame_link
    phase   = which way it points in the tool frame
    centre  = where gripper_frame_link itself actually went

One measurement, valid everywhere, no eyeballing.

The test point is table (8,-3)cm: reach 22cm, tilt only 4.2deg (so the circle
lies nearly flat on the table and can be marked on paper), and every wrist angle
from -150 to +150deg solves there -- checked with no robot before writing this.

The gripper is held nearly CLOSED for this, so each touch leaves one clean mark
rather than two jaw prints. That is also the point a block gets clamped to when
the jaws close on it, which is what the grasp target needs to line up with.

SAFETY: same protocol as tcp_offset_probe.py, which has run live on this arm --
arm torque capped for the duration, hover first, 5mm descent increments, a hard
floor clamp, and Ctrl+C halts the arm where it stands without cutting torque.

Usage:
    python project/scripts/tools/tcp_circle_probe.py --dry-run   # 계산만, 안 움직임
    python project/scripts/tools/tcp_circle_probe.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "robot"))
import new_pick_and_place as npp  # noqa: E402

from lerobot.grad_project.control.grasp_pitch_model import GraspPitchModel, orientation_from_tilt
from lerobot.grad_project.control.table_frame_calibration import apply_table_to_robot, load_calibration
from lerobot.grad_project.control.table_ik import load_kinematics
from lerobot.robots import make_robot_from_config
from lerobot.robots.so_follower import SO101FollowerConfig

TEST_POINT_CM = (8.0, -3.0)
WRIST_ANGLES_DEG = [0.0, -150.0, -100.0, -50.0, 50.0, 100.0, 150.0]
GRIPPER_MARK_POS = 15.0          # nearly closed: one clean mark, not two jaw prints
ARM_TORQUE_PCT = 20
TRACK_ERR_DEG, TRACK_GRACE = 10.0, 3


def build_targets():
    kin = load_kinematics()
    calib = load_calibration("project/config/table_robot_calibration.json")
    pitch = GraspPitchModel.load()
    floor_z = npp.floor_z_from_calibration(calib)

    x_cm, y_cm = TEST_POINT_CM
    target = apply_table_to_robot(x_cm, y_cm, calib)
    reach = float(np.linalg.norm(target[:2]))
    tilt = pitch.tilt_for(reach) if pitch else 0.0
    azimuth = float(np.degrees(np.arctan2(target[1], target[0])))
    base = orientation_from_tilt(target, tilt)
    return kin, floor_z, target, reach, tilt, azimuth, base


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="로봇에 연결하지 않고 계산만 합니다.")
    ap.add_argument("--gripper-pos", type=float, default=GRIPPER_MARK_POS)
    args = ap.parse_args()

    kin, floor_z, target, reach, tilt, azimuth, base = build_targets()
    seed = npp.load_observe_pose()
    hover = target + np.array([0.0, 0.0, npp.HOVER_M])

    print(f"측정 지점: table {TEST_POINT_CM}cm -> robot {np.round(target, 4)}")
    print(f"  reach {reach*100:.1f}cm, tilt {tilt:.1f}도, floor_z {floor_z*1000:.1f}mm")
    print(f"  손목 각도 {len(WRIST_ANGLES_DEG)}개: {[int(a) for a in WRIST_ANGLES_DEG]}\n")

    plan = []
    for ang in WRIST_ANGLES_DEG:
        ori = npp.rotate_about_local_z(base, np.deg2rad(ang + azimuth))
        try:
            q_hover, _ = npp.checked_solve(kin, seed, hover, floor_z=floor_z, seed=seed,
                                           keep_orientation=base)
            q_turn, _ = npp.checked_solve(kin, q_hover, hover, floor_z=floor_z, seed=q_hover,
                                          keep_orientation=ori)
            q_touch, err = npp.checked_solve(kin, q_turn, target, floor_z=floor_z, seed=q_turn,
                                             keep_orientation=ori)
        except npp.IKDivergedError as e:
            print(f"  손목 {ang:+.0f}도 : 계산 실패, 건너뜀 ({str(e).splitlines()[0][:60]})")
            continue
        plan.append((ang, ori, q_hover, q_turn))
        print(f"  손목 {ang:+.0f}도 : OK (IK 오차 {err*1000:.2f}mm)")

    if len(plan) < 4:
        print("\n!! 쓸 수 있는 각도가 4개 미만입니다. 원을 못 맞춥니다. 중단.")
        sys.exit(1)

    if args.dry_run:
        print(f"\n{len(plan)}개 각도 모두 계산됨. 실제로 하려면 --dry-run 빼고 실행하세요.")
        return

    print("\n종이를 측정 지점에 깔고, 전원 스위치 근처에 손을 두세요.")
    print("각 각도마다 그리퍼가 종이에 닿습니다. 닿은 자리에 번호를 적어 표시하고 Enter.\n")

    robot = make_robot_from_config(SO101FollowerConfig(
        port="/dev/so101_follower", id="follower", disable_torque_on_disconnect=False,
        max_relative_target=15.0, max_tracking_error=TRACK_ERR_DEG,
        tracking_error_grace_steps=TRACK_GRACE))
    robot.connect()

    prev_torque: dict[str, int] = {}
    stalled = False
    try:
        npp.set_arm_overload_torque(robot, ARM_TORQUE_PCT, prev_torque)
        print(f"팔 토크를 {ARM_TORQUE_PCT}%로 낮췄습니다 (끝나면 복원).\n")
        current = npp.read_joints(robot)

        for i, (ang, ori, q_hover, q_turn) in enumerate(plan, 1):
            print(f"[{i}/{len(plan)}] 손목 {ang:+.0f}도")
            q_hover = q_hover.copy(); q_hover[npp.GRIPPER_IDX] = args.gripper_pos
            current = npp.ramp_to(robot, current, q_hover, npp.RAMP_STEPS, False)
            q_turn = q_turn.copy(); q_turn[npp.GRIPPER_IDX] = args.gripper_pos
            current = npp.ramp_to(robot, current, q_turn, 60, False)
            current = npp.safe_descend(kin, robot, current, target, floor_z, ori, False,
                                       gripper_goal=args.gripper_pos)
            input(f"   -> 닿은 자리에 '{i}' 이라고 적고 Enter ")
            current = npp.ramp_to(robot, current, q_turn, 60, False)

        print("\n관측 자세로 복귀합니다...")
        current = npp.ramp_to(robot, current, npp.load_observe_pose(), npp.RAMP_STEPS, False)

    except (RuntimeError, KeyboardInterrupt) as e:
        stalled = True
        npp.halt_in_place(robot)
        print(f"\n!! 중단됨, 팔을 그 자리에 정지시켰습니다: {e}")
        print("   토크 상한은 낮춘 채로 둡니다.")
    finally:
        if prev_torque and not stalled:
            npp.restore_arm_overload_torque(robot, prev_torque)
        elif prev_torque:
            print(f"!! 팔 Overload_Torque를 {ARM_TORQUE_PCT}%로 둔 채 종료합니다. 원래값: {prev_torque}")
        robot.disconnect()

    print("\n=== 이제 자로 재세요 ===")
    print("타깃존 사각형의 '로봇에 가까운 쪽 긴 변 왼쪽 끝'을 원점(0,0)으로,")
    print("그 변을 따라가는 방향을 x, 로봇에서 멀어지는 방향을 y 로 해서 (cm 단위)")
    print("표시한 점들의 좌표를 알려주세요:\n")
    for i, (ang, *_rest) in enumerate(plan, 1):
        print(f"  {i}번 (손목 {ang:+.0f}도) : x = ____ cm,  y = ____ cm")
    print("\n이 점들이 그리는 원의 반지름과 방향이 곧 TCP 편차입니다.")


if __name__ == "__main__":
    main()
