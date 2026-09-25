#!/usr/bin/env python3
"""Probe the real-world accuracy of table(cm) -> robot(m) -> IK -> physical position.

This does NOT try to measure gripper geometry or guess a TCP offset in advance.
Instead it commands the arm to touch a handful of KNOWN table-frame (cm) points
and lets a human mark + later measure where it actually landed. Comparing
"commanded table cm" vs "physically measured table cm" gives the TOTAL
real-world error (TCP offset, wrist-roll tilt, table-fit error, whatever) in
one number per point/axis, without needing to isolate the cause first.
See YOLO_하드코딩_개발_이슈정리.md for why this replaces hand-tuned corrections.

`apply_table_to_robot` returns [x, y, z] in the robot base frame -- Z is NOT
skipped, it comes from the same affine fit (the table is slightly tilted, so
the 4 known corners already have different real Z heights). This script
therefore tests Z accuracy (the actual crash risk) just as much as X/Y.

SAFETY (this project has bent a gripper finger and slammed the arm into the
table before -- see the issues log):
  - Never jumps straight to a touch target. Always: hover first, then
    descend in small MAX_STEP_M increments with a short pause between each.
  - Hard floor clamp: will not command Z below the lowest Z seen among the
    already-physically-verified calibration corners, minus FLOOR_MARGIN_M.
    If a computed target would go below that, the descent stops early and
    prints a warning instead of continuing.
  - Ctrl+C is handled explicitly: it commands the arm to hold where it
    physically is (cancelling the still-live servo goal, which otherwise keeps
    driving the arm while Python unwinds) and leaves the reduced torque cap in
    place rather than restoring it. Until 2026-09-01 this line claimed Ctrl+C
    was safe while nothing caught KeyboardInterrupt at all -- it descends from
    BaseException, so it skipped every handler, RESTORED full torque, and left
    the arm pushing harder than before. Do not trust a documented safety
    behaviour that has no handler behind it.
  - Keep a hand near the power switch / e-stop while running this. Watch
    the first descent closely before trusting later ones.

Usage:
    python project/scripts/tools/tcp_offset_probe.py
    python project/scripts/tools/tcp_offset_probe.py --points 10,5 20,0 0,0

Defaults to the table center (10,5cm -- a held-out interpolation test, not
one of the 4 fit points) plus the 4 known calibration corners (tl/tr/br/bl),
all inside the zone that's already been physically touched safely before.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from lerobot.grad_project.control.grasp_pitch_model import GraspPitchModel, orientation_from_tilt
from lerobot.grad_project.control.table_frame_calibration import apply_table_to_robot, load_calibration
from lerobot.grad_project.control.table_ik import load_kinematics, solve_to_position
from lerobot.grad_project.paths import lerobot_root
from lerobot.robots import make_robot_from_config, so_follower  # noqa: F401
from lerobot.robots.so_follower import SO101FollowerConfig

JOINT_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
RAMP_STEPS = 300
FPS = 30.0

HOVER_M = 0.05  # hover 5cm above the computed table point before descending
MAX_STEP_M = 0.005  # never descend more than 5mm per increment
FLOOR_MARGIN_M = 0.003  # extra 3mm safety margin below the lowest known-safe Z
GRIPPER_TOUCH_POS = 15.0  # degrees; nearly-closed, just enough to leave a clear mark -- adjust to your gripper
# --gripper-pos lets us re-touch the SAME calibrated points with the gripper held OPEN
# (as it actually is during new_pick_and_place.py's real descent) instead of nearly-closed,
# to check whether the touch point itself shifts with gripper angle -- see issues log,
# "그리퍼 열림/닫힘 상태에 따른 접촉점 편차" investigation.
MAX_IK_ERR_M = 0.005  # abort (don't move) if IK didn't converge within 5mm of the requested target

# Arm joints (everything except gripper) had NO overload-torque cap in the existing code --
# only the gripper does, added after a jaw broke. That gripper fix is exactly what we're
# borrowing here for the arm, since this test's whole point is "we don't fully trust the
# computed position yet": cap arm torque low for the DURATION OF THIS SCRIPT ONLY, so even
# a wrong descent can only push weakly, then restore whatever was there before on exit.
ARM_OVERLOAD_TORQUE_PCT = 20
# 3.0/1-step was too strict: at 20% torque the arm (esp. shoulder_lift, carrying the arm
# against gravity) normally lags the commanded ramp position by a couple degrees -- that's
# harmless motor lag, not a stall, but it tripped the watchdog on the very first move. Real
# stalls (actually stuck against something) show much larger, PERSISTENT error, not a single
# transient reading a couple degrees over. Loosen the threshold and require it to persist.
TEST_MAX_TRACKING_ERROR = 10.0  # degrees; still much stricter than the 35.0 used elsewhere
TEST_TRACKING_GRACE_STEPS = 3  # require 3 consecutive violating steps, not just one


class IKDivergedError(RuntimeError):
    """Raised when solve_to_position did not converge -- never move on a bad solution."""


def checked_solve(kin, current: np.ndarray, target_xyz: np.ndarray, **kwargs) -> tuple[np.ndarray, float]:
    """solve_to_position, but refuses to hand back a solution that didn't actually converge.

    This is the fix for "no code errors, but the arm slammed into the table anyway": that
    happens when IK fails to converge (unreachable point, near-singularity) and the caller
    moves to the bad solution regardless. Here we raise instead, so the caller MUST stop.
    """
    solved, err = solve_to_position(kin, current, target_xyz, max_iters=kwargs.pop("max_iters", 60), **kwargs)
    if err > MAX_IK_ERR_M:
        raise IKDivergedError(
            f"IK did not converge: target={np.round(target_xyz, 4)} err={err * 1000:.2f}mm "
            f"(limit {MAX_IK_ERR_M * 1000:.0f}mm). Refusing to move -- point may be unreachable."
        )
    return solved, err


DEFAULT_POINTS_CM = [
    ("center", 10.0, 5.0),  # held-out point, not one of the 4 fit corners
    ("tl", 0.0, 0.0),
    ("tr", 20.0, 0.0),
    ("br", 20.0, 10.0),
    ("bl", 0.0, 10.0),
]


def read_joints(robot) -> np.ndarray:
    """The ONE way to learn where the arm really is -- never infer it from what
    was last commanded (a max_relative_target clamp or a watchdog freeze makes
    those two disagree). Same helper/rationale as new_pick_and_place.py."""
    obs = robot.get_observation()
    return np.array([float(obs[f"{n}.pos"]) for n in JOINT_ORDER])


def halt_in_place(robot) -> None:
    """Command the arm to hold exactly where it physically is right now.

    Needed on Ctrl+C: the last send_action goal stays live in the servos and the
    arm keeps driving toward it while the Python exception unwinds. Re-commanding
    the present position cancels that without cutting torque (which would just
    drop the arm). Same helper/rationale as new_pick_and_place.py."""
    try:
        here = read_joints(robot)
        robot.send_action({f"{n}.pos": float(here[i]) for i, n in enumerate(JOINT_ORDER)})
    except Exception as e:  # noqa: BLE001
        print(f"!! 팔을 제자리에 정지시키지 못했습니다 ({e}) -- 아직 움직이면 전원을 차단하세요.")


def ramp_to(robot, current: np.ndarray, target_joints: np.ndarray, steps: int = RAMP_STEPS) -> np.ndarray:
    for step in range(1, steps + 1):
        alpha = step / steps
        robot.send_action(
            {
                f"{n}.pos": float((1 - alpha) * current[i] + alpha * target_joints[i])
                for i, n in enumerate(JOINT_ORDER)
            }
        )
        time.sleep(1.0 / FPS)
    time.sleep(0.3)
    # Return where the arm REALLY ended up, not the commanded target -- this
    # script's whole output is a physical measurement, so silently substituting
    # "what we asked for" wherever a clamp shortened a move would corrupt the
    # very numbers being measured (see issues log 2026-09-01).
    return read_joints(robot)


def safe_descend(
    kin, robot, current: np.ndarray, xy_xyz: np.ndarray, floor_z: float, orientation: np.ndarray
) -> np.ndarray:
    """Descend from wherever `current` is down toward xy_xyz, in small steps, never below floor_z.

    `orientation` must be the reach-appropriate tilt (see orientation_for_target) -- NOT
    derived from whatever `current`'s FK happens to be, since that couples this descent's
    feasibility to how hover was reached rather than to the actual target's own geometry.
    """
    start_pose = kin.forward_kinematics(current)
    start_z = float(start_pose[2, 3])
    target_z = max(float(xy_xyz[2]), floor_z)
    if xy_xyz[2] < floor_z:
        print(
            f"    !! 계산된 목표 높이({xy_xyz[2] * 1000:.1f}mm)가 안전 하한선"
            f"({floor_z * 1000:.1f}mm)보다 낮습니다. 하한선까지만 내려갑니다."
        )

    # Pin the gripper for the whole descent. solve_to_position's output gripper
    # component is whatever the IK solver produced, NOT necessarily the value
    # passed in, so without this the gripper drifts a little on every descent
    # step. That matters most for --gripper-pos: the entire point of that flag
    # is measuring where the contact point lands at a KNOWN gripper angle, and
    # a gripper that quietly drifts off 70deg mid-descent corrupts exactly the
    # measurement being taken (see issues log 2026-09-01).
    gripper_idx = JOINT_ORDER.index("gripper")
    gripper_hold = current[gripper_idx]
    n_steps = max(1, int(abs(start_z - target_z) / MAX_STEP_M) + 1)
    for i in range(1, n_steps + 1):
        z = start_z + (target_z - start_z) * (i / n_steps)
        waypoint = np.array([xy_xyz[0], xy_xyz[1], z])
        solved, err = checked_solve(kin, current, waypoint, keep_orientation=orientation)
        solved[gripper_idx] = gripper_hold
        current = ramp_to(robot, current, solved, steps=60)
        print(f"    하강 {i}/{n_steps}: z={z * 1000:.1f}mm (IK 오차 {err * 1000:.2f}mm)")
    return current


def set_arm_overload_torque(robot, pct: int, previous: dict[str, int]) -> None:
    """Set Overload_Torque on every joint except the gripper (which already has its own
    cap). Fills caller-owned `previous` in AS WE GO -- rather than building+returning
    its own dict -- so a mid-loop exception still leaves `previous` usable for
    restore_arm_overload_torque(). The old version returned a fresh dict, which a raised
    exception discarded entirely, leaving already-lowered motors stuck at pct% with no
    way to restore them (see YOLO_하드코딩_개발_이슈정리.md, 2026-09-01).
    NOT verified on real hardware -- if `robot.bus.read`/`write` signatures differ, this
    will raise; check it works (e.g. read one value back) before trusting it in a real run.
    """
    for motor in robot.bus.motors:
        if motor == "gripper":
            continue
        previous[motor] = robot.bus.read("Overload_Torque", motor)
        if previous[motor] == pct:
            # A run that exited with the arm stalled deliberately leaves the cap
            # on. If that message was missed, the arm stays weak forever after
            # and `previous` now records the cap as if it were the original.
            print(f"!! {motor}: 이미 Overload_Torque={pct}% 상태입니다 -- 이전 실행이 팔이 걸린 채 "
                  f"종료했을 수 있습니다. 원래 값으로 되돌린 뒤 이 결과를 신뢰하세요.")
        robot.bus.write("Overload_Torque", motor, pct)


def restore_overload_torque(robot, previous: dict[str, int]) -> None:
    for motor, val in previous.items():
        robot.bus.write("Overload_Torque", motor, val)


def orientation_for_target(model: "GraspPitchModel | None", target_xyz: np.ndarray) -> np.ndarray:
    """Reach-dependent wrist tilt, same as the real pick-and-place pipeline uses
    (GraspPitchModel: close-in grasps are steeper/near-vertical, far grasps lean over
    more). A single fixed orientation for every point is NOT physically reachable
    across the full workspace -- that's why tl/tr/br/bl failed with a fixed orientation
    while center (closer, less extreme reach) happened to pass.
    """
    reach_m = float(np.linalg.norm(target_xyz[:2]))
    tilt_deg = model.tilt_for(reach_m) if model is not None else 0.0
    return orientation_from_tilt(target_xyz, tilt_deg)


def load_seed_pose(pose_name: str = "observe") -> np.ndarray:
    """Real joint pose from runtime.json, e.g. the "observe" rest pose -- used as the
    dry-run's fake "current position" so IK is seeded from somewhere the arm actually
    sits, not an all-zero pose whose orientation is meaningless for this arm."""
    runtime = json.loads((lerobot_root() / "project/config/runtime.json").read_text())
    pose = runtime["poses"][pose_name]
    return np.array([pose[f"{n}.pos"] for n in JOINT_ORDER])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--points",
        nargs="+",
        default=None,
        help="table-frame points as x,y in cm, e.g. --points 10,5 20,0. Defaults to center + 4 calibration corners.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute every IK target/error and print it. Does NOT connect to the robot or move anything. "
        "Run this FIRST and read the errors before ever running for real.",
    )
    parser.add_argument(
        "--gripper-pos",
        type=float,
        default=GRIPPER_TOUCH_POS,
        help=f"Gripper angle (deg) to hold during touch, default {GRIPPER_TOUCH_POS} (nearly-closed). "
        "Pass 70 (GRIPPER_OPEN_POS in new_pick_and_place.py) to check whether the touch point shifts "
        "when the gripper is held open, as it is during a real pick's descent.",
    )
    args = parser.parse_args()
    gripper_touch_pos = args.gripper_pos

    if args.points:
        points = [(f"pt{i}", *map(float, p.split(","))) for i, p in enumerate(args.points)]
    else:
        points = DEFAULT_POINTS_CM

    kin = load_kinematics()
    calib = load_calibration()

    known_z = [apply_table_to_robot(x, y, calib)[2] for _, x, y in DEFAULT_POINTS_CM[1:]]  # the 4 real corners
    floor_z = min(known_z) - FLOOR_MARGIN_M
    print(f"안전 하한선(floor_z) = {floor_z * 1000:.1f}mm (기존 실측 모서리 중 최저 높이 - {FLOOR_MARGIN_M * 1000:.0f}mm)")

    pitch_model = GraspPitchModel.load()
    if pitch_model is None:
        print("!! GraspPitchModel 샘플(grasp_correction_samples.json)을 못 찾음 -- 모든 지점 tilt=0(수직)으로 대체함.")
        print("   (실제 파이프라인은 거리별로 손목을 기울이므로, 이 상태의 결과는 참고용으로만 쓸 것.)")
    else:
        print(f"GraspPitchModel 로드됨: {pitch_model.describe()}")

    if args.dry_run:
        # All-zero joints is NOT a pose this arm ever actually sits in, and with no
        # keep_orientation passed, solve_to_position tries to hold whatever (likely
        # degenerate/singular) orientation that zero-pose happens to have -- that alone
        # can make a perfectly reachable table point look like an IK failure. Seed from
        # the real "observe" pose instead, the same one the robot actually rests at.
        seed = load_seed_pose()
        print("\n=== DRY RUN (로봇에 연결하지 않음, 아무것도 움직이지 않음) ===")
        try:
            for label, x_cm, y_cm in points:
                target_xyz = apply_table_to_robot(x_cm, y_cm, calib)
                hover_xyz = target_xyz + np.array([0.0, 0.0, HOVER_M])

                # Reach-appropriate tilt for THIS point, same as the real pipeline uses --
                # a single fixed orientation is not reachable across the whole workspace.
                orientation = orientation_for_target(pitch_model, target_xyz)
                reach_cm = float(np.linalg.norm(target_xyz[:2])) * 100
                solved_hover, err_hover = checked_solve(kin, seed, hover_xyz, keep_orientation=orientation)

                touch_z = max(float(target_xyz[2]), floor_z)
                touch_xyz = np.array([target_xyz[0], target_xyz[1], touch_z])
                _, err_touch = checked_solve(kin, solved_hover, touch_xyz, keep_orientation=orientation)

                print(
                    f"[{label}] ({x_cm},{y_cm})cm -> robot {np.round(target_xyz, 4)} "
                    f"reach={reach_cm:.1f}cm : "
                    f"hover 오차 {err_hover * 1000:.2f}mm, touch 오차 {err_touch * 1000:.2f}mm"
                )
            print("\n모든 지점 IK 수렴 확인됨. 실제로 실행하려면 --dry-run 빼고 다시 실행하세요.")
        except IKDivergedError as e:
            print(f"\n!! IK 수렴 실패, 이 지점은 실제로 위험할 수 있습니다: {e}")
            sys.exit(1)
        return

    config = SO101FollowerConfig(
        port="/dev/so101_follower",
        id="follower",
        disable_torque_on_disconnect=False,
        max_relative_target=15.0,
        max_tracking_error=TEST_MAX_TRACKING_ERROR,
        tracking_error_grace_steps=TEST_TRACKING_GRACE_STEPS,
    )
    robot = make_robot_from_config(config)
    robot.connect()

    print(f"\n{len(points)}개 지점을 테스트합니다. (그리퍼 각도: {gripper_touch_pos}도)")
    print("전원 스위치/비상정지 근처에 손을 두고, 첫 하강은 특히 자세히 지켜보세요.")
    print("각 지점에서 그리퍼가 닿으면, 실제 닿은 자리를 연필로 표시하고 Enter를 누르세요.\n")

    prev_torque: dict[str, int] = {}
    try:
        set_arm_overload_torque(robot, ARM_OVERLOAD_TORQUE_PCT, prev_torque)
        print(f"팔 관절 토크를 임시로 {ARM_OVERLOAD_TORQUE_PCT}%로 낮췄습니다 (테스트 끝나면 원래대로 복원).")
    except Exception as e:  # noqa: BLE001
        print(f"!! 팔 토크 제한 설정 실패 ({e}) -- 이 안전장치 없이는 계속 진행하지 않습니다.")
        if prev_torque:  # partial application happened before the failure -- undo it
            restore_overload_torque(robot, prev_torque)
        robot.disconnect()
        sys.exit(1)

    results = []
    stalled_on_exit = False  # set when we bail out with the arm possibly jammed
    try:
        current = read_joints(robot)
        gripper_idx = JOINT_ORDER.index("gripper")
        # NOTE: do NOT overwrite current[gripper_idx] here. `current` must hold the REAL
        # observed gripper position so ramp_to's interpolation closes it gradually over
        # RAMP_STEPS, synchronized with the arm move. Setting it to GRIPPER_TOUCH_POS here
        # instead made the very first send_action() jump the gripper in one step (real pos
        # ~44deg -> target 15deg with zero ramp), which is exactly what tripped the
        # tracking-error watchdog on the first run -- the watchdog did its job correctly.

        for label, x_cm, y_cm in points:
            target_xyz = apply_table_to_robot(x_cm, y_cm, calib)
            hover_xyz = target_xyz + np.array([0.0, 0.0, HOVER_M])

            orientation = orientation_for_target(pitch_model, target_xyz)
            reach_cm = float(np.linalg.norm(target_xyz[:2])) * 100

            print(
                f"[{label}] 목표 테이블 좌표: ({x_cm}cm, {y_cm}cm) -> 로봇좌표 {np.round(target_xyz, 4)} "
                f"(reach={reach_cm:.1f}cm)"
            )

            solved_hover, err_hover = checked_solve(kin, current, hover_xyz, keep_orientation=orientation)
            # Only the ARM's target came from IK; the gripper target is ours to set. Setting
            # it on the SOLVED target (not on `current`) means ramp_to closes it gradually,
            # from wherever it really is right now, over the full ramp -- not in one jump.
            solved_hover[gripper_idx] = gripper_touch_pos
            current = ramp_to(robot, current, solved_hover)
            print(f"  hover 도착 (IK 오차 {err_hover * 1000:.2f}mm)")

            current = safe_descend(kin, robot, current, target_xyz, floor_z, orientation)

            input("  -> 실제 닿은 자리를 표시했으면 Enter (다음 지점으로 이동) ")

            solved_hover_back, _ = checked_solve(kin, current, hover_xyz, keep_orientation=orientation)
            solved_hover_back[gripper_idx] = current[gripper_idx]  # same IK-gripper-drift pin as safe_descend
            current = ramp_to(robot, current, solved_hover_back)
            results.append((label, x_cm, y_cm))

    except (IKDivergedError, RuntimeError) as e:
        print(f"\n!! 안전장치 발동, 즉시 정지했습니다: {e}")
        print("지금까지 표시한 지점까지의 결과는 아래에서 유효합니다. 원인 파악 후 재시도하세요.")
    except KeyboardInterrupt:
        # KeyboardInterrupt is a BaseException, so it was NOT caught above --
        # before 2026-09-01 Ctrl+C skipped every handler, hit `finally`, and
        # RESTORED the full torque cap while the servos still held the last
        # commanded goal. A person presses Ctrl+C because the arm is going
        # somewhere wrong, so that path made a jam push HARDER. Halt the arm
        # where it is and leave the reduced cap on.
        stalled_on_exit = True
        halt_in_place(robot)
        print("\n!! 사용자가 중단했습니다 -- 팔을 현재 위치에 정지시켰습니다.")
        print("   토크 상한은 낮춘 상태로 둡니다(팔이 뭔가에 걸려 있을 수 있으므로).")
    finally:
        if prev_torque:
            if stalled_on_exit:
                print(f"!! 팔 Overload_Torque를 {ARM_OVERLOAD_TORQUE_PCT}%로 낮춘 채 종료합니다 "
                      f"(복원 안 함 -- 팔이 아직 걸려 있을 수 있음). 원래 값: {prev_torque}")
            else:
                try:
                    restore_overload_torque(robot, prev_torque)
                    print("팔 관절 토크를 원래 값으로 복원했습니다.")
                except Exception as e:  # noqa: BLE001
                    print(f"!! 토크 복원 실패 ({e}) -- 다음 사용 전에 수동으로 확인하세요.")
        robot.disconnect()

    print("\n=== 결과 ===")
    print("아래 명령값들을, 표시된 실제 지점을 자/줄자로 재서(원점 = tl 모서리 = 0,0cm)")
    print("비교할 수 있게 (측정된 x_cm, 측정된 y_cm)를 정리해서 알려주세요:\n")
    for label, x_cm, y_cm in results:
        print(f"  {label}: 명령값 ({x_cm}, {y_cm})cm")


if __name__ == "__main__":
    main()
