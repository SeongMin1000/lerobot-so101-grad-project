#!/usr/bin/env python3
"""Verify the arm torque safety net actually works on THIS hardware. Moves nothing.

WHY THIS EXISTS: new_pick_and_place.py and tcp_offset_probe.py both cap arm
Overload_Torque to ARM_OVERLOAD_TORQUE_PCT so a wrong descent can only push
weakly. That cap was written from the bus API's documented signatures and
carried the comment "NOT verified on real hardware" in both files -- nobody had
ever confirmed the read/write round-trips on this robot.

That gap matters asymmetrically:
  - tcp_offset_probe.py refuses to run if the cap fails.
  - new_pick_and_place.py only PRINTS a warning and keeps going
    ("continuing without this safety net"), with P_Coefficient already raised
    to 32. A missed warning line means the arm runs stiff AND uncapped -- the
    worst combination, and silently.

So confirm it here, once, deliberately, before trusting either script.

WHAT THIS DOES: connects, reads each arm joint's Overload_Torque, writes the
cap, READS IT BACK to prove the write actually took (a write that silently
no-ops would look identical to success from the caller's side), then restores
the original value and reads that back too. Same round-trip for P_Coefficient,
since set_arm_p_gain() swallows its exceptions the same way.

WHAT THIS NEVER DOES: send_action(), Goal_Position, or anything else that
commands a position. The arm does not move. Torque stays as the servos already
have it.

Usage:
    python project/scripts/tools/check_torque_control.py
"""

from __future__ import annotations

import sys

from lerobot.robots import make_robot_from_config, so_follower  # noqa: F401
from lerobot.robots.so_follower import SO101FollowerConfig

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent / "robot"))
import new_pick_and_place as npp  # noqa: E402  (constants only; never calls its main())

# Read-modify-read-restore-read on these two registers only.
REGISTERS = [
    ("Overload_Torque", npp.ARM_OVERLOAD_TORQUE_PCT),
    ("P_Coefficient", npp.ARM_P_GAIN),
]


def roundtrip(robot, register: str, test_value: int) -> bool:
    """Returns True only if every arm joint read the ORIGINAL, accepted the test
    value on readback, and read the original again after restore."""
    print(f"\n--- {register} ---")
    originals: dict[str, int] = {}
    ok = True

    try:
        for motor in npp.ARM_JOINTS:
            originals[motor] = robot.bus.read(register, motor)
        print(f"  현재 값: {originals}")
    except Exception as e:  # noqa: BLE001
        print(f"  !! 읽기 자체가 실패했습니다: {e}")
        print(f"     -> {register} 기반 안전장치는 이 로봇에서 동작하지 않습니다.")
        return False

    try:
        for motor in npp.ARM_JOINTS:
            robot.bus.write(register, motor, test_value)
        # The whole point: a write that silently does nothing looks exactly like
        # a successful one to the caller. Only the readback can tell them apart.
        readback = {m: robot.bus.read(register, m) for m in npp.ARM_JOINTS}
        print(f"  {test_value} 쓴 뒤 실제 값: {readback}")
        for motor, val in readback.items():
            if val != test_value:
                print(f"  !! {motor}: {test_value}을 썼는데 {val}로 읽힙니다 -- 쓰기가 먹지 않았습니다.")
                ok = False
    except Exception as e:  # noqa: BLE001
        print(f"  !! 쓰기 실패: {e}")
        ok = False
    finally:
        # Restore ALWAYS, including when the check above failed partway. Leaving
        # the arm capped at a test value is exactly the failure mode
        # tcp_offset_probe.py's original torque helper had.
        try:
            for motor, val in originals.items():
                robot.bus.write(register, motor, val)
            restored = {m: robot.bus.read(register, m) for m in npp.ARM_JOINTS}
            if restored == originals:
                print(f"  원래 값으로 복원 확인: {restored}")
            else:
                print(f"  !! 복원했는데 값이 다릅니다: {restored} (원래 {originals})")
                print("     -> 다음 실행 전에 수동으로 확인하세요.")
                ok = False
        except Exception as e:  # noqa: BLE001
            print(f"  !! 복원 실패: {e} -- 원래 값 {originals}, 수동 복구 필요")
            ok = False

    return ok


def main() -> None:
    print("이 스크립트는 팔을 움직이지 않습니다 (send_action 호출 없음).")
    print("레지스터를 읽고 -> 테스트 값 쓰고 -> 되읽어 확인하고 -> 원래대로 되돌립니다.\n")

    config = SO101FollowerConfig(
        port="/dev/so101_follower",
        id="follower",
        disable_torque_on_disconnect=False,
        # No max_relative_target / watchdog needed: nothing here ever commands a
        # position, so there is no goal for them to police.
    )
    robot = make_robot_from_config(config)
    robot.connect()
    print(f"연결됨. 팔 관절: {npp.ARM_JOINTS}")

    results = {}
    try:
        for register, test_value in REGISTERS:
            results[register] = roundtrip(robot, register, test_value)
    finally:
        robot.disconnect()
        print("\n연결 해제했습니다.")

    print("\n=== 결과 ===")
    for register, ok in results.items():
        print(f"  {register}: {'정상' if ok else '실패'}")

    if all(results.values()):
        print("\n토크 안전장치가 이 로봇에서 정상 동작합니다. 다음 단계로 진행하세요.")
        return

    print("\n!! 안전장치 중 하나가 동작하지 않습니다.")
    print("   new_pick_and_place.py는 이 실패를 경고만 하고 그냥 진행하므로,")
    print("   원인을 잡기 전에는 실제 픽업을 돌리지 마세요.")
    sys.exit(1)


if __name__ == "__main__":
    main()
