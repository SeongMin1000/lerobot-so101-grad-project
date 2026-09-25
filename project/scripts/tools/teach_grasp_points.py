#!/usr/bin/env python3
"""Teach grasp points by hand with the leader arm, one block at a time.

WHY: the calibration in use was mined from recorded demos, which means a script
had to GUESS, from the gripper's position trace, which frame was "the moment it
grasped". Improving that guess got held-out error from 14.7mm to 12.5mm, and
even the cleanest subset of those samples bottoms out near 11mm -- the operator
simply does not put the gripper in the same place twice while teleoperating at
speed. No amount of re-analysis removes that.

Here there is nothing to guess. You drive the follower with the leader until the
block is properly between the jaws, and that pose is recorded because you said
so. Two consequences worth the robot time:

  - The recorded pose is where the arm ACTUALLY has to be, so the TCP offset
    between gripper_frame_link and the jaws is absorbed automatically. The
    hand-set TCP_JAW_OFFSET_M / TCP_FORWARD_OFFSET_M can go back to 0.
  - You teach with the wrist angle this pipeline commands, not whatever angle a
    demonstrator happened to use, so there is no frame mismatch to correct for.

WHAT IT RECORDS, per point: the block's pixel position and angle as the detector
sees it, the follower's joint state, and the forward-kinematics position that
state gives. fit_taught_points.py turns a set of these into the calibration.

HOW MANY: aim for 30-50 spread over the whole area blocks actually get placed,
not clustered near the middle. The fit is only as good as its coverage -- the
mined calibration's far half was empty at one point and everything out there
came out 4cm wrong.

SAFETY: the follower only ever mirrors the leader, so nothing moves unless you
move it. No IK, no autonomous motion, no torque changes.

Usage:
    python project/scripts/tools/teach_grasp_points.py
    python project/scripts/tools/teach_grasp_points.py --out my_points.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "robot"))
import new_pick_and_place as npp  # noqa: E402

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.grad_project.control.table_ik import load_kinematics
from lerobot.grad_project.paths import lerobot_root
from lerobot.robots import make_robot_from_config
from lerobot.robots.so_follower import SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

OUT_PATH = "project/config/taught_grasp_points.json"
FPS = 30.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT_PATH)
    ap.add_argument("--follower-port", default="/dev/so101_follower")
    ap.add_argument("--leader-port", default="/dev/so101_leader")
    ap.add_argument("--color", default=None,
                    help="이 색 블록만 교시 대상으로 씁니다. 여러 개가 보일 때 유용합니다.")
    args = ap.parse_args()

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = lerobot_root() / out_path

    kin = load_kinematics()
    detector = npp.ObbBlockDetector(npp.OBB_MODEL_PATH, npp.DETECTOR_CONFIG_PATH)

    robot = make_robot_from_config(SO101FollowerConfig(
        port=args.follower_port, id="follower",
        disable_torque_on_disconnect=False,
        # No max_relative_target and no tracking watchdog: those exist to police
        # poses a script computed. Here every target is a human hand on the
        # leader, and clamping or halting that just fights the operator.
        cameras={npp.TOP_CAM_KEY: OpenCVCameraConfig(
            index_or_path="/dev/cam_top", width=640, height=480, fps=30)},
    ))
    leader = SO101Leader(SO101LeaderConfig(port=args.leader_port, id="leader"))

    robot.connect()
    leader.connect()

    points: list[dict] = []
    if out_path.is_file():
        points = json.loads(out_path.read_text()).get("points", [])
        print(f"기존 파일에서 {len(points)}개 이어서 시작합니다: {out_path}")

    print("\n" + "=" * 66)
    print("리더암을 움직이면 팔로워가 따라갑니다. 스크립트는 아무것도 안 움직입니다.")
    print()
    print("  블록을 하나 놓고 → 리더암으로 제대로 물릴 때까지 조종 →")
    print("  이 창에서 Enter  (그 자세가 저장됩니다)")
    print()
    print("  s + Enter : 이 블록 건너뛰기      q + Enter : 저장하고 종료")
    print("=" * 66 + "\n")

    try:
        while True:
            # Mirror the leader continuously so the arm is live while you position
            # it; the prompt below is what actually captures a point.
            print(f"[{len(points)}개 저장됨] 블록을 놓고 리더암으로 물린 뒤 Enter "
                  f"(s=건너뛰기, q=종료): ", end="", flush=True)

            import select
            while True:
                robot.send_action(leader.get_action())
                time.sleep(1.0 / FPS)
                if select.select([sys.stdin], [], [], 0.0)[0]:
                    answer = sys.stdin.readline().strip().lower()
                    break

            if answer == "q":
                break
            if answer == "s":
                print("  건너뜁니다.\n")
                continue

            obs = robot.get_observation()
            frame = obs[npp.TOP_CAM_KEY]
            joints = np.array([float(obs[f"{n}.pos"]) for n in npp.JOINT_ORDER])

            detections = [b for b in detector.detect(frame)
                          if args.color is None or b.color == args.color]
            if not detections:
                print("  !! 블록이 안 보입니다. 다시 놓고 시도하세요.\n")
                continue
            if len(detections) > 1:
                print(f"  !! 블록이 {len(detections)}개 보입니다 "
                      f"({', '.join(b.color for b in detections)}). 하나만 두거나 --color 를 쓰세요.\n")
                continue

            block = detections[0]
            tcp = kin.forward_kinematics(joints)[:3, 3]
            points.append({
                "color": block.color,
                "pixel": [float(block.cx), float(block.cy)],
                "block_angle_deg": float(block.angle_deg),
                "raw_angle_deg": float(block.raw_angle_deg),
                "joints_deg": [float(v) for v in joints],
                "tcp_xyz_m": [float(v) for v in tcp],
            })
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps({
                "note": ("Grasp points taught by hand with the leader arm. Each entry is a block the "
                         "operator confirmed was properly held: the detector's view of it, and the "
                         "follower pose that held it. No inference about when a grasp happened, unlike "
                         "the demo-mined calibration this replaces."),
                "taught_on": str(date.today()),
                "points": points,
            }, indent=2, ensure_ascii=False), encoding="utf-8")

            print(f"  저장: {block.color}  픽셀({block.cx:.0f},{block.cy:.0f})  "
                  f"각도 {block.angle_deg:+.1f}도  TCP({tcp[0]*1000:.0f},{tcp[1]*1000:.0f},{tcp[2]*1000:.0f})mm")
            print(f"  누적 {len(points)}개\n")

    except KeyboardInterrupt:
        print("\n중단합니다. 지금까지 저장된 점들은 파일에 남아 있습니다.")
    finally:
        robot.disconnect()
        leader.disconnect()

    print(f"\n총 {len(points)}개 저장: {out_path}")
    if points:
        px = np.array([p["pixel"] for p in points])
        print(f"  픽셀 범위  x {px[:,0].min():.0f}~{px[:,0].max():.0f}, "
              f"y {px[:,1].min():.0f}~{px[:,1].max():.0f}")
        if len(points) < 30:
            print(f"  !! {len(points)}개는 적습니다. 30~50개를 화면 전체에 고루 퍼뜨려 주세요 --")
            print("     비어 있는 구역은 외삽이 되고, 거기서 오차가 급격히 커집니다.")
    print("\n다음: python project/scripts/tools/fit_taught_points.py")


if __name__ == "__main__":
    main()
