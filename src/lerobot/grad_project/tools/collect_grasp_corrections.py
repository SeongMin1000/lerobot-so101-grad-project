#!/usr/bin/env python3
"""Teach the true pixel -> robot grasp mapping by leader-arm demonstration.

The corner calibration in `save_table_calibration_point.py` maps the TABLE
PLANE, but YOLO boxes a block's TOP FACE ~2.5cm above it. With the camera
mounted over the far side of the board, that parallax drags every detection
toward the robot, so the gripper stops short -- and the error grows with
distance from the camera's nadir, which no single scalar offset can fix.

This tool skips the plane model. For each block you drive the follower to the
position where the block ACTUALLY grasps, and it records (block pixel, true
robot xy). A homography fitted over >=4 such pairs maps pixel -> robot
directly for objects at block height, absorbing the parallax exactly because
every block is the same height.

Three subcommands, each a single non-interactive run:

  snapshot            Photograph the board with the arm parked out of the way
                      and remember where every block is. Do this FIRST, and
                      again whenever the blocks move.

  teach --color=red   Mirror the leader onto the follower for --seconds so you
                      can place the gripper exactly where it would close on
                      that block, then record the pair. The block's pixel comes
                      from the snapshot, so it does not matter that the arm is
                      now covering the block.

  fit                 Fit and save project/config/grasp_pixel_to_robot.json.

Spread the taught blocks across the whole board (near/far, left/right). Points
clustered in one region fit fine and generalise badly.
"""

import argparse
import json
import time
from typing import Any

import cv2
import numpy as np

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.grad_project.config_io import save_json_atomic
from lerobot.grad_project.control.table_ik import load_kinematics
from lerobot.grad_project.paths import lerobot_root
from lerobot.robots import make_robot_from_config, so_follower  # noqa: F401
from lerobot.robots.so_follower import SO101FollowerConfig
from lerobot.teleoperators.so_leader import SO101Leader, SO101LeaderConfig

JOINT_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
STATE_PATH = "project/config/grasp_correction_samples.json"
FIT_PATH = "project/config/grasp_pixel_to_robot.json"
TELEOP_FPS = 30.0


def _load_state() -> dict:
    path = lerobot_root() / STATE_PATH
    if not path.is_file():
        return {"snapshot": {}, "samples": []}
    data = json.loads(path.read_text())
    data.setdefault("snapshot", {})
    data.setdefault("samples", [])
    return data


def _save_state(state: dict) -> None:
    save_json_atomic(lerobot_root() / STATE_PATH, state)


def _make_follower(port: str, with_camera: bool):
    cameras = (
        {"top": OpenCVCameraConfig(index_or_path="/dev/cam_top", width=640, height=480, fps=30)}
        if with_camera
        else {}
    )
    return make_robot_from_config(
        SO101FollowerConfig(port=port, id="follower", disable_torque_on_disconnect=False, cameras=cameras)
    )


def cmd_snapshot(args) -> None:
    from lerobot.grad_project.perception.yolo_block_detector import YoloBlockDetector

    detector = YoloBlockDetector.load(
        str(lerobot_root() / args.yolo_model_path),
        lerobot_root() / "project/config/detector.json",
        frame_color="rgb",
        conf=args.conf,
    )
    robot = _make_follower(args.follower_port, with_camera=True)
    robot.connect()
    try:
        frame = robot.get_observation()["top"]
    finally:
        robot.disconnect()

    result = detector.detect(frame)
    state = _load_state()
    # A stray false positive occasionally shares a colour with a real block.
    # Keeping the largest box per colour picks the real one -- the spurious
    # detections are noticeably smaller than a 2.5cm block at this distance.
    largest: dict[str, Any] = {}
    for block in result.blocks:
        if block.color not in largest or block.area > largest[block.color].area:
            largest[block.color] = block
    state["snapshot"] = {c: [float(b.cx), float(b.cy)] for c, b in largest.items()}
    _save_state(state)

    cv2.imwrite(str(lerobot_root() / "var/dryrun_debug/snapshot.jpg"), detector.draw(frame, result))
    print(f"snapshot: {len(result.blocks)} blocks")
    for color, (cx, cy) in state["snapshot"].items():
        print(f"  {color:7s} pixel=({cx:6.1f},{cy:6.1f})")
    already = {s["color"] for s in state["samples"]}
    print(f"\nalready taught: {sorted(already) or 'none'}  (total samples: {len(state['samples'])})")


def _tilt_from_vertical_deg(kin, joints_deg: np.ndarray) -> float:
    """Angle between the gripper's approach axis and straight down.

    0 deg is a pure overhead grasp; ~57 deg is what the old fixed reference
    pose used, which is shallow enough to shove blocks sideways.
    """
    approach = kin.forward_kinematics(joints_deg)[:3, 2]
    approach = approach / np.linalg.norm(approach)
    return float(np.degrees(np.arccos(np.clip(-approach[2], -1.0, 1.0))))


def cmd_teach(args) -> None:
    state = _load_state()
    if args.color not in state["snapshot"]:
        raise SystemExit(
            f"'{args.color}' is not in the last snapshot ({sorted(state['snapshot'])}). Run snapshot first."
        )
    pixel = state["snapshot"][args.color]

    kin = load_kinematics()
    follower = _make_follower(args.follower_port, with_camera=False)

    if args.handjog:
        # Some poses are outside what the leader arm's own joint range can
        # mirror, so teleop can never reach them -- hand-jogging the follower
        # directly (torque off) sidesteps that limit.
        follower.connect()
        try:
            follower.bus.disable_torque()
            if args.seconds <= 0:
                # Zero seconds means the arm was already placed by hand before
                # this ran. A countdown only works if the operator can see it
                # start, and a message printed by a command that has not
                # returned yet is invisible until it is too late.
                print(f"Reading {args.color} from where the arm is now.")
            else:
                print(f"Torque off for {args.seconds:.0f}s -- hand-move the gripper onto {args.color}, then hold.")
                time.sleep(args.seconds)
            final = follower.get_observation()
            joints = np.array([float(final[f"{n}.pos"]) for n in JOINT_ORDER])
            xyz = kin.forward_kinematics(joints)[:3, 3]
            follower.bus.enable_torque()
        finally:
            follower.disconnect()
    else:
        leader = SO101Leader(SO101LeaderConfig(port=args.leader_port, id="leader"))
        follower.connect()
        leader.connect()
        try:
            print(f"Teleop active for {args.seconds:.0f}s -- put the gripper where it would close on {args.color}.")
            deadline = time.perf_counter() + args.seconds
            while time.perf_counter() < deadline:
                follower.send_action(leader.get_action())
                time.sleep(1.0 / TELEOP_FPS)

            final = follower.get_observation()
            joints = np.array([float(final[f"{n}.pos"]) for n in JOINT_ORDER])
            xyz = kin.forward_kinematics(joints)[:3, 3]
        finally:
            leader.disconnect()
            follower.disconnect()

    # Keep the joint vector, not just the resulting xyz. The wrist angle the
    # operator naturally used is the part the pipeline cannot guess -- it is
    # what decides whether the jaws come down on the block or skate into it --
    # and it is recoverable only from the full configuration.
    tilt_deg = _tilt_from_vertical_deg(kin, joints)
    reach_m = float(np.hypot(xyz[0], xyz[1]))

    state["samples"] = [s for s in state["samples"] if s["color"] != args.color or s["pixel"] != pixel]
    state["samples"].append(
        {
            "label": f"{args.color}@{int(pixel[0])},{int(pixel[1])}",
            "color": args.color,
            "pixel": pixel,
            "robot_xyz_m": [float(v) for v in xyz],
            "joints_deg": [float(v) for v in joints],
            "reach_m": reach_m,
            "tilt_from_vertical_deg": tilt_deg,
        }
    )
    _save_state(state)
    print(
        f"[OK] {args.color}: pixel=({pixel[0]:.0f},{pixel[1]:.0f}) -> "
        f"robot=({xyz[0]:.4f},{xyz[1]:.4f},{xyz[2]:.4f})m  "
        f"reach={reach_m:.3f}m tilt={tilt_deg:.1f}deg   [{len(state['samples'])} samples]"
    )


def fit_pixel_to_robot(samples: list[dict]) -> np.ndarray:
    """Homography mapping block-centre pixels to robot-frame xy (metres)."""
    if len(samples) < 4:
        raise SystemExit(f"Need >=4 samples to fit a homography, have {len(samples)}.")
    src = np.array([s["pixel"] for s in samples], dtype=np.float64)
    dst = np.array([s["robot_xyz_m"][:2] for s in samples], dtype=np.float64)
    homography, _ = cv2.findHomography(src, dst, method=0)
    if homography is None:
        raise SystemExit("Homography fit failed -- are the sample points collinear?")

    projected = cv2.perspectiveTransform(src.reshape(-1, 1, 2), homography).reshape(-1, 2)
    residuals = np.linalg.norm(projected - dst, axis=1)
    for sample, residual in zip(samples, residuals, strict=True):
        print(f"  {sample['label']:>18s}: residual {residual * 1000:5.1f}mm")
    print(f"  max {residuals.max() * 1000:.1f}mm, mean {residuals.mean() * 1000:.1f}mm")
    if residuals.max() > 0.01:
        print("  WARNING: >10mm residual -- recheck those samples or spread them out more.")
    return homography


def cmd_fit(_args) -> None:
    samples = _load_state()["samples"]
    print(f"fitting from {len(samples)} samples")
    homography = fit_pixel_to_robot(samples)
    # z varies across the board (the table is not perfectly level), so fit a
    # plane z = a*x + b*y + c over the taught grasps rather than averaging.
    xy = np.array([s["robot_xyz_m"][:2] for s in samples], dtype=np.float64)
    z_values = np.array([s["robot_xyz_m"][2] for s in samples], dtype=np.float64)
    design = np.column_stack([xy, np.ones(len(samples))])
    z_plane, *_ = np.linalg.lstsq(design, z_values, rcond=None)
    z_resid = np.abs(design @ z_plane - z_values)
    print(f"  z-plane residual: max {z_resid.max() * 1000:.1f}mm, mean {z_resid.mean() * 1000:.1f}mm")

    out = lerobot_root() / FIT_PATH
    out.write_text(
        json.dumps(
            {
                "homography_pixel_to_robot_xy_m": homography.tolist(),
                "grasp_z_plane_abc": z_plane.tolist(),
                "grasp_z_mean_m": float(np.mean(z_values)),
                "grasp_z_min_m": float(np.min(z_values)),
                "grasp_z_max_m": float(np.max(z_values)),
                "sample_count": len(samples),
                "note": (
                    "Learned from leader-arm demonstrated grasps, so block-height parallax "
                    "is already baked in. Maps top-camera pixel -> robot xy (m). Re-collect "
                    "if the camera, table, or robot base moves."
                ),
            },
            indent=2,
        )
    )
    print(f"[OK] wrote {out}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["snapshot", "teach", "fit"])
    parser.add_argument("--color")
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--handjog", action="store_true", help="hand-move the follower instead of leader teleop")
    parser.add_argument("--follower_port", default="/dev/so101_follower")
    parser.add_argument("--leader_port", default="/dev/so101_leader")
    parser.add_argument("--yolo_model_path", default="project/models/yolo_block_detector/best.pt")
    parser.add_argument("--conf", type=float, default=0.5)
    args = parser.parse_args()

    if args.command == "teach" and not args.color:
        raise SystemExit("teach needs --color")
    {"snapshot": cmd_snapshot, "teach": cmd_teach, "fit": cmd_fit}[args.command](args)


if __name__ == "__main__":
    main()
