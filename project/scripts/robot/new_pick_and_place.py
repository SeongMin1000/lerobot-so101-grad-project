#!/usr/bin/env python3
"""Ground-up rebuild of the hardcoded YOLO pick-and-place motion pipeline.

Written independently -- NOT based on pick_and_place_yolo_angle.py, per
explicit instruction. It reuses only pure infrastructure that isn't itself
"the hardcoded grasp motion logic" under suspicion:
  - table_ik.py: generic IK solver wrapper around the URDF (no hand-tuned
    pick-and-place behavior in it).
  - table_robot_calibration.json: the table<->robot affine, RECALIBRATED
    2026-08-27 from tcp_offset_probe.py's physical touch-point measurements
    (see YOLO_하드코딩_개발_이슈정리.md) -- this absorbed the real TCP/
    orientation error empirically.
  - GraspPitchModel: reach-vs-tilt fit from taught demonstrations, not
    something invented by trial-and-error hardcoding.
  - pixel_to_table.py: pure pixel->table-cm homography.

UPDATED 2026-08-28: position+color detection now comes from a YOLOv8-OBB
checkpoint (trained today, see YOLO_하드코딩_개발_이슈정리.md) instead of the
original position-only YOLO plus a separate OpenCV Otsu-threshold angle step
-- that two-stage approach turned out unreliable (checkerboard/background
confusion). See ObbBlockDetector below.

All motion sequencing (hover/approach/grasp/retract/drop), safety checks,
and block-angle detection below are new code, written today.

SAFETY, carried over from what tcp_offset_probe.py already validated live on
this robot:
  - checked_solve(): refuses to move on an IK solution that didn't converge.
  - safe_descend(): approaches the table in small (5mm) increments, never
    past a floor clamp derived from the calibration's own measured points.
  - A tracking-error watchdog tight enough to catch a real stall (not the
    very loose values used elsewhere).
  - P_Coefficient raised to 32 on the arm joints -- the historically
    validated fix for gravity droop (see issues log, "servo P게인 32로 상향").

This has NOT been run yet. Read through it, then run with --dry-run first
(no robot connection, no movement), then with --max-blocks 1 before trusting
it with the full run.

Usage:
    python project/scripts/robot/new_pick_and_place.py --dry-run
    python project/scripts/robot/new_pick_and_place.py --max-blocks 1
    python project/scripts/robot/new_pick_and_place.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import cv2
import numpy as np

from pathlib import Path

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.grad_project.control.grasp_pitch_model import GraspPitchModel, orientation_from_tilt
from lerobot.grad_project.control.table_frame_calibration import apply_table_to_robot, load_calibration
from lerobot.grad_project.control.table_ik import load_kinematics, solve_to_position
from lerobot.grad_project.paths import lerobot_root
from lerobot.grad_project.perception.opencv_block_detector import _point_in_polygon
from lerobot.grad_project.perception.pixel_to_table import load_homography, pixel_to_table_xy
from lerobot.grad_project.perception.yolo_block_detector import _as_polygon, _shrink_polygon
from lerobot.robots import make_robot_from_config, so_follower  # noqa: F401
from lerobot.robots.so_follower import SO101FollowerConfig

JOINT_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ARM_JOINTS = JOINT_ORDER[:-1]
GRIPPER_IDX = JOINT_ORDER.index("gripper")

TOP_CAM_KEY = "top"
# Trained 2026-08-28 (Colab, yolov8n-obb.pt fine-tuned on ~200 hand-labeled
# frames -- see YOLO_하드코딩_개발_이슈정리.md). Copy the downloaded best.pt
# here before running for real -- NOT present on disk yet as of this commit.
OBB_MODEL_PATH = "project/models/yolo_block_detector_obb/best.pt"
DETECTOR_CONFIG_PATH = "project/config/detector.json"
CAMERA_CALIBRATION_PATH = "project/config/camera_calibration.json"
RUNTIME_CONFIG_PATH = "project/config/runtime.json"

# UNVERIFIED on the real robot yet. The recalibrated table->robot affine
# (2026-08-27) has a negative-determinant linear part -- i.e. table/camera
# frame and robot frame are mirror images of each other, not just rotated.
# That means a detected image-angle likely needs its SIGN flipped (not just
# an additive offset) to become the correct wrist_roll direction. Default
# guess is "flip"; confirm with one real block of known rotation before
# trusting this, and flip to +1.0 if the wrist turns the wrong way.
ANGLE_SIGN = -1.0

RAMP_STEPS = 200
FPS = 30.0
HOVER_M = 0.08  # height above the grasp point to fly to before descending
MAX_STEP_M = 0.005  # max per-increment descent, matches the validated probe
FLOOR_MARGIN_M = 0.003
MAX_IK_ERR_M = 0.005

GRIPPER_OPEN_POS = 70.0
GRIPPER_CLOSED_POS = 0.0
GRIP_SUCCESS_MIN_POS = 10.0  # gripper stopped before fully closed => holding something

ARM_P_GAIN = 32
TRACKING_ERROR_DEG = 15.0
TRACKING_GRACE_STEPS = 2

MIN_REACH_M = 0.13
MAX_REACH_M = 0.45

MAX_ATTEMPTS = 15

# Drop slots: 5 evenly spaced points along the target zone's width, near its
# far edge from the tl origin. Simple fixed grid, independent of any prior
# drop-slot code -- refine later if blocks land too close together.
DROP_SLOTS_TABLE_CM = [(3.0, 8.0), (7.0, 8.0), (10.0, 8.0), (13.0, 8.0), (17.0, 8.0)]


class IKDivergedError(RuntimeError):
    """Raised when solve_to_position did not converge -- never move on a bad solution."""


def checked_solve(kin, current: np.ndarray, target_xyz: np.ndarray, **kwargs) -> tuple[np.ndarray, float]:
    solved, err = solve_to_position(kin, current, target_xyz, max_iters=kwargs.pop("max_iters", 60), **kwargs)
    if err > MAX_IK_ERR_M:
        raise IKDivergedError(
            f"IK did not converge: target={np.round(target_xyz, 4)} err={err * 1000:.2f}mm "
            f"(limit {MAX_IK_ERR_M * 1000:.0f}mm)."
        )
    return solved, err


def ramp_to(robot, current: np.ndarray, target_joints: np.ndarray, steps: int, dry_run: bool) -> np.ndarray:
    for step in range(1, steps + 1):
        alpha = step / steps
        action = {
            f"{n}.pos": float((1 - alpha) * current[i] + alpha * target_joints[i])
            for i, n in enumerate(JOINT_ORDER)
        }
        if not dry_run:
            robot.send_action(action)
            time.sleep(1.0 / FPS)
    return target_joints


def safe_descend(
    kin, robot, current: np.ndarray, xy_xyz: np.ndarray, floor_z: float, orientation: np.ndarray, dry_run: bool
) -> np.ndarray:
    """Descend to xy_xyz in small increments, never past floor_z. Mirrors the
    protocol validated live by tcp_offset_probe.py earlier today."""
    start_z = float(kin.forward_kinematics(current)[2, 3])
    target_z = max(float(xy_xyz[2]), floor_z)
    if xy_xyz[2] < floor_z:
        print(f"    !! target z {xy_xyz[2] * 1000:.1f}mm below floor clamp {floor_z * 1000:.1f}mm, capping.")
    n_steps = max(1, int(abs(start_z - target_z) / MAX_STEP_M) + 1)
    for i in range(1, n_steps + 1):
        z = start_z + (target_z - start_z) * (i / n_steps)
        waypoint = np.array([xy_xyz[0], xy_xyz[1], z])
        solved, _ = checked_solve(kin, current, waypoint, keep_orientation=orientation)
        current = ramp_to(robot, current, solved, steps=40, dry_run=dry_run)
    return current


class BlockDetection:
    __slots__ = ("color", "cx", "cy", "angle_deg", "conf", "in_target")

    def __init__(self, color, cx, cy, angle_deg, conf, in_target):
        self.color, self.cx, self.cy = color, cx, cy
        self.angle_deg, self.conf, self.in_target = angle_deg, conf, in_target


class ObbBlockDetector:
    """Wraps the OBB checkpoint trained 2026-08-28 (position + rotation angle
    in one model) -- replaces the old position-only YOLO detector plus a
    separate OpenCV angle-from-contour step. That two-stage approach turned
    out unreliable: plain Otsu threshold on a tight crop kept confusing the
    checkerboard background with the block itself (see issues log, same date),
    which is exactly why an OBB model trained on real photos was worth the
    relabeling effort instead of patching the OpenCV heuristic further.
    """

    def __init__(self, model_path: str, detector_config_path: str, conf: float = 0.5):
        from ultralytics import YOLO  # local import: only needed at runtime, not for --dry-run-less tooling

        root = lerobot_root()
        resolved_model = model_path if Path(model_path).is_absolute() else root / model_path
        if not Path(resolved_model).is_file():
            raise FileNotFoundError(
                f"OBB weights not found: {resolved_model}. Copy the best.pt trained in Colab there first."
            )
        self.model = YOLO(str(resolved_model))
        self.conf = conf

        cfg_path = Path(detector_config_path)
        cfg_path = cfg_path if cfg_path.is_absolute() else root / cfg_path
        cfg = json.loads(cfg_path.read_text())
        target_polygon = _as_polygon(cfg.get("target_polygon"))
        # Same 9px inset rationale as the old detector: a block straddling the
        # boundary reads as "already placed" on a plain point-in-polygon test.
        self._inset_polygon = _shrink_polygon(target_polygon, float(cfg.get("target_inset_px", 9.0)))

    def detect(self, frame_rgb: np.ndarray) -> list[BlockDetection]:
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        results = self.model.predict(source=frame_bgr, conf=self.conf, verbose=False)
        result = results[0]
        obb = result.obb
        if obb is None or len(obb) == 0:
            return []
        names = result.names

        best_per_color: dict[str, BlockDetection] = {}
        for i in range(len(obb)):
            color = names[int(obb.cls[i].item())]
            conf = float(obb.conf[i].item())
            if color in best_per_color and best_per_color[color].conf >= conf:
                continue
            cx, cy, _w, _h, r = obb.xywhr[i].tolist()
            angle_deg = np.degrees(r)
            angle_deg = ((angle_deg + 45.0) % 90.0) - 45.0  # square block, 4-fold symmetry
            in_target = self._inset_polygon is not None and _point_in_polygon(cx, cy, self._inset_polygon)
            best_per_color[color] = BlockDetection(color, cx, cy, angle_deg, conf, in_target)

        return list(best_per_color.values())


def rotate_about_local_z(orientation: np.ndarray, angle_rad: float) -> np.ndarray:
    """Spin the gripper's jaw-closing direction by `angle_rad` around the tool's
    own approach axis, without changing the approach/tilt direction itself."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return orientation @ rz


def load_observe_pose() -> np.ndarray:
    runtime = json.loads((lerobot_root() / RUNTIME_CONFIG_PATH).read_text())
    pose = runtime["poses"]["observe"]
    return np.array([pose[f"{n}.pos"] for n in JOINT_ORDER])


def floor_z_from_calibration(calib_path: str) -> float:
    data = json.loads((lerobot_root() / calib_path).read_text())
    zs = [p["robot_xyz_m"][2] for p in data["source_points"]]
    return min(zs) - FLOOR_MARGIN_M


def set_arm_p_gain(robot, gain: int) -> None:
    for motor in ARM_JOINTS:
        try:
            robot.bus.write("P_Coefficient", motor, gain)
        except Exception as e:  # noqa: BLE001
            print(f"!! failed to set P_Coefficient on {motor}: {e}")


def pick_one_block(
    kin, robot, detector, homography, calib, pitch_model,
    current: np.ndarray, floor_z: float, dry_run: bool,
    unreachable: set[str], placed_count: int,
) -> tuple[np.ndarray, str, str | None]:
    """Detect, pick, and drop one block.

    Returns (updated joint state, outcome, block_color), outcome in:
      "empty"        -- nothing left outside the target zone, caller should stop
      "unreachable"  -- block_color can't be reached, caller should add it to `unreachable`
      "missed"       -- grasp attempt failed, block was not moved
      "diverged"     -- IK failed mid-sequence, aborted safely
      "placed"       -- successfully picked and dropped
    block_color is None only for "empty".
    """
    frame = robot.get_observation()[TOP_CAM_KEY] if not dry_run else np.zeros((480, 640, 3), dtype=np.uint8)
    detections = detector.detect(frame)
    in_target_count = sum(1 for b in detections if b.in_target)

    candidates = [b for b in detections if not b.in_target and b.color not in unreachable]
    if not candidates:
        print(f"nothing left to pick (target_count={in_target_count}, unreachable={sorted(unreachable)})")
        return current, "empty", None
    block = candidates[0]

    table_x_cm, table_y_cm = pixel_to_table_xy(block.cx, block.cy, homography)
    target_xyz = apply_table_to_robot(table_x_cm, table_y_cm, calib)
    reach = float(np.linalg.norm(target_xyz[:2]))

    print(
        f"picking {block.color} at pixel({block.cx:.0f},{block.cy:.0f}) -> "
        f"table({table_x_cm:.1f},{table_y_cm:.1f})cm -> robot{np.round(target_xyz, 4)} reach={reach*100:.1f}cm"
    )

    if not (MIN_REACH_M <= reach <= MAX_REACH_M):
        print(f"  unreachable: reach {reach*100:.1f}cm outside working range [{MIN_REACH_M*100:.0f},{MAX_REACH_M*100:.0f}]cm")
        return current, "unreachable", block.color

    tilt_deg = pitch_model.tilt_for(reach) if pitch_model else 0.0
    base_orientation = orientation_from_tilt(target_xyz, tilt_deg)
    # ANGLE_SIGN: see the big comment at its definition -- unverified sign flip
    # for the topcam/robot mirror relationship found in today's recalibration.
    orientation = rotate_about_local_z(base_orientation, np.deg2rad(ANGLE_SIGN * block.angle_deg))
    print(f"  tilt={tilt_deg:.1f}deg block_angle={block.angle_deg:+.1f}deg (image) -> "
          f"wrist_roll uses {ANGLE_SIGN * block.angle_deg:+.1f}deg")

    hover_xyz = target_xyz + np.array([0.0, 0.0, HOVER_M])

    try:
        solved_hover, _ = checked_solve(kin, current, hover_xyz, keep_orientation=orientation)
    except IKDivergedError as e:
        print(f"  unreachable: hover point doesn't converge: {e}")
        return current, "unreachable", block.color

    try:
        solved_hover[GRIPPER_IDX] = GRIPPER_OPEN_POS
        current = ramp_to(robot, current, solved_hover, RAMP_STEPS, dry_run)

        current = safe_descend(kin, robot, current, target_xyz, floor_z, orientation, dry_run)

        closed = current.copy()
        closed[GRIPPER_IDX] = GRIPPER_CLOSED_POS
        current = ramp_to(robot, current, closed, 60, dry_run)

        held = True if dry_run else float(robot.get_observation()["gripper.pos"]) > GRIP_SUCCESS_MIN_POS
        print(f"  grasp {'held' if held else 'MISSED'}")

        solved_hover_back, _ = checked_solve(kin, current, hover_xyz, keep_orientation=orientation)
        solved_hover_back[GRIPPER_IDX] = current[GRIPPER_IDX]
        current = ramp_to(robot, current, solved_hover_back, RAMP_STEPS, dry_run)

        if not held:
            reopened = current.copy()
            reopened[GRIPPER_IDX] = GRIPPER_OPEN_POS
            current = ramp_to(robot, current, reopened, 60, dry_run)
            return current, "missed", block.color

        drop_x_cm, drop_y_cm = DROP_SLOTS_TABLE_CM[placed_count % len(DROP_SLOTS_TABLE_CM)]
        drop_xyz = apply_table_to_robot(drop_x_cm, drop_y_cm, calib)
        drop_tilt = pitch_model.tilt_for(float(np.linalg.norm(drop_xyz[:2]))) if pitch_model else 0.0
        drop_orientation = orientation_from_tilt(drop_xyz, drop_tilt)
        drop_hover = drop_xyz + np.array([0.0, 0.0, HOVER_M])

        solved_drop_hover, _ = checked_solve(kin, current, drop_hover, keep_orientation=drop_orientation)
        solved_drop_hover[GRIPPER_IDX] = GRIPPER_CLOSED_POS
        current = ramp_to(robot, current, solved_drop_hover, RAMP_STEPS, dry_run)

        current = safe_descend(kin, robot, current, drop_xyz, floor_z, drop_orientation, dry_run)

        opened = current.copy()
        opened[GRIPPER_IDX] = GRIPPER_OPEN_POS
        current = ramp_to(robot, current, opened, 60, dry_run)

        solved_drop_hover_back, _ = checked_solve(kin, current, drop_hover, keep_orientation=drop_orientation)
        current = ramp_to(robot, current, solved_drop_hover_back, RAMP_STEPS, dry_run)

    except IKDivergedError as e:
        print(f"  !! IK diverged mid-sequence, aborting this block: {e}")
        return current, "diverged", block.color

    return current, "placed", block.color


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-blocks", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true", help="No robot connection, no movement.")
    args = parser.parse_args()

    kin = load_kinematics()
    calib_path = "project/config/table_robot_calibration.json"
    calib = load_calibration(calib_path)
    homography = load_homography(CAMERA_CALIBRATION_PATH)
    pitch_model = GraspPitchModel.load()
    floor_z = floor_z_from_calibration(calib_path)
    print(f"floor_z={floor_z*1000:.1f}mm  GraspPitchModel={'loaded' if pitch_model else 'MISSING (tilt=0 fallback)'}")

    # ObbBlockDetector.detect() expects RGB frames (lerobot camera convention)
    # and converts to BGR internally for its own cv2 use.
    detector = ObbBlockDetector(OBB_MODEL_PATH, DETECTOR_CONFIG_PATH)

    if args.dry_run:
        print("=== DRY RUN: no robot connection, no movement ===")
        current = load_observe_pose()
        robot = None
    else:
        config = SO101FollowerConfig(
            port="/dev/so101_follower",
            id="follower",
            disable_torque_on_disconnect=False,
            max_relative_target=15.0,
            max_tracking_error=TRACKING_ERROR_DEG,
            tracking_error_grace_steps=TRACKING_GRACE_STEPS,
            # Same device path/resolution other scripts in this repo use (goto_table_point.py).
            # Double-check /dev/cam_top still resolves on this machine before trusting it.
            cameras={TOP_CAM_KEY: OpenCVCameraConfig(index_or_path="/dev/cam_top", width=640, height=480, fps=30)},
        )
        robot = make_robot_from_config(config)
        robot.connect()
        set_arm_p_gain(robot, ARM_P_GAIN)
        obs = robot.get_observation()
        current = np.array([float(obs[f"{n}.pos"]) for n in JOINT_ORDER])

    observe_pose = load_observe_pose()
    unreachable: set[str] = set()
    miss_counts: dict[str, int] = {}
    max_misses_before_giving_up = 2
    placed_count = 0

    try:
        for attempt in range(MAX_ATTEMPTS):
            if placed_count >= args.max_blocks:
                print(f"\n{placed_count} placed, reached --max-blocks={args.max_blocks}. Stopping.")
                break

            print(f"\n--- attempt {attempt + 1}/{MAX_ATTEMPTS} (placed {placed_count}/{args.max_blocks}) ---")
            # observe_pose is already a joint vector (from runtime.json), no IK needed.
            current = ramp_to(robot, current, observe_pose, RAMP_STEPS, args.dry_run)

            current, outcome, color = pick_one_block(
                kin, robot, detector, homography, calib, pitch_model,
                current, floor_z, args.dry_run, unreachable, placed_count,
            )

            if outcome == "empty":
                break
            if outcome == "unreachable":
                unreachable.add(color)
            elif outcome in ("missed", "diverged"):
                miss_counts[color] = miss_counts.get(color, 0) + 1
                if miss_counts[color] >= max_misses_before_giving_up:
                    print(f"  {color} missed {miss_counts[color]}x, giving up on it for this run.")
                    unreachable.add(color)
            elif outcome == "placed":
                placed_count += 1
    finally:
        if robot is not None:
            robot.disconnect()


if __name__ == "__main__":
    main()
