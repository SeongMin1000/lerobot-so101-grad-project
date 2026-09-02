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

# Constant wrist twist added on top of the block's own angle, in degrees.
# Set to 15.0 for one run on 2026-09-01 after the jaws sat ~15deg off on a
# block the detector read as unrotated -- then REVERTED to 0, because the very
# next observation was that the skew shows up on the FIRST attempt of a run and
# not on the ones after it, with the same block in the same place. A constant
# cannot do that; something that depends on where the IK solve started can.
# Leaving a constant in would have biased every attempt after the first.
# Keep this at 0 unless a skew is confirmed to be the same on every attempt.
WRIST_OFFSET_DEG = 0.0

# Interpolation steps for a full pose-to-pose move, sent at FPS.
# 200 (6.7s per move) was needlessly slow: the largest joint travel in a pick is
# shoulder_lift's ~60-93deg lift, which at 200 steps crawls at 9-14 deg/s while
# max_relative_target would allow 450. 140 puts it at 13-20 deg/s -- still ~20x
# under that ceiling, and the arm is watched by the tracking watchdog either way.
# Do NOT raise further while shoulder_lift is torque-limited: going faster costs
# more torque, and that motor is already the one that trips the watchdog.
RAMP_STEPS = 140

# Interpolation steps for ONE 5mm descent increment. Was 40 (1.3s per 5mm, so a
# 5cm descent took ~17s). 5mm of tool travel is only a couple of degrees at the
# joints, so 40 interpolation points for it was far finer than anything needed;
# 15 keeps the descent at a deliberate ~10mm/s while cutting the descent to ~7s.
# The safety property here is MAX_STEP_M -- the arm still re-solves IK and
# re-checks the floor clamp every 5mm -- not how finely each 5mm is subdivided.
DESCENT_RAMP_STEPS = 15

# The last stretch above the target descends at the original, slower rate
# (40 steps per 5mm, ~3.5mm/s). This is the only part of the whole sequence
# where the tool is about to touch something, and impact energy scales with the
# square of speed -- so 3x slower here is ~9x gentler on a contact, at a cost of
# about 3 extra seconds per pick. Everything above this band is free air.
SLOW_APPROACH_M = 0.015
DESCENT_RAMP_STEPS_SLOW = 40
FPS = 30.0
# Height above the grasp point to fly to before descending. Was 0.08 until
# 2026-09-01, when the first real --max-blocks 1 run refused to move at all:
# the hover solve came back at 6.39mm against the 5mm gate, and all six retry
# seeds failed too. Measured afterwards with no robot (placo only), sweeping
# hover height at the failing block and four other table points:
#
#     height  |  0cm   2cm   4cm   5cm   6cm   8cm
#     worst   |  0.2   0.2   1.5   3.0   4.6   8.0  mm
#
# The DESCENT target itself converges to 0.2mm everywhere -- only the hover was
# ever infeasible. Holding an 18-22deg wrist tilt that high stretches a 5-DOF
# arm past what it can do, and it gets worse with reach and tilt.
#
# 0.05 clears every tested point with the worst case at 3.0mm (60% of the gate),
# and it is the value tcp_offset_probe.py has been using on this exact robot all
# along -- already validated live across its 5 touch points. TRADEOFF: transit
# clearance between the pick and drop hovers drops from ~5.5cm to ~2.5cm over a
# ~2.5cm block. Still clears, and path_lowest_z() guards the ramp itself, but do
# not lower this further without re-checking that.
HOVER_M = 0.05
MAX_STEP_M = 0.005  # max per-increment descent, matches the validated probe
FLOOR_MARGIN_M = 0.003
MAX_IK_ERR_M = 0.005

# This arm has 5 joints that affect the TCP (gripper opens the jaw on a separate
# URDF branch and doesn't move gripper_frame_link at all), so a full 6-DOF pose
# -- 3 position + 3 orientation -- is NOT generally achievable. That is very
# likely WHY table_ik.py defaults orientation_weight to 0.01 against
# position_weight 1.0: orientation is deliberately the soft constraint. So do
# NOT raise IK_ORIENTATION_WEIGHT without measuring first (ik_orientation_check.py
# does exactly that, with no robot) -- trading position accuracy away to chase an
# orientation the arm can't reach anyway would make things worse, not better.
IK_ORIENTATION_WEIGHT = 0.01  # unchanged from table_ik.py's default, on purpose
# Set from ik_orientation_check.py's actual output (2026-09-01, weight 0.01):
# every solve that converged properly landed within 0.4deg of the requested
# orientation, while the ones that failed were off by 39.8deg. There is no grey
# band between those, so 10deg both clears every good solve by 25x and rejects
# every observed bad one decisively.
MAX_IK_ORIENT_ERR_DEG = 10.0

GRIPPER_OPEN_POS = 70.0
GRIPPER_CLOSED_POS = 0.0
GRIP_SUCCESS_MIN_POS = 10.0  # gripper stopped before fully closed => holding something

ARM_P_GAIN = 32
TRACKING_ERROR_DEG = 15.0
TRACKING_GRACE_STEPS = 2

MIN_REACH_M = 0.13
# Restored to 0.45 on 2026-09-01 after being wrongly lowered to 0.36 the same
# day. The 34.6cm "measured limit" behind that change came from sweeping the
# workspace through IK -- but with a calibration whose far half had no training
# data at all, so the far targets it was rejecting were at made-up coordinates.
# The demo dataset settles it: 423 of 1369 real teleoperated grasps (31%) were
# at reach over 36cm, out to 45.9cm. The arm reaches; the gate was refusing
# legitimate blocks, e.g. one at table y=+14.7 reported as reach 37.9cm.
MAX_REACH_M = 0.45

# Steepest lean allowed when searching for a reachable approach (see the tilt
# loop in pick_one_block). Past this the gripper is coming in almost sideways.
MAX_TILT_DEG = 75.0

# Generous bounding box around the known table workspace (see TABLE_CORNERS_CM).
# MIN/MAX_REACH alone can miss a wild homography extrapolation whose distance
# from the base happens to land in-range by coincidence even though the point
# itself is nowhere near the real table (see issues log 2026-09-01).
# Set from where blocks ACTUALLY were across the 1369 mined demo samples:
# x -25.9..51.4cm, y -29.8..22.2cm. The first values here were (-10,30) and
# (-10,20), guessed as a "generous box around the table" without checking, and
# they cut off 64% of the real working area -- a blue block at x=42.5 (reach
# 34.7cm, a perfectly ordinary target) was thrown out as a bad detection. Keep a
# small margin past the observed range; this gate exists only to catch homography
# extrapolation gone wild, not to define the workspace.
TABLE_X_RANGE_CM = (-30.0, 56.0)
TABLE_Y_RANGE_CM = (-34.0, 27.0)

MAX_ATTEMPTS = 15

# The task is not "pick the 5 blocks", it's "pick them IN THIS ORDER" -- it's
# spelled out in the demo dataset's own task string ("Pick up the 5 blocks in
# sequence (red, yellow, wood, green, blue), then place each at the target
# area."). Picking whatever the detector happened to list first satisfies the
# first half and fails the second. mine_calibration_from_demos.py already
# depended on this order to match grasp events to colors; this is the shared
# definition both use.
COLOR_SEQUENCE = ["red", "yellow", "wood", "green", "blue"]

# Drop slots inside the 20x10cm target zone, as two rows: 3 blocks across the
# row nearer the zone's y=0 edge (the top of the top-camera image), then 2
# staggered between them on the far row. Was a single line of 5 at y=8.0, which
# is not the layout the task asks for. Filled in list order, so the top row
# completes first. ~6cm apart against a ~2.5cm block leaves room for placement
# error without blocks touching.
DROP_SLOTS_TABLE_CM = [
    (4.0, 3.0), (10.0, 3.0), (16.0, 3.0),   # top row (nearer y=0 edge)
    (7.0, 7.5), (13.0, 7.5),                # bottom row, staggered
]

# Release the block this far above the computed table height instead of driving
# all the way down to it. safe_descend() has no contact sensing on the way down
# -- close_gripper_until_resistance() covers the grasp, but nothing covered the
# placement -- so once the block bottomed out on the table the arm just kept
# working through its remaining 5mm steps, and with nowhere left to go that came
# out as the tool creeping forward against the table. Letting go a few
# millimetres up costs nothing (the block just settles) and removes the push.
DROP_CLEARANCE_M = 0.012

# How far to slide the grasp point along the gripper's own jaw axis, in metres.
# Positive moves toward the jaw on the arm's LEFT (robot +y when the wrist is at
# its nominal angle, which is the top camera's RIGHT). Observed live 2026-09-01:
# with the wrist angle itself correct, the left jaw kept landing on the block's
# edge rather than beside it, meaning the whole gripper needed to travel further
# that way for the block to sit between the jaws.
#
# 0.012 is a first estimate -- roughly half a block. Tune from one observation:
# if the same jaw still lands on the block, raise it; if the OTHER jaw now does,
# lower it. Because this rides the jaw axis it stays correct as the wrist turns,
# so unlike a robot-frame constant it should hold across the whole table -- if
# it does not, the problem is not a TCP offset and this should go back to 0.
# Raised 0.012 -> 0.018 after 0.012 left the grasp still marginal in the same
# direction ("아슬아슬", same jaw).

# The second tool-frame offset, along tool_x -- the horizontal axis at right
# angles to the jaws. Measured (and identical at the left, centre and right of
# the table, which is the point of doing this in the tool frame): +1cm of tool_x
# moves the grip 1.08cm toward the TOP of the top-camera image -- and the camera
# sits on the far side of the table from the arm, so image-top is the side
# NEARER the robot. The gripper was closing on the block's far edge (the one
# nearer the camera, which reads as the BOTTOM of the image), so it has to come
# back toward the robot -- positive. About half a block as a first estimate;
# tune the same way as the jaw offset.
#
# Sign in plain terms, to avoid the top/bottom confusion that got this backwards
# once already: POSITIVE moves the grip toward the robot, NEGATIVE away from it.
TCP_JAW_OFFSET_M = 0.012
TCP_FORWARD_OFFSET_M = 0.010


class IKDivergedError(RuntimeError):
    """Raised when solve_to_position did not converge -- never move on a bad solution."""


def orientation_error_deg(kin, solved: np.ndarray, requested: np.ndarray) -> float:
    """Geodesic angle (deg) between the orientation we ASKED for and the one the
    solved joints actually produce. solve_to_position's own returned error is
    POSITION ONLY (see table_ik.py: err = norm(fk[:3,3] - target)), so without
    this a solution whose wrist points somewhere else entirely still reports as
    converged -- which is exactly what makes an ANGLE_SIGN test meaningless if
    it isn't measured (see issues log 2026-09-01)."""
    achieved = kin.forward_kinematics(solved)[:3, :3]
    r_err = requested.T @ achieved
    cos = (np.trace(r_err) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def path_lowest_z(kin, start_joints: np.ndarray, end_joints: np.ndarray, samples: int = 25) -> float:
    """Lowest TCP height along the joint-space straight line ramp_to will actually
    execute between these two poses.

    Nothing checked the TRANSIT path before this (see issues log, open item 7):
    safe_descend clamps the descent, but a ramp between two perfectly safe end
    poses can still dip through the table in the middle, because interpolating
    joints linearly does NOT move the tool in a straight line. That matters most
    for a solution found from a PERTURBED seed, which can come back in a
    different arm configuration (elbow up vs down) far from where we are now."""
    lowest = float("inf")
    for i in range(samples + 1):
        alpha = i / samples
        q = (1.0 - alpha) * np.asarray(start_joints, dtype=float) + alpha * np.asarray(end_joints, dtype=float)
        lowest = min(lowest, float(kin.forward_kinematics(q)[2, 3]))
    return lowest


# This check is for catching a MID-ramp dip through the table, not for
# re-adjudicating the endpoint. safe_descend deliberately ends exactly at the
# floor clamp when a target is below it, so an exact >= floor_z comparison would
# reject its own last descent step on floating-point noise alone. The clamp
# already sits FLOOR_MARGIN_M above the lowest measured table point, so this
# tolerance still leaves the path above the real table.
PATH_FLOOR_TOLERANCE_M = 0.002


# Seeds to retry from when the solve from `current` fails. ik_orientation_check.py
# (2026-09-01) showed the failures it found were NOT physical limits -- the same
# target/orientation solved fine from a perturbed seed, i.e. the iterative solver
# had fallen into a local minimum. Without this, a block at ~28.8cm reach rotated
# 45deg is silently reported "unreachable" and skipped.
RETRY_SEED_OFFSETS_DEG = [
    [0.0, -25.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 25.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, -25.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 25.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, 45.0, 0.0],
    [0.0, 0.0, 0.0, 0.0, -45.0, 0.0],
]


def _solve_and_grade(kin, seed, target_xyz, kwargs, requested_orientation):
    """Returns (solved, pos_err, ori_err). ori_err is 0.0 when no orientation was requested."""
    solved, err = solve_to_position(kin, seed, target_xyz, **kwargs)
    ori_err = orientation_error_deg(kin, solved, requested_orientation) if requested_orientation is not None else 0.0
    return solved, err, ori_err


def checked_solve(
    kin, current: np.ndarray, target_xyz: np.ndarray, floor_z: float | None = None,
    seed: np.ndarray | None = None, **kwargs
) -> tuple[np.ndarray, float]:
    """Solve, and refuse to hand back anything that fails position, orientation,
    or (when floor_z is given) transit-path safety. A retry from a different seed
    must clear the SAME three gates -- retrying only ever turns a refusal into a
    verified-good solution, never into a looser one."""
    requested_orientation = kwargs.get("keep_orientation")
    kwargs.setdefault("orientation_weight", IK_ORIENTATION_WEIGHT)
    kwargs.setdefault("max_iters", 60)

    # `seed` is only where the solver STARTS from; `current` stays the arm's real
    # pose and is what the transit-path check measures from. Splitting them lets
    # a caller ask for a repeatable solve: solve_to_position iterates 60 steps
    # from its seed with orientation as a soft constraint, so wrist_roll -- the
    # least constrained joint -- lands somewhere different for seeds that differ
    # by a fraction of a degree. Seeding from the MEASURED pose therefore made
    # the first pick of a run come out visibly twisted compared with the ones
    # after it, because the arm reaches the observe pose from a different place
    # the first time (see issues log 2026-09-01).
    solve_from = current if seed is None else seed
    solved, err, ori_err = _solve_and_grade(kin, solve_from, target_xyz, kwargs, requested_orientation)
    ok_pos, ok_ori = err <= MAX_IK_ERR_M, ori_err <= MAX_IK_ORIENT_ERR_DEG
    ok_path = True
    if ok_pos and ok_ori and floor_z is not None:
        ok_path = path_lowest_z(kin, current, solved) >= floor_z - PATH_FLOOR_TOLERANCE_M

    # Only retry when floor_z was supplied. A retry deliberately starts from a
    # perturbed seed, so its solution can be a different arm configuration than
    # the one we're standing in -- and without floor_z there is no way to check
    # the arm doesn't sweep through the table reaching it. Refusing beats
    # guessing: with no floor_z, behave exactly as before.
    if not (ok_pos and ok_ori and ok_path) and floor_z is not None:
        for offset in RETRY_SEED_OFFSETS_DEG:
            retry_seed = np.asarray(solve_from, dtype=float) + np.asarray(offset, dtype=float)
            cand, cand_err, cand_ori = _solve_and_grade(kin, retry_seed, target_xyz, kwargs, requested_orientation)
            if cand_err > MAX_IK_ERR_M or cand_ori > MAX_IK_ORIENT_ERR_DEG:
                continue
            if floor_z is not None and path_lowest_z(kin, current, cand) < floor_z - PATH_FLOOR_TOLERANCE_M:
                # Solves the pose, but the arm would sweep through the table getting
                # there -- exactly the case this check exists for. Keep looking.
                continue
            print(f"    (IK retried from a perturbed seed: {err*1000:.1f}mm/{ori_err:.1f}deg "
                  f"-> {cand_err*1000:.1f}mm/{cand_ori:.1f}deg)")
            return cand, cand_err

    if err > MAX_IK_ERR_M:
        raise IKDivergedError(
            f"IK did not converge: target={np.round(target_xyz, 4)} err={err * 1000:.2f}mm "
            f"(limit {MAX_IK_ERR_M * 1000:.0f}mm), and no retry seed converged either."
        )
    if not ok_ori:
        raise IKDivergedError(
            f"IK hit the position but NOT the requested orientation: "
            f"orientation error {ori_err:.1f}deg (limit {MAX_IK_ORIENT_ERR_DEG:.0f}deg), "
            f"and no retry seed did better. The wrist would not actually be pointing "
            f"where this move assumes."
        )
    if not ok_path:
        raise IKDivergedError(
            f"IK solved the pose, but the ramp to it dips to "
            f"{path_lowest_z(kin, current, solved) * 1000:.1f}mm -- below the floor clamp "
            f"({floor_z * 1000:.1f}mm). The arm would sweep through the table on the way."
        )
    return solved, err


SETTLE_S = 0.15  # let the last commanded step actually arrive before reading back


def read_joints(robot) -> np.ndarray:
    """The ONE way to learn where the arm really is. Never infer the arm's
    position from what was last commanded -- a clamp (max_relative_target), a
    watchdog freeze, or a raised exception all leave the commanded value and
    the real value disagreeing, and every safety decision downstream (floor
    clamp, IK seed, emergency retract) is only as good as this number.

    Reads the motor bus directly rather than going through get_observation(),
    which also grabs a frame from every camera -- pure waste when all we want
    is joint angles, and it runs after every single ramp (11+ times per
    descent alone)."""
    pos = robot.bus.sync_read("Present_Position")
    return np.array([float(pos[n]) for n in JOINT_ORDER])


def halt_in_place(robot) -> None:
    """Command the arm to hold exactly where it physically is right now.

    Needed on Ctrl+C specifically. The tracking-error watchdog freezes the arm
    itself before raising, but Ctrl+C does not: the last send_action goal is
    still live in the servos, so the arm keeps driving toward it while the
    Python exception unwinds. Re-commanding the present position cancels that
    pending motion without cutting torque (which would just drop the arm)."""
    try:
        here = read_joints(robot)
        robot.send_action({f"{n}.pos": float(here[i]) for i, n in enumerate(JOINT_ORDER)})
    except Exception as e:  # noqa: BLE001
        print(f"!! could not halt the arm in place ({e}) -- cut power manually if it is still moving.")


def current_after_abort(robot, current: np.ndarray, dry_run: bool) -> np.ndarray:
    """Call this in EVERY except: block that can fire after the arm has already
    moved, before handing `current` back to anyone.

    The rule this enforces: when an exception interrupts a motion, the
    `current = ramp_to(...)` / `current = safe_descend(...)` assignment never
    happened, so the local `current` still holds the pose from BEFORE that
    motion while the arm is physically somewhere else. Whoever gets that stale
    value next ramps from a fake starting point, and since ramp_to interpolates
    from `current`, the very first commanded step jumps the whole difference --
    a max_relative_target-limited lurch at ~450deg/s. This has now bitten twice
    (the watchdog handler in main(), then the IKDivergedError handler in
    pick_one_block); see issues log 2026-09-01 C and D."""
    if dry_run:
        return current
    return read_joints(robot)


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
    if dry_run:
        return target_joints
    # Read the REAL position back instead of trusting target_joints -- if
    # max_relative_target (or anything else) clamped a step short, the next
    # IK call must seed from where the arm actually is, not where this
    # function assumed it ended up (see issues log 2026-09-01). Settle first,
    # or the readback catches the arm still mid-move.
    time.sleep(SETTLE_S)
    return read_joints(robot)


def safe_descend(
    kin, robot, current: np.ndarray, xy_xyz: np.ndarray, floor_z: float, orientation: np.ndarray, dry_run: bool,
    gripper_goal: float | None = None,
) -> np.ndarray:
    """Descend to xy_xyz in small increments, never past floor_z. Mirrors the
    protocol validated live by tcp_offset_probe.py earlier today."""
    start_z = float(kin.forward_kinematics(current)[2, 3])
    target_z = max(float(xy_xyz[2]), floor_z)
    if xy_xyz[2] < floor_z:
        print(f"    !! target z {xy_xyz[2] * 1000:.1f}mm below floor clamp {floor_z * 1000:.1f}mm, capping.")
    # solve_to_position's gripper output isn't guaranteed to preserve the input
    # gripper value (see issues log 2026-09-01) -- pin it to whatever it was
    # when descent started so the gripper doesn't drift open/closed on its own
    # over the course of the descent.
    # Default to whatever the gripper is doing now, but let a caller carrying a
    # block pass the squeezing goal instead -- current[] is a MEASURED value and
    # for a gripper stalled on a block that is not the same as its goal.
    gripper_hold = current[GRIPPER_IDX] if gripper_goal is None else gripper_goal
    # Chain the IK seed through the SOLVED poses, not the measured ones. The
    # measured pose differs run to run by fractions of a degree (droop, backlash,
    # where the arm came from), and with orientation as a soft constraint that is
    # enough to walk wrist_roll somewhere else over 10+ successive solves -- the
    # remaining half of "the angle is right sometimes and wrong other times".
    # `current` still carries the real pose, so the transit-path check and the
    # ramp itself are unaffected.
    seed = current
    n_steps = max(1, int(abs(start_z - target_z) / MAX_STEP_M) + 1)
    for i in range(1, n_steps + 1):
        z = start_z + (target_z - start_z) * (i / n_steps)
        waypoint = np.array([xy_xyz[0], xy_xyz[1], z])
        solved, _ = checked_solve(kin, current, waypoint, floor_z=floor_z, seed=seed,
                                  keep_orientation=orientation)
        seed = solved
        solved[GRIPPER_IDX] = gripper_hold
        # Two speeds, because the risk is not uniform down the descent. Well
        # above the target the gripper is in free air and there is nothing to
        # hit; the last centimetre or so is where it either touches the block,
        # or -- if the calibration is off -- the table. Collision energy goes
        # with the square of speed, so the approach is where slow actually buys
        # something, and the upper part is where slow only costs time.
        near_contact = abs(z - target_z) <= SLOW_APPROACH_M
        steps = DESCENT_RAMP_STEPS_SLOW if near_contact else DESCENT_RAMP_STEPS
        current = ramp_to(robot, current, solved, steps=steps, dry_run=dry_run)
    return current


GRIPPER_STEP_DEG = 4.0  # per-step close increment
GRIPPER_RESISTANCE_DEG = 8.0  # present_pos lagging the commanded step by this much = holding something
GRIPPER_SETTLE_S = 0.15
# How far INSIDE the block to keep commanding once resistance is felt. Setting
# the goal to the measured position (what this did before) leaves the servo with
# zero position error, so it produces no holding force at all -- the block is
# only resting between the jaws and slips out on the way to the drop. Keeping
# the goal a few degrees closed of where the jaws actually are gives a steady
# squeeze. 6deg stays well under the 15deg tracking-error threshold, so the
# watchdog does not read the deliberate, permanent lag as a stall.
GRIPPER_HOLD_SQUEEZE_DEG = 10.0


def close_gripper_until_resistance(robot, current: np.ndarray, dry_run: bool) -> tuple[np.ndarray, bool]:
    """Close the gripper toward GRIPPER_CLOSED_POS in small steps, and STOP as
    soon as resistance appears -- that IS a successful grasp, not a stall.

    Root-caused live today: the old code ramped the gripper all the way to
    GRIPPER_CLOSED_POS regardless of what was inside it. On every ACTUAL
    successful grasp, the gripper physically can't reach that fully-closed
    target (it's holding a block) -- present_pos falls further and further
    behind the commanded goal every step, and after 2 consecutive steps over
    the robot's global 15deg tracking-error threshold, the safety watchdog
    kills the whole script. So the bug was: the better the grasp, the more
    likely the run crashes. This checks for resistance itself, in small
    steps, and stops well before the global watchdog would ever trip.
    """
    if dry_run:
        current = current.copy()
        current[GRIPPER_IDX] = GRIPPER_CLOSED_POS
        return current, True

    current = current.copy()
    while current[GRIPPER_IDX] > GRIPPER_CLOSED_POS + GRIPPER_STEP_DEG:
        step_target = max(GRIPPER_CLOSED_POS, current[GRIPPER_IDX] - GRIPPER_STEP_DEG)
        current[GRIPPER_IDX] = step_target
        robot.send_action({f"{n}.pos": float(current[i]) for i, n in enumerate(JOINT_ORDER)})
        time.sleep(GRIPPER_SETTLE_S)
        present = float(robot.bus.sync_read("Present_Position")["gripper"])
        if present - step_target > GRIPPER_RESISTANCE_DEG:
            # Keep asking for slightly MORE closed than the jaws can reach, so the
            # servo keeps a real grip on the block instead of merely resting on it.
            current[GRIPPER_IDX] = max(GRIPPER_CLOSED_POS, present - GRIPPER_HOLD_SQUEEZE_DEG)
            return current, True

    # Reached fully closed with no resistance anywhere along the way -- nothing was grasped.
    present = float(robot.bus.sync_read("Present_Position")["gripper"])
    held = present > GRIP_SUCCESS_MIN_POS
    current[GRIPPER_IDX] = (max(GRIPPER_CLOSED_POS, present - GRIPPER_HOLD_SQUEEZE_DEG)
                            if held else present)
    return current, held


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


# Same 4 nominal table-cm corners as tcp_offset_probe.py's DEFAULT_POINTS_CM.
# floor_z MUST be derived by projecting these through the CURRENT calibration's
# affine, not by reading calib["source_points"] directly (2026-09-01 bug: after
# swapping to the demo-mined calibration, source_points became ~1369 raw grasp
# joint-states, some mid-air, not table touches -- min(z) over those is not a
# floor at all, and the safety clamp silently stopped doing anything).
TABLE_CORNERS_CM = [(0.0, 0.0), (20.0, 0.0), (20.0, 10.0), (0.0, 10.0)]


def floor_z_from_calibration(calib) -> float:
    zs = [apply_table_to_robot(x, y, calib)[2] for x, y in TABLE_CORNERS_CM]
    return min(zs) - FLOOR_MARGIN_M


def set_arm_p_gain(robot, gain: int) -> None:
    for motor in ARM_JOINTS:
        try:
            robot.bus.write("P_Coefficient", motor, gain)
        except Exception as e:  # noqa: BLE001
            print(f"!! failed to set P_Coefficient on {motor}: {e}")


# ARM_P_GAIN=32 makes the arm push harder against gravity droop -- but nothing
# in this script capped arm torque the way tcp_offset_probe.py does while
# validating a fresh calibration. Missing here entirely until 2026-09-01: this
# file is the one that actually drives the arm on every real run, so it needs
# this safety net at least as much as the one-off test tool did.
#
# Started at 20 (copied from tcp_offset_probe.py). The first real run never got
# past the very first move: ramping observe -> hover, shoulder_lift was
# commanded 31.3deg and physically managed 15.7deg -- half -- and the tracking
# watchdog correctly stopped everything at 15.6deg of lag.
#
# That is not a bad block position, it is the pose this arm starts every pick
# from. observe sits at shoulder_lift = -87.4deg, deeply folded, so reaching any
# hover means lifting the whole arm against gravity:
#
#     block reach 18.6cm -> +59.5deg     (the one that failed)
#     block reach 22.7cm -> +84.2deg
#     block reach 25.0cm -> +93.1deg
#
# 20% cannot do that lift. It is not a conservative setting, it is a
# non-functional one -- and a safety limit that blocks normal operation is one
# that eventually gets switched off entirely.
#
# 40 got the arm through one full lift but shoulder_lift still stalled on the
# next attempt (Feetech overload protection latches, so a struggle on one move
# leaves the motor capped for the ones after it). 50 gives more headroom while
# staying well under the servos' original 80.
#
# The real protection against driving into the table was never this number: it
# is the tracking watchdog (15deg / 2 steps), which has now stopped the arm
# correctly twice, plus the 5mm descent increments and the floor clamp. The
# gripper keeps its own separate 25% cap -- that is the one guarding the jaws
# that broke before. Raise this to 60 if shoulder_lift still lags; do NOT jump
# straight to 80.
ARM_OVERLOAD_TORQUE_PCT = 70


def set_arm_overload_torque(robot, pct: int, previous: dict[str, int]) -> None:
    """Fills caller-owned `previous` in as it goes (rather than building+returning
    its own dict) so a mid-loop exception still leaves `previous` usable for
    restore_arm_overload_torque -- tcp_offset_probe.py's original version lost the
    whole dict on such an exception, leaving already-lowered motors stuck at
    pct% until someone noticed (see issues log 2026-09-01)."""
    for motor in ARM_JOINTS:
        previous[motor] = robot.bus.read("Overload_Torque", motor)
        if previous[motor] == pct:
            # We deliberately leave the cap in place when a run exits with the
            # arm possibly jammed (see stalled_on_exit), printing the originals
            # so they can be restored by hand. If that message got missed, the
            # arm silently runs weak/droopy forever after -- and "previous" now
            # records the capped value as if it were the original, making the
            # next clean exit restore it to the cap too. Say so out loud.
            print(f"!! {motor} was ALREADY at Overload_Torque={pct}% -- a previous run probably "
                  f"exited with the arm stalled. Restore its real value before trusting this run.")
        robot.bus.write("Overload_Torque", motor, pct)


def restore_arm_overload_torque(robot, previous: dict[str, int]) -> None:
    for motor, val in previous.items():
        try:
            robot.bus.write("Overload_Torque", motor, val)
        except Exception as e:  # noqa: BLE001
            print(f"!! failed to restore Overload_Torque on {motor}: {e}")


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
    # Detection order is dictionary/confidence order and varies frame to frame;
    # the task requires COLOR_SEQUENCE order. Unknown colors sort last rather
    # than raising -- an unexpected class should not stop the run.
    candidates.sort(key=lambda b: COLOR_SEQUENCE.index(b.color) if b.color in COLOR_SEQUENCE else len(COLOR_SEQUENCE))
    block = candidates[0]

    table_x_cm, table_y_cm = pixel_to_table_xy(block.cx, block.cy, homography)
    if not (TABLE_X_RANGE_CM[0] <= table_x_cm <= TABLE_X_RANGE_CM[1]
            and TABLE_Y_RANGE_CM[0] <= table_y_cm <= TABLE_Y_RANGE_CM[1]):
        print(f"  unreachable: table coords ({table_x_cm:.1f},{table_y_cm:.1f})cm way outside "
              f"the known workspace -- treating as a bad detection, not moving toward it")
        return current, "unreachable", block.color
    target_xyz = apply_table_to_robot(table_x_cm, table_y_cm, calib)
    reach = float(np.linalg.norm(target_xyz[:2]))

    print(
        f"picking {block.color} at pixel({block.cx:.0f},{block.cy:.0f}) -> "
        f"table({table_x_cm:.1f},{table_y_cm:.1f})cm -> robot{np.round(target_xyz, 4)} reach={reach*100:.1f}cm"
    )

    if not (MIN_REACH_M <= reach <= MAX_REACH_M):
        print(f"  unreachable: reach {reach*100:.1f}cm outside working range [{MIN_REACH_M*100:.0f},{MAX_REACH_M*100:.0f}]cm")
        return current, "unreachable", block.color

    # GraspPitchModel under-predicts the lean the arm needs at long reach: it has
    # a single taught sample past 0.37m (0.43m -> 47deg) and interpolates linearly
    # into the gap, so at 0.39m it asks for ~33deg when the arm physically needs
    # 50deg or more. Measured with no robot at three far blocks: the model tilt
    # left 18-23mm of position error (refused), while 50deg solved to 0.1-0.4mm.
    # Near blocks are unaffected -- their first candidate is the model's own tilt
    # and it solves immediately. Leaning further only ever gets tried when the
    # arm could not otherwise reach at all.
    model_tilt = pitch_model.tilt_for(reach) if pitch_model else 0.0
    tilt_candidates = [model_tilt] + [t for t in (model_tilt + 10.0, model_tilt + 20.0,
                                                  model_tilt + 30.0, model_tilt + 40.0)
                                      if t <= MAX_TILT_DEG]

    solved_hover = solved_turn = None
    last_err: IKDivergedError | None = None
    for tilt_deg in tilt_candidates:
        base_orientation = orientation_from_tilt(target_xyz, tilt_deg)
        azimuth_deg = float(np.degrees(np.arctan2(target_xyz[1], target_xyz[0])))
        wrist_deg = ANGLE_SIGN * block.angle_deg + WRIST_OFFSET_DEG + azimuth_deg
        wrist_deg = ((wrist_deg + 45.0) % 90.0) - 45.0
        orientation = rotate_about_local_z(base_orientation, np.deg2rad(wrist_deg))
        grasp_xyz = (target_xyz + orientation[:, 1] * TCP_JAW_OFFSET_M
                     + orientation[:, 0] * TCP_FORWARD_OFFSET_M)
        hover_xyz = grasp_xyz + np.array([0.0, 0.0, HOVER_M])
        try:
            solved_hover, _ = checked_solve(kin, current, hover_xyz, floor_z=floor_z,
                                            seed=load_observe_pose(), keep_orientation=base_orientation)
            solved_turn, _ = checked_solve(kin, solved_hover, hover_xyz, floor_z=floor_z,
                                           seed=solved_hover, keep_orientation=orientation)
            break
        except IKDivergedError as e:
            last_err = e
            solved_hover = solved_turn = None

    try:
        if solved_hover is None:
            raise last_err if last_err is not None else IKDivergedError("no tilt candidate solved")
        if tilt_deg > model_tilt:
            print(f"  (leaned further to reach it: tilt {model_tilt:.1f} -> {tilt_deg:.1f}deg)")
        print(f"  tilt={tilt_deg:.1f}deg block_angle={block.angle_deg:+.1f}deg (image) -> "
              f"wrist {ANGLE_SIGN * block.angle_deg:+.1f} + azimuth {azimuth_deg:+.1f} = {wrist_deg:+.1f}deg")
        target_xyz = grasp_xyz
    except IKDivergedError as e:
        print(f"  unreachable: hover point doesn't converge: {e}")
        return current, "unreachable", block.color

    try:
        print("  moving to hover (wrist not yet turned)...")
        solved_hover[GRIPPER_IDX] = GRIPPER_OPEN_POS
        current = ramp_to(robot, current, solved_hover, RAMP_STEPS, dry_run)

        print(f"  turning wrist {wrist_deg:+.1f}deg in place -- watch which way it goes")
        solved_turn[GRIPPER_IDX] = GRIPPER_OPEN_POS
        current = ramp_to(robot, current, solved_turn, 60, dry_run)

        print("  descending to grasp...")
        current = safe_descend(kin, robot, current, target_xyz, floor_z, orientation, dry_run)

        current, held = close_gripper_until_resistance(robot, current, dry_run)
        # Remember the GOAL. ramp_to() hands back measured joints, and a gripper
        # stalled on a block reads back at the block's width -- so re-deriving the
        # gripper target from current[] on each later move commanded it to exactly
        # where it already was, zero position error, zero holding force. The grip
        # was silently released on the first move after the grasp and the block
        # was only resting between the jaws for the whole carry (see issues log
        # 2026-09-01, "제대로 잡아도 계속 놓침").
        grip_goal = float(current[GRIPPER_IDX])
        print(f"  grasp {'held' if held else 'MISSED'}")

        print("  lifting back to hover...")
        solved_hover_back, _ = checked_solve(kin, current, hover_xyz, floor_z=floor_z, keep_orientation=orientation)
        solved_hover_back[GRIPPER_IDX] = grip_goal
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

        print(f"  carrying to drop slot ({drop_x_cm:.0f},{drop_y_cm:.0f})cm...")
        solved_drop_hover, _ = checked_solve(kin, current, drop_hover, floor_z=floor_z, keep_orientation=drop_orientation)
        solved_drop_hover[GRIPPER_IDX] = grip_goal  # keep squeezing; don't force-close to 0
        current = ramp_to(robot, current, solved_drop_hover, RAMP_STEPS, dry_run)

        print("  descending to place...")
        place_xyz = drop_xyz + np.array([0.0, 0.0, DROP_CLEARANCE_M])
        current = safe_descend(kin, robot, current, place_xyz, floor_z, drop_orientation, dry_run,
                               gripper_goal=grip_goal)

        print("  releasing...")
        opened = current.copy()
        opened[GRIPPER_IDX] = GRIPPER_OPEN_POS
        current = ramp_to(robot, current, opened, 60, dry_run)

        print("  retreating...")
        solved_drop_hover_back, _ = checked_solve(kin, current, drop_hover, floor_z=floor_z, keep_orientation=drop_orientation)
        solved_drop_hover_back[GRIPPER_IDX] = current[GRIPPER_IDX]
        current = ramp_to(robot, current, solved_drop_hover_back, RAMP_STEPS, dry_run)

    except IKDivergedError as e:
        print(f"  !! IK diverged mid-sequence, aborting this block: {e}")
        # The arm has already moved by this point (hover, and possibly part of
        # the descent), so `current` is stale -- see current_after_abort.
        return current_after_abort(robot, current, dry_run), "diverged", block.color
    except RuntimeError as e:
        # Not IKDivergedError -- almost certainly the tracking-error watchdog
        # tripping mid-motion. Don't swallow this and try the next block --
        # propagate so main() stops the whole run and asks a human what to do
        # (see main()'s own RuntimeError handler for why: never touch torque
        # automatically here, the arm could be mid-air holding a block).
        print(f"  !! safety watchdog tripped mid-sequence: {e}")
        raise

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
    floor_z = floor_z_from_calibration(calib)
    print(f"floor_z={floor_z*1000:.1f}mm  GraspPitchModel={'loaded' if pitch_model else 'MISSING (tilt=0 fallback)'}")

    # ObbBlockDetector.detect() expects RGB frames (lerobot camera convention)
    # and converts to BGR internally for its own cv2 use.
    detector = ObbBlockDetector(OBB_MODEL_PATH, DETECTOR_CONFIG_PATH)

    prev_arm_torque: dict[str, int] = {}

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
        try:
            set_arm_overload_torque(robot, ARM_OVERLOAD_TORQUE_PCT, prev_arm_torque)
            print(f"arm overload torque capped at {ARM_OVERLOAD_TORQUE_PCT}% (restored on exit)")
        except Exception as e:  # noqa: BLE001
            print(f"!! failed to cap arm overload torque ({e}) -- continuing without this safety net.")
        current = read_joints(robot)

    observe_pose = load_observe_pose()
    unreachable: set[str] = set()
    miss_counts: dict[str, int] = {}
    max_misses_before_giving_up = 2
    placed_count = 0
    stalled_on_exit = False  # set when we bail out with the arm possibly jammed

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

        # The ramp to observe_pose lives at the TOP of the loop, so it only ever
        # runs BEFORE a pick. Whenever the loop ends -- max-blocks reached, or
        # nothing left to pick -- the arm was simply left standing at the last
        # drop hover, and then disconnected with torque still on. Park it.
        print("\nreturning to observe pose...")
        current = ramp_to(robot, current, observe_pose, RAMP_STEPS, args.dry_run)
    except (RuntimeError, KeyboardInterrupt) as e:
        # RuntimeError: almost certainly the tracking-error watchdog -- it already
        # froze the arm at its current position (hold_pos) BEFORE raising, so it
        # is not actively driving into whatever it hit. Cutting torque
        # unconditionally here (an earlier version of this handler did exactly
        # that) is WRONG when the trip happens mid-air -- e.g. carrying a block
        # between the pick and drop hovers, well above the table -- since losing
        # torque there just drops the arm/block. Never touch torque
        # automatically; ask a human who can actually see the arm first (see
        # issues log 2026-09-01, same failure mode as the old broken-gripper-
        # finger incident: watchdog trips mid-air + torque cut together).
        #
        # KeyboardInterrupt: MUST be caught here too. It descends from
        # BaseException, not Exception, so `except RuntimeError` never saw it --
        # Ctrl+C therefore skipped straight to `finally` with stalled_on_exit
        # still False, which RESTORED the arm's original (higher) torque cap and
        # then disconnected with torque still on. A person hits Ctrl+C precisely
        # when the arm is going somewhere wrong, i.e. exactly the jam case
        # stalled_on_exit exists to protect -- and the old path let it push
        # HARDER instead. Treat Ctrl+C as the human declaring a stall.
        stalled_on_exit = True
        interrupted = isinstance(e, KeyboardInterrupt)
        if interrupted and robot is not None and not args.dry_run:
            # Unlike the watchdog, Ctrl+C leaves the last commanded goal live in
            # the servos -- stop the arm before doing anything else.
            halt_in_place(robot)
            print("\n!! interrupted -- arm commanded to hold where it is.")
        else:
            print(f"\n!! safety watchdog stopped the run: {e}")
        print("   The arm is holding its current position -- do NOT assume it's resting on the table.")
        print("   Look at the arm now: is it hanging in the air, or resting on/against something solid?")
        try:
            choice = input("   [h]old as-is and exit (torque stays on) / [r]etract slowly to observe pose first: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            # No interactive stdin (piped/nohup), or a second Ctrl+C at the
            # prompt. Hold is the safe default -- never move an arm nobody is
            # watching, and a second Ctrl+C means "stop asking, just stop".
            choice = "h"
            print("   (no answer -- defaulting to hold)")
        if choice == "r" and robot is not None and not args.dry_run:
            try:
                # MUST re-read: `current` is stale here. pick_one_block raised
                # before its `current = ...` assignment ever happened, so the
                # local still holds the observe pose from the top of this loop
                # iteration while the arm is physically somewhere else
                # entirely. Ramping observe_pose -> observe_pose would command
                # the full observe pose from step one and fling the arm at
                # max_relative_target speed -- in the one code path where a
                # sudden move is least acceptable (see issues log 2026-09-01).
                current = current_after_abort(robot, current, args.dry_run)
                current = ramp_to(robot, current, observe_pose, RAMP_STEPS, args.dry_run)
                print("   retracted to observe pose.")
                stalled_on_exit = False
            except Exception as retract_err:  # noqa: BLE001
                print(f"   !! retract failed too ({retract_err}) -- leaving arm as-is, intervene manually.")
        else:
            print("   leaving torque on, arm holds its current position after disconnect.")
    finally:
        if robot is not None:
            if prev_arm_torque:
                if stalled_on_exit:
                    # Do NOT raise the torque cap back up on an arm that may
                    # still be jammed against something -- that just lets it
                    # push harder and cook the motor. Leave it capped and say so.
                    print(f"!! arm left with Overload_Torque capped at {ARM_OVERLOAD_TORQUE_PCT}% "
                          f"(not restored, arm may still be stalled). Original values: {prev_arm_torque}")
                else:
                    restore_arm_overload_torque(robot, prev_arm_torque)
            robot.disconnect()


if __name__ == "__main__":
    main()
