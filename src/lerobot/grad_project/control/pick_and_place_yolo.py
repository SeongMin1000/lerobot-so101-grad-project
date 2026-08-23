#!/usr/bin/env python3
"""End-to-end YOLO pick-and-place FSM for SO-101: single-neural-net (YOLO)
color/position detection + fully hardcoded IK motion control.

  observe pose -> top camera frame -> YOLO detect -> pick nearest outside
  block -> pixel->table(cm) -> table->robot(m) -> IK -> hover -> descend ->
  close gripper -> lift -> move to drop slot -> open gripper -> repeat

Every piece below already exists and is individually tested
(pixel_to_table.py, table_ik.py, table_frame_calibration.py,
yolo_block_detector.py); this script is the first place they're wired
together into a live loop.

Before running against the real arm, two one-time physical setup steps are
required (both raise a clear error here if missing):
  1) Train/download YOLO weights -> --yolo_model_path
  2) Table<->robot alignment -> `save_table_calibration_point.py` (see its
     docstring), which writes project/config/table_robot_calibration.json

Example:
  python -m lerobot.grad_project.control.pick_and_place_yolo \\
    --robot.type=so101_follower --robot.port=/dev/so101_follower \\
    --robot.id=follower --robot.disable_torque_on_disconnect=false \\
    --robot.max_relative_target=15 --robot.max_tracking_error=8 \\
    --yolo_model_path=project/models/yolo_block_detector/best.pt \\
    --max_blocks=5 --dry_run=false
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat
from typing import Any

import cv2
import draccus
import numpy as np

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.grad_project.control.table_frame_calibration import (
    DEFAULT_CALIBRATION_PATH,
    apply_table_to_robot,
    load_calibration,
)
from lerobot.grad_project.control.grasp_pitch_model import GraspPitchModel, orientation_from_tilt
from lerobot.grad_project.control.table_ik import DEFAULT_URDF_PATH, load_kinematics, solve_to_position
from lerobot.grad_project.paths import detector_calib_path, lerobot_root, runtime_config_path
from lerobot.grad_project.perception.pixel_to_table import DEFAULT_CALIBRATION_PATH as PIXEL_CALIB_PATH
from lerobot.grad_project.perception.pixel_to_table import (
    TARGET_HEIGHT_CM,
    TARGET_WIDTH_CM,
    load_homography,
    pixel_to_table_xy,
)
from lerobot.grad_project.perception.yolo_block_detector import YoloBlockDetector
from lerobot.robots import (  # noqa: F401
    Robot,
    RobotConfig,
    bi_so_follower,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    so_follower,
)
from lerobot.utils.import_utils import register_third_party_plugins

logger = logging.getLogger(__name__)

JOINT_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ARM_JOINT_ORDER = JOINT_ORDER[:-1]

def _rotation_about(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues rotation matrix for `angle_rad` about a (non-unit) axis."""
    axis = axis / np.linalg.norm(axis)
    skew = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return np.eye(3) + np.sin(angle_rad) * skew + (1.0 - np.cos(angle_rad)) * (skew @ skew)



@dataclass
class PickAndPlaceYoloConfig:
    robot: RobotConfig

    yolo_model_path: str = ""
    yolo_conf: float = 0.5
    yolo_device: str | None = None

    detector_calib: str = str(detector_calib_path(must_exist=False))
    runtime_config: str = str(runtime_config_path(must_exist=False))
    pixel_calibration: str = PIXEL_CALIB_PATH
    table_calibration: str = DEFAULT_CALIBRATION_PATH
    # Leader-arm-taught pixel -> robot mapping (collect_grasp_corrections.py).
    # Preferred when present: it is fitted from real grasps at block height,
    # so it already contains the parallax the table-plane model cannot see.
    grasp_calibration: str = "project/config/grasp_pixel_to_robot.json"
    urdf_path: str = DEFAULT_URDF_PATH

    top_key: str = "top"
    max_blocks: int = 5

    # Safety hop height above the table plane (robot frame, meters) used
    # both when approaching a block and before/after moving to the drop
    # slot, so the gripper never drags across other blocks in transit.
    # Assumes the robot base frame's z-axis points up (true for a flat,
    # level mount) -- re-check after calibration if the arm sits tilted.
    # Standoff measured along the jaw axis, not vertically -- at 33 deg that
    # 8cm buys ~4.3cm of clearance over a 2.5cm block, enough to survive the
    # arm's droop without a long low skim into the target.
    hover_height_m: float = 0.08
    grasp_z_offset_m: float = 0.0

    # Height the gripper is raised to before crossing the board to the drop
    # slot, so a carried block clears the ones still lying on the table.
    transit_height_m: float = 0.12

    # How far above the taught drop pose the block is let go, so it drops the
    # last few mm instead of being pressed into whatever is already there.
    release_height_m: float = 0.02

    # Radial (outward from the robot base) correction applied to every
    # detected block position -- see pixel_to_robot_xyz for why.
    grasp_radial_offset_m: float = 0.0

    # Constant robot-frame nudge on every grasp target, for bias the taught
    # calibration cannot see: gear backlash in shoulder_pan and flex under
    # load put the jaws slightly off where the encoders say they are.
    # +y is the operator's LEFT when standing behind the arm.
    grasp_offset_x_m: float = 0.0
    grasp_offset_y_m: float = 0.0

    # Extra travel along the jaw axis past the detected block centre. This
    # earned its keep when the jaws came in nearly horizontally (~33-57 deg
    # off vertical): they met the block's near face and needed a nudge to seat
    # around it. The demonstrated angles are near-vertical now, so the same
    # push just drives past the centre and into the block. Zero means the
    # descent is purely vertical onto the detected centre, which is what the
    # taught grasps actually did.
    grasp_axial_overshoot_m: float = 0.0

    # The servos lift the folded arm at only ~6 deg/s, and commanding faster
    # than that trips the tracking watchdog mid-move. Durations are stretched
    # so no joint is ever asked to exceed this.
    max_joint_speed_deg_s: float = 75.0

    # Servo position gain for the five arm joints (0 = leave as configured).
    arm_p_gain: int = 32

    # Radius below which the arm has to fold in on itself. A standoff inside
    # this is unusable, and a grasp target inside it is unreachable at all.
    min_standoff_reach_m: float = 0.22
    # Blocks nearer than this are refused outright. Demonstrated grasps reach
    # in to 121mm, so anything above that throws away board the arm can work;
    # the IK feasibility check is what actually decides the hard cases.
    min_grasp_reach_m: float = 0.13
    max_grasp_reach_m: float = 0.45

    # Bounding box (robot frame, m) a real block can occupy, sized to the
    # board itself -- NOT to where teaching samples happened to land, which cut
    # off a third of the play area and refused blocks genuinely sitting on it.
    # Off-board false positives are caught upstream by the detector's
    # workspace_polygon; this is the backstop for a wild extrapolation.
    workspace_x_min_m: float = -0.12
    workspace_x_max_m: float = 0.55
    workspace_y_min_m: float = -0.42
    workspace_y_max_m: float = 0.43

    # Relaxing a demonstrated wrist angle that IK cannot reach (see
    # grasp_orientation_for): step size and the error that counts as reached.
    pitch_relax_step_deg: float = 5.0
    pitch_relax_tolerance_m: float = 0.005

    # Lowering the high transit waypoint until IK can reach it (see
    # _reachable_approach). Looser than the grasp tolerance on purpose: this
    # point only has to clear the block, not line up with it.
    approach_reach_tolerance_m: float = 0.02
    approach_lower_step_m: float = 0.02
    # Straight-line moves (see move_straight_to): spacing of the IK waypoints
    # along the line, and the per-waypoint error worth warning about.
    straight_step_m: float = 0.01
    straight_tolerance_m: float = 0.005
    straight_max_steps: int = 24
    straight_hold_s: float = 0.4
    # A carried block must clear the ones still on the table (25mm tall), so
    # the lift is never traded below this no matter what IK says.
    min_lift_clearance_m: float = 0.035

    # Table surface guard (see clamp_to_table). The corner points touched
    # during calibration define the floor; the margin keeps the jaws just off
    # it rather than skimming it.
    table_points_file: str = "project/config/table_calibration_points.json"
    table_floor_margin_m: float = 0.002

    # Droop compensation for the grasp height (see grasp_waypoints). Measured
    # directly in clear air by measure_vertical_sag.py -- 10mm at 18cm reach,
    # 18mm by 24cm -- rather than inferred from "settled Nmm short" log lines,
    # which are 3D distances and overstated the vertical part by about half.
    # Degrees added to the wrist_roll joint for the three grasp moves
    # (approach, hover, descend). Turns the gripper clockwise seen from the
    # wrist looking out along the arm, which drops the right jaw; without it
    # that jaw rides over the block's top face and only strokes it. Flip the
    # sign to turn the other way.
    wrist_roll_offset_deg: float = 0.0

    droop_base_m: float = 0.010
    droop_per_reach_m: float = 0.067
    droop_per_azimuth_m: float = 0.0
    droop_max_m: float = 0.020

    # Fallback tilt ramp, used only until demonstrations carry wrist angles:
    # at pitch_ramp_far_m the old reference tilt is used as-is, ramping to
    # max_extra_pitch_deg steeper by pitch_ramp_near_m.
    pitch_ramp_far_m: float = 0.30
    pitch_ramp_near_m: float = 0.18
    max_extra_pitch_deg: float = 30.0

    # How close (in pixels) a block must be to a slot to count as placed.
    # ~40px is roughly 3.5cm on this board: wider than the arm's placement
    # error, tighter than the gap between neighbouring slots.
    slot_tolerance_px: float = 40.0

    gripper_close_value: float = 0.0
    # How wide the jaws sit while descending onto a block. The runtime's
    # "open" value is only just wider than a block, so the jaws catch its
    # edge on the way down; opening further gives clearance.
    # Jaw opening on the way down. A block reads ~27 when properly held, so
    # 45 leaves only ~9 either side -- less than the calibration's own ~20mm
    # scatter, which is why a slightly-off target catches an edge instead of
    # straddling the block. Opening wider costs nothing and buys tolerance.
    gripper_approach_open: float = 70.0

    # Five blocks do not fit in one row across the 20x10cm zone, so lay them
    # out in two rows -- 3 in the front row, 2 staggered behind. Slots are
    # inset from the boundary by drop_margin_cm so a block's own width (and
    # the placement error) stays inside the lines rather than straddling them.
    # In table coordinates +x is the operator's left and +y is away from them.
    drop_row_counts: tuple[int, ...] = (3, 2)
    # Slot order by colour: slot 0,1,2 are the near row left-to-right, then
    # 3,4 the far row. Fixed so the arrangement repeats run to run.
    drop_slot_colors: tuple[str, ...] = ()
    # Left/right inset for the outermost slots in a row. Smaller spreads the
    # row wider: at 5cm the three blocks sat 22mm apart, close enough to knock
    # into each other on a slightly-off placement, and 3cm opens that to 41mm.
    # It costs nothing at the edges -- the tightest margin in the zone is set
    # by the row spacing in y, not by x, until this drops under about 2cm.
    drop_margin_x_cm: float = 5.0
    # Right-hand inset, separate from the left. The leftmost slot lands well
    # where it is, so widening the row symmetrically would move a good slot to
    # fix the other two; shrinking only this side spreads slots 2 and 3 and
    # leaves slot 1 alone.
    drop_margin_right_cm: float = 5.0
    # Measured zone depth is 8.9cm, not the nominal 10, so table-y 2.0 puts the
    # first row only 1.8cm from the near edge -- 5mm of margin for a 2.5cm
    # block. The arm also settles short of every command, i.e. toward the robot,
    # which is that same edge, so the front row was landing half outside. Both
    # rows sit deeper now, still far enough apart not to touch. Pushing both
    # rows in by the full 5cm would have traded the front row's problem for the
    # back row's: it lands 13mm from the FAR edge. A 36mm pitch also let the
    # back row ride up onto the front row's blocks, so these trade a little
    # edge margin (27 -> 22mm, still comfortably over a block's 12.5mm
    # half-width) for a 44mm pitch that clears a 25mm block outright.
    # Table-frame y, where y=0 is the TL-TR edge (the one nearest the ROBOT,
    # top of the camera image) and y grows toward BL-BR (away from the robot,
    # toward the camera). Both rows shifted up-y: the near row now sits 26.6mm
    # off the TL-TR edge instead of 17.7mm, and the pitch is 44.4mm instead of
    # 35.5mm so the far row stops riding up onto the near row's blocks.
    drop_row_y_cm: tuple[float, ...] = (3.0, 8.0)
    # Table-frame trim for the whole layout, tuned from where blocks land.
    drop_offset_x_cm: float = 0.0
    drop_offset_y_cm: float = 0.0

    # Closing onto air runs the jaws all the way shut; closing onto a block
    # stalls them part-open. Anything at or below this counts as "missed".
    # Measured on this arm: empty and fully shut reads 1.6, a properly seated
    # block reads 27. The old 5.0 also passed the 10-12 readings you get when
    # the jaws catch only a corner -- those look like successes, then the block
    # slips out on the lift and the run counts a placement that never happened.
    # Sitting between the corner grips and a real hold rejects them.
    grasp_detect_threshold: float = 18.0
    # A miss re-detects and tries again instead of miming a drop and burning
    # a slot. Blocks nudged by the failed attempt get a fresh YOLO fix.
    max_grasp_retries: int = 2
    # Cheap retries at the same coordinates, before falling back to a full
    # observe-and-re-detect cycle.
    in_place_retries: int = 1

    # Gravity droop correction (see move_to_cartesian). The arm settles short
    # of every commanded pose -- 27 times in one run, median 15mm -- and short
    # means toward the base, so whatever this leaves uncorrected shows up as
    # the jaws closing on the near side of a block instead of its centre. One
    # pass stopping at 9mm left up to 9mm of that on a 25mm block, which is
    # visibly off-centre; two passes at 4mm bring it inside a millimetre or two.
    position_correction_passes: int = 2
    position_tolerance_mm: float = 4.0
    correction_duration_s: float = 0.8
    # Guards on the droop correction itself: how far past the target it may
    # lead, and the IK residual above which the corrected pose is refused.
    # Without these an unreachable lead point makes the arm lurch into the desk.
    max_correction_lead_m: float = 0.03
    correction_max_residual_m: float = 0.01

    observe_pose_name: str = "observe"
    fallback_drop_pose_name: str = "drop_center"

    # Grasp orientation reference. A fixed world orientation is NOT feasible
    # across the whole board (the arm's approach direction naturally rotates
    # with shoulder_pan), so instead we take one hand-taught pregrasp pose's
    # orientation and rotate it about world z by the difference in azimuth
    # between that pose and the target. Measured max IK residual across the
    # 5 board positions: A2 = 1.6mm, versus 16-27mm for any fixed orientation.
    grasp_orientation_ref: str = "A2"

    fps: float = 30.0
    hover_duration_s: float = 1.2
    # 75mm of descent in 0.8s is 94mm/s arriving at a block -- fast enough to
    # feel like a drop and to hit hard if the height is off at all. Slower
    # costs about half a second per grasp.
    descend_duration_s: float = 0.8
    gripper_move_duration_s: float = 0.8
    lift_duration_s: float = 0.8
    drop_duration_s: float = 1.5
    settle_s: float = 0.15
    post_place_wait_s: float = 0.2

    debug_save_dir: str = ""
    dry_run: bool = False


class PickAndPlaceYolo:
    def __init__(self, cfg: PickAndPlaceYoloConfig):
        self.cfg = cfg
        self.robot: Robot = make_robot_from_config(cfg.robot)
        self.robot.connect()

        # The driver's configure() sets every servo to P=16, tuned to keep an
        # unloaded arm from shaking. Stretched out over the far half of the
        # board that is far too soft: the arm settles 2-4cm below the
        # commanded pose and the gripper closes on air. Stiffen the joints
        # that carry the gravity load; the gripper is left alone so it still
        # yields against a block instead of crushing it.
        if cfg.arm_p_gain:
            for motor in ARM_JOINT_ORDER:
                self.robot.bus.write("P_Coefficient", motor, cfg.arm_p_gain)
            logger.info("Raised arm P gain to %d (gripper left at default)", cfg.arm_p_gain)

        if not cfg.yolo_model_path:
            raise SystemExit("--yolo_model_path is required (path to trained best.pt)")

        self.detector = YoloBlockDetector.load(
            cfg.yolo_model_path,
            detector_calib_path(cfg.detector_calib, must_exist=True),
            frame_color="rgb",  # frames come from robot.get_observation(), which yields RGB
            conf=cfg.yolo_conf,
            device=cfg.yolo_device,
        )

        self.homography = load_homography(cfg.pixel_calibration)
        self.table_to_robot = load_calibration(cfg.table_calibration)

        grasp_path = Path(cfg.grasp_calibration)
        if not grasp_path.is_absolute():
            grasp_path = lerobot_root() / grasp_path
        if grasp_path.is_file():
            data = json.loads(grasp_path.read_text())
            self.grasp_homography = np.array(data["homography_pixel_to_robot_xy_m"], dtype=np.float64)
            self.grasp_z_plane = np.array(data["grasp_z_plane_abc"], dtype=np.float64)
            logger.info("Using taught grasp calibration (%d samples) from %s", data["sample_count"], grasp_path)
        else:
            self.grasp_homography = None
            self.grasp_z_plane = None
            logger.warning(
                "No taught grasp calibration at %s -- falling back to the table-plane model, which "
                "ignores block-height parallax and typically stops ~2-3cm short. Run "
                "lerobot.grad_project.tools.collect_grasp_corrections to fix.",
                grasp_path,
            )
        self.table_floor_plane = self._load_table_floor_plane()
        self.kin = load_kinematics(cfg.urdf_path)

        self.runtime = self._load_runtime(runtime_config_path(cfg.runtime_config))
        self._ref_joints: np.ndarray | None = None
        self._ref_orientation, self._ref_azimuth_rad = self._load_grasp_orientation_ref()
        self._pitch_model = GraspPitchModel.load()
        if self._pitch_model is not None:
            logger.info("Using demonstrated wrist angles: %s", self._pitch_model.describe())
        else:
            logger.warning(
                "No demonstrated wrist angles yet (re-teach with collect_grasp_corrections to "
                "record them) -- falling back to the fixed tilt ramp."
            )

        self.debug_save_dir = Path(cfg.debug_save_dir) if cfg.debug_save_dir else None
        if self.debug_save_dir:
            self.debug_save_dir.mkdir(parents=True, exist_ok=True)
        self._detect_count = 0
        self.placed_count = 0
        self._queue: list = []
        self._unreachable: set[str] = set()
        self._skipped_this_attempt = False

        logger.info("PickAndPlaceYolo ready.")

    @staticmethod
    def _load_runtime(path: Path) -> dict[str, Any]:
        import json

        if not path.is_file():
            raise FileNotFoundError(
                f"runtime config not found: {path}. Create it with hybrid_save_pose.py "
                "(needs 'observe' pose, 'gripper.open', and drop_slots/poses)."
            )
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("poses", {})
        data.setdefault("drop_slots", [])
        data.setdefault("gripper", {})
        return data

    def close(self) -> None:
        self.robot.disconnect()

    # ------------------------------------------------------------------
    # Pose helpers
    # ------------------------------------------------------------------
    def get_current_pose(self) -> dict[str, float]:
        obs = self.robot.get_observation()
        return {f"{name}.pos": float(obs[f"{name}.pos"]) for name in JOINT_ORDER}

    def current_joint_deg(self) -> np.ndarray:
        pose = self.get_current_pose()
        return np.array([pose[f"{name}.pos"] for name in JOINT_ORDER], dtype=float)

    def _load_grasp_orientation_ref(self) -> tuple[np.ndarray, float]:
        """Return (rotation, azimuth_rad) of the reference pregrasp pose."""
        label = self.cfg.grasp_orientation_ref
        for point in self.runtime.get("pregrasp_points", []):
            if point.get("label") == label:
                joints = np.array([float(point["pose"][f"{n}.pos"]) for n in JOINT_ORDER])
                pose = self.kin.forward_kinematics(joints)
                self._ref_joints = joints
                return pose[:3, :3].copy(), float(np.arctan2(pose[1, 3], pose[0, 3]))
        available = [p.get("label") for p in self.runtime.get("pregrasp_points", [])]
        raise KeyError(
            f"grasp_orientation_ref '{label}' not found in runtime_config pregrasp_points. "
            f"Available: {available}"
        )

    def _load_table_floor_plane(self) -> np.ndarray | None:
        """Fit z = a*x + b*y + c through the four touched table corners.

        These come from putting the gripper tip ON the table, so they are the
        surface itself -- the height below which the jaws are pressing into it
        rather than closing on a block.
        """
        path = lerobot_root() / self.cfg.table_points_file
        if not path.is_file():
            return None
        points = json.loads(path.read_text()).get("points", [])
        if len(points) < 3:
            return None
        xyz = np.array([p["robot_xyz_m"] for p in points], dtype=np.float64)
        design = np.column_stack([xyz[:, :2], np.ones(len(xyz))])
        plane, *_ = np.linalg.lstsq(design, xyz[:, 2], rcond=None)
        logger.info(
            "Table floor from %d touched corners: %.1f to %.1fmm",
            len(points), xyz[:, 2].min() * 1000, xyz[:, 2].max() * 1000,
        )
        return plane

    def clamp_to_table(self, xyz: np.ndarray, label: str, sag_m: float = 0.0) -> np.ndarray:
        """Never command the jaws below the table surface.

        The taught grasp heights scatter over ~50mm and some were recorded with
        the gripper pressed into the table, so the fitted grasp plane can sit
        up to 25mm low. Descending to that drives the jaws into the table and
        holds them there under the raised arm gain -- enough to snap a printed
        jaw. The block sits ON the table, so clipping at the surface can only
        help the grasp.
        """
        if self.table_floor_plane is None:
            return xyz
        # Raise ONLY what would otherwise end up under the table. Adding the
        # sag estimate to every grasp lifted heights that were landing fine and
        # the jaws started catching the top edge of the block; the floor is the
        # only place the sag has to be paid for, so pay it only there.
        floor = (
            float(self.table_floor_plane @ np.array([xyz[0], xyz[1], 1.0]))
            + self.cfg.table_floor_margin_m
            + sag_m
        )
        if xyz[2] >= floor:
            return xyz
        logger.warning(
            "%s target %.1fmm is below the table (%.1fmm) -- clamping so the jaws do not press into it",
            label, xyz[2] * 1000, floor * 1000,
        )
        clamped = xyz.copy()
        clamped[2] = floor
        return clamped

    def grasp_waypoints(
        self, target_xyz_m: np.ndarray, orientation: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The (grasp, hover, approach_high) chain for grasping at this pose.

        Both the feasibility check and the actual move have to agree on where
        the arm will be asked to go, so the geometry lives here rather than
        being spelled out twice.
        """
        approach_axis = orientation[:, 2] / np.linalg.norm(orientation[:, 2])
        # Command the grasp HIGHER than the taught height by however far the
        # arm is going to sag getting there. The taught z came from reading
        # where the gripper actually WAS while the operator held it on the
        # block, but the pipeline feeds that number back as a command -- and
        # the arm settles below every command it is given. Measured across
        # runs: 17mm of sag under 30cm reach, 24mm at full stretch, against a
        # table clearance that shrinks from 16mm to 5mm over the same span. So
        # the jaws were being driven into the table everywhere, worst at full
        # extension, which is exactly where it was seen.
        reach = float(np.hypot(target_xyz_m[0], target_xyz_m[1]))
        # Sag also grows as the arm swings off centre -- 17mm straight ahead
        # against 20-22mm out to either side, and worse in the tail -- so a
        # reach-only estimate under-compensates exactly the side positions
        # where the jaws were still hitting the table. Sized off the 90th
        # percentile rather than the median: too high costs a retry, too low
        # drives the jaws into the table.
        azimuth_deg = abs(float(np.degrees(np.arctan2(target_xyz_m[1], target_xyz_m[0]))))
        sag = (
            self.cfg.droop_base_m
            + self.cfg.droop_per_reach_m * max(0.0, reach - 0.15)
            + self.cfg.droop_per_azimuth_m * azimuth_deg
        )
        sag = float(np.clip(sag, 0.0, self.cfg.droop_max_m))
        grasp_xyz = self.clamp_to_table(
            target_xyz_m
            + np.array([0.0, 0.0, self.cfg.grasp_z_offset_m])
            + approach_axis * self.cfg.grasp_axial_overshoot_m,
            "grasp",
            sag,
        )
        hover_xyz = grasp_xyz + np.array([0.0, 0.0, self.cfg.hover_height_m])
        approach_xyz = np.array([hover_xyz[0], hover_xyz[1], hover_xyz[2] + self.cfg.transit_height_m])
        return grasp_xyz, hover_xyz, approach_xyz

    def _approach_is_reachable(self, target_xyz_m: np.ndarray, orientation: np.ndarray) -> bool:
        """Can the arm hold this wrist angle where the grasp actually happens?

        Only the grasp point and the hover directly above it are checked. The
        high transit waypoint is deliberately excluded: near the base it is out
        of range at ANY wrist angle (position limit, not orientation), so
        gating on it rejected every near-vertical grasp and relaxed a taught
        2deg wrist out to 37deg -- discarding the whole point of the taught
        angles. `_reachable_approach` keeps that waypoint safe instead.
        """
        grasp_xyz, hover_xyz, _ = self.grasp_waypoints(target_xyz_m, orientation)
        for waypoint in (grasp_xyz, hover_xyz):
            _, err = self.solve_ik(self._ref_joints, waypoint, orientation)
            if err > self.cfg.pitch_relax_tolerance_m:
                return False
        return True

    def _reachable_approach(
        self, hover_xyz: np.ndarray, approach_xyz: np.ndarray, orientation: np.ndarray
    ) -> np.ndarray:
        """Lower the high transit waypoint until the arm can actually get there.

        This waypoint only exists to cross above the block before dropping onto
        it, so trading height for reachability costs nothing. Asking for a
        height the arm cannot make is what let IK return a pose 350mm from the
        request and swing the arm off the board.
        """
        candidate = approach_xyz.copy()
        floor = hover_xyz[2]
        while candidate[2] > floor:
            _, err = self.solve_ik(self._ref_joints, candidate, orientation)
            if err <= self.cfg.approach_reach_tolerance_m:
                if candidate[2] < approach_xyz[2]:
                    logger.info(
                        "lowered waypoint %.0fmm -> %.0fmm to stay reachable",
                        approach_xyz[2] * 1000, candidate[2] * 1000,
                    )
                return candidate
            candidate[2] = max(floor, candidate[2] - self.cfg.approach_lower_step_m)

        # Even the floor is out of range. Returning it is still the least-bad
        # option, but say so -- this is the case that drags a block instead of
        # carrying it, and it means the block sits at the edge of the arm's
        # reach rather than that anything is misconfigured.
        _, err = self.solve_ik(self._ref_joints, candidate, orientation)
        if err > self.cfg.approach_reach_tolerance_m:
            logger.warning(
                "no reachable height down to %.0fmm (best miss %.0fmm) -- the arm may not clear the "
                "table here; block is at the edge of its reach",
                floor * 1000, err * 1000,
            )
        return candidate

    def grasp_orientation_for(self, target_xyz_m: np.ndarray) -> np.ndarray:
        """Wrist angle for this grasp: demonstrated first, IK-feasible second.

        Preferred source is the tilt-vs-reach model fitted from teleoperated
        grasps -- an angle a human already succeeded with beats one derived
        from a single old reference pose. The arm cannot always honour it
        though (near full stretch a steep wrist is unreachable), so the
        demonstrated angle is walked back toward the shallow reference until
        IK can actually hit the point. Falls back to the old fixed ramp
        entirely when no demonstrations carry wrist data yet.
        """
        if self._pitch_model is not None:
            reach = float(np.hypot(target_xyz_m[0], target_xyz_m[1]))
            wanted = self._pitch_model.tilt_for(reach)
            for tilt in np.arange(wanted, 90.0, self.cfg.pitch_relax_step_deg):
                orientation = orientation_from_tilt(target_xyz_m, float(tilt))
                if self._approach_is_reachable(target_xyz_m, orientation):
                    if tilt > wanted:
                        logger.info(
                            "reach %.0fmm: taught tilt %.0fdeg unreachable, relaxed to %.0fdeg",
                            reach * 1000, wanted, tilt,
                        )
                    else:
                        logger.info("reach %.0fmm -> taught tilt %.0fdeg", reach * 1000, tilt)
                    return orientation
            logger.warning(
                "reach %.0fmm: no tilt from %.0fdeg to vertical-ish was reachable, using the old ramp",
                reach * 1000, wanted,
            )

        delta = float(np.arctan2(target_xyz_m[1], target_xyz_m[0])) - self._ref_azimuth_rad
        cos_d, sin_d = np.cos(delta), np.sin(delta)
        rot_z = np.array([[cos_d, -sin_d, 0.0], [sin_d, cos_d, 0.0], [0.0, 0.0, 1.0]])
        orientation = rot_z @ self._ref_orientation

        reach = float(np.hypot(target_xyz_m[0], target_xyz_m[1]))
        span = self.cfg.pitch_ramp_far_m - self.cfg.pitch_ramp_near_m
        ramp = (self.cfg.pitch_ramp_far_m - reach) / span if span > 1e-6 else 0.0
        ramp = min(max(ramp, 0.0), 1.0)
        if ramp <= 0.0:
            return orientation

        extra_deg = ramp * self.cfg.max_extra_pitch_deg
        logger.info("reach %.0fmm -> pitching the approach %.0f deg steeper", reach * 1000, extra_deg)
        # Pitch about the horizontal axis perpendicular to the reach direction,
        # so the jaws tip downward without yawing off the block.
        radial = np.array([target_xyz_m[0], target_xyz_m[1], 0.0])
        radial /= np.linalg.norm(radial)
        pitch_axis = np.cross(np.array([0.0, 0.0, 1.0]), radial)
        return _rotation_about(pitch_axis, np.deg2rad(extra_deg)) @ orientation

    def pose(self, name: str) -> dict[str, float]:
        """Named pose from the runtime config, with the wrist bias already in.

        Pre-rotating here is what keeps the bias affordable: hold it from the
        observe pose onward and every later move is a small adjustment, so the
        follower's max_relative_target clamp never sees a big wrist step. Adding
        it only at the grasp meant asking for 28 degrees in one command, and the
        clamp cut that to 15 -- the wrist visibly failed to finish turning.
        """
        return self._pose_with_wrist_bias(self._raw_pose(name))

    def _pose_with_wrist_bias(self, pose: dict[str, float]) -> dict[str, float]:
        if self.cfg.wrist_roll_offset_deg and "wrist_roll.pos" in pose:
            pose = dict(pose)
            pose["wrist_roll.pos"] += self.cfg.wrist_roll_offset_deg
        return pose

    def _raw_pose(self, name: str) -> dict[str, float]:
        poses = self.runtime.get("poses", {})
        if name not in poses:
            raise KeyError(f"Pose '{name}' not found in runtime_config. Available: {list(poses)}")
        return {k: float(v) for k, v in poses[name].items()}

    def gripper_open_value(self) -> float:
        value = self.runtime.get("gripper", {}).get("open")
        if value is None:
            value = self.pose(self.cfg.observe_pose_name).get("gripper.pos")
        if value is None:
            raise KeyError("No gripper.open in runtime_config and observe pose has no gripper.pos")
        return float(value)

    def send_action(self, action: dict[str, float]) -> None:
        if self.cfg.dry_run:
            logger.info("[DRY RUN] action: %s", {k: round(v, 2) for k, v in action.items()})
            return
        self.robot.send_action(action)

    def move_to_pose(self, target_pose: dict[str, float], duration_s: float, name: str = "") -> None:
        current = self.get_current_pose()
        keys = [k for k in JOINT_ORDER if f"{k}.pos" in target_pose]
        keys = [f"{k}.pos" for k in keys]

        # Stretch the ramp if any joint would have to move faster than it
        # physically can -- otherwise the goal runs away from the measured
        # position and the follower's tracking watchdog aborts the run.
        travel = max((abs(float(target_pose[k]) - current[k]) for k in keys), default=0.0)
        needed = travel / max(self.cfg.max_joint_speed_deg_s, 1e-6)
        if needed > duration_s:
            logger.info("%s: stretching %.1fs -> %.1fs for %.0f deg of travel", name or "pose", duration_s, needed, travel)
            duration_s = needed
        steps = max(2, int(duration_s * self.cfg.fps))
        logger.info("Move to %s over %.2fs (%d steps)", name or "pose", duration_s, steps)
        dt = 1.0 / self.cfg.fps
        for i in range(1, steps + 1):
            alpha = i / steps
            action = dict(current)
            for k in keys:
                action[k] = (1.0 - alpha) * current[k] + alpha * float(target_pose[k])
            self.send_action(action)
            if not self.cfg.dry_run:
                time.sleep(dt)
        if not self.cfg.dry_run:
            time.sleep(self.cfg.settle_s)

    def move_straight_to(
        self,
        target_xyz_m: np.ndarray,
        keep_orientation: np.ndarray | None,
        duration_s: float,
        name: str = "",
        gripper_override: float | None = None,
        wrist_roll_offset_deg: float | None = None,
    ) -> None:
        """Move the gripper along a STRAIGHT LINE in space to `target_xyz_m`.

        move_to_pose interpolates joint angles, which does not move the tip in
        a straight line -- the tip bows outward along an arc between the two
        end poses. For a descent onto a block that arc is a sideways sweep
        through the block: measured 5-8mm on mid-board blocks and 31mm out at
        the edge, which is more than enough to knock one out from under the
        jaws. Solving IK per waypoint along the line keeps the tip on it, so a
        vertical descent is actually vertical the whole way down.
        """
        if wrist_roll_offset_deg is None:
            wrist_roll_offset_deg = self.cfg.wrist_roll_offset_deg
        joints = self.current_joint_deg()
        start = self.kin.forward_kinematics(joints)[:3, 3]
        distance = float(np.linalg.norm(target_xyz_m - start))
        steps = int(np.clip(round(distance / self.cfg.straight_step_m), 2, self.cfg.straight_max_steps))
        gripper = gripper_override if gripper_override is not None else float(joints[-1])

        logger.info("Move straight to %s: %.0fmm in %d steps", name or "pose", distance * 1000, steps)
        waypoints = []
        seed = joints
        worst = 0.0
        for i in range(1, steps + 1):
            point = start + (target_xyz_m - start) * (i / steps)
            # Continue from the previous waypoint's solution only. solve_ik
            # also tries a neutral and a reference seed and keeps whichever
            # lands closest, which is right for a single move but wrong along a
            # path: consecutive points pick different IK branches and the wrist
            # visibly swings back and forth (measured +/-6 deg mid-descent).
            seed, err = solve_to_position(self.kin, seed, point, keep_orientation=keep_orientation)
            worst = max(worst, err)
            waypoints.append(seed.copy())
        # One line, not one per waypoint: a bowing path is a property of the
        # whole move, and at the workspace edge every waypoint would report it.
        if worst > self.cfg.straight_tolerance_m:
            logger.warning(
                "%s: path bows up to %.1fmm -- the arm cannot hold a straight line here",
                name or "pose", worst * 1000,
            )

        # Stream the whole path as one motion. Handing each waypoint to
        # move_to_pose would stop and settle at every centimetre, turning a
        # smooth descent into a stutter and adding a second of dead time.
        # Same speed guard move_to_pose applies: command a joint faster than it
        # can physically turn and the goal outruns the measured position until
        # the follower's tracking watchdog aborts the run.
        travel = max(float(np.max(np.abs(waypoints[-1][:5] - joints[:5]))), 0.0)
        needed = travel / max(self.cfg.max_joint_speed_deg_s, 1e-6)
        if needed > duration_s:
            logger.info("%s: stretching %.1fs -> %.1fs for %.0f deg of travel", name or "pose", duration_s, needed, travel)
            duration_s = needed

        # The wrist bias is added to every commanded pose below, so the pose
        # this interpolation STARTS from has to be free of it. `joints` is
        # measured, and therefore already biased -- blending from it and then
        # adding the bias again applied it twice, flicking the wrist a full
        # offset past target on the first tick before it settled back. That is
        # the "wrist snaps round and returns" seen on every descent.
        joints_base = joints.copy()
        joints_base[JOINT_ORDER.index("wrist_roll")] -= wrist_roll_offset_deg

        ticks = max(2, int(round(duration_s * self.cfg.fps)))
        dt = 1.0 / self.cfg.fps
        for tick in range(1, ticks + 1):
            position = (tick / ticks) * steps
            index = min(int(position), steps - 1)
            blend = position - index
            lower = waypoints[index - 1] if index > 0 else joints_base
            solved = lower + (waypoints[index] - lower) * blend
            action = {f"{n}.pos": float(v) for n, v in zip(JOINT_ORDER, solved, strict=True)}
            action["wrist_roll.pos"] += wrist_roll_offset_deg
            action["gripper.pos"] = float(gripper)
            self.send_action(action)
            if not self.cfg.dry_run:
                time.sleep(dt)
        # Hold the last waypoint until the servos actually arrive. Streaming
        # gives each intermediate point only one tick, which is fine while
        # passing through, but the endpoint needs dwell time: commanding it
        # once left the descent short of the block and the jaws closed around
        # its top edge. move_to_pose got this for free from its ramp.
        final = {f"{n}.pos": float(v) for n, v in zip(JOINT_ORDER, waypoints[-1], strict=True)}
        final["wrist_roll.pos"] += wrist_roll_offset_deg
        final["gripper.pos"] = float(gripper)
        holds = max(1, int(round(self.cfg.straight_hold_s * self.cfg.fps)))
        for _ in range(holds):
            self.send_action(final)
            if not self.cfg.dry_run:
                time.sleep(dt)
        if not self.cfg.dry_run:
            time.sleep(self.cfg.settle_s)

    def move_gripper_to(self, value: float, duration_s: float | None = None) -> None:
        current = self.get_current_pose()
        current["gripper.pos"] = float(value)
        self.move_to_pose(current, duration_s or self.cfg.gripper_move_duration_s, "gripper")

    def solve_ik(
        self,
        current: np.ndarray,
        target_xyz_m: np.ndarray,
        keep_orientation: np.ndarray | None,
    ) -> tuple[np.ndarray, float]:
        """Solve IK from several seeds and keep the best.

        placo's solver is a local optimiser, and the folded `observe` pose is a
        pathological seed -- from there it parks in a local minimum ~250mm off
        the target and extra iterations never recover. Seeding additionally
        from a neutral and a taught grasp configuration costs a few ms and
        turns those failures back into sub-millimetre solves. Ties break toward
        the smallest joint travel so the arm does not take a scenic route.
        """
        best: tuple[np.ndarray, float] | None = None
        for seed in self._ik_seeds(current):
            solved, err = solve_to_position(self.kin, seed, target_xyz_m, keep_orientation=keep_orientation)
            if best is None or err < best[1] - 1e-5:
                best = (solved, err)
            elif err < best[1] + 1e-5:
                travel = float(np.linalg.norm(solved[:5] - current[:5]))
                if travel < float(np.linalg.norm(best[0][:5] - current[:5])):
                    best = (solved, err)
        assert best is not None
        return best

    def _ik_seeds(self, current: np.ndarray) -> list[np.ndarray]:
        seeds = [current, np.zeros(6)]
        if self._ref_joints is not None:
            seeds.append(self._ref_joints)
        return seeds

    def move_to_cartesian(
        self,
        target_xyz_m: np.ndarray,
        keep_orientation: np.ndarray | None,
        duration_s: float,
        name: str = "",
        gripper_override: float | None = None,
        correct: bool = True,
        correct_vertical_only: bool = False,
        wrist_roll_offset_deg: float | None = None,
    ) -> None:
        if wrist_roll_offset_deg is None:
            wrist_roll_offset_deg = self.cfg.wrist_roll_offset_deg
        current = self.current_joint_deg()
        solved, err_m = self.solve_ik(current, target_xyz_m, keep_orientation)
        if err_m > 0.005:
            logger.warning("IK residual %.1fmm for target %s (%s) -- may be near a workspace edge", err_m * 1000, target_xyz_m, name)
        target_pose = {f"{n}.pos": float(v) for n, v in zip(JOINT_ORDER, solved, strict=True)}
        target_pose["wrist_roll.pos"] += wrist_roll_offset_deg
        if gripper_override is not None:
            target_pose["gripper.pos"] = float(gripper_override)
        else:
            target_pose["gripper.pos"] = current[-1]
        self.move_to_pose(target_pose, duration_s, name)

        # The servos hold position with a soft gain, so at long reach the arm
        # settles a centimetre or two short of the commanded angles under its
        # own weight -- worst exactly where the far blocks are. Measure where
        # the gripper actually ended up and close the gap.
        #
        # At block level the horizontal half of this correction is destructive:
        # closing a 3D gap there drags the jaws SIDEWAYS across the table and
        # shoves the block out from under them. The vertical half is still
        # needed, though -- dropping it entirely leaves the descent short and
        # the jaws close around the top edge of the block instead of its body.
        # So the descent corrects in z only: all the way down, never sideways.
        previous_gap_mm = float("inf")
        for _ in range(self.cfg.position_correction_passes if correct else 0):
            if self.cfg.dry_run:
                break
            reached_joints = self.current_joint_deg()
            reached = self.kin.forward_kinematics(reached_joints)[:3, 3]
            gap = target_xyz_m - reached
            if correct_vertical_only:
                gap = np.array([0.0, 0.0, gap[2]])
            gap_mm = float(np.linalg.norm(gap)) * 1000
            if gap_mm <= self.cfg.position_tolerance_mm:
                break

            # Stop the moment a pass makes things worse. The correction aims at
            # target+gap to lead the droop, but at the workspace edge that lead
            # point is unreachable, IK returns a far-away pose, and the arm
            # lurches into the desk -- logged as 33mm short becoming 52mm
            # short. A correction that is diverging will not converge next pass.
            if gap_mm >= previous_gap_mm:
                logger.warning(
                    "%s: correction diverging (%.1fmm -> %.1fmm) -- stopping before the arm lurches",
                    name or "pose", previous_gap_mm, gap_mm,
                )
                break
            previous_gap_mm = gap_mm
            logger.info("%s: settled %.1fmm short, correcting", name or "pose", gap_mm)
            # Aim from where the arm ACTUALLY is, not from the nominal target:
            # steering back to the target's xy would be a sideways move at
            # block level, which is exactly what this mode exists to avoid.
            base = np.array([reached[0], reached[1], target_xyz_m[2]]) if correct_vertical_only else target_xyz_m
            # Cap the lead so an already-marginal target cannot be pushed past
            # the arm's reach, and refuse a correction IK cannot actually hit.
            lead = gap * min(1.0, self.cfg.max_correction_lead_m / max(float(np.linalg.norm(gap)), 1e-9))
            corrected, corrected_err = self.solve_ik(reached_joints, base + lead, keep_orientation)
            if corrected_err > self.cfg.correction_max_residual_m:
                logger.warning(
                    "%s: correction target unreachable (%.1fmm residual) -- leaving it where it is",
                    name or "pose", corrected_err * 1000,
                )
                break
            correction_pose = {f"{n}.pos": float(v) for n, v in zip(JOINT_ORDER, corrected, strict=True)}
            # Re-apply the wrist bias. IK solves for the orientation alone and
            # knows nothing about it, so without this the correction quietly
            # spun the wrist back to neutral right after the main move had set
            # it -- the jaws turned, then untuned themselves on the way down.
            correction_pose["wrist_roll.pos"] += wrist_roll_offset_deg
            correction_pose["gripper.pos"] = target_pose["gripper.pos"]
            self.move_to_pose(correction_pose, self.cfg.correction_duration_s, f"{name}_fix")

    # ------------------------------------------------------------------
    # Perception
    # ------------------------------------------------------------------
    def detect_top(self):
        raw = self.robot.get_observation()
        if self.cfg.top_key not in raw:
            raise KeyError(f"top_key '{self.cfg.top_key}' not in observation keys={list(raw)}")
        result = self.detector.detect(raw[self.cfg.top_key])
        logger.info(
            "YOLO detect: blocks=%d outside=%d target_count=%d chosen=%s",
            len(result.blocks),
            len(result.outside_blocks),
            result.target_count,
            None if result.chosen is None else f"{result.chosen.color}@({result.chosen.cx:.1f},{result.chosen.cy:.1f})",
        )
        if self.debug_save_dir:
            overlay = self.detector.draw(raw[self.cfg.top_key], result)
            out = self.debug_save_dir / f"detect_{self._detect_count:04d}.jpg"
            cv2.imwrite(str(out), overlay)
            self._detect_count += 1
        return result

    def pixel_to_robot_xyz(self, px: float, py: float) -> np.ndarray:
        if self.grasp_homography is not None:
            point = np.array([[[px, py]]], dtype=np.float64)
            x, y = cv2.perspectiveTransform(point, self.grasp_homography)[0, 0]
            z = float(self.grasp_z_plane @ np.array([x, y, 1.0]))
            xyz = np.array([float(x), float(y), z])
        else:
            table_x_cm, table_y_cm = pixel_to_table_xy(px, py, self.homography)
            xyz = apply_table_to_robot(table_x_cm, table_y_cm, self.table_to_robot)

        # The homography maps the TABLE PLANE, but YOLO boxes the block's top
        # face, which sits ~2.5cm above it. The camera is mounted over the far
        # side of the board, so that parallax drags every detection toward the
        # robot and the gripper stops short of the block. Push the target back
        # out radially (the arm's own approach direction) to compensate.
        offset = self.cfg.grasp_radial_offset_m
        if offset:
            radial = np.array([xyz[0], xyz[1], 0.0])
            norm = float(np.linalg.norm(radial))
            if norm > 1e-6:
                xyz = xyz + (radial / norm) * offset
        return xyz + np.array([self.cfg.grasp_offset_x_m, self.cfg.grasp_offset_y_m, 0.0])

    def slot_index_for(self, color: str) -> int:
        """Which slot this colour belongs in.

        A fixed slot per colour, rather than "next free slot", so the layout is
        the same every run and a block re-picked after rolling out goes back
        where it was instead of taking someone else's place.
        """
        order = list(self.cfg.drop_slot_colors)
        if color in order:
            return order.index(color)
        return self.placed_count % max(sum(self.cfg.drop_row_counts), 1)

    def drop_slot_table_cm(self, placed_count: int) -> tuple[str, float, float]:
        """Table-frame (cm) centre of the nth slot in the 3+2 grid."""
        rows = [n for n in self.cfg.drop_row_counts if n > 0]
        total = sum(rows)
        index = placed_count % total
        row = 0
        while index >= rows[row]:
            index -= rows[row]
            row += 1

        count = rows[row]
        margin = self.cfg.drop_margin_x_cm
        span = TARGET_WIDTH_CM - margin - self.cfg.drop_margin_right_cm
        x_cm = margin + (span / 2 if count == 1 else span * index / (count - 1))
        y_cm = self.cfg.drop_row_y_cm[min(row, len(self.cfg.drop_row_y_cm) - 1)]

        x_cm = min(max(x_cm + self.cfg.drop_offset_x_cm, 0.0), TARGET_WIDTH_CM)
        y_cm = min(max(y_cm + self.cfg.drop_offset_y_cm, 0.0), TARGET_HEIGHT_CM)
        return f"slot_r{row + 1}c{index + 1}", x_cm, y_cm

    def drop_xyz_for_count(self, placed_count: int) -> tuple[str, np.ndarray]:
        """Robot-frame release point for the nth block.

        xy comes from the corner calibration (a direct table-to-robot
        correspondence, no camera involved), while z reuses the taught grasp
        plane so the block is let go at the same height it was picked from.
        """
        name, x_cm, y_cm = self.drop_slot_table_cm(placed_count)
        xyz = apply_table_to_robot(x_cm, y_cm, self.table_to_robot)
        if self.grasp_z_plane is not None:
            xyz[2] = float(self.grasp_z_plane @ np.array([xyz[0], xyz[1], 1.0]))
        # Backlash and flex bias the release exactly as they bias the grasp,
        # so the same robot-frame trim applies -- without it the blocks land
        # scattered and overlapping even though the slots are laid out evenly.
        xyz = xyz + np.array([self.cfg.grasp_offset_x_m, self.cfg.grasp_offset_y_m, 0.0])

        # No droop compensation here, deliberately. The correction pass on the
        # drop_high move already pulls this in from 10-15mm to 4-7mm, so adding
        # a ~24mm radial push on top over-corrected by a factor of four: it
        # skewed each row into a diagonal (every slot pushed along its own
        # radius, not the row's normal) and commanded the far row past the far
        # edge of an 8.9cm-deep zone.
        return name, xyz

    # ------------------------------------------------------------------
    # FSM
    # ------------------------------------------------------------------
    def grasp_succeeded(self) -> bool:
        if self.cfg.dry_run:
            return True
        width = self.get_current_pose()["gripper.pos"]
        held = width > self.cfg.grasp_detect_threshold
        logger.info("gripper closed to %.1f -> %s", width, "holding a block" if held else "EMPTY")
        return held

    def slot_pixels(self) -> list[np.ndarray]:
        """Pixel position of each drop slot, for judging what is already placed."""
        inverse = np.linalg.inv(self.homography)
        pixels = []
        for index in range(sum(self.cfg.drop_row_counts)):
            _, x_cm, y_cm = self.drop_slot_table_cm(index)
            point = np.array([[[x_cm, y_cm]]], dtype=np.float64)
            pixels.append(cv2.perspectiveTransform(point, inverse)[0, 0])
        return pixels

    def is_placed(self, block) -> bool:
        """True when a block is sitting in one of our slots.

        Both tests have to pass. Slot proximity alone accepted a block up to
        40px -- around 5cm -- from its slot, which is far enough to be wholly
        outside the zone; a run ended reporting every block placed while its
        own detector still said two were outside. The polygon test alone
        flickers for a block resting on the boundary line. Together: near the
        slot it was aimed at, AND actually in the zone.
        """
        if not block.in_target:
            return False
        centre = np.array([block.cx, block.cy])
        return any(float(np.linalg.norm(centre - slot)) <= self.cfg.slot_tolerance_px for slot in self.slot_pixels())

    def refresh_queue(self) -> int:
        """Park at `observe`, look once, and queue every block not yet in a slot."""
        self.move_to_pose(self.pose(self.cfg.observe_pose_name), self.cfg.hover_duration_s, self.cfg.observe_pose_name)
        result = self.detect_top()
        self._queue = [b for b in result.blocks if not self.is_placed(b) and b.color not in self._unreachable]
        self._queue.sort(key=lambda b: (self.detector.prefer_colors.index(b.color) if b.color in self.detector.prefer_colors else 999, -b.cy, b.cx))
        return len(self._queue)

    def pick_and_place_one(self, attempt: int) -> bool | None:
        """True = block placed, None = grasp missed (retry), False = nothing left."""
        # One look serves the whole board. Returning to `observe` between every
        # block costs a full arm fold-and-unfold each time, which is most of
        # the cycle time; the grasp check re-looks only when something moved.
        if not self._queue and self.refresh_queue() == 0:
            # refresh_queue drops blocks already marked unreachable, so an empty
            # queue means "nothing left I am willing to try", which is not the
            # same as "everything is placed". Reporting the latter hid two
            # blocks still sitting outside the zone.
            if self._unreachable:
                logger.warning(
                    "Nothing left to try, but %s never made it in (skipped as unreachable).",
                    ", ".join(sorted(self._unreachable)),
                )
            else:
                logger.info("Every block is inside the target zone.")
            return False

        chosen = self._queue.pop(0)
        logger.info("===== attempt %d: %s, %d left in queue =====", attempt + 1, chosen.color, len(self._queue))
        target_xyz = self.pixel_to_robot_xyz(chosen.cx, chosen.cy)

        # The pixel->robot homography is only trustworthy inside the region it
        # was taught on -- outside that it extrapolates, and a projective
        # transform can extrapolate to nearly anything. A block detected near
        # the image edge, where teaching coverage is thin, can come back as a
        # point off the board entirely (has happened: computed target beside
        # the robot, off the play surface). Refuse anything outside the known
        # board footprint rather than trust the number and swing there.
        if not (self.cfg.workspace_x_min_m <= target_xyz[0] <= self.cfg.workspace_x_max_m) or not (
            self.cfg.workspace_y_min_m <= target_xyz[1] <= self.cfg.workspace_y_max_m
        ):
            logger.warning(
                "%s mapped to (%.3f,%.3f)m, outside the trusted board area -- skipping rather than "
                "swinging there. Likely thin calibration coverage for that camera region.",
                chosen.color,
                target_xyz[0],
                target_xyz[1],
            )
            self._unreachable.add(chosen.color)
            self._skipped_this_attempt = True
            return None

        # Tucked in against the base the arm simply cannot put the jaws on the
        # table. Say so and move on, rather than burning retries opening and
        # closing on thin air above an unreachable block.
        reach = float(np.hypot(target_xyz[0], target_xyz[1]))
        if reach < self.cfg.min_grasp_reach_m or reach > self.cfg.max_grasp_reach_m:
            logger.warning(
                "%s sits %.0fmm from the base, outside the arm's %.0f-%.0fmm working range -- "
                "skipping it. Move it and run again.",
                chosen.color,
                reach * 1000,
                self.cfg.min_grasp_reach_m * 1000,
                self.cfg.max_grasp_reach_m * 1000,
            )
            self._unreachable.add(chosen.color)
            self._skipped_this_attempt = True
            return None

        keep_orientation = self.grasp_orientation_for(target_xyz)

        # Back off along the gripper's own axis rather than world z, so the
        # final move onto the block is purely axial and the jaws close on it
        # without ever shoving it sideways. The standoff then sits straight up
        # from there, which keeps it clear of the base -- backing off along the
        # tilted axis dragged it inside the arm's minimum reach for near
        # blocks, folding the elbow so the jaws never reached the table at all.
        # Three stages: cross high above the standoff, drop onto it vertically,
        # then close the short axial gap.
        grasp_xyz, hover_xyz, approach_xyz = self.grasp_waypoints(target_xyz, keep_orientation)
        approach_xyz = self._reachable_approach(hover_xyz, approach_xyz, keep_orientation)
        open_value = self.gripper_open_value()
        approach_open = self.cfg.gripper_approach_open
        self.move_to_cartesian(approach_xyz, keep_orientation, self.cfg.hover_duration_s, "approach_high", gripper_override=approach_open, wrist_roll_offset_deg=self.cfg.wrist_roll_offset_deg)
        # Retry in place before paying for a full observe-and-re-detect round
        # trip: an axial approach rarely moves the block, so the second try at
        # the same coordinates usually just needs the jaws seated better.
        held = False
        for retry in range(self.cfg.in_place_retries + 1):
            label = "pick" if retry == 0 else f"pick_retry{retry}"
            self.move_to_cartesian(hover_xyz, keep_orientation, self.cfg.hover_duration_s, f"{label}_hover", gripper_override=approach_open, wrist_roll_offset_deg=self.cfg.wrist_roll_offset_deg)
            self.move_straight_to(grasp_xyz, keep_orientation, self.cfg.descend_duration_s, f"{label}_descend", gripper_override=approach_open, wrist_roll_offset_deg=self.cfg.wrist_roll_offset_deg)
            self.move_gripper_to(self.cfg.gripper_close_value)
            if self.grasp_succeeded():
                held = True
                break
            logger.warning("Grasp missed %s (try %d).", chosen.color, retry + 1)
            self.move_gripper_to(approach_open)

        if not held:
            # A miss usually means the block shifted, so the rest of the queue
            # may be stale too -- drop it and look again from scratch.
            logger.warning("Giving up on %s in place -- re-detecting.", chosen.color)
            self._queue.clear()
            self.move_to_cartesian(approach_xyz, keep_orientation, self.cfg.lift_duration_s, "miss_retreat", gripper_override=approach_open)
            return None

        # Straight up from where the block was grasped. Retracing the axial
        # standoff first would swing the arm out and back for nothing, and a
        # held block only needs to clear the ones still on the table.
        #
        # Clamped to a reachable height for the same reason the approach is.
        # Commanding a lift the arm cannot make does not produce a smaller
        # lift -- IK returns whatever it got closest to, which left the jaws on
        # the table and dragged the block across the board to the drop zone
        # (observed: lift short by 161mm). Better a lower lift that happens.
        # The floor here is a real clearance, not the grasp height: clamping
        # all the way back down would "succeed" by not lifting at all.
        lift_floor = grasp_xyz.copy()
        lift_floor[2] = grasp_xyz[2] + self.cfg.min_lift_clearance_m
        transit_xyz = self._reachable_approach(
            lift_floor,
            np.array([target_xyz[0], target_xyz[1], target_xyz[2] + self.cfg.transit_height_m]),
            keep_orientation,
        )
        self.move_to_cartesian(transit_xyz, keep_orientation, self.cfg.lift_duration_s, "lift", gripper_override=self.cfg.gripper_close_value)

        # Check again now that the block has been picked up. Closing on a block
        # only proves the jaws met it, not that they are holding it: a grip on
        # the top edge measures exactly as wide as a proper one and lets go the
        # moment the arm takes the weight. Without this the run carries nothing
        # to the drop slot and counts a placement, which is how blocks that
        # were never really picked up got reported as placed.
        if not self.grasp_succeeded():
            logger.warning("%s slipped out during the lift -- re-detecting.", chosen.color)
            self.move_gripper_to(approach_open)
            self._queue.clear()
            return None

        # Approach the slot from transit height and lower straight down --
        # travelling at drop height scrapes the carried block along the board.
        # Index by our own placement count. Counting what the camera sees in
        # the zone looks more robust but is not: one false positive there and
        # the index jumps, dropping this block on top of an earlier one.
        drop_name, drop_xyz = self.drop_xyz_for_count(self.slot_index_for(chosen.color))
        drop_orientation = self.grasp_orientation_for(drop_xyz)

        drop_high = np.array([drop_xyz[0], drop_xyz[1], drop_xyz[2] + self.cfg.transit_height_m])
        drop_release = np.array([drop_xyz[0], drop_xyz[1], drop_xyz[2] + self.cfg.release_height_m])
        self.move_to_cartesian(drop_high, drop_orientation, self.cfg.drop_duration_s, f"{drop_name}_high", gripper_override=self.cfg.gripper_close_value)
        self.move_to_cartesian(drop_release, drop_orientation, self.cfg.descend_duration_s, f"{drop_name}_release", gripper_override=self.cfg.gripper_close_value, correct=False)

        self.move_gripper_to(approach_open)
        self.move_to_cartesian(drop_high, drop_orientation, self.cfg.lift_duration_s, f"{drop_name}_clear", gripper_override=approach_open)
        if not self.cfg.dry_run:
            time.sleep(self.cfg.post_place_wait_s)
        self.placed_count += 1
        return True

    def run(self) -> None:
        """Keep going until the camera sees nothing left outside the zone.

        A fixed count would abandon any block a failed grasp left behind, so
        the loop is driven by what is still outside instead. `max_attempts`
        only exists so a block the arm genuinely cannot grasp -- out of reach,
        wedged against the frame -- does not spin here forever.
        """
        max_attempts = self.cfg.max_blocks * (1 + self.cfg.max_grasp_retries)
        consecutive_misses = 0
        try:
            for attempt in range(max_attempts):
                outcome = self.pick_and_place_one(attempt)
                if outcome is False:
                    break
                if outcome is None:
                    if self._skipped_this_attempt:
                        self._skipped_this_attempt = False
                        continue
                    consecutive_misses += 1
                    if consecutive_misses > self.cfg.max_grasp_retries:
                        logger.warning(
                            "Missed the same block %d times in a row -- giving up on it.",
                            consecutive_misses,
                        )
                        break
                    continue
                consecutive_misses = 0
            else:
                logger.warning("Hit the %d-attempt cap with blocks still outside.", max_attempts)

            if self._unreachable:
                logger.warning(
                    "Done. placed %d block(s); %s left outside (%s).",
                    self.placed_count, len(self._unreachable), ", ".join(sorted(self._unreachable)),
                )
            else:
                logger.info("Done. placed %d block(s).", self.placed_count)
            self.move_to_pose(self.pose(self.cfg.observe_pose_name), self.cfg.hover_duration_s, self.cfg.observe_pose_name)
        finally:
            self.close()


@draccus.wrap()
def main(cfg: PickAndPlaceYoloConfig) -> None:
    # force=True: lerobot/draccus configure logging on import, so a plain
    # basicConfig here would be a no-op and swallow every INFO line.
    logging.basicConfig(level=logging.INFO, force=True)
    logging.info("Config:\n%s", pformat(cfg))
    fsm = PickAndPlaceYolo(cfg)
    fsm.run()


if __name__ == "__main__":
    register_third_party_plugins()
    main()
