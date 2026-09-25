#!/usr/bin/env python3
"""End-to-end YOLO + Block Angle (minAreaRect) pick-and-place FSM for SO-101.

Features:
  1) YOLO color/position detection from top camera.
  2) cv2.minAreaRect contour analysis inside each YOLO crop to compute block
     yaw angle (-45° to +45°).
  3) Dynamic wrist_roll adjustment during approach/hover/descend so the gripper jaws
     align perfectly parallel with the block's flat faces before grasping.
  4) Visual debug overlay showing bounding box + detected orientation crosshair.

Example:
  python -m lerobot.grad_project.control.pick_and_place_yolo_angle \
    --robot.type=so101_follower --robot.port=/dev/so101_follower \
    --robot.id=follower --robot.disable_torque_on_disconnect=false \
    --robot.max_relative_target=15.0 --robot.max_tracking_error=150.0 \
    --robot.cameras='{"top": {"type": "opencv", "index_or_path": "/dev/cam_top", "width": 640, "height": 480, "fps": 30}}' \
    --yolo_model_path=project/models/yolo_block_detector/best.pt \
    --enable_angle_alignment=true \
    --dry_run=false
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import draccus
import numpy as np

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.grad_project.control.grasp_pitch_model import GraspPitchModel, orientation_from_tilt
from lerobot.grad_project.control.table_frame_calibration import (
    DEFAULT_CALIBRATION_PATH,
    apply_table_to_robot,
    load_calibration,
)
from lerobot.grad_project.control.table_ik import DEFAULT_URDF_PATH, load_kinematics, solve_to_position
from lerobot.grad_project.paths import detector_calib_path, lerobot_root, runtime_config_path
from lerobot.grad_project.perception.opencv_block_detector import (
    BlockDetection,
    DetectionResult,
    _ensure_bgr,
    _point_in_polygon,
)
from lerobot.grad_project.perception.pixel_to_table import DEFAULT_CALIBRATION_PATH as PIXEL_CALIB_PATH
from lerobot.grad_project.perception.pixel_to_table import (
    TARGET_HEIGHT_CM,
    TARGET_WIDTH_CM,
    load_homography,
    pixel_to_table_xy,
)
from lerobot.grad_project.perception.yolo_block_detector import (
    DEFAULT_CONF,
    DEFAULT_IOU,
    YoloBlockDetector,
    _as_polygon,
    _shrink_polygon,
)
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

MIN_FILL = 0.15
MAX_ASPECT = 1.6
COLOUR_TOLERANCE = 34.0


def _rotation_about(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues rotation matrix for `angle_rad` about a (non-unit) axis."""
    axis = axis / np.linalg.norm(axis)
    skew = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return np.eye(3) + np.sin(angle_rad) * skew + (1.0 - np.cos(angle_rad)) * (skew @ skew)


def block_angle_deg(frame_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[float, str]:
    """Rotation of the block inside `bbox` using LAB chroma segmentation + minAreaRect.

    Returns:
      (angle_in_degrees [-45, 45), status_string)
    """
    height, width = frame_bgr.shape[:2]
    x, y, w, h = bbox
    pad = 4
    left, top = max(0, x - pad), max(0, y - pad)
    right, bottom = min(width, x + w + pad), min(height, y + h + pad)
    if right - left < 8 or bottom - top < 8:
        return 0.0, "box too small"

    crop = frame_bgr[top:bottom, left:right]
    lab = cv2.cvtColor(cv2.GaussianBlur(crop, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.int16)
    mid_y, mid_x = lab.shape[0] // 2, lab.shape[1] // 2
    patch = lab[max(0, mid_y - 3) : mid_y + 4, max(0, mid_x - 3) : mid_x + 4]
    reference = np.median(patch.reshape(-1, 3), axis=0)
    distance = np.linalg.norm(lab - reference, axis=2)
    mask = (distance < COLOUR_TOLERANCE).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, "no contour"

    holding = [c for c in contours if cv2.pointPolygonTest(c, (float(mid_x), float(mid_y)), False) >= 0]
    largest = max(holding or contours, key=cv2.contourArea)
    fill = cv2.contourArea(largest) / max((right - left) * (bottom - top), 1)
    if fill < MIN_FILL:
        return 0.0, f"contour only {fill:.0%} of crop"

    (_, _), (rect_w, rect_h), angle = cv2.minAreaRect(largest)
    short, long = min(rect_w, rect_h), max(rect_w, rect_h)
    if short < 4:
        return 0.0, "rect too thin"
    if long / max(short, 1e-6) > MAX_ASPECT:
        return 0.0, f"aspect {long / short:.1f} not square"

    # Normalize to [-45, 45) range since square block has 4-fold 90° symmetry
    normalized_angle = float(((angle + 45.0) % 90.0) - 45.0)
    return normalized_angle, "ok"


class YoloAngleBlockDetector(YoloBlockDetector):
    """Extends YoloBlockDetector with automatic block yaw estimation and crosshair overlays."""

    def detect(self, frame: np.ndarray) -> DetectionResult:
        frame_bgr = _ensure_bgr(frame, self.frame_color)
        results = self.model.predict(
            source=frame_bgr,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )
        result = results[0]
        names = result.names

        best_box: dict[str, tuple[float, Any]] = {}
        for box in result.boxes if result.boxes is not None else []:
            color = str(names[int(box.cls[0].item())])
            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

            if self.workspace_polygon is not None and not _point_in_polygon(cx, cy, self.workspace_polygon):
                continue
            if color in best_box and best_box[color][0] >= confidence:
                continue
            best_box[color] = (confidence, (x1, y1, x2, y2))

        blocks: list[BlockDetection] = []
        for color, (_confidence, (x1, y1, x2, y2)) in best_box.items():
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w / 2.0, y1 + h / 2.0
            in_target = _point_in_polygon(cx, cy, self._inset_polygon)
            bbox = (int(round(x1)), int(round(y1)), int(round(w)), int(round(h)))

            # Compute block yaw rotation angle
            angle_deg, _note = block_angle_deg(frame_bgr, bbox)

            blocks.append(
                BlockDetection(
                    color=color,
                    cx=cx,
                    cy=cy,
                    area=w * h,
                    angle_deg=angle_deg,
                    in_target=in_target,
                    bbox=bbox,
                )
            )

        outside = [block for block in blocks if not block.in_target]
        target_count = sum(1 for block in blocks if block.in_target)
        chosen = self.choose_next(outside)
        return DetectionResult(blocks, outside, target_count, chosen)

    def draw(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        output = _ensure_bgr(frame, self.frame_color).copy()
        chosen = result.chosen
        for block in result.blocks:
            x, y, w, h = block.bbox
            is_chosen = (
                chosen is not None and abs(block.cx - chosen.cx) < 1.0 and abs(block.cy - chosen.cy) < 1.0
            )
            draw_color = (180, 180, 180) if block.in_target else (0, 255, 0)
            if is_chosen:
                draw_color = (0, 165, 255)

            cv2.rectangle(output, (x, y), (x + w, y + h), draw_color, 2)

            # Draw angle crosshairs
            rad = np.radians(block.angle_deg)
            length = min(w, h) * 0.4
            cx, cy = block.cx, block.cy
            # Primary face direction (red)
            cv2.line(
                output,
                (int(cx - length * np.cos(rad)), int(cy - length * np.sin(rad))),
                (int(cx + length * np.cos(rad)), int(cy + length * np.sin(rad))),
                (0, 0, 255),
                2,
            )
            # Orthogonal face direction (magenta)
            cv2.line(
                output,
                (int(cx + length * np.sin(rad)), int(cy - length * np.cos(rad))),
                (int(cx - length * np.sin(rad)), int(cy + length * np.cos(rad))),
                (255, 0, 255),
                2,
            )

            tag = f"{block.color} {'IN' if block.in_target else 'OUT'} ({block.angle_deg:+.1f}°)"
            cv2.putText(
                output,
                tag,
                (x, max(15, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                draw_color,
                1,
            )

        cv2.putText(
            output,
            f"YOLO+Angle  target={result.target_count} outside={len(result.outside_blocks)}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )
        return output


@dataclass
class PickAndPlaceYoloAngleConfig:
    robot: RobotConfig

    yolo_model_path: str = ""
    yolo_conf: float = 0.5
    yolo_device: str | None = None

    # Angle alignment settings
    enable_angle_alignment: bool = True
    angle_scale: float = 1.0  # Set to -1.0 if wrist_roll rotates counter to camera angle
    max_angle_align_deg: float = 45.0

    detector_calib: str = str(detector_calib_path(must_exist=False))
    runtime_config: str = str(runtime_config_path(must_exist=False))
    pixel_calibration: str = PIXEL_CALIB_PATH
    table_calibration: str = DEFAULT_CALIBRATION_PATH
    grasp_calibration: str = "project/config/grasp_pixel_to_robot.json"
    urdf_path: str = DEFAULT_URDF_PATH

    top_key: str = "top"
    max_blocks: int = 5

    hover_height_m: float = 0.08
    grasp_z_offset_m: float = 0.0
    transit_height_m: float = 0.12
    release_height_m: float = 0.02
    grasp_radial_offset_m: float = 0.0
    grasp_offset_x_m: float = 0.0
    grasp_offset_y_m: float = 0.0
    grasp_axial_overshoot_m: float = 0.0

    max_joint_speed_deg_s: float = 75.0
    arm_p_gain: int = 32

    min_standoff_reach_m: float = 0.22
    min_grasp_reach_m: float = 0.13
    max_grasp_reach_m: float = 0.45

    workspace_x_min_m: float = -0.12
    workspace_x_max_m: float = 0.55
    workspace_y_min_m: float = -0.42
    workspace_y_max_m: float = 0.43

    pitch_relax_step_deg: float = 5.0
    pitch_relax_tolerance_m: float = 0.005
    approach_reach_tolerance_m: float = 0.02
    approach_lower_step_m: float = 0.02
    straight_step_m: float = 0.01
    straight_tolerance_m: float = 0.005
    straight_max_steps: int = 24
    straight_hold_s: float = 0.4
    min_lift_clearance_m: float = 0.035

    table_points_file: str = "project/config/table_calibration_points.json"
    table_floor_margin_m: float = 0.002

    wrist_roll_offset_deg: float = 0.0
    droop_base_m: float = 0.010
    droop_per_reach_m: float = 0.067
    droop_per_azimuth_m: float = 0.0
    droop_max_m: float = 0.020

    pitch_ramp_far_m: float = 0.30
    pitch_ramp_near_m: float = 0.18
    max_extra_pitch_deg: float = 30.0

    grasp_orientation_ref: str = "A2"
    slot_tolerance_px: float = 40.0
    gripper_close_value: float = 0.0
    gripper_approach_open: float = 70.0

    drop_row_counts: tuple[int, ...] = (3, 2)
    drop_slot_colors: tuple[str, ...] = ()
    drop_margin_x_cm: float = 5.0
    drop_margin_y_cm: float = 3.0
    drop_stagger_offset_x_cm: float = 0.0
    drop_z_bias_m: float = 0.0

    hover_duration_s: float = 1.0
    descend_duration_s: float = 0.8
    close_duration_s: float = 0.4
    lift_duration_s: float = 0.8
    drop_duration_s: float = 1.5
    observe_duration_s: float = 1.2
    correction_duration_s: float = 0.35
    gripper_move_duration_s: float = 0.8

    post_place_wait_s: float = 0.2
    settle_s: float = 0.15
    fps: float = 30.0

    position_correction_passes: int = 2
    position_tolerance_mm: float = 2.0
    max_correction_lead_m: float = 0.020
    correction_max_residual_m: float = 0.008

    gripper_held_threshold: float = 10.0
    max_attempts: int = 15
    in_place_retries: int = 1

    dry_run: bool = False
    debug_save_dir: str = ""


class PickAndPlaceYoloAngleFSM:
    def __init__(self, cfg: PickAndPlaceYoloAngleConfig):
        self.cfg = cfg
        register_third_party_plugins()
        self.robot: Robot = make_robot_from_config(cfg.robot)
        self.robot.connect()

        if cfg.arm_p_gain:
            for motor in ARM_JOINT_ORDER:
                try:
                    self.robot.bus.write("P_Coefficient", motor, cfg.arm_p_gain)
                except Exception as e:
                    logger.warning("Failed to write P_gain to %s: %s", motor, e)
            logger.info("Raised arm P gain to %d", cfg.arm_p_gain)

        if not cfg.yolo_model_path:
            raise SystemExit("--yolo_model_path is required (path to trained best.pt)")

        self.detector = YoloAngleBlockDetector.load(
            cfg.yolo_model_path,
            detector_calib_path(cfg.detector_calib, must_exist=True),
            frame_color="rgb",
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
            logger.warning("No taught grasp calibration at %s, using table-plane fallback.", grasp_path)

        self.table_floor_plane = self._load_table_floor_plane()
        self.kin = load_kinematics(cfg.urdf_path)

        self.runtime = self._load_runtime(runtime_config_path(cfg.runtime_config))
        self._ref_joints: np.ndarray | None = None
        self._ref_orientation, self._ref_azimuth_rad = self._load_grasp_orientation_ref()
        self._pitch_model = GraspPitchModel.load()
        if self._pitch_model is not None:
            logger.info("Using demonstrated wrist angles: %s", self._pitch_model.describe())
        else:
            logger.warning("No demonstrated wrist angles -- using fixed tilt ramp.")

        self.debug_save_dir = Path(cfg.debug_save_dir) if cfg.debug_save_dir else None
        if self.debug_save_dir:
            self.debug_save_dir.mkdir(parents=True, exist_ok=True)
        self._detect_count = 0
        self.placed_count = 0
        self._queue: list = []
        self._unreachable: set[str] = set()
        self._skipped_this_attempt = False

        logger.info("PickAndPlaceYoloAngle ready.")

    @staticmethod
    def _load_runtime(path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(f"runtime config not found: {path}")
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def close(self) -> None:
        try:
            if self.robot.is_connected:
                self.robot.disconnect()
        except Exception as e:
            logger.warning("Error disconnecting robot: %s", e)

    def get_current_pose(self) -> dict[str, float]:
        obs = self.robot.get_observation()
        return {f"{n}.pos": float(obs[f"{n}.pos"]) for n in JOINT_ORDER}

    def current_joint_deg(self) -> np.ndarray:
        obs = self.robot.get_observation()
        return np.array([float(obs[f"{n}.pos"]) for n in JOINT_ORDER])

    def _load_grasp_orientation_ref(self) -> tuple[np.ndarray, float]:
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
        pts_path = lerobot_root() / self.cfg.table_points_file
        if not pts_path.is_file():
            return None
        data = json.loads(pts_path.read_text())
        pts = data.get("points", {})
        corners = [k for k in ("bottom_left", "bottom_right", "top_right", "top_left") if k in pts]
        if len(corners) < 3:
            return None
        xyzs = np.array(
            [self.kin.forward_kinematics(np.array([pts[c][f"{n}.pos"] for n in JOINT_ORDER]))[:3, 3] for c in corners]
        )
        centroid = xyzs.mean(axis=0)
        _, _, vh = np.linalg.svd(xyzs - centroid)
        normal = vh[2]
        if normal[2] < 0:
            normal = -normal
        return np.array([normal[0], normal[1], normal[2], -float(normal @ centroid)])

    def clamp_to_table(self, xyz: np.ndarray, label: str, sag_m: float = 0.0) -> np.ndarray:
        if self.table_floor_plane is None:
            return xyz
        a, b, c, d = self.table_floor_plane
        floor = (-d - a * xyz[0] - b * xyz[1]) / c + self.cfg.table_floor_margin_m + sag_m
        clamped = xyz.copy()
        clamped[2] = max(float(floor), xyz[2])
        return clamped

    def grasp_waypoints(
        self, target_xyz_m: np.ndarray, orientation: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        approach_axis = orientation[:, 2] / np.linalg.norm(orientation[:, 2])
        reach = float(np.hypot(target_xyz_m[0], target_xyz_m[1]))
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
        grasp_xyz, hover_xyz, _ = self.grasp_waypoints(target_xyz_m, orientation)
        for waypoint in (grasp_xyz, hover_xyz):
            _, err = self.solve_ik(self._ref_joints, waypoint, orientation)
            if err > self.cfg.pitch_relax_tolerance_m:
                return False
        return True

    def _reachable_approach(
        self, hover_xyz: np.ndarray, approach_xyz: np.ndarray, orientation: np.ndarray
    ) -> np.ndarray:
        candidate = approach_xyz.copy()
        floor = hover_xyz[2]
        while candidate[2] > floor:
            _, err = self.solve_ik(self._ref_joints, candidate, orientation)
            if err <= self.cfg.approach_reach_tolerance_m:
                return candidate
            candidate[2] = max(floor, candidate[2] - self.cfg.approach_lower_step_m)
        return candidate

    def grasp_orientation_for(self, target_xyz_m: np.ndarray) -> np.ndarray:
        if self._pitch_model is not None:
            reach = float(np.hypot(target_xyz_m[0], target_xyz_m[1]))
            wanted = self._pitch_model.tilt_for(reach)
            for tilt in np.arange(wanted, 90.0, self.cfg.pitch_relax_step_deg):
                orientation = orientation_from_tilt(target_xyz_m, float(tilt))
                if self._approach_is_reachable(target_xyz_m, orientation):
                    return orientation

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
        radial = np.array([target_xyz_m[0], target_xyz_m[1], 0.0])
        radial /= np.linalg.norm(radial)
        pitch_axis = np.cross(np.array([0.0, 0.0, 1.0]), radial)
        return _rotation_about(pitch_axis, np.deg2rad(extra_deg)) @ orientation

    def pose(self, name: str) -> dict[str, float]:
        raw = self._raw_pose(name)
        return self._pose_with_wrist_bias(raw)

    def _pose_with_wrist_bias(self, pose: dict[str, float]) -> dict[str, float]:
        pose = dict(pose)
        if self.cfg.wrist_roll_offset_deg and "wrist_roll.pos" in pose:
            pose["wrist_roll.pos"] += self.cfg.wrist_roll_offset_deg
        return pose

    def _raw_pose(self, name: str) -> dict[str, float]:
        poses = self.runtime["poses"]
        if name not in poses:
            raise KeyError(f"named pose '{name}' not in runtime.json poses ({list(poses)})")
        return {k: float(v) for k, v in poses[name].items()}

    def gripper_open_value(self) -> float:
        val = self.runtime.get("gripper_open")
        if val is None:
            return 45.0
        return float(val)

    def send_action(self, action: dict[str, float]) -> None:
        if not self.cfg.dry_run:
            self.robot.send_action(action)

    def move_to_pose(self, target_pose: dict[str, float], duration_s: float, name: str = "") -> None:
        joints_now = self.current_joint_deg()
        target_arr = np.array([target_pose[f"{n}.pos"] for n in JOINT_ORDER])

        travel = max(float(np.max(np.abs(target_arr[:5] - joints_now[:5]))), 0.0)
        needed = travel / max(self.cfg.max_joint_speed_deg_s, 1e-6)
        if needed > duration_s:
            duration_s = needed

        ticks = max(2, int(round(duration_s * self.cfg.fps)))
        dt = 1.0 / self.cfg.fps
        for tick in range(1, ticks + 1):
            alpha = tick / ticks
            blended = (1.0 - alpha) * joints_now + alpha * target_arr
            action = {f"{n}.pos": float(blended[i]) for i, n in enumerate(JOINT_ORDER)}
            action["gripper.pos"] = float(target_pose["gripper.pos"])
            self.send_action(action)
            if not self.cfg.dry_run:
                time.sleep(dt)

        if not self.cfg.dry_run and self.cfg.settle_s > 0:
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
        if wrist_roll_offset_deg is None:
            wrist_roll_offset_deg = self.cfg.wrist_roll_offset_deg
        joints = self.current_joint_deg()
        start = self.kin.forward_kinematics(joints)[:3, 3]
        distance = float(np.linalg.norm(target_xyz_m - start))
        steps = int(np.clip(round(distance / self.cfg.straight_step_m), 2, self.cfg.straight_max_steps))
        gripper = gripper_override if gripper_override is not None else float(joints[-1])

        waypoints = []
        seed = joints
        for i in range(1, steps + 1):
            point = start + (target_xyz_m - start) * (i / steps)
            seed, err = solve_to_position(self.kin, seed, point, keep_orientation=keep_orientation)
            waypoints.append(seed.copy())

        travel = max(float(np.max(np.abs(waypoints[-1][:5] - joints[:5]))), 0.0)
        needed = travel / max(self.cfg.max_joint_speed_deg_s, 1e-6)
        if needed > duration_s:
            duration_s = needed

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

        if not self.cfg.dry_run and self.cfg.settle_s > 0:
            time.sleep(self.cfg.settle_s)

    def move_gripper_to(self, value: float, duration_s: float | None = None) -> None:
        duration = self.cfg.gripper_move_duration_s if duration_s is None else duration_s
        target = self.get_current_pose()
        target["gripper.pos"] = float(value)
        self.move_to_pose(target, duration, "gripper")

    def solve_ik(
        self, current: np.ndarray, target_xyz_m: np.ndarray, keep_orientation: np.ndarray | None
    ) -> tuple[np.ndarray, float]:
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
        return [current, np.zeros(6), self._ref_joints]

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
        target_pose = {f"{n}.pos": float(v) for n, v in zip(JOINT_ORDER, solved, strict=True)}
        target_pose["wrist_roll.pos"] += wrist_roll_offset_deg
        if gripper_override is not None:
            target_pose["gripper.pos"] = float(gripper_override)
        else:
            target_pose["gripper.pos"] = current[-1]
        self.move_to_pose(target_pose, duration_s, name)

        if not correct:
            return

        previous_gap_mm = float("inf")
        for _ in range(self.cfg.position_correction_passes):
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
            if gap_mm >= previous_gap_mm:
                break
            previous_gap_mm = gap_mm
            base = np.array([reached[0], reached[1], target_xyz_m[2]]) if correct_vertical_only else target_xyz_m
            lead = gap * min(1.0, self.cfg.max_correction_lead_m / max(float(np.linalg.norm(gap)), 1e-9))
            corrected, corrected_err = self.solve_ik(reached_joints, base + lead, keep_orientation)
            if corrected_err > self.cfg.correction_max_residual_m:
                break
            correction_pose = {f"{n}.pos": float(v) for n, v in zip(JOINT_ORDER, corrected, strict=True)}
            correction_pose["wrist_roll.pos"] += wrist_roll_offset_deg
            correction_pose["gripper.pos"] = target_pose["gripper.pos"]
            self.move_to_pose(correction_pose, self.cfg.correction_duration_s, f"{name}_fix")

    def detect_top(self) -> DetectionResult:
        raw = self.robot.get_observation()
        if self.cfg.top_key not in raw:
            raise KeyError(f"top_key '{self.cfg.top_key}' not in observation keys={list(raw)}")
        result = self.detector.detect(raw[self.cfg.top_key])
        logger.info(
            "YOLO+Angle detect: blocks=%d outside=%d target_count=%d chosen=%s",
            len(result.blocks),
            len(result.outside_blocks),
            result.target_count,
            None if result.chosen is None else f"{result.chosen.color}@({result.chosen.cx:.1f},{result.chosen.cy:.1f}, {result.chosen.angle_deg:+.1f}°)",
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

        offset = self.cfg.grasp_radial_offset_m
        if offset:
            radial = np.array([xyz[0], xyz[1], 0.0])
            norm = float(np.linalg.norm(radial))
            if norm > 1e-6:
                xyz = xyz + (radial / norm) * offset
        return xyz + np.array([self.cfg.grasp_offset_x_m, self.cfg.grasp_offset_y_m, 0.0])

    def slot_index_for(self, color: str) -> int:
        if self.cfg.drop_slot_colors:
            try:
                return self.cfg.drop_slot_colors.index(color)
            except ValueError:
                pass
        return self.placed_count

    def drop_slot_table_cm(self, placed_count: int) -> tuple[str, float, float]:
        row_counts = self.cfg.drop_row_counts
        total_slots = sum(row_counts)
        slot_idx = placed_count % total_slots
        row_idx = 0
        cum = 0
        for count in row_counts:
            if slot_idx < cum + count:
                break
            cum += count
            row_idx += 1
        col_idx = slot_idx - cum
        count_in_row = row_counts[row_idx]
        if count_in_row == 1:
            x_cm = TARGET_WIDTH_CM / 2.0
        else:
            x_min = self.cfg.drop_margin_x_cm
            x_max = TARGET_WIDTH_CM - self.cfg.drop_margin_x_cm
            x_cm = x_min + (x_max - x_min) * (col_idx / (count_in_row - 1))
        if row_idx == 1 and self.cfg.drop_stagger_offset_x_cm:
            x_cm += self.cfg.drop_stagger_offset_x_cm
        y_min = self.cfg.drop_margin_y_cm
        y_max = TARGET_HEIGHT_CM - self.cfg.drop_margin_y_cm
        y_cm = y_min if len(row_counts) == 1 else y_min + (y_max - y_min) * (row_idx / (len(row_counts) - 1))
        return f"slot{slot_idx}_r{row_idx}c{col_idx}", x_cm, y_cm

    def drop_xyz_for_count(self, placed_count: int) -> tuple[str, np.ndarray]:
        slot_name, table_x_cm, table_y_cm = self.drop_slot_table_cm(placed_count)
        if self.grasp_homography is not None:
            H_inv = np.linalg.inv(self.homography)
            pt_table = np.array([[[table_x_cm, table_y_cm]]], dtype=np.float64)
            px, py = cv2.perspectiveTransform(pt_table, H_inv)[0, 0]
            pt_px = np.array([[[px, py]]], dtype=np.float64)
            x, y = cv2.perspectiveTransform(pt_px, self.grasp_homography)[0, 0]
            z = float(self.grasp_z_plane @ np.array([x, y, 1.0]))
            xyz = np.array([float(x), float(y), z])
        else:
            xyz = apply_table_to_robot(table_x_cm, table_y_cm, self.table_to_robot)
        xyz[2] += self.cfg.drop_z_bias_m
        return slot_name, xyz

    def grasp_succeeded(self) -> bool:
        if self.cfg.dry_run:
            return True
        obs = self.robot.get_observation()
        grip = float(obs["gripper.pos"])
        return grip > self.cfg.gripper_held_threshold

    def refresh_queue(self) -> int:
        self.move_to_pose(self.pose("observe"), self.cfg.observe_duration_s, "observe")
        result = self.detect_top()
        candidates = [b for b in result.outside_blocks if b.color not in self._unreachable]
        self._queue = candidates
        return len(self._queue)

    def pick_and_place_one(self, attempt: int) -> bool | None:
        if not self._queue and self.refresh_queue() == 0:
            if self._unreachable:
                logger.warning(
                    "Nothing left to try, but %s never made it in.",
                    ", ".join(sorted(self._unreachable)),
                )
            else:
                logger.info("Every block is inside the target zone.")
            return False

        chosen = self._queue.pop(0)
        logger.info(
            "===== attempt %d: %s (angle=%.1f°), %d left in queue =====",
            attempt + 1,
            chosen.color,
            chosen.angle_deg,
            len(self._queue),
        )
        target_xyz = self.pixel_to_robot_xyz(chosen.cx, chosen.cy)

        if not (self.cfg.workspace_x_min_m <= target_xyz[0] <= self.cfg.workspace_x_max_m) or not (
            self.cfg.workspace_y_min_m <= target_xyz[1] <= self.cfg.workspace_y_max_m
        ):
            logger.warning("%s outside trusted workspace, skipping.", chosen.color)
            self._unreachable.add(chosen.color)
            return None

        reach = float(np.hypot(target_xyz[0], target_xyz[1]))
        if reach < self.cfg.min_grasp_reach_m or reach > self.cfg.max_grasp_reach_m:
            logger.warning("%s reach %.0fmm outside working range, skipping.", chosen.color, reach * 1000)
            self._unreachable.add(chosen.color)
            return None

        keep_orientation = self.grasp_orientation_for(target_xyz)
        grasp_xyz, hover_xyz, approach_xyz = self.grasp_waypoints(target_xyz, keep_orientation)
        approach_xyz = self._reachable_approach(hover_xyz, approach_xyz, keep_orientation)
        approach_open = self.cfg.gripper_approach_open

        # -------------------------------------------------------------
        # Dynamic wrist_roll calculation based on detected block angle
        # -------------------------------------------------------------
        block_rot = chosen.angle_deg * self.cfg.angle_scale if self.cfg.enable_angle_alignment else 0.0
        block_rot = float(np.clip(block_rot, -self.cfg.max_angle_align_deg, self.cfg.max_angle_align_deg))
        total_wrist_roll = self.cfg.wrist_roll_offset_deg + block_rot
        logger.info(
            "🎯 [Angle Alignment] %s: detected=%.1f° -> commanding wrist_roll offset=%.1f°",
            chosen.color,
            chosen.angle_deg,
            total_wrist_roll,
        )

        self.move_to_cartesian(
            approach_xyz,
            keep_orientation,
            self.cfg.hover_duration_s,
            "approach_high",
            gripper_override=approach_open,
            wrist_roll_offset_deg=total_wrist_roll,
        )

        held = False
        for retry in range(self.cfg.in_place_retries + 1):
            label = "pick" if retry == 0 else f"pick_retry{retry}"
            self.move_to_cartesian(
                hover_xyz,
                keep_orientation,
                self.cfg.hover_duration_s,
                f"{label}_hover",
                gripper_override=approach_open,
                wrist_roll_offset_deg=total_wrist_roll,
            )
            self.move_straight_to(
                grasp_xyz,
                keep_orientation,
                self.cfg.descend_duration_s,
                f"{label}_descend",
                gripper_override=approach_open,
                wrist_roll_offset_deg=total_wrist_roll,
            )
            self.move_gripper_to(self.cfg.gripper_close_value)
            if self.grasp_succeeded():
                held = True
                break
            logger.warning("Grasp missed %s (try %d).", chosen.color, retry + 1)
            self.move_gripper_to(approach_open)

        if not held:
            logger.warning("Giving up on %s in place -- re-detecting.", chosen.color)
            self._queue.clear()
            self.move_to_cartesian(
                approach_xyz,
                keep_orientation,
                self.cfg.lift_duration_s,
                "miss_retreat",
                gripper_override=approach_open,
                wrist_roll_offset_deg=total_wrist_roll,
            )
            return None

        # Lift
        lift_floor = grasp_xyz.copy()
        lift_floor[2] = grasp_xyz[2] + self.cfg.min_lift_clearance_m
        transit_xyz = self._reachable_approach(
            lift_floor,
            np.array([target_xyz[0], target_xyz[1], target_xyz[2] + self.cfg.transit_height_m]),
            keep_orientation,
        )
        self.move_to_cartesian(
            transit_xyz,
            keep_orientation,
            self.cfg.lift_duration_s,
            "lift",
            gripper_override=self.cfg.gripper_close_value,
            wrist_roll_offset_deg=total_wrist_roll,
        )

        if not self.grasp_succeeded():
            logger.warning("%s slipped out during lift -- re-detecting.", chosen.color)
            self.move_gripper_to(approach_open)
            self._queue.clear()
            return None

        # Move to Drop Slot (use clean standard drop orientation)
        drop_name, drop_xyz = self.drop_xyz_for_count(self.slot_index_for(chosen.color))
        drop_orientation = self.grasp_orientation_for(drop_xyz)

        drop_high = np.array([drop_xyz[0], drop_xyz[1], drop_xyz[2] + self.cfg.transit_height_m])
        drop_release = np.array([drop_xyz[0], drop_xyz[1], drop_xyz[2] + self.cfg.release_height_m])
        self.move_to_cartesian(
            drop_high,
            drop_orientation,
            self.cfg.drop_duration_s,
            f"{drop_name}_high",
            gripper_override=self.cfg.gripper_close_value,
        )
        self.move_to_cartesian(
            drop_release,
            drop_orientation,
            self.cfg.descend_duration_s,
            f"{drop_name}_release",
            gripper_override=self.cfg.gripper_close_value,
            correct=False,
        )

        self.move_gripper_to(approach_open)
        self.move_to_cartesian(
            drop_high,
            drop_orientation,
            self.cfg.lift_duration_s,
            f"{drop_name}_clear",
            gripper_override=approach_open,
        )
        if not self.cfg.dry_run:
            time.sleep(self.cfg.post_place_wait_s)
        self.placed_count += 1
        return True

    def run(self) -> None:
        t0 = time.time()
        logger.info("=== Starting YOLO + Block Angle Pick & Place (dry_run=%s) ===", self.cfg.dry_run)
        attempt = 0
        try:
            while self.placed_count < self.cfg.max_blocks and attempt < self.cfg.max_attempts:
                res = self.pick_and_place_one(attempt)
                if res is False:
                    break
                attempt += 1

            dur = time.time() - t0
            logger.info("Finished: %d/%d placed in %.1fs (%d attempts)", self.placed_count, self.cfg.max_blocks, dur, attempt)
            self.move_to_pose(self.pose("observe"), self.cfg.observe_duration_s, "final_observe")
        finally:
            self.close()


@draccus.wrap()
def main(cfg: PickAndPlaceYoloAngleConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    fsm = PickAndPlaceYoloAngleFSM(cfg)
    fsm.run()


if __name__ == "__main__":
    main()
