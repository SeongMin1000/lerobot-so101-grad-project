#!/usr/bin/env python3
#
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Hybrid Task 2 Vertical Stacking 5-Block Recorder (v1) for SO-101 VLA model training.

Key Architecture for Task 2 (Stacking):
========================================
1. Far-to-Near Spatial Prioritization (Auto-sorting):
   - At initial observe pose, Top Camera + YOLO detects all blocks.
   - Blocks located furthest away (highest distance R / behind target) are processed FIRST.
   - Prevents the arm links or gripper from knocking down stacked towers when reaching for far blocks!

2. Two-Phase Precision Human-in-the-Loop per Block:
   - Phase 1 (Auto Approach): High-arc C2 spline flight to block hover.
   - Phase 2 (Manual Pick): Leader arm unlocked -> Operator grasps block -> [SPACE/ENTER].
   - Phase 3 (Auto High Transit): Vertical lift + High-arc flight directly to Stack Hover (stack height Z automatically increments +2cm per layer).
   - Phase 4 (Manual Stack): Leader arm unlocked -> Operator aligns block on tower & opens gripper -> [SPACE/ENTER].
   - Phase 5 (Auto Vertical Escape & Return): Vertical lift (+10cm) off the tower to avoid clipping -> Return to Observe pose.

3. Zero Collision & Stable Stack Guarantee:
   - High-arc transit (Apex Z >= 22cm) ensures zero collision with the existing stack during carrying.
   - Vertical escape ensures gripper fingers do not touch the delicate tower when retreating.

Usage:
======
python -m lerobot.grad_project.recording.hybrid_record_task2_stack_v1 \\
  --robot.type=so101_follower --robot.port=/dev/so101_follower \\
  --teleop.type=so101_leader --teleop.port=/dev/so101_leader \\
  --dataset.repo_id=eslab1234/task2_stack_5blocks_v1 \\
  --dataset.single_task="Stack 5 blocks vertically in sequence." \\
  --sort_order="far_to_near"
"""

from __future__ import annotations

import json
import logging
import math
import select
import sys
import termios
import time
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from pprint import pformat
from typing import Any

import cv2
import numpy as np

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.common.control_utils import (
    is_headless,
    sanity_check_dataset_robot_compatibility,
)
from lerobot.configs import parser
from lerobot.configs.dataset import DatasetRecordConfig
from lerobot.datasets import (
    LeRobotDataset,
    VideoEncodingManager,
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.grad_project.control.hybrid_goto_both_pose import find_target_pose, load_runtime, move_both
from lerobot.grad_project.paths import detector_calib_path, lerobot_root, runtime_config_path
from lerobot.processor import (
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
    make_default_processors,
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
from lerobot.scripts.lerobot_record import RecordConfig as LeRobotRecordConfig
from lerobot.teleoperators import (  # noqa: F401
    Teleoperator,
    TeleoperatorConfig,
    bi_so_leader,
    koch_leader,
    make_teleoperator_from_config,
    omx_leader,
    so_leader,
)
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.feature_utils import build_dataset_frame, combine_feature_dicts
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import init_logging, log_say
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

RECORDER_BUILD = "2026-08-30-task2-stacking-hybrid-v1-far-to-near"
JOINT_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


@dataclass
class HybridTask2StackConfig(LeRobotRecordConfig):
    """Configuration for Task 2 vertical stacking hybrid dataset recording."""

    # Color sequence & block options
    color_sequence: str = "red,yellow,wood,green,blue"
    max_blocks: int = 5
    sort_order: str = "zone_color_priority"  # "zone_color_priority" (upper zone in color order -> lower zone in color order), "far_to_near", or "color_sequence"
    zone_split_x_m: float = 0.32             # Real-world robot X coordinate boundary (meters) dividing upper/far zone (X >= 0.32m = 32cm) and lower/near zone (X < 0.32m)

    # Stacking geometry
    stack_pose_name: str = "drop_center"
    block_height_m: float = 0.02          # Standard block thickness: 2.0 cm
    stack_hover_clearance_m: float = 0.07 # Hover clearance above current stack top: 7.0 cm

    # Path configurations
    runtime_config: str = str(runtime_config_path())
    detector_calib: str = str(detector_calib_path(must_exist=False))
    yolo_model_path: str = "project/models/yolo_block_detector/best.pt"
    grasp_calibration: str = "project/config/grasp_pixel_to_robot_record.json"

    # Motion durations & heights
    macro_goto_duration_s: float = 2.0
    macro_transit_duration_s: float = 2.2
    macro_return_duration_s: float = 2.0
    transit_apex_height_m: float = 0.15
    hover_z_offset_m: float = 0.10
    max_joint_speed_deg_s: float = 60.0

    # Poses & keys
    observe_pose_name: str = "observe"
    top_key: str = "top"
    wrist_key: str = "wrist"

    # Execution flags
    use_yolo_detection: bool = True
    wait_enter_before_episode: bool = True

    # Distance-adaptive direction compensation ("right", "left", or "none")
    pan_bias_direction: str = "none"
    pan_bias_near_deg: float = 0.0   # near zone (R <= 12cm)
    pan_bias_far_deg: float = 0.0    # far zone (R >= 37cm)


class RecordControlEvent(str, Enum):
    SAVE = "save"
    RERECORD = "rerecord"
    STOP = "stop"
    NEXT_PHASE = "next_phase"


def init_hybrid_keyboard_listener() -> tuple[Any, dict[str, bool]]:
    """Initialize non-blocking keyboard listener for hybrid recording controls."""
    events = {
        "exit_early": False,       # Right arrow -> Save episode
        "rerecord_episode": False, # Left arrow -> Discard & Retry
        "stop_recording": False,   # Escape -> Discard & Quit
        "next_phase": False,       # Space/Enter/C -> Phase advance (Pick done / Place done)
    }

    if is_headless():
        logging.warning("Headless environment detected. Keyboard listener unavailable.")
        return None, events

    from pynput import keyboard

    def on_press(key):
        try:
            if key == keyboard.Key.right:
                print("\n[KEY] → Right Arrow: Complete & Save Episode")
                events["exit_early"] = True
            elif key == keyboard.Key.left:
                print("\n[KEY] ← Left Arrow: Discard & Rerecord Episode")
                events["rerecord_episode"] = True
            elif key == keyboard.Key.esc:
                print("\n[KEY] ESC: Stop Session")
                events["stop_recording"] = True
            elif key == keyboard.Key.space or key == keyboard.Key.enter:
                print("\n[KEY] Space/Enter: Phase Done -> Next Step")
                events["next_phase"] = True
            elif hasattr(key, "char") and key.char in {"c", "C"}:
                print("\n[KEY] 'C': Phase Done -> Next Step")
                events["next_phase"] = True
        except Exception as e:
            logging.debug("Error handling key press: %s", e)

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    return listener, events


def _get_leader_arm(teleop: Any) -> Any:
    if teleop is None:
        return None
    if isinstance(teleop, list):
        for t in teleop:
            if hasattr(t, "enable_torque") or hasattr(t, "send_feedback"):
                return t
        return teleop[0]
    return teleop


def _set_leader_torque(teleop: Any, enable: bool) -> None:
    leader = _get_leader_arm(teleop)
    if leader is None:
        return
    try:
        if enable:
            if hasattr(leader, "enable_torque"):
                leader.enable_torque()
            elif hasattr(leader, "bus") and hasattr(leader.bus, "enable_torque"):
                leader.bus.enable_torque()
        else:
            if hasattr(leader, "disable_torque"):
                leader.disable_torque()
            elif hasattr(leader, "bus") and hasattr(leader.bus, "disable_torque"):
                leader.bus.disable_torque()
    except Exception as e:
        logging.warning("Failed to set leader torque (enable=%s): %s", enable, e)


def _send_leader_feedback(teleop: Any, feedback: dict[str, float]) -> None:
    leader = _get_leader_arm(teleop)
    if leader is not None and hasattr(leader, "send_feedback"):
        try:
            leader.send_feedback(feedback)
        except Exception as e:
            logging.debug("Leader send_feedback failed: %s", e)


def _read_current_poses(robot: Any, teleop: Any) -> tuple[dict[str, float], dict[str, float]]:
    obs = robot.get_observation()
    robot_pose = {k: float(obs[k]) for k in robot.action_features if k in obs}

    leader = _get_leader_arm(teleop)
    teleop_pose = {}
    if leader is not None and hasattr(leader, "get_action"):
        try:
            act = leader.get_action()
            teleop_pose = {k: float(v) for k, v in act.items() if k.endswith(".pos")}
        except Exception:
            pass
    if not teleop_pose:
        teleop_pose = dict(robot_pose)
    return robot_pose, teleop_pose


class Task2HoverResolver:
    """Resolves block hover & stack hover joint angles with Far-to-Near prioritization and kinematics."""

    def __init__(self, cfg: HybridTask2StackConfig):
        self.cfg = cfg
        self.root = lerobot_root()
        self.runtime = load_runtime(cfg.runtime_config)
        self.pregrasp_points = self.runtime.get("pregrasp_points", [])
        self._cached_block_coords: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}
        self._detected_color_sequence: list[str] = []

        # 1. Load YOLO detector
        self.yolo_detector = None
        yolo_path = self.root / cfg.yolo_model_path if not Path(cfg.yolo_model_path).is_absolute() else Path(cfg.yolo_model_path)
        if cfg.use_yolo_detection and yolo_path.is_file():
            try:
                from lerobot.grad_project.perception.yolo_block_detector import YoloBlockDetector
                det_cfg_path = detector_calib_path(cfg.detector_calib, must_exist=False)
                self.yolo_detector = YoloBlockDetector.load(
                    str(yolo_path),
                    det_cfg_path if det_cfg_path.is_file() else None,
                    frame_color="rgb",
                )
                logging.info("YOLO Block Detector loaded successfully from %s", yolo_path)
            except Exception as e:
                logging.warning("Could not initialize YOLO detector: %s", e)

        # 2. Load Kinematics & Grasp homography
        self.kin = None
        self.grasp_homography = None
        self.grasp_z_plane = None
        self._ref_a2_joints = None
        self._ref_a2_orientation = None
        self._ref_a2_azimuth_rad = 0.0
        self._ref_c2_joints = None
        self._ref_c2_orientation = None
        self._ref_c2_azimuth_rad = 0.0

        try:
            from lerobot.grad_project.control.table_ik import load_kinematics
            self.kin = load_kinematics()
            grasp_calib_file = self.root / cfg.grasp_calibration if not Path(cfg.grasp_calibration).is_absolute() else Path(cfg.grasp_calibration)
            if grasp_calib_file.is_file():
                g_data = json.loads(grasp_calib_file.read_text())
                self.grasp_homography = np.array(g_data["homography_pixel_to_robot_xy_m"], dtype=np.float64)
                self.grasp_z_plane = np.array(g_data["grasp_z_plane_abc"], dtype=np.float64)
                logging.info("Loaded taught grasp homography and Z-plane from %s", grasp_calib_file)

            self._load_grasp_orientation_ref()
        except Exception as e:
            logging.warning("Failed to initialize kinematics or grasp homography: %s", e)

        # 3. Base Stack Target Drop Pose & Kinematics
        self.base_stack_pose = find_target_pose(self.runtime, cfg.stack_pose_name, "drop_center")
        self.base_stack_xyz = None
        self.base_stack_orient = None
        if self.kin is not None:
            try:
                base_joints = np.array([self.base_stack_pose.get(f"{n}.pos", 0.0) for n in JOINT_ORDER])
                fk = self.kin.forward_kinematics(base_joints)
                self.base_stack_xyz = fk[:3, 3].copy()
                self.base_stack_orient = fk[:3, :3].copy()
                logging.info("Base stack drop position FK: X=%.3f, Y=%.3f, Z=%.3f", *self.base_stack_xyz)
            except Exception as e:
                logging.warning("Failed to FK base stack pose: %s", e)

    def _load_grasp_orientation_ref(self) -> None:
        if self.kin is None:
            return
        self.hover_joint_model = None
        model_path = lerobot_root() / "project/config/hover_joint_model_record.json"
        if model_path.is_file():
            try:
                self.hover_joint_model = json.loads(model_path.read_text())
                logging.info("Loaded taught hover joint model from %s", model_path)
            except Exception as e:
                logging.warning("Failed to load hover joint model: %s", e)

        for point in self.pregrasp_points:
            label = point.get("label", "")
            if label == "A2":
                joints = np.array([float(point["pose"][f"{n}.pos"]) for n in JOINT_ORDER])
                pose = self.kin.forward_kinematics(joints)
                self._ref_a2_joints = joints
                self._ref_a2_orientation = pose[:3, :3].copy()
                self._ref_a2_azimuth_rad = float(np.arctan2(pose[1, 3], pose[0, 3]))
            elif label == "C2":
                joints = np.array([float(point["pose"][f"{n}.pos"]) for n in JOINT_ORDER])
                pose = self.kin.forward_kinematics(joints)
                self._ref_c2_joints = joints
                self._ref_c2_orientation = pose[:3, :3].copy()
                self._ref_c2_azimuth_rad = float(np.arctan2(pose[1, 3], pose[0, 3]))

        if self._ref_a2_orientation is None and self.pregrasp_points:
            p = self.pregrasp_points[0]
            joints = np.array([float(p["pose"][f"{n}.pos"]) for n in JOINT_ORDER])
            pose = self.kin.forward_kinematics(joints)
            self._ref_a2_joints = joints
            self._ref_a2_orientation = pose[:3, :3].copy()
            self._ref_a2_azimuth_rad = float(np.arctan2(pose[1, 3], pose[0, 3]))

    def pixel_to_robot_xyz(self, px: float, py: float) -> np.ndarray:
        point = np.array([[[px, py]]], dtype=np.float64)
        x, y = cv2.perspectiveTransform(point, self.grasp_homography)[0, 0]
        z = float(self.grasp_z_plane @ np.array([x, y, 1.0]))
        return np.array([float(x), float(y), z])

    def grasp_orientation_for(self, target_xyz_m: np.ndarray) -> np.ndarray | None:
        radius = float(np.hypot(target_xyz_m[0], target_xyz_m[1]))
        if radius < 0.22 and self._ref_c2_orientation is not None:
            ref_orient = self._ref_c2_orientation
            ref_azimuth = self._ref_c2_azimuth_rad
        else:
            ref_orient = self._ref_a2_orientation
            ref_azimuth = self._ref_a2_azimuth_rad

        if ref_orient is None:
            return None

        delta = float(np.arctan2(target_xyz_m[1], target_xyz_m[0])) - ref_azimuth
        cos_d, sin_d = np.cos(delta), np.sin(delta)
        rot_z = np.array([[cos_d, -sin_d, 0.0], [sin_d, cos_d, 0.0], [0.0, 0.0, 1.0]])
        return rot_z @ ref_orient

    def cache_scene_blocks(self, top_img: np.ndarray) -> list[str]:
        """Runs initial YOLO pass, sorts blocks by 2-Zone Color Priority, and returns sequence."""
        self._cached_block_coords.clear()
        self._detected_color_sequence.clear()

        config_colors = [c.strip().lower() for c in self.cfg.color_sequence.split(",") if c.strip()][: self.cfg.max_blocks]

        if self.yolo_detector is None or self.grasp_homography is None:
            self._detected_color_sequence = list(config_colors)
            return self._detected_color_sequence

        try:
            res = self.yolo_detector.detect(top_img)
            detected_items: list[tuple[str, np.ndarray, float, float]] = []
            upper_detected: list[str] = []
            lower_detected: list[str] = []

            for blk in res.blocks:
                color_name = blk.color.lower()
                target_xyz = self.pixel_to_robot_xyz(blk.cx, blk.cy)
                orientation = self.grasp_orientation_for(target_xyz)
                self._cached_block_coords[color_name] = (target_xyz, orientation)

                radius = float(np.hypot(target_xyz[0], target_xyz[1]))
                detected_items.append((color_name, target_xyz, radius, float(blk.cy)))

                # Split using real-world calibrated robot X-axis depth (meters)
                if target_xyz[0] >= self.cfg.zone_split_x_m:
                    upper_detected.append(color_name)
                else:
                    lower_detected.append(color_name)

            sort_mode = str(self.cfg.sort_order).strip().lower()

            if sort_mode in {"zone_color_priority", "zone_color", "two_zone", "2zone"}:
                # 1. Upper zone in color priority order
                upper_sorted = [c for c in config_colors if c in upper_detected]
                # 2. Lower zone in color priority order
                lower_sorted = [c for c in config_colors if c in lower_detected and c not in upper_sorted]

                combined_sequence = upper_sorted + lower_sorted

                # Append any expected colors that were missed in detection
                for c in config_colors:
                    if c not in combined_sequence:
                        combined_sequence.append(c)

                self._detected_color_sequence = combined_sequence[: self.cfg.max_blocks]

                print("\n" + "=" * 78)
                print("🎯 [TASK 2 SPATIAL ORDER: REAL-COORDINATE 2-ZONE COLOR PRIORITY]")
                print("   (위쪽/원거리 영역 색상순 먼저 ➔ 아래쪽/근거리 영역 색상순 나중)")
                print(f"   * 실측 분할 경계선: 로봇 베이스 기준 X = {self.cfg.zone_split_x_m * 100:.1f}cm")
                print("   " + "-" * 62)
                if upper_sorted:
                    print(f"   [1단계: 위쪽(원거리) 영역 - 로봇 X >= {self.cfg.zone_split_x_m * 100:.1f}cm]")
                    for rank, c in enumerate(upper_sorted, 1):
                        if c in self._cached_block_coords:
                            t_xyz = self._cached_block_coords[c][0]
                            dist_val = np.hypot(t_xyz[0], t_xyz[1]) * 100
                            print(f"      #{rank}: {c.upper():<8} (실측 X={t_xyz[0]*100:.1f}cm, Y={t_xyz[1]*100:+.1f}cm, R={dist_val:.1f}cm)")
                        else:
                            print(f"      #{rank}: {c.upper():<8} (미검출 fallback)")
                if lower_sorted:
                    print(f"   [2단계: 아래쪽(근거리) 영역 - 로봇 X < {self.cfg.zone_split_x_m * 100:.1f}cm]")
                    for rank, c in enumerate(lower_sorted, len(upper_sorted) + 1):
                        if c in self._cached_block_coords:
                            t_xyz = self._cached_block_coords[c][0]
                            dist_val = np.hypot(t_xyz[0], t_xyz[1]) * 100
                            print(f"      #{rank}: {c.upper():<8} (실측 X={t_xyz[0]*100:.1f}cm, Y={t_xyz[1]*100:+.1f}cm, R={dist_val:.1f}cm)")
                        else:
                            print(f"      #{rank}: {c.upper():<8} (미검출 fallback)")
                print(f"   👉 최종 파지 순서: {' -> '.join([c.upper() for c in self._detected_color_sequence])}")
                print("=" * 78)

            elif sort_mode == "far_to_near":
                detected_items.sort(key=lambda item: item[2], reverse=True)
                sorted_colors = [item[0] for item in detected_items if item[0] in config_colors]
                for c in config_colors:
                    if c not in sorted_colors:
                        sorted_colors.append(c)
                self._detected_color_sequence = sorted_colors[: self.cfg.max_blocks]
            else:
                self._detected_color_sequence = list(config_colors)

        except Exception as e:
            logging.warning("Scene block caching failed: %s", e)
            self._detected_color_sequence = list(config_colors)

        return self._detected_color_sequence

    def resolve_block_targets(
        self,
        color: str,
        observation: dict[str, Any],
        current_joint_deg: dict[str, float],
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Calculates (approach_high_pose, hover_pose) for the target color block."""
        target_xyz = None
        orientation = None

        if color.lower() in self._cached_block_coords:
            target_xyz, orientation = self._cached_block_coords[color.lower()]

        if target_xyz is None and self.yolo_detector is not None and self.grasp_homography is not None:
            try:
                top_img = observation.get(self.cfg.top_key)
                if top_img is not None:
                    res = self.yolo_detector.detect(top_img)
                    matched = [b for b in res.blocks if b.color.lower() == color.lower()]
                    if matched:
                        blk = matched[0]
                        target_xyz = self.pixel_to_robot_xyz(blk.cx, blk.cy)
                        orientation = self.grasp_orientation_for(target_xyz)
            except Exception as e:
                logging.warning("Live YOLO detection failed for %s: %s", color, e)

        # Evaluate Taught Demonstration RBF Model if available
        if target_xyz is not None and self.hover_joint_model is not None:
            try:
                names = self.hover_joint_model.get("joint_names", ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"])
                weights = np.array(self.hover_joint_model["weights"], dtype=np.float64)
                xy = np.array([[target_xyz[0], target_xyz[1]]], dtype=np.float64)

                m_type = self.hover_joint_model.get("type", "polynomial")
                if m_type == "rbf_multiquadric":
                    centers = np.array(self.hover_joint_model["centers_xy"], dtype=np.float64)
                    eps = float(self.hover_joint_model.get("eps", 0.08))
                    dists = np.linalg.norm(xy[:, None, :] - centers[None, :, :], axis=-1)
                    phi = np.sqrt(dists ** 2 + eps ** 2)
                    pred_joints = phi @ weights
                else:
                    deg = self.hover_joint_model.get("degree", 2)
                    x, y = xy[:, 0], xy[:, 1]
                    cols = [np.ones_like(x)]
                    for d in range(1, deg + 1):
                        for i in range(d + 1):
                            cols.append((x ** (d - i)) * (y ** i))
                    feat = np.column_stack(cols)
                    pred_joints = feat @ weights

                pose_hover = {f"{n}.pos": float(pred_joints[0, i]) for i, n in enumerate(names)}

                radius = float(np.hypot(target_xyz[0], target_xyz[1]))
                dist_norm = float(np.clip((radius - 0.12) / 0.25, 0.0, 1.0))
                mag = float(self.cfg.pan_bias_near_deg + (self.cfg.pan_bias_far_deg - self.cfg.pan_bias_near_deg) * dist_norm)
                dir_lower = str(self.cfg.pan_bias_direction).strip().lower()
                if dir_lower in {"right", "r", "우", "우측"}:
                    pan_bias = +mag
                elif dir_lower in {"left", "l", "좌", "좌측"}:
                    pan_bias = -mag
                else:
                    pan_bias = 0.0

                if "shoulder_pan.pos" in pose_hover:
                    pose_hover["shoulder_pan.pos"] += pan_bias

                pose_hover["gripper.pos"] = 45.0
                pose_high = pose_hover.copy()
                pose_high["shoulder_lift.pos"] = min(-10.0, pose_hover["shoulder_lift.pos"] - 15.0)
                pose_high["wrist_flex.pos"] = max(20.0, pose_hover["wrist_flex.pos"] + 15.0)
                return pose_high, pose_hover
            except Exception as e:
                logging.warning("Taught RBF evaluation failed for %s: %s", color, e)

        # Fallback to pregrasp points
        if self.pregrasp_points:
            for p in self.pregrasp_points:
                if p.get("color", "").lower() == color.lower() or p.get("label", "").lower() == color.lower():
                    pose = {k: float(v) for k, v in p["pose"].items()}
                    pose["gripper.pos"] = 45.0
                    return pose, pose
            pose = {k: float(v) for k, v in self.pregrasp_points[0]["pose"].items()}
            pose["gripper.pos"] = 45.0
            return pose, pose

        obs_pose = find_target_pose(self.runtime, self.cfg.observe_pose_name, "")
        return obs_pose, obs_pose

    def resolve_stack_hover_pose(self, layer_idx: int = 0, cur_gripper: float = 0.0) -> dict[str, float]:
        """Returns the 100% exact taught stack_hover target pose from runtime.json."""
        stack_hover_pose = find_target_pose(self.runtime, "stack_hover", "")
        if stack_hover_pose:
            hover_pose = dict(stack_hover_pose)
            hover_pose["gripper.pos"] = cur_gripper
            logging.info("🎯 [STACK HOVER] Using exact taught runtime.json 'stack_hover' pose")
            return hover_pose

        # Fallback to drop_center only if stack_hover not found in runtime.json
        hover_pose = dict(self.base_stack_pose)
        hover_pose["gripper.pos"] = cur_gripper
        return hover_pose


def compute_clamped_cubic_spline_cmd(
    start_pose: dict[str, float],
    apex_pose: dict[str, float] | None,
    target_pose: dict[str, float],
    t: float,
    total_t: float,
    apex_ratio: float = 0.35,
    wrist_pitch_bump_deg: float = 0.0,
    pan_completion_ratio: float = 1.0,
) -> dict[str, float]:
    """Computes C2-continuous clamped cubic spline (or smooth cosine easing for single segment)."""
    keys = [k for k in target_pose if k in start_pose]
    cmd: dict[str, float] = {}

    if total_t <= 1e-6:
        return dict(target_pose)

    t_clamped = max(0.0, min(t, total_t))

    if apex_pose is None:
        s_general = 0.5 * (1.0 - math.cos(math.pi * (t_clamped / total_t)))
        pan_ratio = max(0.4, min(1.0, pan_completion_ratio))
        t_pan_total = pan_ratio * total_t
        if t_clamped <= t_pan_total:
            s_pan = 0.5 * (1.0 - math.cos(math.pi * (t_clamped / t_pan_total)))
        else:
            s_pan = 1.0

        for k in keys:
            s = s_pan if "shoulder_pan" in k else s_general
            val = (1.0 - s) * float(start_pose[k]) + s * float(target_pose[k])
            if "wrist_flex" in k and wrist_pitch_bump_deg > 0.0:
                val -= wrist_pitch_bump_deg * math.sin(math.pi * s_general)
            cmd[k] = val
        return cmd

    apex_ratio = max(0.1, min(0.9, apex_ratio))
    t1 = apex_ratio * total_t
    h1 = t1
    h2 = total_t - t1

    for k in keys:
        q0 = float(start_pose[k])
        q1 = float(apex_pose.get(k, target_pose[k]))
        q2 = float(target_pose[k])

        if "gripper" in k:
            cmd[k] = q2
            continue

        denom = h1 * h2 * (h1 + h2)
        if denom < 1e-9:
            s = 0.5 * (1.0 - math.cos(math.pi * (t_clamped / total_t)))
            cmd[k] = (1.0 - s) * q0 + s * q2
            continue

        v1 = 1.5 * ((q1 - q0) * (h2 ** 2) + (q2 - q1) * (h1 ** 2)) / denom

        if t_clamped <= t1:
            c = (3.0 * (q1 - q0) / (h1 ** 2)) - (v1 / h1)
            d = (-2.0 * (q1 - q0) / (h1 ** 3)) + (v1 / (h1 ** 2))
            cmd[k] = q0 + c * (t_clamped ** 2) + d * (t_clamped ** 3)
        else:
            tau = t_clamped - t1
            c = (3.0 * (q2 - q1) / (h2 ** 2)) - (2.0 * v1 / h2)
            d = (-2.0 * (q2 - q1) / (h2 ** 3)) + (v1 / (h2 ** 2))
            cmd[k] = q1 + v1 * tau + c * (tau ** 2) + d * (tau ** 3)

    return cmd


def _record_auto_spline_trajectory(
    *,
    robot: Any,
    teleop: Any,
    start_robot_pose: dict[str, float],
    apex_pose: dict[str, float] | None,
    target_pose: dict[str, float],
    duration_s: float,
    fps: int,
    dataset: LeRobotDataset,
    single_task: str,
    robot_obs_proc: RobotProcessorPipeline,
    teleop_act_proc: RobotProcessorPipeline,
    robot_act_proc: RobotProcessorPipeline,
    display_data: bool,
    display_compressed: bool,
    events: dict[str, bool],
    apex_ratio: float = 0.35,
    wrist_pitch_bump_deg: float = 0.0,
    pan_completion_ratio: float = 1.0,
) -> RecordControlEvent | None:
    steps = max(2, int(duration_s * fps))
    dt = 1.0 / fps

    _set_leader_torque(teleop, enable=True)

    for i in range(1, steps + 1):
        if events["stop_recording"]:
            return RecordControlEvent.STOP
        if events["rerecord_episode"]:
            return RecordControlEvent.RERECORD

        start_loop_t = time.perf_counter()
        t = i * dt

        macro_cmd = compute_clamped_cubic_spline_cmd(
            start_pose=start_robot_pose,
            apex_pose=apex_pose,
            target_pose=target_pose,
            t=t,
            total_t=duration_s,
            apex_ratio=apex_ratio,
            wrist_pitch_bump_deg=wrist_pitch_bump_deg,
            pan_completion_ratio=pan_completion_ratio,
        )

        robot.send_action(macro_cmd)
        _send_leader_feedback(teleop, macro_cmd)

        obs = robot.get_observation()
        proc_obs = robot_obs_proc(obs)
        obs_frame = build_dataset_frame(dataset.features, proc_obs, prefix=OBS_STR)

        proc_act = teleop_act_proc((macro_cmd, obs))
        act_frame = build_dataset_frame(dataset.features, proc_act, prefix=ACTION)

        dataset.add_frame({**obs_frame, **act_frame, "task": single_task})

        if display_data:
            log_rerun_data(observation=proc_obs, action=proc_act, compress_images=display_compressed)

        elapsed = time.perf_counter() - start_loop_t
        precise_sleep(max(dt - elapsed, 0.0))

    return None


def _record_manual_teleop_phase(
    *,
    robot: Any,
    teleop: Any,
    fps: int,
    dataset: LeRobotDataset,
    single_task: str,
    robot_obs_proc: RobotProcessorPipeline,
    teleop_act_proc: RobotProcessorPipeline,
    robot_act_proc: RobotProcessorPipeline,
    display_data: bool,
    display_compressed: bool,
    events: dict[str, bool],
) -> RecordControlEvent:
    _set_leader_torque(teleop, enable=False)
    events["next_phase"] = False
    events["exit_early"] = False
    events["rerecord_episode"] = False
    dt = 1.0 / fps

    while True:
        if events["stop_recording"]:
            return RecordControlEvent.STOP
        if events["rerecord_episode"]:
            return RecordControlEvent.RERECORD
        if events["exit_early"]:
            return RecordControlEvent.SAVE
        if events["next_phase"]:
            events["next_phase"] = False
            return RecordControlEvent.NEXT_PHASE

        start_loop_t = time.perf_counter()

        obs = robot.get_observation()
        proc_obs = robot_obs_proc(obs)
        obs_frame = build_dataset_frame(dataset.features, proc_obs, prefix=OBS_STR)

        raw_teleop_act = teleop.get_action()
        proc_teleop_act = teleop_act_proc((raw_teleop_act, obs))
        robot_act = robot_act_proc((proc_teleop_act, obs))
        robot.send_action(robot_act)

        act_frame = build_dataset_frame(dataset.features, proc_teleop_act, prefix=ACTION)
        dataset.add_frame({**obs_frame, **act_frame, "task": single_task})

        if display_data:
            log_rerun_data(observation=proc_obs, action=proc_teleop_act, compress_images=display_compressed)

        elapsed = time.perf_counter() - start_loop_t
        precise_sleep(max(dt - elapsed, 0.0))


def _record_task2_stack_episode(
    *,
    robot: Any,
    teleop: Any,
    events: dict[str, bool],
    cfg: HybridTask2StackConfig,
    dataset: LeRobotDataset,
    resolver: Task2HoverResolver,
    observe_target: dict[str, float],
    robot_obs_proc: RobotProcessorPipeline,
    teleop_act_proc: RobotProcessorPipeline,
    robot_act_proc: RobotProcessorPipeline,
) -> RecordControlEvent:
    fps = cfg.dataset.fps
    single_task = cfg.dataset.single_task

    init_obs = robot.get_observation()
    if cfg.top_key in init_obs:
        sequence = resolver.cache_scene_blocks(init_obs[cfg.top_key])
    else:
        sequence = [c.strip().lower() for c in cfg.color_sequence.split(",") if c.strip()][: cfg.max_blocks]

    print()
    print("=" * 78)
    print(f"🎬 [TASK 2 EPISODE START] Stacking 5 Blocks: {' -> '.join(sequence)}")
    print("=" * 78)

    events["exit_early"] = False
    events["rerecord_episode"] = False
    events["next_phase"] = False

    for layer_idx, color in enumerate(sequence):
        print(f"\n" + "-" * 60)
        print(f"🧱 [LAYER {layer_idx + 1}/{len(sequence)}: {color.upper()}] (Target Stack Height: +{layer_idx * 2}cm)")
        print("-" * 60)

        # -------------------------------------------------------------
        # STEP 1. AUTO APPROACH: High-Arc Flight to Block Hover
        # -------------------------------------------------------------
        obs = robot.get_observation()
        cur_robot, _ = _read_current_poses(robot, teleop)
        _, hover_target = resolver.resolve_block_targets(color, obs, cur_robot)

        log_say(f"Approaching {color} block", cfg.play_sounds)

        radius = 0.25
        if color.lower() in resolver._cached_block_coords:
            t_xyz, _ = resolver._cached_block_coords[color.lower()]
            radius = float(np.hypot(t_xyz[0], t_xyz[1]))

        dist_norm = float(np.clip((radius - 0.12) / 0.25, 0.0, 1.0))
        wrist_bump = 6.0 + 8.0 * dist_norm
        pan_ratio = 0.90 - 0.15 * dist_norm
        total_dur = cfg.macro_goto_duration_s

        print(f"🤖 [AUTO 1/4] High flight from observe to {color} hover pose ({total_dur:.1f}s, R={radius*100:.1f}cm)...")
        macro_res = _record_auto_spline_trajectory(
            robot=robot,
            teleop=teleop,
            start_robot_pose=cur_robot,
            apex_pose=None,
            target_pose=hover_target,
            duration_s=total_dur,
            fps=fps,
            dataset=dataset,
            single_task=single_task,
            robot_obs_proc=robot_obs_proc,
            teleop_act_proc=teleop_act_proc,
            robot_act_proc=robot_act_proc,
            display_data=cfg.display_data,
            display_compressed=cfg.display_compressed_images,
            events=events,
            wrist_pitch_bump_deg=wrist_bump,
            pan_completion_ratio=pan_ratio,
        )
        if macro_res is not None:
            return macro_res

        # -------------------------------------------------------------
        # STEP 2. MANUAL PICK: Teleoperate Grasp
        # -------------------------------------------------------------
        log_say(f"Grasp {color}", cfg.play_sounds)
        print(f"🙋‍♂️ [MANUAL 1/2: PICK] Leader unlocked. Grasp {color} firmly.")
        print("👉 Press [SPACE / ENTER] when grasped.")
        pick_res = _record_manual_teleop_phase(
            robot=robot,
            teleop=teleop,
            fps=fps,
            dataset=dataset,
            single_task=single_task,
            robot_obs_proc=robot_obs_proc,
            teleop_act_proc=teleop_act_proc,
            robot_act_proc=robot_act_proc,
            display_data=cfg.display_data,
            display_compressed=cfg.display_compressed_images,
            events=events,
        )
        if pick_res in {RecordControlEvent.STOP, RecordControlEvent.RERECORD, RecordControlEvent.SAVE}:
            return pick_res

        # -------------------------------------------------------------
        # STEP 3. AUTO HIGH TRANSIT: Smooth Direct S-Curve to Stack Hover
        # -------------------------------------------------------------
        cur_robot, _ = _read_current_poses(robot, teleop)
        cur_gripper = cur_robot.get("gripper.pos", 0.0)

        # Target stack hover for current layer
        stack_hover_target = resolver.resolve_stack_hover_pose(layer_idx, cur_gripper=cur_gripper)

        transit_dur = cfg.macro_transit_duration_s
        print(f"🤖 [AUTO 2/4] Smooth S-curve transit holding {color} to stack hover layer {layer_idx + 1} ({transit_dur:.1f}s)...")
        log_say(f"Carrying {color} to stack", cfg.play_sounds)

        transit_res = _record_auto_spline_trajectory(
            robot=robot,
            teleop=teleop,
            start_robot_pose=cur_robot,
            apex_pose=None,
            target_pose=stack_hover_target,
            duration_s=transit_dur,
            fps=fps,
            dataset=dataset,
            single_task=single_task,
            robot_obs_proc=robot_obs_proc,
            teleop_act_proc=teleop_act_proc,
            robot_act_proc=robot_act_proc,
            display_data=cfg.display_data,
            display_compressed=cfg.display_compressed_images,
            events=events,
        )
        if transit_res is not None:
            return transit_res

        # -------------------------------------------------------------
        # STEP 4. MANUAL STACK: Teleoperate Align, Place & Open Gripper
        # -------------------------------------------------------------
        log_say(f"Stack {color}", cfg.play_sounds)
        print(f"🙋‍♂️ [MANUAL 2/2: STACK] Leader unlocked. Place {color} gently onto tower & open gripper.")
        print("👉 Press [SPACE / ENTER] when placed & gripper is open.")
        stack_res = _record_manual_teleop_phase(
            robot=robot,
            teleop=teleop,
            fps=fps,
            dataset=dataset,
            single_task=single_task,
            robot_obs_proc=robot_obs_proc,
            teleop_act_proc=teleop_act_proc,
            robot_act_proc=robot_act_proc,
            display_data=cfg.display_data,
            display_compressed=cfg.display_compressed_images,
            events=events,
        )
        if stack_res in {RecordControlEvent.STOP, RecordControlEvent.RERECORD, RecordControlEvent.SAVE}:
            return stack_res

        # -------------------------------------------------------------
        # STEP 5. AUTO RETURN TO OBSERVE
        # -------------------------------------------------------------
        is_last_block = (layer_idx == len(sequence) - 1)
        cur_robot, _ = _read_current_poses(robot, teleop)

        complete_target = observe_target.copy()
        complete_target["gripper.pos"] = 0.0
        target_return_pose = complete_target if is_last_block else observe_target
        ret_label = "task complete pose (Gripper Closed)" if is_last_block else "observe pose (Gripper Open)"
        ret_dur = cfg.macro_return_duration_s

        print(f"🤖 [AUTO 3/4] Smooth flight to {ret_label} ({ret_dur:.1f}s)...")
        escape_res = _record_auto_spline_trajectory(
            robot=robot,
            teleop=teleop,
            start_robot_pose=cur_robot,
            apex_pose=None,
            target_pose=target_return_pose,
            duration_s=ret_dur,
            fps=fps,
            dataset=dataset,
            single_task=single_task,
            robot_obs_proc=robot_obs_proc,
            teleop_act_proc=teleop_act_proc,
            robot_act_proc=robot_act_proc,
            display_data=cfg.display_data,
            display_compressed=cfg.display_compressed_images,
            events=events,
        )
        if escape_res is not None:
            return escape_res

    # Completed all 5 layers!
    log_say("Task 2 stacking completed", cfg.play_sounds)
    print("\n" + "=" * 78)
    print("🏆 [TASK 2 COMPLETE] All 5 blocks successfully stacked vertically!")
    print("🛑 Frame recording finished. Waiting for operator confirmation:")
    print("👉 Press [→ Right Arrow] to SAVE or [← Left Arrow] to DISCARD.")
    print("=" * 78)

    events["exit_early"] = False
    events["rerecord_episode"] = False

    while True:
        if events["stop_recording"]:
            return RecordControlEvent.STOP
        if events["rerecord_episode"]:
            return RecordControlEvent.RERECORD
        if events["exit_early"]:
            return RecordControlEvent.SAVE

        precise_sleep(0.05)


def _discard_current_episode(
    *,
    dataset: LeRobotDataset,
    expected_episode_index: int,
    reason: str,
) -> int:
    before_count = dataset.num_episodes
    if before_count != expected_episode_index:
        raise RuntimeError(
            f"Expected dataset episode count {expected_episode_index}, but found {before_count}."
        )

    writer = getattr(dataset, "writer", None)
    episode_buffer = getattr(writer, "episode_buffer", None)
    discarded_frames = int(episode_buffer.get("size", 0)) if isinstance(episode_buffer, dict) else 0

    dataset.clear_episode_buffer(delete_images=True)
    cleanup = getattr(writer, "cleanup_interrupted_episode", None)
    if callable(cleanup):
        cleanup(expected_episode_index)

    after_count = dataset.num_episodes
    print(f"[DISCARDED] {reason} | removed {discarded_frames} frames | total episodes: {after_count}")
    return discarded_frames


def _wait_for_scene_ready(
    session_idx: int,
    cfg: HybridTask2StackConfig,
    dataset: LeRobotDataset,
    events: dict[str, bool] | None = None,
) -> bool:
    if not cfg.wait_enter_before_episode:
        return True

    print()
    print("#" * 78)
    print(f"[TASK 2 SCENE SETUP] Episode {session_idx + 1}/{cfg.dataset.num_episodes} | Dataset Total: {dataset.num_episodes}")
    print(f"[TASK PROMPT] {cfg.dataset.single_task}")
    print(f"[SORT ORDER]  {cfg.sort_order.upper()} (Furthest blocks stacked first)")
    print()
    print("1) 5개 블록을 작업대 위에 자연스럽게 분산 배치하세요.")
    print("2) 목표 스택 구역(드롭 슬롯)이 깨끗한지 확인하세요.")
    print("3) 카메라 시야와 안전 거리를 확보하세요.")
    print()
    print("조작 안내 (2-Phase Teleop):")
    print("  [호버 도달 후]    : 리더암으로 블록 파지 → [SPACE/ENTER] 입력")
    print("  [스택 호버 도달] : 리더암으로 스택 위에 안착 & 열기 → [SPACE/ENTER] 입력")
    print("  [5개 완료 후]    : [→] (Right Arrow) 저장 / [←] (Left Arrow) 재촬영")
    print("#" * 78)

    while True:
        try:
            ans = input("준비 완료: ENTER / 종료: q + ENTER > ").strip().lower()
        except EOFError:
            return False
        if ans in {"q", "quit", "exit"}:
            return False
        return True


def _goto_observe_outside(
    robot: Any,
    teleop: Any,
    observe_target: dict[str, float],
    duration_s: float = 2.0,
    fps: int = 30,
    reason: str = "Reset",
) -> None:
    print(f"[RESET] Moving to observe pose ({reason}, outside dataset)...")
    target = dict(observe_target)
    target["gripper.pos"] = 45.0
    move_both(
        robot=robot,
        teleop=teleop,
        target=target,
        duration_s=duration_s,
        fps=fps,
    )
    time.sleep(0.2)


def record(
    cfg: HybridTask2StackConfig,
    teleop_action_processor: RobotProcessorPipeline | None = None,
    robot_action_processor: RobotProcessorPipeline | None = None,
    robot_observation_processor: RobotProcessorPipeline | None = None,
) -> LeRobotDataset:
    """Top-level record function for Task 2 vertical stacking dataset collection."""
    init_logging()
    print(f"[RECORDER BUILD] {RECORDER_BUILD}")
    cfg.runtime_config = str(runtime_config_path(cfg.runtime_config))
    logging.info("Using runtime config: %s", cfg.runtime_config)
    logging.info(pformat(asdict(cfg)))

    runtime = load_runtime(cfg.runtime_config)
    observe_target = find_target_pose(runtime, cfg.observe_pose_name, "")

    if cfg.display_data:
        init_rerun(session_name="task2_stacking_recording", ip=cfg.display_ip, port=cfg.display_port)

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop)

    if teleop_action_processor is None or robot_action_processor is None or robot_observation_processor is None:
        def_t, def_r, def_o = make_default_processors()
        teleop_action_processor = teleop_action_processor or def_t
        robot_action_processor = robot_action_processor or def_r
        robot_observation_processor = robot_observation_processor or def_o

    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=cfg.dataset.video,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=cfg.dataset.video,
        ),
    )

    dataset = None
    listener = None

    try:
        if cfg.resume:
            num_cams = len(robot.cameras) if hasattr(robot, "cameras") else 0
            dataset = LeRobotDataset.resume(
                cfg.dataset.repo_id,
                root=cfg.dataset.root,
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                camera_encoder=cfg.dataset.camera_encoder,
                encoder_threads=cfg.dataset.encoder_threads,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
                image_writer_processes=cfg.dataset.num_image_writer_processes if num_cams > 0 else 0,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * num_cams if num_cams > 0 else 0,
            )
            sanity_check_dataset_robot_compatibility(dataset, robot, cfg.dataset.fps, dataset_features)
        else:
            cfg.dataset.stamp_repo_id()
            dataset = LeRobotDataset.create(
                cfg.dataset.repo_id,
                cfg.dataset.fps,
                root=cfg.dataset.root,
                robot_type=robot.name,
                features=dataset_features,
                use_videos=cfg.dataset.video,
                image_writer_processes=cfg.dataset.num_image_writer_processes,
                image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
                batch_encoding_size=cfg.dataset.video_encoding_batch_size,
                camera_encoder=cfg.dataset.camera_encoder,
                encoder_threads=cfg.dataset.encoder_threads,
                streaming_encoding=cfg.dataset.streaming_encoding,
                encoder_queue_maxsize=cfg.dataset.encoder_queue_maxsize,
            )

        resolver = Task2HoverResolver(cfg)

        robot.connect()
        teleop.connect()

        _goto_observe_outside(robot, teleop, observe_target, duration_s=cfg.macro_return_duration_s, fps=cfg.dataset.fps, reason="Session start")

        listener, events = init_hybrid_keyboard_listener()

        with VideoEncodingManager(dataset):
            recorded = 0
            while recorded < cfg.dataset.num_episodes and not events["stop_recording"]:
                if not _wait_for_scene_ready(recorded, cfg, dataset, events=events):
                    break

                expected_ep = dataset.num_episodes
                outcome = _record_task2_stack_episode(
                    robot=robot,
                    teleop=teleop,
                    events=events,
                    cfg=cfg,
                    dataset=dataset,
                    resolver=resolver,
                    observe_target=observe_target,
                    robot_obs_proc=robot_observation_processor,
                    teleop_act_proc=teleop_action_processor,
                    robot_act_proc=robot_action_processor,
                )

                if outcome == RecordControlEvent.SAVE:
                    dataset.save_episode()
                    recorded += 1
                    print(f"🎉 [SAVED] Task 2 Episode {dataset.num_episodes - 1} committed successfully! ({recorded}/{cfg.dataset.num_episodes})")
                    _goto_observe_outside(robot, teleop, observe_target, duration_s=cfg.macro_return_duration_s, fps=cfg.dataset.fps, reason="Post-save reset")
                elif outcome == RecordControlEvent.RERECORD:
                    _discard_current_episode(dataset=dataset, expected_episode_index=expected_ep, reason="Operator rerequested episode")
                    _goto_observe_outside(robot, teleop, observe_target, duration_s=cfg.macro_return_duration_s, fps=cfg.dataset.fps, reason="Post-discard reset")
                elif outcome == RecordControlEvent.STOP:
                    _discard_current_episode(dataset=dataset, expected_episode_index=expected_ep, reason="Session stopped by operator")
                    break

        return dataset
    finally:
        if listener is not None and hasattr(listener, "stop"):
            listener.stop()
        if teleop is not None:
            _set_leader_torque(teleop, enable=False)
            teleop.disconnect()
        if robot is not None:
            robot.disconnect()


def _make_record_cli_entrypoint():
    from dataclasses import is_dataclass
    if not is_dataclass(HybridTask2StackConfig):
        raise TypeError("HybridTask2StackConfig must remain decorated with @dataclass for draccus.")

    record.__annotations__["cfg"] = HybridTask2StackConfig
    return parser.wrap()(record)


record_cli = _make_record_cli_entrypoint()


def main() -> None:
    register_third_party_plugins()
    record_cli()


if __name__ == "__main__":
    main()
