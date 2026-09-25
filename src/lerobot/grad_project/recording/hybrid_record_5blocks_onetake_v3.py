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

"""Hybrid 5-block one-take dataset recorder (v3) for SO-101 VLA model training.

Key Improvements in v3:
=======================
1. Continuous Parabolic / Arc Spline Trajectory:
   - When transitioning from a placed block to the next target block, instead of
     a disjoint 2-step motion (lift +12cm -> stop -> linear transit to hover),
     a C2-continuous clamped cubic spline is generated through (Start -> Apex Clear -> Target Hover).
   - Zero starting velocity, smooth non-zero velocity across the +12cm apex,
     and gentle deceleration at the hover target.
   - Smooth continuous recording with zero motion stuttering or sharp stops.

2. Cosine-Eased Approach & Return:
   - Initial approach from observe and final return to observe are smoothly eased
     with zero starting/ending acceleration.

Usage:
======
python -m lerobot.grad_project.recording.hybrid_record_5blocks_onetake_v3 \\
  --robot.type=so101_follower --robot.port=/dev/so101_follower \\
  --teleop.type=so101_leader --teleop.port=/dev/so101_leader \\
  --dataset.repo_id=eslab1234/task1_hybrid_5blocks_v3 \\
  --dataset.single_task="Pick and place 5 blocks in sequence" \\
  --color_sequence="red,yellow,wood,green,blue"
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

RECORDER_BUILD = "2026-08-22-hybrid-5blocks-onetake-v3-parabolic-spline"
JOINT_ORDER = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


@dataclass
class HybridOneTakeRecordConfig(LeRobotRecordConfig):
    """Configuration for hybrid auto+manual continuous dataset recording."""

    # Color sequence for the 5-block task
    color_sequence: str = "red,yellow,wood,green,blue"
    max_blocks: int = 5

    # Path configurations
    runtime_config: str = str(runtime_config_path())
    detector_calib: str = str(detector_calib_path(must_exist=False))
    yolo_model_path: str = "project/models/yolo_block_detector/best.pt"
    grasp_calibration: str = "project/config/grasp_pixel_to_robot_record.json"

    # Motion durations & heights
    macro_goto_duration_s: float = 2.0
    macro_return_duration_s: float = 2.0
    transit_height_m: float = 0.12
    hover_z_offset_m: float = 0.10
    parabolic_apex_ratio: float = 0.30
    max_joint_speed_deg_s: float = 60.0
    use_transit_flight: bool = True

    # Poses & keys
    observe_pose_name: str = "observe"
    top_key: str = "top"
    wrist_key: str = "wrist"

    # Execution flags
    use_yolo_detection: bool = True
    return_to_observe_each_block: bool = True
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
        "next_phase": False,       # Space/Enter/C -> Toggle Auto/Manual
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
                print("\n[KEY] Space/Enter: Switch Phase (Auto ↔ Manual)")
                events["next_phase"] = True
            elif hasattr(key, "char") and key.char in {"c", "C"}:
                print("\n[KEY] 'C': Switch Phase (Auto ↔ Manual)")
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


class TargetHoverResolver:
    """Resolves target block hover joint angles using YOLO, projection, orientation, and multi-seed IK."""

    def __init__(self, cfg: HybridOneTakeRecordConfig):
        self.cfg = cfg
        self.root = lerobot_root()
        self.runtime = load_runtime(cfg.runtime_config)
        self.pregrasp_points = self.runtime.get("pregrasp_points", [])
        self._cached_block_coords: dict[str, tuple[np.ndarray, np.ndarray | None]] = {}

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
                logging.warning("Could not initialize YOLO detector: %s. Falling back to pregrasp points.", e)

        # 2. Load Kinematics & Grasp homography
        self.kin = None
        self.grasp_homography = None
        self.grasp_z_plane = None
        self._ref_joints = None
        self._ref_orientation = None
        self._ref_azimuth_rad = 0.0

        try:
            from lerobot.grad_project.control.table_ik import load_kinematics
            self.kin = load_kinematics()
            grasp_calib_file = self.root / cfg.grasp_calibration if not Path(cfg.grasp_calibration).is_absolute() else Path(cfg.grasp_calibration)
            if grasp_calib_file.is_file():
                g_data = json.loads(grasp_calib_file.read_text())
                self.grasp_homography = np.array(g_data["homography_pixel_to_robot_xy_m"], dtype=np.float64)
                self.grasp_z_plane = np.array(g_data["grasp_z_plane_abc"], dtype=np.float64)
                logging.info("Loaded taught grasp homography and Z-plane from %s", grasp_calib_file)

            # Load multi-zone grasp orientation references (A2 far, C2 near)
            self._load_grasp_orientation_ref()
        except Exception as e:
            logging.warning("Failed to initialize kinematics or grasp homography: %s", e)

    def cache_scene_blocks(self, top_img: np.ndarray) -> None:
        """Runs initial YOLO pass on observe view and caches all block 3D coordinates & orientations."""
        if self.yolo_detector is None or self.grasp_homography is None:
            return
        try:
            res = self.yolo_detector.detect(top_img)
            self._cached_block_coords.clear()
            for blk in res.blocks:
                target_xyz = self.pixel_to_robot_xyz(blk.cx, blk.cy)
                orientation = self.grasp_orientation_for(target_xyz)
                self._cached_block_coords[blk.color.lower()] = (target_xyz, orientation)
            logging.info("Cached %d block coordinates from initial observe frame: %s", len(self._cached_block_coords), list(self._cached_block_coords.keys()))
        except Exception as e:
            logging.warning("Scene block caching failed: %s", e)

    def _load_grasp_orientation_ref(self) -> None:
        """Loads both A2 (far/mid zone 60°) and C2 (near zone steep) reference poses."""
        if self.kin is None:
            return

        self._ref_a2_joints = None
        self._ref_a2_orientation = None
        self._ref_a2_azimuth_rad = 0.0

        self._ref_c2_joints = None
        self._ref_c2_orientation = None
        self._ref_c2_azimuth_rad = 0.0

        # Load taught hover demonstration model if available
        self.hover_joint_model = None
        model_path = lerobot_root() / "project/config/hover_joint_model_record.json"
        if model_path.is_file():
            try:
                import json
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

        # Fallback if A2 not found
        if self._ref_a2_orientation is None and self.pregrasp_points:
            p = self.pregrasp_points[0]
            joints = np.array([float(p["pose"][f"{n}.pos"]) for n in JOINT_ORDER])
            pose = self.kin.forward_kinematics(joints)
            self._ref_a2_joints = joints
            self._ref_a2_orientation = pose[:3, :3].copy()
            self._ref_a2_azimuth_rad = float(np.arctan2(pose[1, 3], pose[0, 3]))

    def grasp_orientation_for(self, target_xyz_m: np.ndarray) -> np.ndarray | None:
        """Calculates distance-adaptive 3D grasp orientation based on 2D radius R = sqrt(X^2 + Y^2)."""
        radius = float(np.hypot(target_xyz_m[0], target_xyz_m[1]))

        # Near zone (R < 22cm): use C2 near-field steep pitch reference
        if radius < 0.22 and self._ref_c2_orientation is not None:
            ref_orient = self._ref_c2_orientation
            ref_azimuth = self._ref_c2_azimuth_rad
        else:
            # Mid/Far zone (R >= 22cm): use A2 standard 60° pitch reference
            ref_orient = self._ref_a2_orientation
            ref_azimuth = self._ref_a2_azimuth_rad

        if ref_orient is None:
            return None

        delta = float(np.arctan2(target_xyz_m[1], target_xyz_m[0])) - ref_azimuth
        cos_d, sin_d = np.cos(delta), np.sin(delta)
        rot_z = np.array([[cos_d, -sin_d, 0.0], [sin_d, cos_d, 0.0], [0.0, 0.0, 1.0]])
        return rot_z @ ref_orient

    def pixel_to_robot_xyz(self, px: float, py: float) -> np.ndarray:
        """Converts (px, py) to robot base XYZ (m) using calibrated homography and Z-plane."""
        point = np.array([[[px, py]]], dtype=np.float64)
        x, y = cv2.perspectiveTransform(point, self.grasp_homography)[0, 0]
        z = float(self.grasp_z_plane @ np.array([x, y, 1.0]))
        return np.array([float(x), float(y), z])

    def solve_ik_multiseed(
        self,
        current_joints: np.ndarray,
        target_xyz_m: np.ndarray,
        keep_orientation: np.ndarray | None,
    ) -> tuple[np.ndarray, float]:
        """Solves IK with multi-seeds including current, neutral, A2 (far) and C2 (near)."""
        from lerobot.grad_project.control.table_ik import solve_to_position
        seeds = [current_joints, np.zeros(6)]
        if self._ref_a2_joints is not None:
            seeds.append(self._ref_a2_joints)
        if self._ref_c2_joints is not None:
            seeds.append(self._ref_c2_joints)

        best: tuple[np.ndarray, float] | None = None
        for seed in seeds:
            solved, err = solve_to_position(self.kin, seed, target_xyz_m, keep_orientation=keep_orientation, max_iters=50)
            if best is None or err < best[1] - 1e-5:
                best = (solved, err)
            elif err < best[1] + 1e-5:
                travel = float(np.linalg.norm(solved[:5] - current_joints[:5]))
                if travel < float(np.linalg.norm(best[0][:5] - current_joints[:5])):
                    best = (solved, err)
        assert best is not None
        return best

    def resolve_lift_clear_pose(self, cur_robot: dict[str, float]) -> dict[str, float]:
        """Calculates a pose lifted vertically by transit_height_m (+12cm) from the current position."""
        if self.kin is not None:
            try:
                cur_arr = np.array([cur_robot.get(f"{n}.pos", 0.0) for n in JOINT_ORDER])
                cur_xyz = self.kin.forward_kinematics(cur_arr)[:3, 3]
                clear_xyz = np.array([cur_xyz[0], cur_xyz[1], cur_xyz[2] + self.cfg.transit_height_m])
                solved, err = self.solve_ik_multiseed(cur_arr, clear_xyz, keep_orientation=None)
                if err < 0.02:
                    pose = {f"{n}.pos": float(solved[i]) for i, n in enumerate(JOINT_ORDER)}
                    pose["gripper.pos"] = 45.0
                    return pose
            except Exception as e:
                logging.debug("Kinematics clear pose solve failed: %s", e)

        # Fallback: lift shoulder
        lift_pose = dict(cur_robot)
        lift_pose["shoulder_lift.pos"] = cur_robot.get("shoulder_lift.pos", 0.0) - 25.0
        lift_pose["gripper.pos"] = 45.0
        return lift_pose

    def resolve_block_targets(
        self,
        color: str,
        observation: dict[str, Any],
        current_joint_deg: dict[str, float],
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Returns (approach_high_pose, hover_pose) for the target color block."""
        target_xyz = None
        orientation = None

        # Check cached coordinates first
        if color.lower() in self._cached_block_coords:
            target_xyz, orientation = self._cached_block_coords[color.lower()]

        # If not cached, run YOLO on the current frame
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

        # If taught demonstration model exists, evaluate it directly (ground-truth mapping)
        if target_xyz is not None and self.hover_joint_model is not None:
            try:
                names = self.hover_joint_model.get("joint_names", ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"])
                weights = np.array(self.hover_joint_model["weights"], dtype=np.float64)
                xy = np.array([[target_xyz[0], target_xyz[1]]], dtype=np.float64)

                m_type = self.hover_joint_model.get("type", "polynomial")
                if m_type == "rbf_multiquadric":
                    centers = np.array(self.hover_joint_model["centers_xy"], dtype=np.float64)
                    eps = float(self.hover_joint_model.get("eps", 0.08))
                    dists = np.linalg.norm(xy[:, None, :] - centers[None, :, :], axis=-1)  # (1, N)
                    phi = np.sqrt(dists ** 2 + eps ** 2)  # (1, N)
                    pred_joints = phi @ weights  # (1, 5)
                else:
                    deg = self.hover_joint_model.get("degree", 2)
                    x, y = xy[:, 0], xy[:, 1]
                    cols = [np.ones_like(x)]
                    for d in range(1, deg + 1):
                        for i in range(d + 1):
                            cols.append((x ** (d - i)) * (y ** i))
                    feat = np.column_stack(cols)
                    pred_joints = feat @ weights  # (1, 5)

                pose_hover = {f"{n}.pos": float(pred_joints[0, i]) for i, n in enumerate(names)}

                # Distance-adaptive shoulder_pan direction compensation ("right" adds, "left" subtracts)
                radius = float(np.hypot(target_xyz[0], target_xyz[1]))
                dist_norm = float(np.clip((radius - 0.12) / 0.25, 0.0, 1.0))
                mag = float(self.cfg.pan_bias_near_deg + (self.cfg.pan_bias_far_deg - self.cfg.pan_bias_near_deg) * dist_norm)
                dir_lower = str(self.cfg.pan_bias_direction).strip().lower()
                if dir_lower in {"right", "r", "우", "우측"}:
                    pan_bias = +mag
                    dir_tag = f"RIGHT +{mag:.2f}°"
                elif dir_lower in {"left", "l", "좌", "좌측"}:
                    pan_bias = -mag
                    dir_tag = f"LEFT -{mag:.2f}°"
                else:
                    pan_bias = 0.0
                    dir_tag = "OFF"

                if "shoulder_pan.pos" in pose_hover:
                    pose_hover["shoulder_pan.pos"] += pan_bias

                pose_hover["gripper.pos"] = 45.0

                pose_high = pose_hover.copy()
                pose_high["shoulder_lift.pos"] = min(-10.0, pose_hover["shoulder_lift.pos"] - 15.0)
                pose_high["wrist_flex.pos"] = max(20.0, pose_hover["wrist_flex.pos"] + 15.0)

                logging.info(
                    "🎯 [TAUGHT RBF MODEL] Evaluated demonstration joint model for %s at R=%.2fm (pan_bias=%s)",
                    color, radius, dir_tag
                )
                return pose_high, pose_hover
            except Exception as e:
                logging.warning("Taught model evaluation failed for %s: %s (falling back to IK)", color, e)

        # Solve IK for approach_high and hover
        if target_xyz is not None and self.kin is not None:
            try:
                radius = float(np.hypot(target_xyz[0], target_xyz[1]))
                if orientation is not None:
                    approach_axis = orientation[:, 2] / np.linalg.norm(orientation[:, 2])
                    h_xy = np.array([approach_axis[0], approach_axis[1], 0.0])
                    norm_xy = float(np.linalg.norm(h_xy))
                    h_xy_unit = h_xy / norm_xy if norm_xy > 1e-6 else np.array([1.0, 0.0, 0.0])

                    # Diagonal look-down with compact horizontal offset (2.5cm):
                    # Positions the gripper nicely above the block (+8cm Z, 2.5cm behind),
                    # minimizing excessive horizontal reach and maximizing vertical shoulder motion!
                    h_offset = 0.025 if radius >= 0.22 else 0.015
                    hover_xyz = target_xyz - h_xy_unit * h_offset + np.array([0.0, 0.0, self.cfg.hover_z_offset_m])
                    approach_high_xyz = hover_xyz
                else:
                    hover_xyz = target_xyz + np.array([0.0, 0.0, self.cfg.hover_z_offset_m])
                    approach_high_xyz = target_xyz + np.array([0.0, 0.0, self.cfg.hover_z_offset_m])

                cur_arr = np.array([current_joint_deg.get(f"{n}.pos", 0.0) for n in JOINT_ORDER])

                solved_high, _ = self.solve_ik_multiseed(cur_arr, approach_high_xyz, keep_orientation=orientation)
                solved_hover, err_h = self.solve_ik_multiseed(solved_high, hover_xyz, keep_orientation=orientation)

                # If tight orientation solve had slight error in near zone, retry with position priority
                if err_h >= 0.02:
                    logging.info("Tight IK err=%.1fmm for %s, retrying with relaxed orientation...", err_h * 1000, color)
                    solved_hover_rel, err_h_rel = self.solve_ik_multiseed(solved_high, hover_xyz, keep_orientation=None)
                    if err_h_rel < err_h:
                        solved_hover, err_h = solved_hover_rel, err_h_rel

                if err_h < 0.025:
                    logging.info("✅ Target IK solved for %s at R=%.2fm (hover err=%.1fmm)", color, radius, err_h * 1000)
                    pose_high = {f"{n}.pos": float(solved_high[i]) for i, n in enumerate(JOINT_ORDER)}
                    pose_high["gripper.pos"] = 45.0
                    pose_hover = {f"{n}.pos": float(solved_hover[i]) for i, n in enumerate(JOINT_ORDER)}
                    pose_hover["gripper.pos"] = 45.0
                    return pose_high, pose_hover
                else:
                    logging.warning("⚠️ IK error too large (%.1fmm) for %s at R=%.2fm", err_h * 1000, color, radius)
            except Exception as e:
                logging.warning("Target IK calculation failed for %s: %s", color, e)

        # Fallback to pregrasp
        hover_fallback = self.resolve_hover_pose(color, observation, current_joint_deg)
        return hover_fallback, hover_fallback

    def resolve_hover_pose(
        self,
        color: str,
        observation: dict[str, Any],
        current_joint_deg: dict[str, float],
    ) -> dict[str, float]:
        """Fallback: returns single hover pose."""
        if self.pregrasp_points:
            for p in self.pregrasp_points:
                if p.get("color", "").lower() == color.lower() or p.get("label", "").lower() == color.lower():
                    pose = {k: float(v) for k, v in p["pose"].items()}
                    pose["gripper.pos"] = 45.0
                    return pose
            pose = {k: float(v) for k, v in self.pregrasp_points[0]["pose"].items()}
            pose["gripper.pos"] = 45.0
            return pose
        return find_target_pose(self.runtime, self.cfg.observe_pose_name, "")

    def resolve_distance_adaptive_apex(
        self,
        start_pose: dict[str, float],
        target_pose: dict[str, float],
        color: str,
    ) -> tuple[dict[str, float], float, float]:
        """Calculates distance-proportional parabolic apex pose, duration, and apex ratio.

        Returns: (apex_pose, total_duration_s, apex_ratio)
        """
        radius = 0.25
        if color.lower() in self._cached_block_coords:
            t_xyz, _ = self._cached_block_coords[color.lower()]
            radius = float(np.hypot(t_xyz[0], t_xyz[1]))

        # Normalized distance [0.15m ~ 0.35m] -> [0.0, 1.0]
        dist_norm = float(np.clip((radius - 0.15) / 0.20, 0.0, 1.0))

        # 1. Distance-proportional duration (1.5s near -> 2.4s far)
        total_duration = 1.5 + 0.9 * dist_norm

        # 2. Distance-proportional apex ratio (apex at 35% time mark)
        apex_ratio = 0.35

        # 3. Distance-proportional parabolic apex joint configuration
        apex_pose = {}
        for k in start_pose:
            q_start = float(start_pose[k])
            q_target = float(target_pose.get(k, q_start))

            if "shoulder_pan" in k:
                # Rotate 60% towards target azimuth during ascent
                apex_pose[k] = 0.4 * q_start + 0.6 * q_target
            elif "shoulder_lift" in k:
                # Proportional parabolic lift: far targets lift higher (-35°), near targets lift moderately (-55°)
                apex_lift = -55.0 + 25.0 * dist_norm
                apex_pose[k] = min(q_start, min(q_target, apex_lift))
            elif "elbow_flex" in k:
                # Elbow flexes cleanly at apex to create parabolic arch and prevent table collision
                apex_pose[k] = 0.5 * q_start + 0.5 * q_target - (12.0 + 10.0 * dist_norm)
            elif "wrist_flex" in k:
                # Wrist pitches up to maintain gripper clearance
                apex_pose[k] = max(q_target + 18.0, 0.5 * q_start + 0.5 * q_target)
            elif "wrist_roll" in k:
                apex_pose[k] = 0.3 * q_start + 0.7 * q_target
            elif "gripper" in k:
                apex_pose[k] = 45.0
            else:
                apex_pose[k] = 0.5 * q_start + 0.5 * q_target

        return apex_pose, total_duration, apex_ratio


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


def compute_clamped_cubic_spline_cmd(
    start_pose: dict[str, float],
    apex_pose: dict[str, float] | None,
    target_pose: dict[str, float],
    t: float,
    total_t: float,
    apex_ratio: float = 0.30,
    wrist_pitch_bump_deg: float = 0.0,
    pan_completion_ratio: float = 1.0,
) -> dict[str, float]:
    """Computes C2-continuous clamped cubic spline (or smooth cosine easing for single segment).

    Properties:
    - Zero starting velocity at t=0
    - If apex_pose is given: passes through apex_pose at t1 = apex_ratio * total_t with continuous velocity & acceleration
    - Smooth zero velocity arrival at target_pose at t=total_t
    """
    keys = [k for k in target_pose if k in start_pose]
    cmd: dict[str, float] = {}

    if total_t <= 1e-6:
        return dict(target_pose)

    t_clamped = max(0.0, min(t, total_t))

    if apex_pose is None:
        # Smooth S-curve (Cosine easing) for single-segment transitions
        s_general = 0.5 * (1.0 - math.cos(math.pi * (t_clamped / total_t)))

        # Earlier completion profile for shoulder_pan (aligns direction at pan_completion_ratio * total_t)
        pan_ratio = max(0.4, min(1.0, pan_completion_ratio))
        t_pan_total = pan_ratio * total_t
        if t_clamped <= t_pan_total:
            s_pan = 0.5 * (1.0 - math.cos(math.pi * (t_clamped / t_pan_total)))
        else:
            s_pan = 1.0

        for k in keys:
            s = s_pan if "shoulder_pan" in k else s_general
            val = (1.0 - s) * float(start_pose[k]) + s * float(target_pose[k])
            # Dynamic mid-flight wrist camera elevation for clear target visibility (negative flex = pitch up towards sky)
            if "wrist_flex" in k and wrist_pitch_bump_deg > 0.0:
                val -= wrist_pitch_bump_deg * math.sin(math.pi * s_general)
            cmd[k] = val
        return cmd

    # Parabolic Arch with Apex at t1
    apex_ratio = max(0.1, min(0.9, apex_ratio))
    t1 = apex_ratio * total_t
    h1 = t1
    h2 = total_t - t1

    for k in keys:
        q0 = float(start_pose[k])
        q1 = float(apex_pose.get(k, target_pose[k]))
        q2 = float(target_pose[k])

        # Special easing for gripper to release cleanly during ascent to apex
        if "gripper" in k:
            target_grip = q2
            if t_clamped <= t1:
                sg = 0.5 * (1.0 - math.cos(math.pi * (t_clamped / t1)))
                cmd[k] = q0 + (target_grip - q0) * sg
            else:
                cmd[k] = target_grip
            continue

        denom = h1 * h2 * (h1 + h2)
        if denom < 1e-9:
            s = 0.5 * (1.0 - math.cos(math.pi * (t_clamped / total_t)))
            cmd[k] = (1.0 - s) * q0 + s * q2
            continue

        # Exact velocity at apex ensuring C2 acceleration continuity:
        # v1 = 1.5 * ((q1 - q0)*h2^2 + (q2 - q1)*h1^2) / (h1 * h2 * (h1 + h2))
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
    apex_ratio: float = 0.30,
    wrist_pitch_bump_deg: float = 0.0,
    pan_completion_ratio: float = 1.0,
) -> RecordControlEvent | None:
    """Executes a single continuous parabolic / arc spline trajectory while recording every frame."""
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

        # Compute smooth spline command
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

        # Follower executes macro
        robot.send_action(macro_cmd)
        # Leader follows follower
        _send_leader_feedback(teleop, macro_cmd)

        # Read and record frame
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
    """Hands control over to the human operator until Space/Enter/C or Arrow keys are pressed."""
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


def _record_one_take_5blocks_episode(
    *,
    robot: Any,
    teleop: Any,
    events: dict[str, bool],
    cfg: HybridOneTakeRecordConfig,
    dataset: LeRobotDataset,
    resolver: TargetHoverResolver,
    observe_target: dict[str, float],
    robot_obs_proc: RobotProcessorPipeline,
    teleop_act_proc: RobotProcessorPipeline,
    robot_act_proc: RobotProcessorPipeline,
) -> RecordControlEvent:
    """Executes the full 5-block one-take sequence with continuous Parabolic/Arc Spline trajectories."""
    colors = [c.strip() for c in cfg.color_sequence.split(",") if c.strip()][: cfg.max_blocks]
    fps = cfg.dataset.fps
    single_task = cfg.dataset.single_task

    print()
    print("=" * 78)
    print(f"🎬 [EPISODE START] Recording 5-Block Sequence: {' -> '.join(colors)}")
    print("=" * 78)

    # Reset all control event flags at the start of each episode
    events["exit_early"] = False
    events["rerecord_episode"] = False
    events["next_phase"] = False

    # 1. At initial observe pose, cache scene block positions
    init_obs = robot.get_observation()
    if cfg.top_key in init_obs:
        resolver.cache_scene_blocks(init_obs[cfg.top_key])

    for block_idx, color in enumerate(colors):
        print(f"\n--- [BLOCK {block_idx + 1}/{len(colors)}: {color.upper()}] ---")

        # -------------------------------------------------------------
        # 1. AUTO APPROACH: Direct Smooth Real Demonstration Transition
        # -------------------------------------------------------------
        obs = robot.get_observation()
        cur_robot, cur_teleop = _read_current_poses(robot, teleop)
        _, hover_target = resolver.resolve_block_targets(color, obs, cur_robot)

        log_say(f"Approaching {color} block", cfg.play_sounds)

        # Determine target radius R for distance-adaptive wrist camera elevation
        radius = 0.25
        if color.lower() in resolver._cached_block_coords:
            t_xyz, _ = resolver._cached_block_coords[color.lower()]
            radius = float(np.hypot(t_xyz[0], t_xyz[1]))

        # Distance-adaptive wrist lift: near (R=12cm) -> +6°, far (R=37cm) -> +14°
        dist_norm = float(np.clip((radius - 0.12) / 0.25, 0.0, 1.0))
        wrist_bump = 6.0 + 8.0 * dist_norm

        # Distance-adaptive earlier pan alignment: near (R=12cm) -> 90% time, far (R=37cm) -> 75% time
        pan_ratio = 0.90 - 0.15 * dist_norm

        is_direct_slot_transit = (block_idx > 0 and not cfg.return_to_observe_each_block)

        if is_direct_slot_transit:
            # 1. Lift vertically above the slot: raise shoulder + lift elbow + curl wrist up cleanly
            lift_apex_pose = cur_robot.copy()
            # Keep slot's exact shoulder_pan so it lifts purely upwards without sideways swing
            lift_apex_pose["shoulder_pan.pos"] = cur_robot.get("shoulder_pan.pos", 0.0)
            # Raise shoulder up (negative degrees is up on SO-101) by 25 degrees
            lift_apex_pose["shoulder_lift.pos"] = cur_robot.get("shoulder_lift.pos", -40.0) - 25.0
            # Lift elbow up (tuck forearm higher) by 15 degrees
            lift_apex_pose["elbow_flex.pos"] = cur_robot.get("elbow_flex.pos", 45.0) - 15.0
            # Pitch wrist DOWNWARDS (increasing degrees points fingers vertically straight down to ground)
            lift_apex_pose["wrist_flex.pos"] = min(98.0, cur_robot.get("wrist_flex.pos", 75.0) + 20.0)
            lift_apex_pose["gripper.pos"] = 45.0

            lift_dur = 0.6
            transit_dur = cfg.macro_goto_duration_s
            total_dur = lift_dur + transit_dur
            apex_ratio = lift_dur / total_dur  # ~0.23

            print(f"🤖 [AUTO] Vertical lift from slot ({lift_dur:.1f}s) + flight to {color} hover pose ({transit_dur:.1f}s, total={total_dur:.1f}s)...")
            macro_res = _record_auto_spline_trajectory(
                robot=robot,
                teleop=teleop,
                start_robot_pose=cur_robot,
                apex_pose=lift_apex_pose,
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
                apex_ratio=apex_ratio,
            )
        else:
            total_dur = cfg.macro_goto_duration_s
            print(f"🤖 [AUTO] Direct smooth transition from observe pose to {color} hover pose ({total_dur:.1f}s, R={radius*100:.1f}cm, pan align @ {pan_ratio*100:.0f}%, mid-flight wrist lift +{wrist_bump:.1f}°)...")
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
        # 2. MANUAL MANIPULATION: Teleoperate pick & place
        # -------------------------------------------------------------
        log_say(f"Manual {color}", cfg.play_sounds)
        print(f"🙋‍♂️ [MANUAL] Leader unlocked. Pick & place {color}. Press [SPACE] when placed.")
        manual_res = _record_manual_teleop_phase(
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
        if manual_res in {RecordControlEvent.STOP, RecordControlEvent.RERECORD, RecordControlEvent.SAVE}:
            return manual_res

        # -------------------------------------------------------------
        # 3. AUTO RETURN: Direct Shortest-Path Return to Observe pose
        # -------------------------------------------------------------
        is_last_block = (block_idx == len(colors) - 1)
        if cfg.return_to_observe_each_block or is_last_block:
            cur_robot, _ = _read_current_poses(robot, teleop)
            total_dur = cfg.macro_return_duration_s

            # Last block returns to Observe pose while smoothly closing gripper (0.0) as task completion signal
            complete_target = observe_target.copy()
            complete_target["gripper.pos"] = 0.0
            target_return_pose = complete_target if is_last_block else observe_target
            ret_label = "task complete pose (Gripper Closed)" if is_last_block else "observe pose (Gripper Open)"

            print(f"🤖 [AUTO] Direct return to {ret_label} ({total_dur:.1f}s)...")
            ret_res = _record_auto_spline_trajectory(
                robot=robot,
                teleop=teleop,
                start_robot_pose=cur_robot,
                apex_pose=None,
                target_pose=target_return_pose,
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
            )
            if ret_res is not None:
                return ret_res

    # Completed all 5 blocks!
    # Frame recording stops immediately as soon as complete_target is reached (0 extra idle frames recorded!)
    log_say("All 5 blocks completed", cfg.play_sounds)
    print("\n" + "=" * 78)
    print("✅ [5 BLOCKS COMPLETE] Task complete pose reached (Gripper Closed).")
    print("🛑 Frame recording finished. Waiting for operator confirmation:")
    print("👉 Press [→ Right Arrow] to SAVE or [← Left Arrow] to DISCARD.")
    print("=" * 78)

    # Wait for final save/discard confirmation WITHOUT recording extra frames
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


def _wait_for_scene_ready(
    session_idx: int,
    cfg: HybridOneTakeRecordConfig,
    dataset: LeRobotDataset,
    events: dict[str, bool] | None = None,
) -> bool:
    if not cfg.wait_enter_before_episode:
        return True

    print()
    print("#" * 78)
    print(f"[SCENE SETUP] Session {session_idx + 1}/{cfg.dataset.num_episodes} | Dataset Episode {dataset.num_episodes}")
    print(f"[TASK] {cfg.dataset.single_task}")
    print(f"[SEQUENCE] {cfg.color_sequence}")
    print()
    print("1) 5개 블록을 작업공간에 분산 배치하세요.")
    print("2) 목표 구역(타겟 슬롯)이 깨끗한지 확인하세요.")
    print("3) 카메라 시야와 조종 안전 거리를 확보하세요.")
    print()
    print("조작 키 안내:")
    print("  [SPACE] / [ENTER] / [C] : 블록 배치 완료 후 Auto 복귀/다음 블록 전환")
    print("  [→] (Right Arrow)       : 5개 블록 완료 후 에피소드 저장")
    print("  [←] (Left Arrow)        : 에피소드 폐기 및 재촬영")
    print("  [ESC]                   : 촬영 종료")
    print("#" * 78)

    while True:
        try:
            ans = input("준비 완료: ENTER / 종료: q + ENTER > ").strip().lower()
        except EOFError:
            if events is not None:
                events["next_phase"] = False
                events["exit_early"] = False
                events["rerecord_episode"] = False
            return True
        if ans == "":
            if events is not None:
                events["next_phase"] = False
                events["exit_early"] = False
                events["rerecord_episode"] = False
            return True
        if ans in {"q", "quit", "exit"}:
            return False


def _goto_observe_outside(
    robot: Any,
    teleop: Any,
    target: dict[str, float],
    duration_s: float = 2.5,
    fps: int = 30,
    reason: str = "initial setup",
) -> None:
    """Moves both follower and leader to observe pose without adding frames to the dataset."""
    print()
    print("=" * 78)
    print(f"[AUTO GOTO OBSERVE] {reason}")
    print(f"Moving follower + leader to neutral observe pose ({duration_s:.1f}s)...")
    print("=" * 78)
    move_both(
        robot=robot,
        teleop=teleop,
        target=target,
        duration_s=duration_s,
        fps=fps,
    )
    time.sleep(0.2)


def record(
    cfg: HybridOneTakeRecordConfig,
    teleop_action_processor: RobotProcessorPipeline | None = None,
    robot_action_processor: RobotProcessorPipeline | None = None,
    robot_observation_processor: RobotProcessorPipeline | None = None,
) -> LeRobotDataset:
    """Top-level record function for hybrid 5-block one-take dataset collection."""
    init_logging()
    print(f"[RECORDER BUILD] {RECORDER_BUILD}")
    cfg.runtime_config = str(runtime_config_path(cfg.runtime_config))
    logging.info("Using runtime config: %s", cfg.runtime_config)
    logging.info(pformat(asdict(cfg)))

    runtime = load_runtime(cfg.runtime_config)
    observe_target = find_target_pose(runtime, cfg.observe_pose_name, "")

    if cfg.display_data:
        init_rerun(session_name="hybrid_recording", ip=cfg.display_ip, port=cfg.display_port)

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

        resolver = TargetHoverResolver(cfg)

        robot.connect()
        teleop.connect()

        # Move to observe pose at session start outside dataset
        _goto_observe_outside(robot, teleop, observe_target, duration_s=cfg.macro_return_duration_s, fps=cfg.dataset.fps, reason="Session start")

        listener, events = init_hybrid_keyboard_listener()

        with VideoEncodingManager(dataset):
            recorded = 0
            while recorded < cfg.dataset.num_episodes and not events["stop_recording"]:
                if not _wait_for_scene_ready(recorded, cfg, dataset, events=events):
                    break

                expected_ep = dataset.num_episodes
                outcome = _record_one_take_5blocks_episode(
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
                    print(f"🎉 [SAVED] Episode {dataset.num_episodes - 1} committed successfully! ({recorded}/{cfg.dataset.num_episodes})")
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
    """Bind the real dataclass before LeRobot builds the CLI wrapper."""
    from dataclasses import is_dataclass
    if not is_dataclass(HybridOneTakeRecordConfig):
        raise TypeError("HybridOneTakeRecordConfig must remain decorated with @dataclass for draccus.")

    record.__annotations__["cfg"] = HybridOneTakeRecordConfig
    return parser.wrap()(record)


record_cli = _make_record_cli_entrypoint()


def main() -> None:
    register_third_party_plugins()
    record_cli()


if __name__ == "__main__":
    main()
