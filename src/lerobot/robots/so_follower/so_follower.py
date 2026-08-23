#!/usr/bin/env python

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

import logging
import time
from functools import cached_property

from lerobot.cameras import make_cameras_from_configs
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from ..utils import ensure_synchronized_goal_position
from .config_so_follower import SOFollowerRobotConfig

logger = logging.getLogger(__name__)


class SOFollower(Robot):
    """
    Generic SO follower base implementing common functionality for SO-100/101/10X.
    Designed to be subclassed with a per-hardware-model `config_class` and `name`.
    """

    config_class = SOFollowerRobotConfig
    name = "so_follower"

    def __init__(self, config: SOFollowerRobotConfig):
        super().__init__(config)
        self.config = config
        # choose normalization mode depending on config if available
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "wrist_roll": Motor(5, "sts3215", norm_mode_body),
                "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
            },
            calibration=self.calibration,
        )
        self.cameras = make_cameras_from_configs(config.cameras)
        self._last_goal_pos: dict[str, float] | None = None
        self._tracking_error_counts: dict[str, int] = {}
        self._last_action_diagnostics: dict[str, object] | None = None

        if self.config.max_tracking_error is not None and self.config.max_tracking_error < 0:
            raise ValueError("max_tracking_error must be non-negative.")
        if self.config.tracking_error_grace_steps < 1:
            raise ValueError("tracking_error_grace_steps must be at least 1.")

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {f"{motor}.pos": float for motor in self.bus.motors}

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._motors_ft

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        """
        We assume that at connection time, arm is in a rest position,
        and torque can be safely disabled to run calibration.
        """

        self.bus.connect()
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        self._reset_action_safety_state()
        logger.info(f"{self} connected.")

    def _reset_action_safety_state(self) -> None:
        self._last_goal_pos = None
        self._tracking_error_counts = {}
        self._last_action_diagnostics = None

    @property
    def last_action_diagnostics(self) -> dict[str, object] | None:
        """Last requested/sent/measured goal snapshot for the async flight recorder."""
        return self._last_action_diagnostics

    def _stop_on_tracking_error(
        self,
        present_pos: dict[str, float],
        commanded_motors: set[str],
    ) -> None:
        """Hold every commanded joint and abort when one motor persistently stalls."""

        if self._last_goal_pos is None or self.config.max_tracking_error is None:
            return

        threshold = self.config.max_tracking_error
        violations: dict[str, dict[str, float | int]] = {}
        for motor in commanded_motors:
            tracking_error = abs(self._last_goal_pos[motor] - present_pos[motor])
            if tracking_error > threshold:
                count = self._tracking_error_counts.get(motor, 0) + 1
                self._tracking_error_counts[motor] = count
                if count >= self.config.tracking_error_grace_steps:
                    violations[motor] = {
                        "last goal_pos": self._last_goal_pos[motor],
                        "present_pos": present_pos[motor],
                        "tracking_error": tracking_error,
                        "consecutive_steps": count,
                    }
            else:
                self._tracking_error_counts[motor] = 0

        if not violations:
            return

        hold_pos = {motor: present_pos[motor] for motor in commanded_motors}
        self.bus.sync_write("Goal_Position", hold_pos)
        self._last_goal_pos = hold_pos
        self._tracking_error_counts = {}
        message = (
            "Motor tracking error exceeded the safety limit; all commanded joints were held and "
            f"inference must stop. Details: {violations}"
        )
        logger.error(message)
        raise RuntimeError(message)

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        if self.calibration:
            # Calibration file exists, ask user whether to use it or run new calibration
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
                self.bus.write_calibration(self.calibration)
                return

        logger.info(f"\nRunning calibration of {self}")
        self.bus.disable_torque()
        for motor in self.bus.motors:
            self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

        input(f"Move {self} to the middle of its range of motion and press ENTER....")
        homing_offsets = self.bus.set_half_turn_homings()

        # Attempt to call record_ranges_of_motion with a reduced motor set when appropriate.
        full_turn_motor = "wrist_roll"
        unknown_range_motors = [motor for motor in self.bus.motors if motor != full_turn_motor]
        print(
            f"Move all joints except '{full_turn_motor}' sequentially through their "
            "entire ranges of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.bus.record_ranges_of_motion(unknown_range_motors)
        range_mins[full_turn_motor] = 1350
        range_maxes[full_turn_motor] = 3110

        self.calibration = {}
        for motor, m in self.bus.motors.items():
            self.calibration[motor] = MotorCalibration(
                id=m.id,
                drive_mode=0,
                homing_offset=homing_offsets[motor],
                range_min=range_mins[motor],
                range_max=range_maxes[motor],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print("Calibration saved to", self.calibration_fpath)

    def configure(self) -> None:
        with self.bus.torque_disabled():
            self.bus.configure_motors()
            for motor in self.bus.motors:
                self.bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
                # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
                self.bus.write("P_Coefficient", motor, 16)
                # Set I_Coefficient and D_Coefficient to default value 0 and 32
                self.bus.write("I_Coefficient", motor, 0)
                self.bus.write("D_Coefficient", motor, 32)

                if motor == "gripper":
                    # gripper_close_value targets the fully-closed position, so a
                    # held block never lets the servo reach it -- it pushes at
                    # this cap continuously for as long as the grasp is held,
                    # not just on contact. 50% was still enough to shatter a
                    # printed jaw, so this caps it well below that.
                    self.bus.write("Max_Torque_Limit", motor, 200)  # 20% of max torque
                    self.bus.write("Protection_Current", motor, 100)  # 20% of max current
                    self.bus.write("Overload_Torque", motor, 25)  # 25% torque when overloaded

    def setup_motors(self) -> None:
        for motor in reversed(self.bus.motors):
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        # Read arm position
        start = time.perf_counter()
        obs_dict = self.bus.sync_read("Present_Position")
        obs_dict = {f"{motor}.pos": val for motor, val in obs_dict.items()}
        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.read_latest()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """Command arm to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Raises:
            RobotDeviceNotConnectedError: if robot is not connected.

        Returns:
            RobotAction: the action sent to the motors, potentially clipped.
        """

        goal_pos = {key.removesuffix(".pos"): val for key, val in action.items() if key.endswith(".pos")}
        requested_goal_pos = goal_pos.copy()
        present_pos: dict[str, float] | None = None
        previous_goal_pos: dict[str, float] | None = None

        # Rate-limit the complete goal vector from the previous command, rather
        # than clipping each joint independently from its measured position.
        # A tracking watchdog stops every joint if one motor cannot keep up.
        if self.config.max_relative_target is not None:
            present_pos = self.bus.sync_read("Present_Position")
            commanded_motors = set(goal_pos)

            if self._last_goal_pos is None or set(self._last_goal_pos) != commanded_motors:
                self._last_goal_pos = {motor: present_pos[motor] for motor in commanded_motors}
                self._tracking_error_counts = dict.fromkeys(commanded_motors, 0)
            else:
                previous_goal_pos = self._last_goal_pos.copy()
                try:
                    self._stop_on_tracking_error(present_pos, commanded_motors)
                except RuntimeError:
                    self._last_action_diagnostics = {
                        "event": "tracking_abort",
                        "requested_goal_pos": requested_goal_pos,
                        "sent_goal_pos": self._last_goal_pos.copy(),
                        "previous_goal_pos": previous_goal_pos,
                        "present_pos": present_pos.copy(),
                        "tracking_error": {
                            motor: abs(previous_goal_pos[motor] - present_pos[motor])
                            for motor in commanded_motors
                        },
                    }
                    raise

            if previous_goal_pos is None:
                previous_goal_pos = self._last_goal_pos.copy()

            goal_reference_pos = {
                motor: (desired_pos, self._last_goal_pos[motor])
                for motor, desired_pos in goal_pos.items()
            }
            goal_pos = ensure_synchronized_goal_position(
                goal_reference_pos, self.config.max_relative_target
            )
            self._last_goal_pos = goal_pos.copy()
        else:
            self._reset_action_safety_state()

        # Send goal position to the arm
        self.bus.sync_write("Goal_Position", goal_pos)
        self._last_action_diagnostics = {
            "event": "command",
            "requested_goal_pos": requested_goal_pos,
            "sent_goal_pos": goal_pos.copy(),
            "previous_goal_pos": previous_goal_pos,
            "present_pos": None if present_pos is None else present_pos.copy(),
            "tracking_error": (
                {}
                if present_pos is None or previous_goal_pos is None
                else {
                    motor: abs(previous_goal_pos[motor] - present_pos[motor])
                    for motor in goal_pos
                }
            ),
        }
        return {f"{motor}.pos": val for motor, val in goal_pos.items()}

    @check_if_not_connected
    def disconnect(self):
        self._reset_action_safety_state()
        self.bus.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")


SO100Follower = SOFollower
SO101Follower = SOFollower
