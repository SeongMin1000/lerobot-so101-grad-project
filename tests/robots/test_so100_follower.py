#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest

from lerobot.robots.so_follower import (
    SO100Follower,
    SO100FollowerConfig,
)
from lerobot.robots.utils import ensure_synchronized_goal_position


def _make_bus_mock() -> MagicMock:
    """Return a bus mock with just the attributes used by the robot."""
    bus = MagicMock(name="FeetechBusMock")
    bus.is_connected = False

    def _connect():
        bus.is_connected = True

    def _disconnect(_disable=True):
        bus.is_connected = False

    bus.connect.side_effect = _connect
    bus.disconnect.side_effect = _disconnect

    @contextmanager
    def _dummy_cm():
        yield

    bus.torque_disabled.side_effect = _dummy_cm

    return bus


@pytest.fixture
def follower():
    bus_mock = _make_bus_mock()

    def _bus_side_effect(*_args, **kwargs):
        bus_mock.motors = kwargs["motors"]
        motors_order: list[str] = list(bus_mock.motors)

        bus_mock.sync_read.return_value = {motor: idx for idx, motor in enumerate(motors_order, 1)}
        bus_mock.sync_write.return_value = None
        bus_mock.write.return_value = None
        bus_mock.disable_torque.return_value = None
        bus_mock.enable_torque.return_value = None
        bus_mock.is_calibrated = True
        return bus_mock

    with (
        patch(
            "lerobot.robots.so_follower.so_follower.FeetechMotorsBus",
            side_effect=_bus_side_effect,
        ),
        patch.object(SO100Follower, "configure", lambda self: None),
    ):
        cfg = SO100FollowerConfig(port="/dev/null")
        robot = SO100Follower(cfg)
        yield robot
        if robot.is_connected:
            robot.disconnect()


def test_connect_disconnect(follower):
    assert not follower.is_connected

    follower.connect()
    assert follower.is_connected

    follower.disconnect()
    assert not follower.is_connected


def test_get_observation(follower):
    follower.connect()
    obs = follower.get_observation()

    expected_keys = {f"{m}.pos" for m in follower.bus.motors}
    assert set(obs.keys()) == expected_keys

    for idx, motor in enumerate(follower.bus.motors, 1):
        assert obs[f"{motor}.pos"] == idx


def test_send_action(follower):
    follower.connect()

    action = {f"{m}.pos": i * 10 for i, m in enumerate(follower.bus.motors, 1)}
    returned = follower.send_action(action)

    assert returned == action

    goal_pos = {m: (i + 1) * 10 for i, m in enumerate(follower.bus.motors)}
    follower.bus.sync_write.assert_called_once_with("Goal_Position", goal_pos)


def test_synchronized_goal_position_preserves_joint_ratio():
    goal_reference_pos = {
        "shoulder_lift": (50.0, 0.0),
        "elbow_flex": (-40.0, 0.0),
    }

    safe_goal = ensure_synchronized_goal_position(goal_reference_pos, 2.0)

    assert safe_goal["shoulder_lift"] == pytest.approx(2.0)
    assert safe_goal["elbow_flex"] == pytest.approx(-1.6)


def test_send_action_stops_all_joints_when_one_motor_stalls(follower):
    follower.config.max_relative_target = 2.0
    follower.config.max_tracking_error = 3.0
    follower.config.tracking_error_grace_steps = 1
    follower.connect()

    motors = list(follower.bus.motors)
    present_0 = dict.fromkeys(motors, 0.0)
    present_1 = {**present_0, "elbow_flex": -1.6}
    present_2 = {**present_0, "elbow_flex": -3.2}
    follower.bus.sync_read.side_effect = [present_0, present_1, present_2]

    action = {f"{motor}.pos": 0.0 for motor in motors}
    action["shoulder_lift.pos"] = 50.0
    action["elbow_flex.pos"] = -40.0

    first_sent = follower.send_action(action)
    second_sent = follower.send_action(action)

    assert first_sent["shoulder_lift.pos"] == pytest.approx(2.0)
    assert first_sent["elbow_flex.pos"] == pytest.approx(-1.6)
    assert second_sent["shoulder_lift.pos"] == pytest.approx(4.0)
    assert second_sent["elbow_flex.pos"] == pytest.approx(-3.2)
    assert follower.last_action_diagnostics["event"] == "command"
    assert follower.last_action_diagnostics["tracking_error"]["shoulder_lift"] == pytest.approx(2.0)

    with pytest.raises(RuntimeError, match="Motor tracking error exceeded"):
        follower.send_action(action)

    follower.bus.sync_write.assert_called_with("Goal_Position", present_2)
    assert follower.last_action_diagnostics["event"] == "tracking_abort"
    assert follower.last_action_diagnostics["tracking_error"]["shoulder_lift"] == pytest.approx(4.0)
