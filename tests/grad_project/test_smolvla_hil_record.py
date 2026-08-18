import torch

from lerobot.async_inference.helpers import TimedAction
from lerobot.grad_project.recording.smolvla_hil_record import (
    HILAsyncRobotClient,
    HILCommand,
    _hold_follower_at_measured_position,
    decode_hil_key_bytes,
)


def test_hil_terminal_key_mapping():
    commands = decode_hil_key_bytes(b" \r\x1b[C\x1b[Dnpq")

    assert commands == [
        HILCommand.TOGGLE_POLICY,
        HILCommand.START_CORRECTION,
        HILCommand.SAVE_CORRECTION,
        HILCommand.DISCARD_CORRECTION,
        HILCommand.NEXT_TRIAL,
        HILCommand.STOP,
    ]


def test_stale_chunk_rejected_as_a_whole_even_when_tail_is_in_future():
    stale_chunk = [
        TimedAction(timestamp=9.9, timestep=10, action=torch.zeros(6)),
        TimedAction(timestamp=10.2, timestep=11, action=torch.zeros(6)),
    ]
    fresh_chunk = [TimedAction(timestamp=10.1, timestep=12, action=torch.zeros(6))]

    assert not HILAsyncRobotClient.action_chunk_is_fresh(stale_chunk, minimum_timestamp=10.0)
    assert HILAsyncRobotClient.action_chunk_is_fresh(fresh_chunk, minimum_timestamp=10.0)


def test_hil_hold_bypasses_old_policy_goal_and_resets_safety_state():
    class FakeBus:
        def __init__(self):
            self.written = None

        def sync_read(self, register):
            assert register == "Present_Position"
            return {"shoulder_pan": 1.5, "gripper": 22.0}

        def sync_write(self, register, values):
            assert register == "Goal_Position"
            self.written = values.copy()

    class FakeRobot:
        def __init__(self):
            self.bus = FakeBus()
            self._last_goal_pos = {"shoulder_pan": 99.0, "gripper": 99.0}
            self._tracking_error_counts = {"shoulder_pan": 4}
            self._last_action_diagnostics = None

    robot = FakeRobot()
    held = _hold_follower_at_measured_position(robot)

    assert held == {"shoulder_pan.pos": 1.5, "gripper.pos": 22.0}
    assert robot.bus.written == {"shoulder_pan": 1.5, "gripper": 22.0}
    assert robot._last_goal_pos == robot.bus.written
    assert robot._tracking_error_counts == {"shoulder_pan": 0, "gripper": 0}
    assert robot._last_action_diagnostics["event"] == "hil_hold"
