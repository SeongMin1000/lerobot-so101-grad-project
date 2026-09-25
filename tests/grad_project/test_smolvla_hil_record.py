import json
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest
import torch

from lerobot.async_inference.helpers import TimedAction
from lerobot.grad_project.recording.smolvla_hil_record import (
    HardcodedHoverResolver,
    HILAsyncRobotClient,
    HILCommand,
    HILPhase,
    HILRecordConfig,
    HILSession,
    TrialOutcome,
    _hold_follower_at_measured_position,
    decode_hil_key_bytes,
)


def test_hil_terminal_key_mapping():
    data = b" lLoOmMcC\r\n\x1b[C\x1b[DrRnNqQ\x1bzZxX"
    commands = decode_hil_key_bytes(data)

    assert commands == [
        HILCommand.PAUSE_INTERVENTION,            # ' '
        HILCommand.START_LOCAL_CORRECTION,        # 'l'
        HILCommand.START_LOCAL_CORRECTION,        # 'L'
        HILCommand.START_OBSERVE_CORRECTION,      # 'o'
        HILCommand.START_OBSERVE_CORRECTION,      # 'O'
        HILCommand.START_STACK_HOVER_CORRECTION,  # 'm'
        HILCommand.START_STACK_HOVER_CORRECTION,  # 'M'
        HILCommand.FREEZE_CORRECTION,             # 'c'
        HILCommand.FREEZE_CORRECTION,             # 'C'
        HILCommand.START_DIRECT_CORRECTION,       # '\r' (Enter)
        HILCommand.START_DIRECT_CORRECTION,       # '\n' (Enter)
        HILCommand.SAVE_FROZEN_CORRECTION,        # Right arrow
        HILCommand.DISCARD_FROZEN_CORRECTION,     # Left arrow
        HILCommand.RESUME_AUTONOMOUS_VIA_OBSERVE, # 'r'
        HILCommand.RESUME_AUTONOMOUS_VIA_OBSERVE, # 'R'
        HILCommand.NEXT_TRIAL,                    # 'n'
        HILCommand.NEXT_TRIAL,                    # 'N'
        HILCommand.STOP,                          # 'q'
        HILCommand.STOP,                          # 'Q'
        HILCommand.STOP,                          # ESC
        HILCommand.START_OBSERVE_CORRECTION,      # 'z' (alias)
        HILCommand.START_OBSERVE_CORRECTION,      # 'Z' (alias)
        HILCommand.START_LOCAL_CORRECTION,        # 'x' (alias)
        HILCommand.START_LOCAL_CORRECTION,        # 'X' (alias)
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


def test_hil_record_config_rac_defaults():
    cfg = HILRecordConfig(
        robot=MagicMock(),
        teleop=MagicMock(),
        dataset=MagicMock(single_task="test task", episode_time_s=60.0),
        server_address="localhost:8080",
        pretrained_name_or_path="dummy",
    )
    assert cfg.record_mode == "corrections_only"
    assert cfg.use_yolo_recovery is True
    assert cfg.yolo_model_path == "project/models/yolo_block_detector/best.pt"
    assert cfg.local_retract_ratio == 0.4
    assert cfg.local_retract_lock_pan is True
    assert cfg.pre_intervention_seconds == 1.5


def test_hil_retract_joint_backtrack():
    cur_robot = {
        "shoulder_pan.pos": -35.0,
        "shoulder_lift.pos": -20.0,
        "elbow_flex.pos": 5.0,
        "wrist_flex.pos": 35.0,
        "wrist_roll.pos": -15.0,
        "gripper.pos": 15.0,
    }
    observe_target = {
        "shoulder_pan.pos": -0.97,
        "shoulder_lift.pos": -87.43,
        "elbow_flex.pos": 16.66,
        "wrist_flex.pos": 95.47,
        "wrist_roll.pos": -7.03,
        "gripper.pos": 45.0,
    }

    ratio = 0.4
    lock_pan = True
    hover_target = {}
    for k in cur_robot:
        if k == "gripper.pos":
            hover_target[k] = 45.0
        elif k == "shoulder_pan.pos" and lock_pan:
            hover_target[k] = cur_robot[k]
        elif k in observe_target:
            hover_target[k] = (1.0 - ratio) * cur_robot[k] + ratio * observe_target[k]

    # Pan is strictly locked to current block direction
    assert hover_target["shoulder_pan.pos"] == -35.0
    # Gripper is fully open
    assert hover_target["gripper.pos"] == 45.0
    # Shoulder lift retreats upwards by 40%
    expected_lift = -20.0 + 0.4 * (-87.43 - (-20.0))
    assert abs(hover_target["shoulder_lift.pos"] - expected_lift) < 1e-4
    # Wrist flex retreats upwards by 40%
    expected_wrist = 35.0 + 0.4 * (95.47 - 35.0)
    assert abs(hover_target["wrist_flex.pos"] - expected_wrist) < 1e-4


def test_hardcoded_hover_resolver_evaluates_without_ik():
    cfg = HILRecordConfig(
        robot=MagicMock(),
        teleop=MagicMock(),
        dataset=MagicMock(fps=30, single_task="Pick up red block", episode_time_s=60.0),
        server_address="localhost:8080",
        pretrained_name_or_path="dummy",
        use_yolo_detection=False,
    )
    resolver = HardcodedHoverResolver(cfg)

    # Test with target_xy coordinates (simulating center table at 22cm forward)
    target_xy = np.array([0.22, 0.0])
    cur_joints = {
        "shoulder_pan.pos": 0.0,
        "shoulder_lift.pos": -80.0,
        "elbow_flex.pos": 20.0,
        "wrist_flex.pos": 90.0,
        "wrist_roll.pos": 0.0,
        "gripper.pos": 45.0,
    }

    hover_pose = resolver.resolve_hover_pose(target_xy, "red", cur_joints)
    assert isinstance(hover_pose, dict)
    assert "shoulder_pan.pos" in hover_pose
    assert "shoulder_lift.pos" in hover_pose
    assert "elbow_flex.pos" in hover_pose
    assert "wrist_flex.pos" in hover_pose
    assert "wrist_roll.pos" in hover_pose
    assert hover_pose["gripper.pos"] == 45.0


def test_correction_only_pre_buffer_bypassed():
    cfg = HILRecordConfig(
        robot=MagicMock(),
        teleop=MagicMock(),
        dataset=MagicMock(fps=30, single_task="test", episode_time_s=60.0),
        server_address="localhost:8080",
        pretrained_name_or_path="dummy",
        record_mode="corrections_only",
        pre_intervention_seconds=1.5,
        use_yolo_recovery=False,
    )
    dataset_mock = MagicMock()
    added_frames = []
    dataset_mock.add_frame.side_effect = lambda f: added_frames.append(f)
    dataset_mock.num_episodes = 0
    client_mock = MagicMock()

    session = HILSession(
        cfg=cfg,
        client=client_mock,
        teleop=MagicMock(),
        dataset=dataset_mock,
        teleop_action_processor=MagicMock(),
        robot_action_processor=MagicMock(),
        robot_observation_processor=MagicMock(),
        display_compressed_images=False,
    )

    # In corrections_only mode, flush should return 0 and add no frames
    session.pre_intervention_buffer.append({"dummy": 1})
    flushed = session._flush_pre_intervention_buffer()
    assert flushed == 0
    assert len(added_frames) == 0


def test_hil_fsm_correction_only_workflow(tmp_path):
    cfg = HILRecordConfig(
        robot=MagicMock(),
        teleop=MagicMock(),
        dataset=MagicMock(
            fps=30,
            single_task="Pick up the red block and place it in the red target slot.",
            episode_time_s=60.0,
            num_episodes=10,
            root=str(tmp_path),
        ),
        server_address="localhost:8080",
        pretrained_name_or_path="dummy",
        record_mode="corrections_only",
        use_yolo_recovery=False,
        macro_return_duration_s=0.1,
        local_retract_duration_s=0.1,
    )

    added_frames = []
    dataset_mock = MagicMock()
    dataset_mock.add_frame.side_effect = lambda f: added_frames.append(f)
    dataset_mock.num_episodes = 0
    dataset_mock.has_pending_frames.side_effect = lambda: len(added_frames) > 0
    dataset_mock.features = {}
    dataset_mock.root = str(tmp_path)

    def mock_save():
        dataset_mock.num_episodes += 1
    dataset_mock.save_episode.side_effect = mock_save
    dataset_mock.clear_episode_buffer.side_effect = lambda *args, **kwargs: added_frames.clear()

    client_mock = MagicMock()
    robot_mock = MagicMock()
    robot_mock.action_features = [
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
    ]
    robot_mock.get_observation.return_value = {
        "shoulder_pan.pos": -10.0,
        "shoulder_lift.pos": -20.0,
        "elbow_flex.pos": 10.0,
        "wrist_flex.pos": 40.0,
        "wrist_roll.pos": -5.0,
        "gripper.pos": 10.0,
    }
    robot_mock.send_action = MagicMock(side_effect=lambda a: a)
    robot_mock.bus = None
    client_mock.robot = robot_mock

    teleop_mock = MagicMock()
    teleop_mock.action_features = robot_mock.action_features
    teleop_mock.get_action.return_value = {
        "shoulder_pan.pos": -10.0,
        "shoulder_lift.pos": -20.0,
        "elbow_flex.pos": 10.0,
        "wrist_flex.pos": 40.0,
        "wrist_roll.pos": -5.0,
        "gripper.pos": 10.0,
    }
    teleop_mock.send_feedback = MagicMock(side_effect=lambda a: a)

    observe_target = {
        "shoulder_pan.pos": 0.0,
        "shoulder_lift.pos": -80.0,
        "elbow_flex.pos": 20.0,
        "wrist_flex.pos": 90.0,
        "wrist_roll.pos": 0.0,
        "gripper.pos": 45.0,
    }

    session = HILSession(
        cfg=cfg,
        client=client_mock,
        teleop=teleop_mock,
        dataset=dataset_mock,
        teleop_action_processor=lambda x: x[0],
        robot_action_processor=lambda x: x[0],
        robot_observation_processor=lambda x: x,
        display_compressed_images=False,
        observe_target=observe_target,
    )

    # 1. Trial starts in AUTONOMOUS
    session._resume_autonomous()
    assert session.phase is HILPhase.AUTONOMOUS
    assert len(added_frames) == 0

    # 2. User presses SPACE -> pauses & aligns leader (no frames recorded!)
    session._handle_command(HILCommand.PAUSE_INTERVENTION)
    assert session.phase is HILPhase.INTERVENTION_PAUSED
    assert len(added_frames) == 0
    assert client_mock.pause_policy_control.called

    # 3. User presses L -> starts Local Correction (retract recorded, enters LOCAL_RECORDING)
    session._handle_command(HILCommand.START_LOCAL_CORRECTION)
    assert session.phase is HILPhase.LOCAL_RECORDING
    retract_frame_count = len(added_frames)
    assert retract_frame_count > 0  # Retract frames are recorded!
    assert teleop_mock.disable_torque.called

    # 4. User presses C -> freezes recording -> REVIEW_PAUSED
    session._handle_command(HILCommand.FREEZE_CORRECTION)
    assert session.phase is HILPhase.REVIEW_PAUSED

    # 5. User presses Left Arrow (←) -> discards current frozen correction
    session._handle_command(HILCommand.DISCARD_FROZEN_CORRECTION)
    assert session.phase is HILPhase.INTERVENTION_PAUSED
    assert dataset_mock.num_episodes == 0

    # 6. User presses O -> starts Observe Correction (observe return + hover flight recorded)
    added_frames.clear()
    session._handle_command(HILCommand.START_OBSERVE_CORRECTION)
    assert session.phase is HILPhase.OBSERVE_RECORDING
    assert len(added_frames) > 0

    # 7. User presses C -> freezes recording -> REVIEW_PAUSED
    session._handle_command(HILCommand.FREEZE_CORRECTION)
    assert session.phase is HILPhase.REVIEW_PAUSED

    # 8. User presses Right Arrow (→) -> saves frozen correction episode
    outcome = session._handle_command(HILCommand.SAVE_FROZEN_CORRECTION)
    assert session.phase is HILPhase.INTERVENTION_PAUSED
    assert dataset_mock.save_episode.called
    assert dataset_mock.num_episodes == 1
    assert session.saved_corrections == 1

    # Verify metadata was written
    meta_path = Path(tmp_path) / "meta" / "episode_interventions.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert "0" in meta
    assert meta["0"]["recovery_type"] == "observe"
    assert meta["0"]["total_frames"] > 0
    assert meta["0"]["intervention_start_frame"] == 0

    # 9. User presses SPACE in INTERVENTION_PAUSED -> resumes directly from current pose
    session.phase = HILPhase.INTERVENTION_PAUSED
    session._handle_command(HILCommand.PAUSE_INTERVENTION)
    assert session.phase is HILPhase.AUTONOMOUS
    assert client_mock.resume_policy_control.called

    # 10. User presses R in AUTONOMOUS -> resets back to OBSERVE and restarts trial from scratch
    frames_before_r = len(added_frames)
    session._handle_command(HILCommand.RESUME_AUTONOMOUS_VIA_OBSERVE)
    assert session.phase is HILPhase.AUTONOMOUS
    assert len(added_frames) == frames_before_r  # OBSERVE return is NOT recorded in dataset!
    assert client_mock.resume_policy_control.called

    # 11. User pauses with SPACE, then presses Enter -> starts DIRECT_RECORDING (no retract)
    session._handle_command(HILCommand.PAUSE_INTERVENTION)
    assert session.phase is HILPhase.INTERVENTION_PAUSED
    session._handle_command(HILCommand.START_DIRECT_CORRECTION)
    assert session.phase is HILPhase.DIRECT_RECORDING
    assert session._current_recovery_type == "direct"

    # Press Enter again while recording -> freezes correction -> REVIEW_PAUSED
    session._handle_command(HILCommand.START_DIRECT_CORRECTION)
    assert session.phase is HILPhase.REVIEW_PAUSED


def test_hil_fsm_stack_hover_workflow(tmp_path):
    cfg = HILRecordConfig(
        robot=MagicMock(),
        teleop=MagicMock(),
        dataset=MagicMock(
            fps=30,
            single_task="Stack the red block onto the yellow block.",
            episode_time_s=60.0,
            num_episodes=10,
            root=str(tmp_path),
        ),
        server_address="localhost:8080",
        pretrained_name_or_path="dummy",
        record_mode="corrections_only",
        use_yolo_recovery=False,
        macro_goto_duration_s=0.1,
    )

    added_frames = []
    dataset_mock = MagicMock()
    dataset_mock.add_frame.side_effect = lambda f: added_frames.append(f)
    dataset_mock.num_episodes = 0
    dataset_mock.has_pending_frames.side_effect = lambda: len(added_frames) > 0
    dataset_mock.features = {}
    dataset_mock.root = str(tmp_path)

    def mock_save():
        dataset_mock.num_episodes += 1

    dataset_mock.save_episode.side_effect = mock_save
    dataset_mock.clear_episode_buffer.side_effect = lambda *args, **kwargs: added_frames.clear()

    client_mock = MagicMock()
    robot_mock = MagicMock()
    robot_mock.action_features = [
        "shoulder_pan.pos",
        "shoulder_lift.pos",
        "elbow_flex.pos",
        "wrist_flex.pos",
        "wrist_roll.pos",
        "gripper.pos",
    ]
    # Current robot pose: holding a block with gripper closed at 18.5 deg
    robot_mock.get_observation.return_value = {
        "shoulder_pan.pos": 10.0,
        "shoulder_lift.pos": -25.0,
        "elbow_flex.pos": 5.0,
        "wrist_flex.pos": 45.0,
        "wrist_roll.pos": -10.0,
        "gripper.pos": 18.5,
    }
    sent_robot_actions = []
    robot_mock.send_action = MagicMock(side_effect=lambda a: sent_robot_actions.append(dict(a)) or a)
    robot_mock.bus = None
    client_mock.robot = robot_mock

    teleop_mock = MagicMock()
    teleop_mock.action_features = robot_mock.action_features
    teleop_mock.get_action.return_value = dict(robot_mock.get_observation.return_value)
    teleop_mock.send_feedback = MagicMock(side_effect=lambda a: a)

    session = HILSession(
        cfg=cfg,
        client=client_mock,
        teleop=teleop_mock,
        dataset=dataset_mock,
        teleop_action_processor=lambda x: x[0],
        robot_action_processor=lambda x: x[0],
        robot_observation_processor=lambda x: x,
        display_compressed_images=False,
    )

    # Verify resolve_stack_hover_pose preserves current gripper angle
    hover_pose = session.resolve_stack_hover_pose(cur_gripper=18.5)
    assert hover_pose["gripper.pos"] == 18.5
    assert "shoulder_pan.pos" in hover_pose
    assert "wrist_flex.pos" in hover_pose

    # 1. Trial starts in AUTONOMOUS
    session._resume_autonomous()
    assert session.phase is HILPhase.AUTONOMOUS
    assert len(added_frames) == 0

    # 2. Press M while running autonomous -> pauses policy control and starts stack hover
    session._handle_command(HILCommand.START_STACK_HOVER_CORRECTION)
    assert client_mock.pause_policy_control.called
    assert session.phase is HILPhase.STACK_RECORDING
    assert session._current_recovery_type == "stack_hover"
    assert session._current_target_block == "stack_target"
    assert teleop_mock.disable_torque.called
    # Transit frames were recorded into dataset
    transit_frame_count = len(added_frames)
    assert transit_frame_count > 0
    # Commanded robot actions to stack_hover firmly grips block at 0.0 deg (since 18.5 <= 27.0 deg)
    assert sent_robot_actions[-1]["gripper.pos"] == pytest.approx(0.0, abs=1e-3)

    # 3. Freeze with C -> REVIEW_PAUSED
    session._handle_command(HILCommand.FREEZE_CORRECTION)
    assert session.phase is HILPhase.REVIEW_PAUSED

    # 4. Save with Right Arrow (→)
    outcome = session._handle_command(HILCommand.SAVE_FROZEN_CORRECTION)
    assert session.phase is HILPhase.INTERVENTION_PAUSED
    assert dataset_mock.save_episode.called
    assert dataset_mock.num_episodes == 1
    assert session.saved_corrections == 1

    # Verify metadata saved with recovery_type="stack_hover" and task_phase="stack"
    meta_path = Path(tmp_path) / "meta" / "episode_interventions.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert "0" in meta
    assert meta["0"]["recovery_type"] == "stack_hover"
    assert meta["0"]["task_phase"] == "stack"
    assert meta["0"]["total_frames"] == transit_frame_count


def test_hil_observe_target_outside_priority():
    from lerobot.grad_project.perception.opencv_block_detector import BlockDetection

    priority_order = ["red", "yellow", "wood", "green", "blue"]
    priority_map = {c: idx for idx, c in enumerate(priority_order)}

    # Case: Red and Yellow already inside target; Wood, Green, Blue outside.
    # Highest priority outside block should be WOOD.
    blocks = [
        BlockDetection(color="red", cx=300, cy=250, area=1000, angle_deg=0, in_target=True, bbox=(0, 0, 20, 20)),
        BlockDetection(color="yellow", cx=350, cy=250, area=1000, angle_deg=0, in_target=True, bbox=(0, 0, 20, 20)),
        BlockDetection(color="blue", cx=100, cy=100, area=1000, angle_deg=0, in_target=False, bbox=(0, 0, 20, 20)),
        BlockDetection(color="wood", cx=200, cy=150, area=1000, angle_deg=0, in_target=False, bbox=(0, 0, 20, 20)),
        BlockDetection(color="green", cx=150, cy=120, area=1000, angle_deg=0, in_target=False, bbox=(0, 0, 20, 20)),
    ]

    outside_blocks = [b for b in blocks if not getattr(b, "in_target", False)]
    sorted_cands = sorted(
        outside_blocks,
        key=lambda b: (priority_map.get(b.color.lower(), 999), -b.cy, b.cx),
    )
    assert sorted_cands[0].color == "wood"

    # Case: Yellow was placed into target first (e.g. out of order by policy);
    # Red, Wood, Green, Blue are outside.
    # Highest priority outside block must remain RED!
    blocks2 = [
        BlockDetection(color="yellow", cx=350, cy=250, area=1000, angle_deg=0, in_target=True, bbox=(0, 0, 20, 20)),
        BlockDetection(color="blue", cx=100, cy=100, area=1000, angle_deg=0, in_target=False, bbox=(0, 0, 20, 20)),
        BlockDetection(color="green", cx=150, cy=120, area=1000, angle_deg=0, in_target=False, bbox=(0, 0, 20, 20)),
        BlockDetection(color="red", cx=200, cy=150, area=1000, angle_deg=0, in_target=False, bbox=(0, 0, 20, 20)),
        BlockDetection(color="wood", cx=250, cy=180, area=1000, angle_deg=0, in_target=False, bbox=(0, 0, 20, 20)),
    ]
    outside_blocks2 = [b for b in blocks2 if not getattr(b, "in_target", False)]
    sorted_cands2 = sorted(
        outside_blocks2,
        key=lambda b: (priority_map.get(b.color.lower(), 999), -b.cy, b.cx),
    )
    assert sorted_cands2[0].color == "red"


