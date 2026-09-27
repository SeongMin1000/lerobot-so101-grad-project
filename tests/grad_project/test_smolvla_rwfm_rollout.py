import json
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest
import torch

from lerobot.async_inference.helpers import TimedAction
from lerobot.grad_project.recording.smolvla_hil_record import (
    HILCommand,
    HILPhase,
    HILRecordConfig,
    HILSession,
    TrialOutcome,
    decode_hil_key_bytes,
)


def _create_mock_session(tmp_path: Path, record_mode: str = "rwfm_rollout", failure_tail_frames: int = 0):
    cfg = HILRecordConfig(
        robot=MagicMock(),
        teleop=MagicMock(),
        dataset=MagicMock(
            fps=30,
            single_task="Pick up the green block and place it in target zone.",
            episode_time_s=60.0,
            num_episodes=10,
            root=str(tmp_path),
        ),
        server_address="localhost:8080",
        pretrained_name_or_path="dummy",
        record_mode=record_mode,
        failure_tail_frames=failure_tail_frames,
        use_yolo_recovery=False,
        use_yolo_detection=False,
        macro_return_duration_s=0.01,
        local_retract_duration_s=0.01,
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
        added_frames.clear()

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
        "shoulder_pan.pos": 0.0,
        "shoulder_lift.pos": -30.0,
        "elbow_flex.pos": 10.0,
        "wrist_flex.pos": 40.0,
        "wrist_roll.pos": 0.0,
        "gripper.pos": 45.0,
    }
    robot_mock.send_action = MagicMock(side_effect=lambda a: a)
    robot_mock.bus = None
    client_mock.robot = robot_mock
    client_mock.actions_available.return_value = True
    client_mock.control_loop_action.return_value = {
        "shoulder_pan.pos": 0.1,
        "shoulder_lift.pos": -30.1,
        "elbow_flex.pos": 10.1,
        "wrist_flex.pos": 40.1,
        "wrist_roll.pos": 0.1,
        "gripper.pos": 45.0,
    }
    client_mock._ready_to_send_observation.return_value = False

    teleop_mock = MagicMock()
    teleop_mock.action_features = robot_mock.action_features
    teleop_mock.get_action.return_value = dict(robot_mock.get_observation.return_value)
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
    return session, added_frames, dataset_mock, client_mock


def test_rwfm_terminal_key_mapping():
    data = b"sSfF "
    commands = decode_hil_key_bytes(data)
    assert commands == [
        HILCommand.SET_SELF_CORRECTION,
        HILCommand.SET_SELF_CORRECTION,
        HILCommand.SET_FAILURE,
        HILCommand.SET_FAILURE,
        HILCommand.PAUSE_INTERVENTION,
    ]


def test_rwfm_rollout_trial_start_and_frame_recording(tmp_path):
    session, added_frames, _, _ = _create_mock_session(tmp_path)
    session.physical_trial = 1
    session.is_trial_recording = True
    session._expected_episode_index = 0
    session._rwfm_segments = []
    session._resume_autonomous()

    assert session.phase is HILPhase.AUTONOMOUS
    assert session.is_trial_recording is True
    assert len(added_frames) == 0

    # Simulate 5 ticks
    for _ in range(5):
        session._autonomous_tick()
    assert len(added_frames) == 5


def test_running_space_evaluation_paused_no_frame_increment(tmp_path):
    session, added_frames, _, client_mock = _create_mock_session(tmp_path)
    session.is_trial_recording = True
    session._expected_episode_index = 0
    session._resume_autonomous()

    for _ in range(10):
        session._autonomous_tick()
    assert len(added_frames) == 10

    # SPACE -> pause
    session._handle_command(HILCommand.PAUSE_INTERVENTION)
    assert session.phase is HILPhase.RWFM_EVALUATION_PAUSED
    assert session.is_trial_recording is False
    assert session._rwfm_paused_frame_count == 10
    assert session._rwfm_pending_label == "normal"
    assert client_mock.pause_policy_control.called

    # Ticks while paused must NOT add any frames
    for _ in range(5):
        session._autonomous_tick()
    assert len(added_frames) == 10


def test_space_space_commits_normal_segment_and_resumes(tmp_path):
    session, added_frames, _, client_mock = _create_mock_session(tmp_path)
    session.is_trial_recording = True
    session._expected_episode_index = 0
    session._resume_autonomous()

    for _ in range(12):
        session._autonomous_tick()

    # First SPACE: pause
    session._handle_command(HILCommand.PAUSE_INTERVENTION)
    assert session.phase is HILPhase.RWFM_EVALUATION_PAUSED

    # Second SPACE: commit default NORMAL and resume
    session._handle_command(HILCommand.PAUSE_INTERVENTION)
    assert session.phase is HILPhase.AUTONOMOUS
    assert session.is_trial_recording is True
    assert len(session._rwfm_segments) == 1
    assert session._rwfm_segments[0] == {"start": 0, "end": 12, "type": "normal"}
    assert session._rwfm_segment_start_frame == 12
    assert client_mock.resume_policy_control.called


def test_space_s_and_space_f_pending_and_commit_resume(tmp_path):
    session, added_frames, _, _ = _create_mock_session(tmp_path)
    session.is_trial_recording = True
    session._expected_episode_index = 0
    session._resume_autonomous()

    for _ in range(8):
        session._autonomous_tick()

    # 1. SPACE -> Pause
    session._handle_command(HILCommand.PAUSE_INTERVENTION)
    assert session.phase is HILPhase.RWFM_EVALUATION_PAUSED

    # 2. Press S -> pending self_correction
    session._handle_command(HILCommand.SET_SELF_CORRECTION)
    assert session._rwfm_pending_label == "self_correction"
    assert session.phase is HILPhase.RWFM_EVALUATION_PAUSED  # still paused!

    # 3. Press F -> toggles to pending failure
    session._handle_command(HILCommand.SET_FAILURE)
    assert session._rwfm_pending_label == "failure"
    assert session.phase is HILPhase.RWFM_EVALUATION_PAUSED  # still paused!

    # 4. Press S -> toggles back to pending self_correction
    session._handle_command(HILCommand.SET_SELF_CORRECTION)
    assert session._rwfm_pending_label == "self_correction"

    # 5. SPACE -> commit self_correction and resume
    session._handle_command(HILCommand.PAUSE_INTERVENTION)
    assert session.phase is HILPhase.AUTONOMOUS
    assert len(session._rwfm_segments) == 1
    assert session._rwfm_segments[0] == {"start": 0, "end": 8, "type": "self_correction"}
    assert session._rwfm_segment_start_frame == 8

    # 6. Record 6 more frames (total 14)
    for _ in range(6):
        session._autonomous_tick()

    # 7. SPACE -> F -> SPACE -> commits failure
    session._handle_command(HILCommand.PAUSE_INTERVENTION)
    session._handle_command(HILCommand.SET_FAILURE)
    session._handle_command(HILCommand.PAUSE_INTERVENTION)

    assert session.phase is HILPhase.AUTONOMOUS
    assert len(session._rwfm_segments) == 2
    assert session._rwfm_segments[1] == {"start": 8, "end": 14, "type": "failure"}
    assert session._rwfm_segment_start_frame == 14


def test_space_s_c_and_space_f_c_finish_rollout_without_resume(tmp_path):
    session, added_frames, _, _ = _create_mock_session(tmp_path)
    session.is_trial_recording = True
    session._expected_episode_index = 0
    session._resume_autonomous()

    for _ in range(15):
        session._autonomous_tick()

    session._handle_command(HILCommand.PAUSE_INTERVENTION)
    session._handle_command(HILCommand.SET_SELF_CORRECTION)
    # C finishes rollout
    session._handle_command(HILCommand.FREEZE_CORRECTION)

    assert session.phase is HILPhase.REVIEW_PAUSED
    assert session.is_trial_recording is False
    assert len(session._rwfm_segments) == 1
    assert session._rwfm_segments[0] == {"start": 0, "end": 15, "type": "self_correction"}
    assert session._rwfm_outcome == "grasp_success"


def test_space_f_c_commits_failure_and_sets_grasp_failure_outcome(tmp_path):
    session, added_frames, _, _ = _create_mock_session(tmp_path)
    session.is_trial_recording = True
    session._expected_episode_index = 0
    session._resume_autonomous()

    for _ in range(18):
        session._autonomous_tick()

    session._handle_command(HILCommand.PAUSE_INTERVENTION)
    session._handle_command(HILCommand.SET_FAILURE)
    # Enter finishes rollout
    session._handle_command(HILCommand.START_DIRECT_CORRECTION)

    assert session.phase is HILPhase.REVIEW_PAUSED
    assert session.is_trial_recording is False
    assert len(session._rwfm_segments) == 1
    assert session._rwfm_segments[0] == {"start": 0, "end": 18, "type": "failure"}
    assert session._rwfm_outcome == "grasp_failure"


def test_running_c_commits_normal_and_enters_review_paused(tmp_path):
    session, added_frames, _, _ = _create_mock_session(tmp_path)
    session.is_trial_recording = True
    session._expected_episode_index = 0
    session._resume_autonomous()

    for _ in range(25):
        session._autonomous_tick()

    # Press C directly from RUNNING
    session._handle_command(HILCommand.FREEZE_CORRECTION)

    assert session.phase is HILPhase.REVIEW_PAUSED
    assert session.is_trial_recording is False
    assert len(session._rwfm_segments) == 1
    assert session._rwfm_segments[0] == {"start": 0, "end": 25, "type": "normal"}
    assert session._rwfm_outcome == "grasp_success"


def test_save_and_discard_atomicity(tmp_path):
    session, added_frames, dataset_mock, _ = _create_mock_session(tmp_path)
    session.is_trial_recording = True
    session._expected_episode_index = 0
    session._current_target_block = "green"
    session._resume_autonomous()

    for _ in range(30):
        session._autonomous_tick()

    session._handle_command(HILCommand.FREEZE_CORRECTION)
    assert session.phase is HILPhase.REVIEW_PAUSED

    # 1. Save with Right Arrow (→)
    session._handle_command(HILCommand.SAVE_FROZEN_CORRECTION)
    assert dataset_mock.num_episodes == 1
    assert session.saved_corrections == 1
    assert session.phase is HILPhase.INTERVENTION_PAUSED

    # Verify annotation json was written atomically
    meta_path = tmp_path / "meta" / "rwfm_rollout_annotations.json"
    assert meta_path.exists()
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    assert meta_data["schema_version"] == 1
    ep0 = meta_data["episodes"]["0"]
    assert ep0["episode_index"] == 0
    assert ep0["target_block"] == "green"
    assert ep0["outcome"] == "grasp_success"
    assert ep0["segments"] == [{"start": 0, "end": 30, "type": "normal"}]

    # 2. Next trial and discard with Left Arrow (←)
    session._expected_episode_index = 1
    session.is_trial_recording = True
    session.phase = HILPhase.AUTONOMOUS
    for _ in range(10):
        session._autonomous_tick()
    session._handle_command(HILCommand.FREEZE_CORRECTION)

    session._handle_command(HILCommand.DISCARD_FROZEN_CORRECTION)
    assert len(added_frames) == 0
    assert len(session._rwfm_segments) == 0
    # ep 0 remains untouched in json
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_data2 = json.load(f)
    assert "0" in meta_data2["episodes"]
    assert "1" not in meta_data2["episodes"]


def test_n_next_trial_from_various_states(tmp_path):
    # Test N from RUNNING
    session, added_frames, _, _ = _create_mock_session(tmp_path)
    session.is_trial_recording = True
    session._expected_episode_index = 0
    session._resume_autonomous()
    for _ in range(10):
        session._autonomous_tick()
    outcome = session._handle_command(HILCommand.NEXT_TRIAL)
    assert outcome is TrialOutcome.NEXT
    assert len(added_frames) == 0
    assert len(session._rwfm_segments) == 0

    # Test N from RWFM_EVALUATION_PAUSED
    session, added_frames, _, _ = _create_mock_session(tmp_path)
    session.is_trial_recording = True
    session._expected_episode_index = 0
    session._resume_autonomous()
    for _ in range(10):
        session._autonomous_tick()
    session._handle_command(HILCommand.PAUSE_INTERVENTION)
    outcome = session._handle_command(HILCommand.NEXT_TRIAL)
    assert outcome is TrialOutcome.NEXT
    assert len(added_frames) == 0

    # Test N from REVIEW_PAUSED (unsaved)
    session, added_frames, _, _ = _create_mock_session(tmp_path)
    session.is_trial_recording = True
    session._expected_episode_index = 0
    session._resume_autonomous()
    for _ in range(10):
        session._autonomous_tick()
    session._handle_command(HILCommand.FREEZE_CORRECTION)
    outcome = session._handle_command(HILCommand.NEXT_TRIAL)
    assert outcome is TrialOutcome.NEXT
    assert len(added_frames) == 0

    # Test N after saving
    session, added_frames, dataset_mock, _ = _create_mock_session(tmp_path)
    session.is_trial_recording = True
    session._expected_episode_index = 0
    session._resume_autonomous()
    for _ in range(10):
        session._autonomous_tick()
    session._handle_command(HILCommand.FREEZE_CORRECTION)
    session._handle_command(HILCommand.SAVE_FROZEN_CORRECTION)
    assert dataset_mock.num_episodes == 1
    # Press N after save
    outcome = session._handle_command(HILCommand.NEXT_TRIAL)
    assert outcome is TrialOutcome.NEXT
    assert dataset_mock.num_episodes == 1  # preserved!


def test_failure_tail_frames_split(tmp_path):
    session, added_frames, _, _ = _create_mock_session(tmp_path, failure_tail_frames=10)
    session.is_trial_recording = True
    session._expected_episode_index = 0
    session._resume_autonomous()

    for _ in range(25):
        session._autonomous_tick()

    session._handle_command(HILCommand.PAUSE_INTERVENTION)
    session._handle_command(HILCommand.SET_FAILURE)
    session._handle_command(HILCommand.PAUSE_INTERVENTION)

    # 25 frames split: [0, 15) normal, [15, 25) failure
    assert len(session._rwfm_segments) == 2
    assert session._rwfm_segments[0] == {"start": 0, "end": 15, "type": "normal"}
    assert session._rwfm_segments[1] == {"start": 15, "end": 25, "type": "failure"}


def test_intervention_start_frame_bug_fixed_in_full_on_intervention(tmp_path):
    session, added_frames, _, _ = _create_mock_session(tmp_path, record_mode="full_on_intervention")
    session.intervention_start_frame = 42

    session._record_episode_intervention_meta(
        episode_index=0,
        parent_rollout_id=1,
        intervention_index=1,
        recovery_type="direct",
        total_frames=100,
    )

    meta_path = tmp_path / "meta" / "episode_interventions.json"
    assert meta_path.exists()
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # Must be 42, not hardcoded 0!
    assert meta["0"]["intervention_start_frame"] == 42
