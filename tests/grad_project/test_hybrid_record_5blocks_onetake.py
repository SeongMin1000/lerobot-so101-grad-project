import pytest
from dataclasses import is_dataclass

from lerobot.grad_project.recording.hybrid_record_5blocks_onetake import (
    HybridOneTakeRecordConfig,
    RecordControlEvent,
    TargetHoverResolver,
    _set_leader_torque,
    _send_leader_feedback,
    _get_leader_arm,
)


def test_hybrid_record_config_is_dataclass():
    assert is_dataclass(HybridOneTakeRecordConfig)
    cfg = HybridOneTakeRecordConfig.__new__(HybridOneTakeRecordConfig)
    assert hasattr(HybridOneTakeRecordConfig, "color_sequence")
    assert hasattr(HybridOneTakeRecordConfig, "max_blocks")
    assert hasattr(HybridOneTakeRecordConfig, "macro_goto_duration_s")
    assert hasattr(HybridOneTakeRecordConfig, "macro_return_duration_s")


def test_record_control_events():
    assert RecordControlEvent.SAVE == "save"
    assert RecordControlEvent.RERECORD == "rerecord"
    assert RecordControlEvent.STOP == "stop"
    assert RecordControlEvent.NEXT_PHASE == "next_phase"


def test_leader_arm_helpers():
    class MockLeader:
        def __init__(self):
            self.torque_enabled = False
            self.last_feedback = None

        def enable_torque(self):
            self.torque_enabled = True

        def disable_torque(self):
            self.torque_enabled = False

        def send_feedback(self, fb):
            self.last_feedback = fb

    leader = MockLeader()
    assert _get_leader_arm(leader) is leader
    assert _get_leader_arm([leader]) is leader

    _set_leader_torque(leader, enable=True)
    assert leader.torque_enabled is True

    _set_leader_torque(leader, enable=False)
    assert leader.torque_enabled is False

    _send_leader_feedback(leader, {"shoulder_pan.pos": 10.0})
    assert leader.last_feedback == {"shoulder_pan.pos": 10.0}


def test_simulate_one_take_episode_flow():
    """Simulates the full Auto -> Manual -> Auto -> Save sequence."""
    from lerobot.grad_project.recording.hybrid_record_5blocks_onetake import (
        _record_one_take_5blocks_episode,
        JOINT_ORDER,
    )
    import numpy as np

    class MockRobot:
        def __init__(self):
            self.action_features = {f"{n}.pos": float for n in JOINT_ORDER}
            self.observation_features = {f"{n}.pos": float for n in JOINT_ORDER}
            self.observation_features["top"] = (480, 640, 3)
            self._pose = {f"{n}.pos": 0.0 for n in JOINT_ORDER}
            self.sent_actions = []

        def get_observation(self):
            return {**self._pose, "top": np.zeros((480, 640, 3), dtype=np.uint8)}

        def send_action(self, action):
            self.sent_actions.append(action)
            for k, v in action.items():
                if k in self._pose:
                    self._pose[k] = v

    class MockTeleop:
        def __init__(self):
            self.torque_enabled = False
            self.feedbacks = []

        def enable_torque(self):
            self.torque_enabled = True

        def disable_torque(self):
            self.torque_enabled = False

        def send_feedback(self, fb):
            self.feedbacks.append(fb)

        def get_action(self):
            return {f"{n}.pos": 5.0 for n in JOINT_ORDER}

    class MockDataset:
        def __init__(self):
            self.features = {
                "observation.images.top": {"dtype": "image", "shape": (480, 640, 3)},
                "action": {"dtype": "float32", "shape": (6,)},
            }
            self.frames = []
            self.num_episodes = 0

        def add_frame(self, frame):
            self.frames.append(frame)

        def save_episode(self):
            self.num_episodes += 1

    class MockResolver:
        def cache_scene_blocks(self, top_img):
            pass

        def resolve_lift_clear_pose(self, cur_robot):
            return {f"{n}.pos": 10.0 for n in JOINT_ORDER}

        def resolve_block_targets(self, color, observation, current_joint_deg):
            high = {f"{n}.pos": 12.0 for n in JOINT_ORDER}
            hover = {f"{n}.pos": 15.0 for n in JOINT_ORDER}
            return high, hover

        def resolve_hover_pose(self, color, observation, current_joint_deg):
            return {f"{n}.pos": 15.0 for n in JOINT_ORDER}

    robot = MockRobot()
    teleop = MockTeleop()
    dataset = MockDataset()
    resolver = MockResolver()

    # Pass fast test config
    cfg = HybridOneTakeRecordConfig.__new__(HybridOneTakeRecordConfig)
    cfg.color_sequence = "red,blue"
    cfg.max_blocks = 2
    cfg.macro_goto_duration_s = 0.05
    cfg.macro_return_duration_s = 0.05
    cfg.hover_z_offset_m = 0.08
    cfg.top_key = "top"
    cfg.play_sounds = False
    cfg.display_data = False
    cfg.display_compressed_images = False

    class MockDatasetCfg:
        fps = 30
        single_task = "Test 2 blocks task"
    cfg.dataset = MockDatasetCfg()

    events = {
        "exit_early": False,
        "rerecord_episode": False,
        "stop_recording": False,
        "next_phase": False,
    }

    # Pass identity processors
    obs_proc = lambda obs: obs
    act_proc = lambda item: item[0]

    observe_target = {f"{n}.pos": 0.0 for n in JOINT_ORDER}

    # Simulate: Auto approach starts, then in manual phase next_phase is triggered
    # We trigger next_phase and finally exit_early (save)
    import threading
    import time

    def operator_sim():
        # Wait for auto approach of block 1, then trigger next_phase
        time.sleep(0.1)
        events["next_phase"] = True
        # Wait for auto return + auto approach of block 2, then trigger next_phase
        time.sleep(0.2)
        events["next_phase"] = True
        # Wait for 2 blocks complete, then trigger save (exit_early)
        time.sleep(0.2)
        events["exit_early"] = True

    sim_thread = threading.Thread(target=operator_sim)
    sim_thread.start()

    outcome = _record_one_take_5blocks_episode(
        robot=robot,
        teleop=teleop,
        events=events,
        cfg=cfg,
        dataset=dataset,
        resolver=resolver,
        observe_target=observe_target,
        robot_obs_proc=obs_proc,
        teleop_act_proc=act_proc,
        robot_act_proc=act_proc,
    )

    sim_thread.join()

    assert outcome == RecordControlEvent.SAVE
    assert len(dataset.frames) > 0, "Frames must be continuously recorded"
    print(f"Simulation recorded {len(dataset.frames)} frames across 2 blocks!")

