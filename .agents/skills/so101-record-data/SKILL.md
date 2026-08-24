---
name: so101-record-data
description: LeRobot SO-101 졸업과제의 SmolVLA/ACT 시연 촬영, episode 재촬영·폐기, offline recovery/HIL 데이터 수집, 데이터셋 검사·병합 계획을 수행한다. Use for record, filming, demonstration, HIL, recovery, dataset collection, episode quality, or coverage questions.
---

# SO-101 Record Data

## Load context

Read before acting:

1. `project/docs/agent-context/project-contract.md`
2. `project/docs/agent-context/hardware-and-paths.md`
3. `project/docs/agent-context/decisions-and-known-failures.md`
4. `project/config/experiment-profiles/active.env`

Read `dataset-model-registry.md` when extending or merging an existing dataset.

## Choose the recording mode

- **Hybrid 5-Block One-Take Recording (`v3`)** [PRIMARY FOR TASK 1]:
  - Records a continuous 5-block pick-and-place episode in one shot (`red, yellow, wood, green, blue`).
  - Auto-approach from Observe to block hover using **Taught RBF Joint Model (`hover_joint_model_record.json`)** with **Distance-Adaptive S-Curve & Wrist Camera Elevation** (`val -= wrist_bump * sin(pi*s)`).
  - Teleoperated pick-and-place into target slot, followed by automatic shortest-path return to Observe pose after each block.
  - Script: `project/scripts/robot/run_hybrid_5blocks_onetake_v3_record.sh`
  - Python: `python -m lerobot.grad_project.recording.hybrid_record_5blocks_onetake_v3`
- Use end-to-end SmolVLA recording for a complete single-block pick/place or stack demonstration.
- Use `teleop_before_episode=true` for an offline recovery demonstration that begins from a model-created failure state.
- Use `hybrid_record_grasp_zone.py` only for the historical short-grasp design.
- Use `hybrid_record_color_sequence.py` only when the requested experiment explicitly uses the legacy OpenCV/FSM sequence.
- Do not call offline recovery recording “online HIL.” Online intervention logging is not implemented in the current inference loop.

## Calibration & Kinematic Mapping Stack

### 1. Top Camera 34-Point Ruler Homography & Distance Calibration
- File: `project/config/grasp_pixel_to_robot_record.json`
- Backup samples: `project/config/ruler_calibration_samples_record.json`
- Maps top camera pixel $(cx, cy) \to$ real-world Cartesian $(X, Y\text{ m})$ and distance $R = \sqrt{X^2 + Y^2}$, azimuth $\theta = \text{atan2}(Y, X)$.
- Fitted with 34 physical ruler grid points across $X \in [0, 35]\text{cm}, Y \in [-30, +30]\text{cm}$.
- Mean residual error: $2.03\text{ mm}$, Max error: $4.36\text{ mm}$.
- Live monitor: `python project/scripts/tools/test_yolo_live.py`

### 2. 45-Point 2-Step Teleoperation Demonstration Joint Mapping (RBF Model)
- Tool: `project/scripts/tools/teach_block_hover_joints.py`
- Samples file: `project/config/hover_demonstration_samples_record.json`
- Model file: `project/config/hover_joint_model_record.json`
- **2-Step Workflow**:
  - `[Step 1: Snapshot]`: Captures unoccluded block pixel $(cx, cy) \to (X, Y)$ using top camera while robot arm is parked behind.
  - `[Step 2: Teach]`: Demonstrator teleoperates follower to hover above the block. Current 6 motor joint positions are paired with Step 1 coordinates.
- **RBF Multiquadric Interpolator**:
  - Centers: 45 $(X, Y)$ points across 9 directions $\times$ 5 distances ($R \in [11, 38]\text{cm}$).
  - Overall Mean Joint Error: **$0.22^\circ$** (`shoulder_pan`: $0.04^\circ$, `shoulder_lift`: $0.34^\circ$, `elbow_flex`: $0.41^\circ$, `wrist_flex`: $0.15^\circ$, `wrist_roll`: $0.16^\circ$).
  - Evaluated in real-time by `TargetHoverResolver` in recorder v3.

### 3. Distance-Adaptive Wrist Camera Elevation & Earlier Pan Alignment
- Observe to Hover transition uses direct Cosine S-curve easing (`apex_pose = None`).
- **Earlier Direction Alignment (`shoulder_pan`)**: $\text{pan\_ratio} = 0.90 - 0.15 \times \text{clip}((R - 0.12) / 0.25, 0, 1)$ (completes pan alignment at $75\%\sim 90\%$ of total flight time while arm smoothly continues final descent).
- Distance-adaptive wrist lift bump: $\text{wrist\_bump} = 6.0^\circ + 8.0^\circ \times \text{clip}((R - 0.12) / 0.25, 0, 1)$ ($+6^\circ$ near $\to +14^\circ$ far).
- Wrist flex elevation formula: `cmd["wrist_flex.pos"] = base_val - wrist_bump * sin(pi * s)` (negative flex tilts wrist UP towards sky/horizon, keeping block in full wrist camera view during approach).

### 4. Leader Arm Wrist Roll Center Calibration
- Config: `project/config/calibration/teleoperators/so_leader/leader.json` & active HF cache.
- `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `gripper`: `range_min: 0, range_max: 4095`.
- `wrist_roll`: `range_min: 2200, range_max: 4000` (Center $\text{mid} = 3100$, $\pm 900$).

## Plan before connecting hardware

1. State the dataset ID, exact task text, number of episodes, zones/angles,
   nominal-versus-recovery ratio, and the intended comparison.
2. Balance position and shoulder-pan direction, not only zone labels. Prioritize
   B-zone extremes and G-zone center for the sparse near-front angle range.
3. Include approach angle, wrist orientation, height, gripper timing, and
   release behavior variation.
4. Keep the task text byte-for-byte consistent with training and inference.
5. Confirm that a new repo ID/root will not overwrite an existing dataset.

## Preflight

Run read-only checks first:

```bash
cd "${LEROBOT_ROOT:-$HOME/lerobot}"
conda activate lerobot
source project/config/experiment-profiles/active.env

python -m lerobot.grad_project.recording.smolvla_record_observe_return --help
python -m lerobot.grad_project.paths
ls -l "$ROBOT_PORT" "$TELEOP_PORT" "$TOP_CAM" "$WRIST_CAM"
```

Then verify each camera really delivers 640x480@30 MJPG (Top camera using factory default V4L2 parameters: Auto WB, brightness 0, contrast 40, saturation 64, gamma 300) and that no other
process owns it. Stop if the wrist camera reports repeated read failures.

Do not move either arm until the user explicitly requests a physical run and a
human confirms the clear workspace, emergency stop, and current observe pose.

## Generate a recording command

Inspect local `--help`, then adapt this template. Do not assume flags from a
different LeRobot release:

```bash
DATASET_NAME="replace_me"
NUM_EPISODES="10"
EPISODE_TIME_S="60"
TELEOP_BEFORE_EPISODE="false"

python -m lerobot.grad_project.recording.smolvla_record_observe_return \
  --robot.type=so101_follower \
  --robot.port="$ROBOT_PORT" \
  --robot.id=follower \
  --robot.disable_torque_on_disconnect=false \
  --teleop.type=so101_leader \
  --teleop.port="$TELEOP_PORT" \
  --teleop.id=leader \
  --robot.cameras="{ top: {type: opencv, index_or_path: '$TOP_CAM', width: $WIDTH, height: $HEIGHT, fps: $FPS, fourcc: 'MJPG'}, wrist: {type: opencv, index_or_path: '$WRIST_CAM', width: $WIDTH, height: $HEIGHT, fps: $FPS, fourcc: 'MJPG'} }" \
  --dataset.repo_id="$HF_USER/$DATASET_NAME" \
  --dataset.single_task="$TASK" \
  --dataset.num_episodes="$NUM_EPISODES" \
  --dataset.episode_time_s="$EPISODE_TIME_S" \
  --dataset.reset_time_s=0 \
  --dataset.fps="$FPS" \
  --dataset.video=true \
  --runtime_config="$LEROBOT_RUNTIME_CONFIG" \
  --teleop_before_episode="$TELEOP_BEFORE_EPISODE"
```

Add upload/push options only if the user explicitly asks to publish. Prefer a
local smoke episode before a long session.

## Operator contract

The custom recorder must behave as follows:

- `Right arrow`: save only after the block is placed/stacked, gripper is open,
  and the endpoint is held for about 0.5-1 second.
- `Left arrow`: discard all buffered frames, return to observe outside the
  dataset, and record the same episode number again.
- `Timeout`: discard; never auto-save an incomplete episode.
- `Escape`: discard and stop without automatic movement.
- Automatic observe return must remain outside the dataset.

If keyboard/X11 input is unavailable, stop. Do not record through a headless
session when save/discard keys cannot be trusted.

## Offline recovery workflow

1. Run inference until one target failure appears; do not wait for a collision.
2. Stop safely and preserve the exact block and follower state.
3. Start the recorder with `teleop_before_episode=true`.
4. Use the preparation teleoperation to align follower and leader without
   recording.
5. Press Enter at the failure state.
6. Record the correction: reopen if needed, retreat/re-approach, grasp, lift,
   place or stack, release, and hold.
7. Save only a successful recovery with right arrow.
8. Label the failure category in the session log.

Target recovery categories:

- lateral/longitudinal approach error;
- block pushed during descent;
- empty gripper close, then reopen and recover;
- wrong wrist/grasp angle;
- slip after lifting;
- wrong target approach;
- late or missing release.

Keep nominal demonstrations in the mixture. A dataset containing only recovery
starts can distort the initial-state distribution.

## Validate every session

Before merging or training, inspect:

- actual episode count and contiguous episode indices;
- task text for every episode;
- FPS and timestamps;
- `observation.state` and `action` shapes/names;
- top and wrist feature keys and orientation (side/belly camera is unused);
- decoded videos at beginning, grasp, transport, release, and final hold;
- non-empty actions and absence of NaN/Inf values;
- incomplete temporary frames from discarded attempts;
- dataset stats and compatibility with the intended model.

Do not merge incompatible datasets. Record the new dataset in
`dataset-model-registry.md` only after validation.
