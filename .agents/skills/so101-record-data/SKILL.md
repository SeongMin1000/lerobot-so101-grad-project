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

- Use end-to-end SmolVLA recording for a complete pick/place or stack
  demonstration.
- Use `teleop_before_episode=true` for an offline recovery demonstration that
  begins from a model-created failure state.
- Use `hybrid_record_grasp_zone.py` only for the historical short-grasp design.
- Use `hybrid_record_color_sequence.py` only when the requested experiment
  explicitly uses the legacy OpenCV/FSM sequence.
- Do not call offline recovery recording “online HIL.” Online intervention
  logging is not implemented in the current inference loop.

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
ls -l "$ROBOT_PORT" "$TELEOP_PORT" "$TOP_CAM" "$WRIST_CAM" "$BELLY_CAM"
```

Then verify each camera really delivers 640x480@30 MJPG and that no other
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
  --robot.cameras="{ top: {type: opencv, index_or_path: '$TOP_CAM', width: $WIDTH, height: $HEIGHT, fps: $FPS, fourcc: 'MJPG'}, wrist: {type: opencv, index_or_path: '$WRIST_CAM', width: $WIDTH, height: $HEIGHT, fps: $FPS, fourcc: 'MJPG'}, belly: {type: opencv, index_or_path: '$BELLY_CAM', width: $WIDTH, height: $HEIGHT, fps: $FPS, fourcc: 'MJPG', rotation: 180} }" \
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
- top, wrist, and belly feature keys and orientation;
- decoded videos at beginning, grasp, transport, release, and final hold;
- non-empty actions and absence of NaN/Inf values;
- incomplete temporary frames from discarded attempts;
- dataset stats and compatibility with the intended model.

Do not merge incompatible datasets. Record the new dataset in
`dataset-model-registry.md` only after validation.
