# Experiment History

This is a compact chronology. Use Git, W&B, Hub metadata, logs, and the source
audit when an exact command or timestamp is required.

## Phase 1: ACT and OpenCV hybrid

- A red-only ACT policy succeeded roughly 20/25 times at a familiar C1 setup,
  but showed rotation and generalization failures.
- OpenCV + FSM + ACT split the top-view workspace into zones and used short ACT
  grasp segments. Color segmentation suffered from yellow/wood confusion,
  green false detections, shadows, and background changes.
- A 100-episode multi-block ACT model was trained, but random five-block
  rollouts failed severely. The project shifted to SmolVLA and single-block
  episodes.

## Phase 2: early SmolVLA red pilots

### 30 episodes

- Recorded red-only end-to-end demonstrations with top, wrist, and belly
  cameras and automatic observe return outside the dataset.
- LoRA policy often rotated right and grasped empty space instead of approaching
  the red block.

### 64 episodes over A-H zones

- Collected eight-zone data and ran a full fine-tune.
- The policy sometimes worked at recorded points but failed at novel locations.
- Reducing actions consumed per chunk improved the severe initial backward
  motion. The interaction between long action chunks and the joint-step clamp
  became a primary control concern.

## Phase 3: 126-episode expansion

- Expanded spatial coverage and merged a red-only 126-episode dataset.
- Used later LoRA r64 training with W&B tracking and mild image transforms.
- The policy began moving toward the red block but approached several
  centimeters to the right.
- Action logs showed the shoulder-pan bias already existed in the model output,
  so camera alignment and joint clamps were not the sole explanation.
- Dataset coverage analysis identified sparse near-front shoulder-pan examples,
  especially the angle range represented by B-zone extremes and G-zone center.

## Phase 4: preprocessing repair and recovery data

- Fixed camera-key handling between `top/wrist/belly` and
  `camera1/camera2/camera3`.
- Preserved 640x480 aspect ratio until SmolVLA performs its own 512x512 padded
  resize. Added exactly-once image normalization checks.
- Fixed policy-server reset so a reconnect clears the observation queue,
  predicted timesteps, and `last_processed_obs`.
- Added client/server matched camera captures, motor traces, a coordinated
  action-vector clamp, and a motor-tracking watchdog.
- Added recovery demonstrations, producing
  `eslab1234/red_full_138ep_recovery_v1` and the current 138-episode LoRA model.
- After the server preprocessing fix, approach accuracy improved greatly and
  color confusion was rare. Remaining failures included grasp-angle errors,
  slow or missing release, block pushing, occasional position error, and a
  residual tendency to turn right.

## Current transition

- `ACTIONS_PER_CHUNK=30`, threshold `0.6`, `latest_only`, and a
  `MAX_RELATIVE_TARGET=0.75` candidate were discussed; 0.75 felt slow and is not
  a final optimum.
- The next major work is structured recovery/HIL data, front-angle coverage,
  all-five-color behavior, then Task 2 stacking with the same neural checkpoint.
- An ACT-era 100-episode dataset was converted/relabelled as a possible
  all-five SmolVLA training source. Its training run and Hub state require
  verification.
- Project-specific source was reorganized into `project/` and
  `src/lerobot/grad_project/`; runtime regression tooling was added.

## Key lessons

1. More episodes do not compensate for missing joint-angle and recovery-state
   coverage.
2. Camera-key, aspect-ratio, normalization, and server-state errors can mimic a
   weak model.
3. Long open-loop chunks interact strongly with physical rate limits.
4. Evaluate model output before blaming camera calibration or motor control.
5. Record correction behavior from actual failure states instead of collecting
   only clean nominal demonstrations.
