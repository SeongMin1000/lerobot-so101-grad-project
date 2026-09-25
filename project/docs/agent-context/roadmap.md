# Project Roadmap

Snapshot date: 2026-08-06. Target project submission is expected in the second
week of September 2026; confirm the official date.

## P0: preserve a reproducible red baseline

Exit criteria:

- Current source, active profile, dataset, model, and W&B run are recorded.
- Runtime regression tool passes all available non-hardware checks.
- A cold server/client rollout and a client-reconnect rollout are separately
  identified.
- Red-block evaluation covers A-H with deliberate emphasis on B extremes and G
  center.
- Failure results are logged with `성/접/집/놓/오/충`, time, and error direction.

Do not tune multiple chunk, clamp, and model variables in the same batch.

## P1: structured recovery data

Collect corrections from actual dominant failure states:

- approach right/left/front/back of the block;
- block pushed during descent;
- gripper closes empty;
- reopen, reposition, and grasp;
- wrong wrist/grasp angle;
- object slips after lift;
- target reached but release is late or missing;
- release outside the intended slot/area.

Start with a small balanced recovery batch per dominant failure, retrain, and
measure whether that specific failure decreases. Add more only from evidence.
Keep nominal demonstrations so the policy does not learn to expect every
episode to start from failure.

## P2: all five colors for Task 1

1. Audit the converted 100-episode dataset's task text, camera keys, frames, and
   action/state statistics.
2. Decide whether to use converted ACT-era episodes, newly recorded SmolVLA
   episodes, or a controlled mixture.
3. Encode color/task intent consistently while training one checkpoint.
4. Evaluate selection errors separately from grasp and placement errors.
5. Add rule-based priority only if it does not hide the checkpoint's inability
   to follow the language/task input.
6. Validate red -> yellow -> wood -> green -> blue in random scenes.

Exit criteria: one checkpoint completes repeated five-color Task 1 scenes
within the 3-minute official limit with documented success rate.

## P3: Task 2 stacking with the same checkpoint

1. Define a Task 2 instruction and dataset scheme compatible with the Task 1
   model interface.
2. Record stable stack demonstrations, near-failure recovery, and controlled
   release/retreat behavior.
3. Decide whether rule-based stack-height/slot verification is needed.
4. Train a joint Task 1 + Task 2 checkpoint; do not create a task-specific
   second neural model.
5. Test five-second stability and the 5-minute limit.

## P4: Jetson and Docker migration

1. Reproduce robot client/camera/joint control on Jetson Orin Nano.
2. Measure camera decode, network, inference round-trip, and control-loop FPS.
3. Build a pinned GPU-server Docker image with the exact source and model
   available offline.
4. Validate against the actual evaluation-server driver/CUDA/runtime.
5. Document device mounts, network ports, model volumes, start command, health
   check, and graceful stop.
6. Perform a fresh-machine rehearsal instead of testing only a long-lived
   development container.

## P5: final evaluation rehearsal

- Recreate the official board and `20 x 10 cm` inside target dimensions.
- Use supplied `4 x 4 x 2 cm` blocks.
- Run separate Task 1 and Task 2 scenes after evaluator-style resets.
- Enforce 3-minute and 5-minute timers.
- Verify five-second stack stability.
- Test offline startup, server reconnect, camera reconnect, Ctrl+C, watchdog,
  and emergency-stop recovery.
- Freeze one checkpoint only after the full rehearsal passes.

## Decision checkpoints

Stop and ask the user before:

- replacing the canonical observe pose;
- changing which joints participate in synchronized limiting;
- changing camera-key or resize/normalization behavior;
- overwriting an output directory or dataset;
- pushing code, datasets, models, or a Docker image;
- adopting a second neural checkpoint for Task 2;
- interpreting a new official requirement that conflicts with v2.0.
