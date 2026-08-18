# SO-101 Graduation Project Contract

Last synchronized: 2026-08-06

This file is the stable project contract for agents. Read it before modifying
project code, generating robot commands, choosing a dataset/model, or planning
the final demonstration.

## Goal

Control one LeRobot SO-101 arm to handle five randomly placed blocks. The team
priority is:

`red -> yellow -> wood -> green -> blue`

The project has two separately evaluated tasks:

1. Move all five blocks into a `20 cm x 10 cm` target area.
2. Stack all five blocks vertically.

The evaluator resets all blocks between Task 1 and Task 2. One neural-network
checkpoint must support both tasks. Task-specific prompts, OpenCV, FSM, safety
logic, and other rule-based code may differ.

## Current technical direction

- Primary policy: SmolVLA.
- Baseline/comparison: ACT.
- Historical legacy path: OpenCV + ACT/Diffusion FSM.
- Current OpenCV role: color/position detection, target verification,
  calibration, debugging, and rule-based coordination.
- Robot PC: camera capture, joint I/O, client runtime, and safety.
- GPU PC: training and async policy inference.
- Final edge device: Jetson Orin Nano communicating with the evaluation server.

Do not silently replace the current hybrid design with a pure end-to-end or a
pure rule-based design. Explain the tradeoff and obtain a project decision.

## Source-of-truth rules

Use the following evidence order for each category.

### Runtime and CLI behavior

1. Current source and local `--help` in this checkout.
2. Active checkpoint config and processor files.
3. Current experiment profile.
4. Historical commands and notes.

This source is LeRobot `0.5.2` and requires Python `>=3.12`. Options from a
different LeRobot release are not automatically valid.

### Active experiment values

1. `project/config/experiment-profiles/active.env`.
2. Current user instruction.
3. Observed launch logs.

The active profile is a snapshot. Values marked `draft`, blank, unresolved, or
in conflict with the checkpoint must be confirmed before use.

### Official evaluation rules

1. Newest official announcement supplied by the user.
2. `졸업과제 세부사항_v2.0`.
3. The older v1.5 notice.

See `graduation-requirements.md`. Team conventions such as color priority and
fixed target slots must not be presented as official rules.

## Canonical repository boundaries

- Put project-owned Python under `src/lerobot/grad_project/`.
- Keep only genuinely shared runtime changes in
  `src/lerobot/async_inference/` and SO-101 hardware modules.
- Put static project configuration in `project/config/`.
- Put launchers in `project/scripts/{gpu,robot,tools}/`.
- Put project documentation in `project/docs/`.
- Put generated logs, captures, traces, calibration output, and backups in
  `var/`; never treat them as source.
- Keep historical OpenCV+ACT/Diffusion clients under
  `src/lerobot/grad_project/inference/legacy/`.

Do not recreate old root-level config paths or imports from
`lerobot.async_inference.<moved_project_module>`.

## Physical safety contract

Reading, explaining, diagnosing, or planning does not authorize physical
movement. Before a requested robot run:

1. Identify whether the command runs on the GPU PC or robot PC.
2. Print and verify the model, exact task text, camera keys, devices, server,
   chunk parameters, joint clamp, tracking watchdog, and timeout.
3. Confirm a clear workspace, five-block scene, emergency-stop access, and a
   human at the robot.
4. Retain the literal `START` prompt. Never default `SKIP_CONFIRM=true`.
5. Keep one hand ready for Ctrl+C or the physical emergency stop.
6. Do not automatically disable torque on graceful exit unless explicitly
   requested and the arm is physically supported.
7. Stop on camera failure, non-finite action, tracking-watchdog abort, unknown
   joint key, or a checkpoint/config mismatch.

The synchronized limiter applies one common scale to the complete commanded
joint vector. The current implementation includes wrist roll and gripper.
Changing that membership requires implementation, tests, and hardware review.

## Data contract

- Record three cameras at `640 x 480`, 30 FPS, MJPG.
- Rotate the belly camera by 180 degrees in the robot configuration.
- Keep the natural-language task byte-for-byte consistent between recording,
  training metadata, and inference unless deliberately relabeling the dataset.
- Save only completed, valid demonstrations. The custom recorder uses right
  arrow to save; left arrow, timeout, and Escape discard.
- Do not include automatic observe-pose return in the dataset.
- Preserve action/observation feature names and normalization statistics.
- Inspect episode count, FPS, camera streams, feature keys, task metadata, and
  video readability before training.
- Do not merge datasets with incompatible camera keys, FPS, state/action shape,
  task semantics, or robot calibration without an explicit conversion.

## Experiment contract

- Change one independent variable per comparison.
- Record dataset ID, model ID, commit, profile, cold/warm server state, scene,
  parameters, result code, time, and qualitative error.
- Use result codes: `성` success, `접` approach failure, `집` grasp failure,
  `놓` placement/release failure, `오` wrong color, `충` collision/safety stop.
- Include B-zone extremes and G-zone center because the near-front
  shoulder-pan range is underrepresented.
- Do not promote a model to `active` based on a few favorable rollouts.
- Do not delete an existing output directory to fix a run-name collision.

## Secrets and remote services

Never write Hugging Face tokens, W&B API keys, SSH private keys, passwords, or
signed URLs into source, profiles, logs, skills, or answers. Using already
configured credentials for a requested Hugging Face/W&B operation is allowed;
copying or repurposing credentials is not.

Do not upload datasets, models, push Git changes, or publish Docker images
unless the user explicitly asks for that external mutation.

## Required references by task

- Hardware or launch command: `hardware-and-paths.md` and `active.env`.
- Dataset/model choice: `dataset-model-registry.md`.
- Root-cause analysis: `decisions-and-known-failures.md`.
- Official evaluation claim: `graduation-requirements.md`.
- Planning: `roadmap.md` and `experiment-history.md`.

If two references disagree, surface the conflict; do not silently choose the
more convenient value.
