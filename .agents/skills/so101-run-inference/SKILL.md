---
name: so101-run-inference
description: SO-101 SmolVLA 비동기 추론의 GPU policy server와 robot client를 안전하게 시작·재시작·종료하고 모델, task, camera keys, chunk, clamp, watchdog 설정을 검증한다. Use for inference, rollout, policy server, robot client, server address, chunk settings, camera key mode, or startup commands.
---

# SO-101 Run Inference

## Load context

Read:

1. `project/docs/agent-context/project-contract.md`
2. `project/docs/agent-context/hardware-and-paths.md`
3. `project/config/experiment-profiles/active.env`

Read `dataset-model-registry.md` when selecting a checkpoint and
`decisions-and-known-failures.md` when reproducing an anomaly.

## Separate the two machines

- GPU PC: run `policy_server`; no robot/camera devices are opened.
- Robot PC: move follower/leader to observe, capture cameras, run safety loop,
  and connect to the GPU server.

The current server receives `MODEL_PATH` from the client during policy setup.
Do not look for a model argument in the GPU launcher unless source has changed.

## Read-only preflight

On both machines:

```bash
cd "${LEROBOT_ROOT:-$HOME/lerobot}"
conda activate lerobot
source project/config/experiment-profiles/active.env
python project/scripts/tools/check_antigravity_setup.py
```

On the GPU PC, confirm port availability, source import, CUDA, free VRAM, and
checkpoint loadability. On the robot PC, confirm all five device paths and the
runtime JSON before movement.

Inspect the checkpoint's image features and saved processor rename map. Set:

- `CAMERA_KEY_MODE=dataset` for raw `top/wrist/belly` plus a valid saved rename;
- `CAMERA_KEY_MODE=policy` for direct `camera1/camera2/camera3` inputs.

Do not guess from an old model name.

## Start the GPU server

When server execution is authorized:

```bash
cd "$LEROBOT_ROOT"
conda activate lerobot
source project/config/experiment-profiles/active.env
bash project/scripts/gpu/run_smolvla_red_policy_server.sh
```

Require the log to show the bind address/port and successful module import.
Keep the terminal or managed process visible for model-load errors.

## Start the robot client

Only after the server is reachable and the user explicitly authorizes physical
movement:

```bash
cd "$LEROBOT_ROOT"
conda activate lerobot
source project/config/experiment-profiles/active.env
bash project/scripts/robot/run_smolvla_red_observe_inference.sh
```

Use the general observe launcher with `active.env`. The file
`run_smolvla_red_latest_safe.sh` assigns its own fixed 126-episode model and
parameters, so it is for reproduction rather than the active profile.

Before typing `START`, verify the launcher's printed values:

- exact model and task;
- top/wrist/belly key mapping and belly rotation;
- server endpoint;
- actions per chunk, threshold, and aggregation;
- maximum coordinated step;
- tracking threshold and grace steps;
- timeout and torque behavior;
- saved observe pose;
- clear workspace and emergency stop.

Never set `SKIP_CONFIRM=true` by default.

## Understand current state reset

The current server `Ready()` path resets:

- shutdown state;
- observation queue;
- predicted timesteps;
- `last_processed_obs`;
- debug capture IDs.

A client reconnect should therefore accept a fresh first observation. Restart
the server process after changing server/helper/preprocessor source, Python
environment, or installed dependencies. A reconnect is not a substitute for
reloading modified code.

## Tune only under an evaluation plan

- `ACTIONS_PER_CHUNK` changes open-loop horizon and reaction speed.
- `CHUNK_SIZE_THRESHOLD` changes when new chunks are requested.
- `latest_only` prioritizes the newest prediction; weighted aggregation mixes
  predictions and changes the experiment.
- `MAX_RELATIVE_TARGET` scales the whole action vector, not each joint
  independently.
- `MAX_TRACKING_ERROR` and grace steps are safety controls, not performance
  tuning knobs to disable casually.

Change one value at a time and log it with `so101-evaluate-experiment`.

## Observe and stop

During rollout, watch:

- server model-load and effective rename-map messages;
- camera read warnings;
- observation/action queue behavior;
- raw model and postprocessed action diagnostics when enabled;
- coordinated-clamp warnings;
- motor tracking error and watchdog aborts.

Stop immediately on collision risk, camera stream loss, repeated stale action,
unexpected model/task/key output, or tracking abort. Use Ctrl+C first when safe;
use the physical emergency stop for imminent danger.

After exit, record the motor-trace path, client/server state, result, and any
captured input directory. Do not reset the scene before saving the evidence
needed for diagnosis.
