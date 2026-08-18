---
name: so101-diagnose-runtime
description: SO-101 SmolVLA/ACT 런타임의 카메라 손상, camera-key mismatch, resize·normalization, stale queue, action bias, chunk/clamp, motor tracking, 네트워크 중단 원인을 증거 순서대로 진단한다. Use for wrong direction, 3-4 cm error, camera failure, KeyError, stale first chunk, clamp, watchdog, crash, or unexplained inference behavior.
---

# SO-101 Diagnose Runtime

## Respect task scope

If the user asks only “why” or “diagnose,” inspect and report the cause without
changing runtime code or moving the robot. Implement a fix only when requested.

Read:

1. `project/docs/agent-context/project-contract.md`
2. `project/docs/agent-context/decisions-and-known-failures.md`
3. `project/docs/agent-context/hardware-and-paths.md`
4. `project/docs/RUNTIME_REGRESSION_CHECK.md`
5. `project/config/experiment-profiles/active.env`

## Preserve evidence first

Before restarting processes or resetting the scene, capture:

- complete client and server logs with timestamps;
- exact model, task, camera mode, chunk, clamp, watchdog, and commit;
- whether server/client were cold-started or reused;
- client and server camera-capture directories;
- motor trace and the final requested/sent/feedback action rows;
- physical symptom, zone, error direction/distance, and failure code.

Do not call a symptom “model error” until the input and action routes are
checked.

## Run the read-only regression suite

```bash
cd "${LEROBOT_ROOT:-$HOME/lerobot}"
conda activate lerobot
bash project/scripts/tools/check_runtime_regressions.sh
```

Interpret `SKIP` as missing runtime evidence, not a pass. The tool can verify
static preprocessing and limiter behavior without a live robot, but matched
camera and motor-trace checks need captured data.

## Trace the pipeline in this order

1. Robot-PC camera frame before serialization.
2. GPU-server raw deserialized frame.
3. Async helper output: key, dtype, range, shape, channel order.
4. Policy preprocessor input and effective rename map.
5. Raw policy action chunk.
6. Postprocessed physical-unit action.
7. Action chosen after queue/aggregation.
8. Coordinated-limiter output actually sent to follower.
9. Measured joint feedback and tracking error.

Find the first stage where expected and observed behavior diverge. Later-stage
symptoms are not root-cause evidence.

## Camera and preprocessing branch

Verify:

- top/wrist physical identity and no swapped USB aliases (side/belly camera is unused);
- 640x480, MJPG, 30 FPS actually negotiated;
- RGB/BGR conversion is correct;
- client and server capture pixel content match;
- image values are normalized to `[0,1]` exactly once;
- SmolVLA keeps aspect ratio and pads to 512x512;
- no intermediate square stretch is reintroduced;
- checkpoint feature keys match `CAMERA_KEY_MODE` and saved rename map.

For repeated wrist-camera read failures, inspect USB bandwidth/power, cable,
hub, exclusive device ownership, MJPG format, and stable udev alias before
changing the model.

## Server-state and queue branch

Verify current `Ready()` clears the observation queue, predicted timesteps, and
`last_processed_obs`. Compare:

- cold server + cold client;
- same server + new client;
- server restarted after source/preprocessor change.

Log observation timestep, action-chunk origin timestep, queue contents, and
`must_go`. Stale observation evidence must be temporal, not visual intuition.

## Action and motor branch

Compare the first wrong shoulder-pan/wrist/gripper value across raw model,
postprocess, queue selection, limiter, sent command, and feedback.

- Bias in raw model output: investigate dataset/task/input/model.
- Correct raw action but wrong postprocess: investigate stats, feature order,
  units, or processor config.
- Correct postprocess but wrong queued action: investigate chunk/aggregation.
- Correct selected action but scaled motion: inspect common limiter scale.
- Correct sent action but lagging feedback: inspect motor, torque, bus, load,
  watchdog, and calibration.

The synchronized limiter includes every action joint currently supplied,
including wrist roll and gripper. Do not diagnose their slower motion as an
independent clamp unless the code actually excludes them.

## Symptom shortcuts

- `KeyError observation.images.*`: inspect dataset/policy key mode and rename
  processor first.
- Strange first chunk after reconnect: inspect server reset and timestep state.
- Consistent right offset with matched inputs: inspect raw shoulder-pan output
  and front-angle data coverage.
- Severe initial backward motion: compare chunk horizon, stale observation, and
  coordinated step limit.
- Slow motion after lowering clamp: expected rate effect; quantify completion
  time and tracking rather than removing the watchdog.
- Release failure: inspect raw gripper action, common scale, actual sent value,
  feedback, and endpoint demonstration frames.
- Adapter/config 404: inspect checkpoint packaging on GPU; do not connect robot.

## Report format

Return:

1. Reproduced symptom and exact profile.
2. Evidence that passed.
3. First failing/divergent stage.
4. Root cause confidence: confirmed, likely, or unresolved.
5. Smallest discriminating next test.
6. Fix proposal only if authorized.
7. Hardware/network checks that were skipped.

Separate observation from inference. Example: “raw shoulder-pan is already
biased right” is evidence; “the dataset lacks front angles” remains a
hypothesis until a controlled retraining comparison.
