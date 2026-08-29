# Decisions, Known Failures, and Diagnostic Evidence

Last synchronized: 2026-08-06

## Settled decisions

- Use SmolVLA as the primary policy and retain ACT as a comparison baseline.
- Use one neural checkpoint for both official tasks.
- Keep automatic observe return outside recorded episodes.
- Use right arrow to save only a completed demonstration; discard left arrow,
  timeout, and Escape attempts.
- Side/belly camera is completely deprecated/unused. Standard setup uses two cameras (Top and Wrist) at 640x480@30 MJPG. Top camera V4L2 parameters use hardware factory defaults (Auto WB, brightness 0, contrast 40, saturation 64, gamma 300).
- Use `latest_only` as the current action aggregation reference.
- Keep physical `START` confirmation and torque enabled on graceful exit.
- Preserve aspect ratio for SmolVLA and normalize image values exactly once.
- Always enable W&B logging (`--wandb.enable=true --wandb.project=lerobot`) and Hugging Face Hub upload (`--policy.push_to_hub=true --policy.repo_id="${HF_USER}/${RUN_NAME}"`) by default in all training commands.
- Keep project code separated from upstream-oriented LeRobot modules.

## Symptom-to-evidence map

| Symptom | Evidence already observed | First checks |
| --- | --- | --- |
| Arm turns right and grasps empty space | Early 30/64-episode policies; sometimes bias already present in raw shoulder-pan output | model output, task/camera mapping, spatial coverage, normalization stats |
| Approach is 1-4 cm right of block | Seen on 126-episode model; preprocessing fix later removed much of the error | matched client/server images, padded input, raw vs postprocessed action |
| Works only at recorded positions | Strong in early SmolVLA tests | zone/angle coverage, camera invariance, recovery data, overfitting |
| Initial arm bends backward or acts on stale intent | Improved when actions consumed per chunk were reduced | chunk length, queue threshold, server reset, last observation, joint clamp |
| Motion becomes too slow | Reported around `MAX_RELATIVE_TARGET=0.75` | change only clamp, measure time and tracking, retain safety threshold |
| Correct approach but bad grasp angle | Current 138-episode observations | wrist-view data, wrist roll/flex coverage, pre-close alignment, grasp examples |
| Block is pushed instead of grasped | Current recovery targets | descend trajectory, gripper-open timing, correction demonstrations |
| Block reaches target but is not released | Current 138-episode observations | endpoint hold/release frames, gripper action, coordinated limiter effect |
| `observation.images.top`/camera `KeyError` | Dataset/policy key mismatch occurred | checkpoint features, saved rename map, `CAMERA_KEY_MODE`, server log |
| First chunk is strange after client restart | Old server retained `last_processed_obs` | verify current `Ready()` reset; restart server after code/preprocessor changes |
| Repeated wrist camera read failure | `OpenCVCamera(/dev/cam_wrist) read failed` occurred | USB bandwidth/power, MJPG mode, device alias, competing process, cable |
| `Required path not found: /dev/cam_belly` | Occurred on 2-camera setups when script enforced 3 cameras | Robot scripts updated to make belly cam optional and support 2-camera dict (`top` + `wrist`) |
| Training command exits on unknown args | Per-transform `tfs.*` and `--eval_freq=0` were rejected | local `lerobot-train --help`; remove unsupported copied flags |
| Training refuses output directory | `FileExistsError` when resume was false | new run/output name or supported resume; never delete by default |
| Resume fails with `A config_path is expected` | LeRobot 0.5.2 `--resume=true` requires explicit path to checkpoint's `train_config.json` | add `--config_path="<output_dir>/checkpoints/<step>/pretrained_model/train_config.json"` or use `--policy.pretrained_path` |
| Pretrained fine-tuning fails with `repo_id argument missing` | Base model config had `push_to_hub: true` | specify `--policy.repo_id="<new_hub_id>"` or `--policy.push_to_hub=false` |
| PEFT adapter config 404 | Adapter/full-checkpoint loading ambiguity | inspect Hub files and `adapter_config.json`; test load before robot run |
| Wrist pitches down instead of up during flight | `wrist_flex` sign inversion in motor convention | Negative bump `cmd["wrist_flex.pos"] -= bump * sin(pi*s)` lifts wrist up towards horizon |
| Camera occlusion by robot arm during joint teaching | Arm covering block caused YOLO to detect arm parts | Use 2-step (Snapshot unoccluded -> Teach follower joints) workflow |
| `Motor tracking error exceeded safety limit` on gripper during grasp | Physical block thickness stops gripper at ~23-25° vs 0-2° goal (~22° error); `MAX_TRACKING_ERROR=20.0` with 5 grace steps triggers false stall abort | Set `MAX_TRACKING_ERROR=35.0` and `TRACKING_ERROR_GRACE_STEPS=10` in inference profile/command |

## Preprocessing facts

The current intended route is:

1. Client captures named camera images.
2. Client serializes raw observations.
3. Server reconstructs feature names.
4. Async helper converts HWC to CHW and normalizes once.
5. For SmolVLA, the helper does not force fixed square resize.
6. SmolVLA resizes with aspect preservation and pads to 512x512.
7. Saved preprocessor applies the checkpoint's rename and normalization rules.

Do not reintroduce an intermediate 256x256 square distortion. Do not divide an
already `[0,1]` float image by 255 again.

## Coordinated limiter fact

For a requested delta vector, a single scale is selected:

`scale = min(1, cap_i / abs(delta_i))` over every capped joint.

Every joint then receives:

`safe_i = previous_command_i + scale * delta_i`

Example: requested deltas `(+50, -40)` with a 2-degree cap give a common scale
`2/50 = 0.04`, so the sent deltas are `(+2.0, -1.6)`. This preserves direction
and timing relationships. The current vector includes shoulder, elbow, wrist,
and gripper keys present in the action.

## Data-distribution hypothesis still under test

The near-front shoulder-pan angle has much less data than wide side angles.
B-zone extremes and G-zone center collectively represent only a few reference
directions, while broader regions contain many more reachable points. The fact
that side positions work better supports this hypothesis, but it is not proof.
Test it by adding controlled front-angle demonstrations and comparing the same
model/training recipe against a matched baseline.

## Recovery/HIL terminology

The current recorder implements **offline recovery demonstrations**:

1. Let inference reach a recognizable failure state.
2. Stop safely without changing the scene.
3. Use pre-record teleoperation to align follower/leader at that failure state.
4. Press Enter and record only the human correction to success.

This is not a complete online intervention logger that seamlessly records model
actions and human takeovers inside one inference process. Do not claim true
online HIL unless the runtime is extended and verified to log intervention
boundaries and executed actions.

## Unresolved engineering decisions

1. The canonical observe pose differs from an older discussed safe candidate.
   Do not change it without physical verification.
2. Whether wrist roll and gripper should be excluded from the common limiter is
   unresolved; current code includes them.
3. Verify whether recorder action data always reflects the action actually sent
   after safety limiting.
4. Verify adapter-only versus full/merged LoRA loading for each Hub model.
5. Determine the best chunk/clamp combination with controlled trials rather
   than intuition.
6. Determine whether the target verifier should become part of the final FSM or
   remain an independent evaluation tool.

## Historical package warning

`hybrid_cv_act_package.zip` is an early snapshot. Its installer can overwrite
current detector and client files. Use it only for source comparison.
