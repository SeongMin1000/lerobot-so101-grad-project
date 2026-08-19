# Hardware, Network, Paths, and Variables

Last synchronized: 2026-08-06

## Machines

| Role | Known hardware and responsibility |
| --- | --- |
| Robot/client PC | No training GPU; connects SO-101 follower/leader and three cameras; records data; runs async client and safety loop |
| GPU PC | RTX 3090 24 GB VRAM, 32 GB RAM; trains models and runs async policy server |
| Network | Tailscale between robot PC and GPU PC; current GPU endpoint `100.85.69.64:8080` |
| Final edge device | Jetson Orin Nano; handles robot/camera-side work and communicates with evaluation server |
| Evaluation server | Provided by course staff; team must deliver a Docker image; `nvidia-driver-580-open` is planned |

The Tailscale address is an active-profile value, not a permanent constant.

## Robot and camera devices

| Purpose | Stable device path |
| --- | --- |
| SO-101 follower | `/dev/so101_follower` |
| SO-101 leader | `/dev/so101_leader` |
| Top camera | `/dev/cam_top` |
| Wrist camera | `/dev/cam_wrist` |
| Belly / Side camera (Deprecated) | `/dev/cam_belly` (Unused) |

Camera contract:

- `640 x 480`
- 30 FPS
- MJPG
- camera count: two (`top`, `wrist`; side/belly camera unused)
- Top camera (`/dev/cam_top`) V4L2 hardware controls set to hardware defaults:
  - `brightness=0`, `contrast=40`, `saturation=64`, `hue=0`, `gamma=300`, `sharpness=50`
  - `white_balance_automatic=1` (Auto White Balance enabled), `power_line_frequency=1`
  - `auto_exposure=3` (Aperture Priority Mode), `backlight_compensation=0`

Use stable udev aliases above. Do not replace them with transient `/dev/videoN`
paths in committed scripts.

## Software baseline

- LeRobot version: `0.5.2`
- Python: `>=3.12`
- Conda environment used across all machines (PC and Jetson edge): `lerobot` (Python 3.12, conda-forge CPU PyTorch 2.11.0 on Jetson)
- Hugging Face namespace: `eslab1234`
- Primary policy: SmolVLA
- Async transport: gRPC/TCP-style LeRobot async inference over Tailscale

Run local CLI help before using flags:

```bash
conda activate lerobot
cd ~/lerobot
lerobot-train --help
python -m lerobot.grad_project.recording.smolvla_record_observe_return --help
python -m lerobot.async_inference.policy_server --help
python -m lerobot.async_inference.robot_client --help
```

## Canonical paths

| Item | Repository-relative path |
| --- | --- |
| Project Python | `src/lerobot/grad_project/` |
| Async client/server | `src/lerobot/async_inference/` |
| SO-101 follower safety | `src/lerobot/robots/so_follower/` and `src/lerobot/robots/utils.py` |
| Runtime poses | `project/config/runtime.json` |
| OpenCV detector | `project/config/detector.json` |
| Target verifier | `project/config/target_verifier.json` |
| Detector background | `project/assets/opencv/top_background.png` |
| Empty target reference | `project/assets/opencv/target_empty_reference.png` |
| Active experiment profile | `project/config/experiment-profiles/active.env` |
| GPU launcher | `project/scripts/gpu/run_smolvla_red_policy_server.sh` |
| Robot launcher | `project/scripts/robot/run_smolvla_red_observe_inference.sh` |
| Fixed historical safety wrapper | `project/scripts/robot/run_smolvla_red_latest_safe.sh` |
| Runtime regression entry point | `project/scripts/tools/check_runtime_regressions.sh` |
| Generated artifacts | `var/` |

`run_smolvla_red_latest_safe.sh` intentionally assigns fixed values and will
override previously exported profile values. Use the general observe launcher
when evaluating `active.env`; use the fixed wrapper only to reproduce its
named 126-episode configuration.

## Core environment variables

| Variable | Meaning |
| --- | --- |
| `LEROBOT_ROOT` | Absolute repository root; normally `$HOME/lerobot` |
| `HF_USER` | Hugging Face namespace, `eslab1234` |
| `LEROBOT_RUNTIME_CONFIG` | Canonical runtime JSON override |
| `LEROBOT_DETECTOR_CALIB` | Canonical detector JSON override |
| `RUNTIME_CONFIG` | Compatibility alias used by some launchers |
| `ROBOT_PORT`, `TELEOP_PORT` | follower and leader serial aliases |
| `TOP_CAM`, `WRIST_CAM`, `BELLY_CAM` | camera aliases |
| `WIDTH`, `HEIGHT`, `FPS` | camera/control rate |
| `SERVER_ADDRESS` | robot-client destination as `host:port` |
| `POLICY_SERVER_HOST`, `POLICY_SERVER_PORT` | GPU bind address and port |
| `INFERENCE_LATENCY` | server target latency; current reference `0.033` s |
| `OBS_QUEUE_TIMEOUT` | server observation wait timeout; current reference `1.0` s |
| `MODEL_PATH` | active Hugging Face model/checkpoint |
| `TASK` | exact task instruction used by the dataset/checkpoint |
| `CAMERA_KEY_MODE` | `dataset` for `top/wrist/belly`, `policy` for `camera1/2/3` |
| `ACTIONS_PER_CHUNK` | number of predicted actions consumed per response |
| `CHUNK_SIZE_THRESHOLD` | client queue refill threshold |
| `AGGREGATE_FN_NAME` | current reference `latest_only` |
| `MAX_RELATIVE_TARGET` | maximum coordinated command-vector step |
| `MAX_TRACKING_ERROR` | watchdog command-to-feedback error threshold |
| `TRACKING_ERROR_GRACE_STEPS` | consecutive error steps before abort |
| `INFERENCE_SECONDS` | `0` means run until stop/watchdog |
| `DISABLE_TORQUE_ON_DISCONNECT` | keep `false` unless arm is supported and behavior is intended |
| `SKIP_CONFIRM` | keep `false` for physical runs |

Training variables live in the same profile but are not direct LeRobot CLI
flags. The training skill must translate them only after inspecting
`lerobot-train --help`.

## Camera-key contract

Two valid naming layers exist:

| Mode | Raw observation keys | Use when |
| --- | --- | --- |
| `dataset` | `top`, `wrist` (historical 3-cam: `top`, `wrist`, `belly`) | Checkpoint preprocessor contains a saved rename map to policy keys |
| `policy` | `camera1`, `camera2` (historical 3-cam: `camera1`, `camera2`, `camera3`) | Checkpoint declares those canonical image features directly |

Do not guess. Inspect checkpoint `config.json`, processor config, dataset
features, and policy-server log of the effective rename map. A mismatch can
raise `KeyError` or route the wrong camera into a policy input.

## Current preprocessing contract

- Camera transport may produce uint8 `[0,255]` or float `[0,1]`.
- `prepare_image` normalizes to float `[0,1]` exactly once.
- Async helper does not force SmolVLA to a square image.
- SmolVLA performs aspect-ratio-preserving resize and zero padding to
  `512 x 512` (`resize_imgs_with_padding`).
- ACT and other fixed-shape policies may use the async helper's fixed resize.
- Bilinear interpolation uses `align_corners=False`.

Run the regression tool after changing any part of this route.

## Current runtime pose snapshot

`project/config/runtime.json` currently contains this observe pose:

| Joint | Value (degrees) |
| --- | ---: |
| `shoulder_pan.pos` | `-0.9670` |
| `shoulder_lift.pos` | `-87.4286` |
| `elbow_flex.pos` | `16.6593` |
| `wrist_flex.pos` | `95.4725` |
| `wrist_roll.pos` | `-7.0330` |
| `gripper.pos` | `26.8519` |

An older discussed observe candidate was around pan `-7.6923`, lift
`-101.0549`, elbow `20.8791`, flex `99.5165`, roll `-1.4945`, gripper `27.0`.
Do not replace the canonical pose with that historical candidate without a
physical verification and explicit user decision.

## Secret-handling rule

Profiles may store public repo IDs, device aliases, ports, and non-secret
hyperparameters. Never store access tokens, API keys, passwords, SSH private
keys, cookies, or signed download URLs.
