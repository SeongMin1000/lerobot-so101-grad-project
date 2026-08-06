#!/usr/bin/env bash
#
# Client-side launcher for the red SmolVLA pilot:
#   1) follower + leader -> the same saved "observe" pose used for recording
#   2) SO-101 camera/robot client -> remote SmolVLA policy server
#
# Run on the robot/client PC after the GPU policy server is ready.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}"
RUNTIME_CONFIG="${LEROBOT_RUNTIME_CONFIG:-${RUNTIME_CONFIG:-$LEROBOT_ROOT/project/config/runtime.json}}"
export LEROBOT_RUNTIME_CONFIG="$RUNTIME_CONFIG"

ROBOT_PORT="${ROBOT_PORT:-/dev/so101_follower}"
TELEOP_PORT="${TELEOP_PORT:-/dev/so101_leader}"
TOP_CAM="${TOP_CAM:-/dev/cam_top}"
WRIST_CAM="${WRIST_CAM:-/dev/cam_wrist}"
BELLY_CAM="${BELLY_CAM:-/dev/cam_belly}"

# Dataset-key mode is used by checkpoints whose policy inputs are named
# top/wrist/belly. Policy-key mode sends camera1/camera2/camera3 directly for
# checkpoints that declare those canonical image features.
CAMERA_KEY_MODE="${CAMERA_KEY_MODE:-dataset}"

SERVER_ADDRESS="${SERVER_ADDRESS:-100.85.69.64:8080}"
MODEL_PATH="${MODEL_PATH:-eslab1234/smolvla_red_full_126ep_lora_r64_20k_v1}"
# Keep this byte-for-byte identical to --dataset.single_task used while recording.
TASK="${TASK:-Pick up the red block and place it in the red target slot.}"

FPS="${FPS:-30}"
WIDTH="${WIDTH:-640}"
HEIGHT="${HEIGHT:-480}"
OBSERVE_DURATION_S="${OBSERVE_DURATION_S:-3.0}"
OBSERVE_SETTLE_S="${OBSERVE_SETTLE_S:-0.5}"

ACTIONS_PER_CHUNK="${ACTIONS_PER_CHUNK:-10}"
CHUNK_SIZE_THRESHOLD="${CHUNK_SIZE_THRESHOLD:-0.5}"
AGGREGATE_FN_NAME="${AGGREGATE_FN_NAME:-latest_only}"

# Maximum step of the coordinated command vector from the previously sent
# target. The safe default is 1 degree; 0 freezes joint motion.
MAX_RELATIVE_TARGET="${MAX_RELATIVE_TARGET:-1.0}"

# Abort coordinated motion when any motor trails the last command by more than
# this many degrees for the configured number of consecutive control steps.
MAX_TRACKING_ERROR="${MAX_TRACKING_ERROR:-5.0}"
TRACKING_ERROR_GRACE_STEPS="${TRACKING_ERROR_GRACE_STEPS:-2}"

# The recorder used false. Keeping torque enabled prevents the arm from
# suddenly dropping when the inference client exits. Set true only when an
# automatic torque-off on a graceful disconnect is specifically desired.
DISABLE_TORQUE_ON_DISCONNECT="${DISABLE_TORQUE_ON_DISCONNECT:-false}"

# Set to 0 to run until Ctrl+C or the motor tracking watchdog stops inference.
INFERENCE_SECONDS="${INFERENCE_SECONDS:-0}"
SKIP_CONFIRM="${SKIP_CONFIRM:-false}"
DEBUG_OBSERVATION_DIR="${DEBUG_OBSERVATION_DIR:-$LEROBOT_ROOT/var/debug/client_camera_inputs}"
DEBUG_OBSERVATION_LIMIT="${DEBUG_OBSERVATION_LIMIT:-1}"
DEBUG_MOTOR_TRACE_DIR="${DEBUG_MOTOR_TRACE_DIR:-$LEROBOT_ROOT/var/debug/motor_traces}"
DEBUG_MOTOR_TRACE_LIMIT="${DEBUG_MOTOR_TRACE_LIMIT:-300}"

fail() {
  printf '[ERROR] %s\n' "$*" >&2
  exit 1
}

case "$CAMERA_KEY_MODE" in
  dataset)
    TOP_CAMERA_KEY=top
    WRIST_CAMERA_KEY=wrist
    BELLY_CAMERA_KEY=belly
    ;;
  policy)
    TOP_CAMERA_KEY=camera1
    WRIST_CAMERA_KEY=camera2
    BELLY_CAMERA_KEY=camera3
    ;;
  *)
    fail "CAMERA_KEY_MODE must be dataset or policy"
    ;;
esac

require_path() {
  local target_path="$1"
  [[ -e "$target_path" ]] || fail "Required path not found: $target_path"
}

[[ "${CONDA_DEFAULT_ENV:-}" == "lerobot" ]] || fail \
  "Activate the environment first: conda activate lerobot"

command -v python >/dev/null 2>&1 || fail "python is not available"
command -v timeout >/dev/null 2>&1 || fail "GNU timeout is not available"

case "$DISABLE_TORQUE_ON_DISCONNECT" in
  true|false) ;;
  *) fail "DISABLE_TORQUE_ON_DISCONNECT must be true or false" ;;
esac

require_path "$LEROBOT_ROOT"
require_path "$RUNTIME_CONFIG"
require_path "$ROBOT_PORT"
require_path "$TELEOP_PORT"
require_path "$TOP_CAM"
require_path "$WRIST_CAM"
require_path "$BELLY_CAM"
require_path "$LEROBOT_ROOT/src/lerobot/grad_project/control/hybrid_goto_both_pose.py"

cd "$LEROBOT_ROOT"

# Validate the saved observe pose before opening any serial device.
python - "$RUNTIME_CONFIG" <<'PY'
import json
import sys
from pathlib import Path

runtime_path = Path(sys.argv[1])
runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
observe = runtime.get("poses", {}).get("observe")
required = {
    "shoulder_pan.pos",
    "shoulder_lift.pos",
    "elbow_flex.pos",
    "wrist_flex.pos",
    "wrist_roll.pos",
    "gripper.pos",
}

if not isinstance(observe, dict):
    raise SystemExit(f"[ERROR] poses.observe is missing from {runtime_path}")

missing = sorted(required - set(observe))
if missing:
    raise SystemExit(f"[ERROR] poses.observe is missing keys: {missing}")

print(f"[OK] observe pose: {runtime_path}")
for key in sorted(required):
    print(f"     {key}={float(observe[key]):.4f}")
PY

# Do not move the robot if the remote server is not already listening.
python - "$SERVER_ADDRESS" <<'PY'
import socket
import sys

address = sys.argv[1]
try:
    host, port_text = address.rsplit(":", 1)
    port = int(port_text)
except ValueError as exc:
    raise SystemExit(f"[ERROR] Invalid SERVER_ADDRESS: {address}") from exc

try:
    with socket.create_connection((host, port), timeout=3):
        pass
except OSError as exc:
    raise SystemExit(
        f"[ERROR] Policy server is not reachable at {address}: {exc}\n"
        "Start policy_server on the GPU PC first."
    ) from exc

print(f"[OK] policy server reachable: {address}")
PY

printf '\nModel:  %s\n' "$MODEL_PATH"
printf 'Task:   %s\n' "$TASK"
printf 'Images: top=%s, wrist=%s, belly=%s(rotated 180 degrees)\n' \
  "$TOP_CAMERA_KEY" "$WRIST_CAMERA_KEY" "$BELLY_CAMERA_KEY"
printf 'Limit:  %ss (Ctrl+C stops earlier)\n\n' "$INFERENCE_SECONDS"
printf 'Torque off on graceful exit: %s\n\n' "$DISABLE_TORQUE_ON_DISCONNECT"
printf 'Chunk:  %s actions, threshold=%s, aggregation=%s\n' \
  "$ACTIONS_PER_CHUNK" "$CHUNK_SIZE_THRESHOLD" "$AGGREGATE_FN_NAME"
printf 'Joint step clamp: %s degree(s) per control step\n\n' "$MAX_RELATIVE_TARGET"
printf 'Tracking watchdog: %s degree(s), %s consecutive step(s)\n\n' \
  "$MAX_TRACKING_ERROR" "$TRACKING_ERROR_GRACE_STEPS"

if [[ "$SKIP_CONFIRM" != "true" ]]; then
  read -r -p "Workspace clear, 5 blocks placed, emergency stop ready? Type START: " answer
  [[ "$answer" == "START" ]] || fail "Cancelled by user"
fi

printf '\n[1/2] Moving follower and leader to observe pose...\n'
python -m lerobot.grad_project.control.hybrid_goto_both_pose \
  --robot.type=so101_follower \
  --robot.port="$ROBOT_PORT" \
  --robot.id=follower \
  --robot.disable_torque_on_disconnect=false \
  --teleop.type=so101_leader \
  --teleop.port="$TELEOP_PORT" \
  --teleop.id=leader \
  --runtime_config="$RUNTIME_CONFIG" \
  --pose_name=observe \
  --duration_s="$OBSERVE_DURATION_S" \
  --fps="$FPS" \
  --settle_s="$OBSERVE_SETTLE_S" \
  --keep_torque_on_disconnect=true

# Older checkpoints use dataset keys (top/wrist/belly). Newer checkpoints can
# use their declared policy keys (camera1/camera2/camera3) directly, avoiding a
# server-side feature lookup before the saved rename processor runs.
CAMERAS="{ $TOP_CAMERA_KEY: {type: opencv, index_or_path: '$TOP_CAM', width: $WIDTH, height: $HEIGHT, fps: $FPS, fourcc: 'MJPG'}, $WRIST_CAMERA_KEY: {type: opencv, index_or_path: '$WRIST_CAM', width: $WIDTH, height: $HEIGHT, fps: $FPS, fourcc: 'MJPG'}, $BELLY_CAMERA_KEY: {type: opencv, index_or_path: '$BELLY_CAM', width: $WIDTH, height: $HEIGHT, fps: $FPS, fourcc: 'MJPG', rotation: 180} }"

robot_safety_args=()
if [[ -n "$MAX_RELATIVE_TARGET" ]]; then
  robot_safety_args+=(
    --robot.max_relative_target="$MAX_RELATIVE_TARGET"
    --robot.max_tracking_error="$MAX_TRACKING_ERROR"
    --robot.tracking_error_grace_steps="$TRACKING_ERROR_GRACE_STEPS"
  )
fi

client_cmd=(
  python -m lerobot.async_inference.robot_client
  --server_address="$SERVER_ADDRESS"
  --robot.type=so101_follower
  --robot.port="$ROBOT_PORT"
  --robot.id=follower
  --robot.disable_torque_on_disconnect="$DISABLE_TORQUE_ON_DISCONNECT"
  "${robot_safety_args[@]}"
  --robot.cameras="$CAMERAS"
  --task="$TASK"
  --policy_type=smolvla
  --pretrained_name_or_path="$MODEL_PATH"
  --policy_device=cuda
  --client_device=cpu
  --actions_per_chunk="$ACTIONS_PER_CHUNK"
  --chunk_size_threshold="$CHUNK_SIZE_THRESHOLD"
  --aggregate_fn_name="$AGGREGATE_FN_NAME"
  --fps="$FPS"
  --debug_visualize_queue_size=false
  --debug_observation_dir="$DEBUG_OBSERVATION_DIR"
  --debug_observation_limit="$DEBUG_OBSERVATION_LIMIT"
  --debug_motor_trace_dir="$DEBUG_MOTOR_TRACE_DIR"
  --debug_motor_trace_limit="$DEBUG_MOTOR_TRACE_LIMIT"
)

printf '\n[2/2] Starting SmolVLA inference. Keep one hand on Ctrl+C.\n'
printf '[DEBUG] First outgoing camera observation will be saved under %s\n' "$DEBUG_OBSERVATION_DIR"
printf '[DEBUG] Recent motor command/feedback trace will be saved under %s\n' "$DEBUG_MOTOR_TRACE_DIR"

set +e
if [[ "$INFERENCE_SECONDS" == "0" ]]; then
  "${client_cmd[@]}"
  client_status=$?
else
  timeout \
    --foreground \
    --signal=INT \
    --kill-after=8s \
    "${INFERENCE_SECONDS}s" \
    "${client_cmd[@]}"
  client_status=$?
fi
set -e

case "$client_status" in
  0)
    printf '[DONE] Robot client exited normally.\n'
    ;;
  124)
    printf '[STOP] Reached the %ss inference safety limit.\n' "$INFERENCE_SECONDS"
    ;;
  130)
    printf '[STOP] Inference stopped with Ctrl+C.\n'
    ;;
  *)
    fail "Robot client exited with status $client_status"
    ;;
esac

if [[ "$DISABLE_TORQUE_ON_DISCONNECT" == "true" ]]; then
  printf 'Follower torque was disabled on graceful disconnect.\n'
else
  printf 'Follower torque remains enabled after graceful disconnect, matching the recording setup.\n'
fi
printf 'Reset the scene manually before another run.\n'
