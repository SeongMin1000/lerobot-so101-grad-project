#!/usr/bin/env bash
#
# Dedicated ACT Observe Inference Launcher for SO-101:
#   1) Smoothly moves follower arm to the saved "observe" pose (3.0s).
#   2) Launches 100% continuous ACT neural network policy inference directly from observe pose.
#
# Runs on the local robot PC after the GPU policy server is ready with policy_type=act.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}"

_USER_MODEL_PATH="${MODEL_PATH:-}"
_USER_POLICY_TYPE="${POLICY_TYPE:-}"
_USER_CAMERA_KEY_MODE="${CAMERA_KEY_MODE:-}"
_USER_BELLY_CAMERA_KEY="${BELLY_CAMERA_KEY:-}"
_USER_CAMERAS="${CAMERAS:-}"
_USER_ACTIONS_PER_CHUNK="${ACTIONS_PER_CHUNK:-}"
_USER_CHUNK_SIZE_THRESHOLD="${CHUNK_SIZE_THRESHOLD:-}"
_USER_AGGREGATE_FN_NAME="${AGGREGATE_FN_NAME:-}"

if [[ -f "$LEROBOT_ROOT/project/config/experiment-profiles/active.env" ]]; then
  source "$LEROBOT_ROOT/project/config/experiment-profiles/active.env"
fi

RUNTIME_CONFIG="${LEROBOT_RUNTIME_CONFIG:-${RUNTIME_CONFIG:-$LEROBOT_ROOT/project/config/runtime.json}}"
if [[ ! -e "$RUNTIME_CONFIG" && -e "$LEROBOT_ROOT/project/config/runtime.json" ]]; then
  RUNTIME_CONFIG="$LEROBOT_ROOT/project/config/runtime.json"
fi
export LEROBOT_RUNTIME_CONFIG="$RUNTIME_CONFIG"

POLICY_TYPE="${_USER_POLICY_TYPE:-${POLICY_TYPE:-act}}"
SERVER_ADDRESS="${SERVER_ADDRESS:-100.85.69.64:8080}"
MODEL_PATH="${_USER_MODEL_PATH:-${MODEL_PATH:-eslab1234/act_five_blocks_full_80ep_c100_a100_100k_v1}}"
TASK="${TASK:-Pick up the red block and place it in the red target slot.}"

ROBOT_PORT="${ROBOT_PORT:-/dev/so101_follower}"
TELEOP_PORT="${TELEOP_PORT:-/dev/so101_leader}"
TOP_CAM="${TOP_CAM:-/dev/cam_top}"
WRIST_CAM="${WRIST_CAM:-/dev/cam_wrist}"
BELLY_CAM="${BELLY_CAM:-/dev/cam_belly}"
BELLY_ROTATION="${BELLY_ROTATION:-0}"

CAMERA_KEY_MODE="${_USER_CAMERA_KEY_MODE:-dataset}"

FPS="${FPS:-30}"
WIDTH="${WIDTH:-640}"
HEIGHT="${HEIGHT:-480}"
OBSERVE_DURATION_S="${OBSERVE_DURATION_S:-3.0}"
OBSERVE_SETTLE_S="${OBSERVE_SETTLE_S:-0.5}"

ACTIONS_PER_CHUNK="${_USER_ACTIONS_PER_CHUNK:-${ACTIONS_PER_CHUNK:-100}}"
CHUNK_SIZE_THRESHOLD="${_USER_CHUNK_SIZE_THRESHOLD:-${CHUNK_SIZE_THRESHOLD:-0.5}}"
AGGREGATE_FN_NAME="${_USER_AGGREGATE_FN_NAME:-${AGGREGATE_FN_NAME:-latest_only}}"

fail() {
  printf '[ERROR] %s\n' "$*" >&2
  exit 1
}

[[ "${CONDA_DEFAULT_ENV:-}" == "lerobot" ]] || fail "Activate first: conda activate lerobot"
command -v python >/dev/null 2>&1 || fail "python command not found"

[[ -e "$ROBOT_PORT" ]] || fail "Follower port $ROBOT_PORT not found"
[[ -e "$TOP_CAM" ]] || fail "Top camera $TOP_CAM not found"
[[ -e "$WRIST_CAM" ]] || fail "Wrist camera $WRIST_CAM not found"
[[ -e "$BELLY_CAM" ]] || fail "Belly camera $BELLY_CAM not found"

case "$CAMERA_KEY_MODE" in
  dataset)
    TOP_CAMERA_KEY=top
    WRIST_CAMERA_KEY=wrist
    BELLY_CAMERA_KEY="${_USER_BELLY_CAMERA_KEY:-side}"
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

cd "$LEROBOT_ROOT"

if [[ -f "$RUNTIME_CONFIG" ]]; then
  printf '[OK] observe pose: %s\n' "$RUNTIME_CONFIG"
  python -c "
import json
with open('$RUNTIME_CONFIG') as f:
    poses = json.load(f).get('poses', {})
obs = poses.get('observe', {})
for k in sorted(obs):
    print(f'     {k}={obs[k]:.4f}')
"
else
  fail "Runtime config missing: $RUNTIME_CONFIG"
fi

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

printf '\n'
printf 'Model:  %s\n' "$MODEL_PATH"
printf 'Policy: %s\n' "$POLICY_TYPE"
printf 'Task:   %s\n' "$TASK"
printf 'Images: top=%s, wrist=%s, belly=%s(rotation %d deg)\n' \
  "$TOP_CAMERA_KEY" "$WRIST_CAMERA_KEY" "$BELLY_CAMERA_KEY" "$BELLY_ROTATION"
printf 'Limit:  %ss (Ctrl+C stops earlier)\n' "$INFERENCE_SECONDS"
printf '\n'
printf 'Torque off on graceful exit: %s\n' "$DISABLE_TORQUE_ON_DISCONNECT"
printf '\n'
printf 'Chunk:  %s actions, threshold=%s, aggregation=%s\n' \
  "$ACTIONS_PER_CHUNK" "$CHUNK_SIZE_THRESHOLD" "$AGGREGATE_FN_NAME"
printf 'Joint step clamp: %s degree(s) per control step\n' "${MAX_RELATIVE_TARGET:-disabled}"
printf '\n'
printf 'Tracking watchdog: %s degree(s), %s consecutive step(s)\n' \
  "${MAX_TRACKING_ERROR:-disabled}" "$TRACKING_ERROR_GRACE_STEPS"
printf '\n'

if [[ "$SKIP_CONFIRM" != "true" ]]; then
  read -r -p "Workspace clear, 5 blocks placed, emergency stop ready? Type START: " CONFIRM
  if [[ "$CONFIRM" != "START" ]]; then
    fail "ACT inference start aborted."
  fi
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

sleep 1.0

CAMERAS="{ $TOP_CAMERA_KEY: {type: opencv, index_or_path: '$TOP_CAM', width: $WIDTH, height: $HEIGHT, fps: $FPS, fourcc: 'MJPG'}, $WRIST_CAMERA_KEY: {type: opencv, index_or_path: '$WRIST_CAM', width: $WIDTH, height: $HEIGHT, fps: $FPS, fourcc: 'MJPG'}, $BELLY_CAMERA_KEY: {type: opencv, index_or_path: '$BELLY_CAM', width: $WIDTH, height: $HEIGHT, fps: $FPS, fourcc: 'MJPG', rotation: ${BELLY_ROTATION:-0}} }"

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
  --policy_type="$POLICY_TYPE"
  --pretrained_name_or_path="$MODEL_PATH"
  --policy_device=cuda
  --client_device=cpu
  --actions_per_chunk="$ACTIONS_PER_CHUNK"
  --chunk_size_threshold="$CHUNK_SIZE_THRESHOLD"
  --aggregate_fn_name="$AGGREGATE_FN_NAME"
  --fps="$FPS"
  --debug_visualize_queue_size=false
)

printf '\n[2/2] Starting ACT continuous inference from observe pose. Keep one hand on Ctrl+C.\n'
"${client_cmd[@]}"

printf '\n[DONE] ACT Robot client exited normally.\n'
