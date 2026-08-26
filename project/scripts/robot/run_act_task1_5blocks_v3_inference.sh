#!/usr/bin/env bash
#
# Client-side launcher for 5-Block Task 1 ACT Model Inference:
#   Model: eslab1234/task1_hybrid_5blocks_v3_223ep_merged_act_b16_v1
#   1) Follower & Leader move to Observe pose
#   2) Connects to GPU Policy Server (100.85.69.64:8080)
#   3) Executes autonomous 5-block pick and place sequence
#

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}"
RUNTIME_CONFIG="${LEROBOT_RUNTIME_CONFIG:-$LEROBOT_ROOT/project/config/runtime.json}"

if [[ -x "$HOME/miniforge3/envs/lerobot/bin/python" ]]; then
  PYTHON_BIN="$HOME/miniforge3/envs/lerobot/bin/python"
elif [[ -x "$HOME/miniconda3/envs/lerobot/bin/python" ]]; then
  PYTHON_BIN="$HOME/miniconda3/envs/lerobot/bin/python"
else
  PYTHON_BIN="python"
fi
export PYTHONPATH="$LEROBOT_ROOT/src:$LEROBOT_ROOT:${PYTHONPATH:-}"

ROBOT_PORT="${ROBOT_PORT:-/dev/so101_follower}"
TELEOP_PORT="${TELEOP_PORT:-/dev/so101_leader}"
TOP_CAM="${TOP_CAM:-/dev/cam_top}"
WRIST_CAM="${WRIST_CAM:-/dev/cam_wrist}"

# Model & Server Setup
SERVER_ADDRESS="${SERVER_ADDRESS:-100.85.69.64:8080}"
MODEL_PATH="${MODEL_PATH:-eslab1234/task1_hybrid_5blocks_v3_223ep_merged_act_b16_v1}"
TASK="${TASK:-Pick and place 5 blocks in sequence (red, yellow, wood, green, blue).}"
POLICY_TYPE="act"

FPS="${FPS:-30}"
WIDTH="${WIDTH:-640}"
HEIGHT="${HEIGHT:-480}"
OBSERVE_DURATION_S="${OBSERVE_DURATION_S:-3.0}"
OBSERVE_SETTLE_S="${OBSERVE_SETTLE_S:-0.5}"

# Chunk & Safety Settings
ACTIONS_PER_CHUNK="${ACTIONS_PER_CHUNK:-30}"
CHUNK_SIZE_THRESHOLD="${CHUNK_SIZE_THRESHOLD:-0.5}"
AGGREGATE_FN_NAME="${AGGREGATE_FN_NAME:-latest_only}"
MAX_RELATIVE_TARGET="${MAX_RELATIVE_TARGET:-1.0}"
MAX_TRACKING_ERROR="${MAX_TRACKING_ERROR:-35.0}"
TRACKING_ERROR_GRACE_STEPS="${TRACKING_ERROR_GRACE_STEPS:-10}"
DISABLE_TORQUE_ON_DISCONNECT="${DISABLE_TORQUE_ON_DISCONNECT:-false}"
INFERENCE_SECONDS="${INFERENCE_SECONDS:-0}"
SKIP_CONFIRM="${SKIP_CONFIRM:-false}"

fail() {
  printf '[ERROR] %s\n' "$*" >&2
  exit 1
}

require_path() {
  local target_path="$1"
  [[ -e "$target_path" ]] || fail "Required path not found: $target_path"
}

printf '\n==============================================================================\n'
printf '🤖 [ACT 5-BLOCKS 223EP INFERENCE CLIENT]\n'
printf '==============================================================================\n'
printf 'Model:          %s\n' "$MODEL_PATH"
printf 'Policy Type:    %s\n' "$POLICY_TYPE"
printf 'Server:         %s\n' "$SERVER_ADDRESS"
printf 'Cameras:        top (/dev/cam_top), wrist (/dev/cam_wrist)\n'
printf 'Chunk Settings: actions=%s, threshold=%s, agg=%s\n' "$ACTIONS_PER_CHUNK" "$CHUNK_SIZE_THRESHOLD" "$AGGREGATE_FN_NAME"
printf 'Safety Clamp:   max_step=%s deg, tracking_error_max=%s deg\n' "$MAX_RELATIVE_TARGET" "$MAX_TRACKING_ERROR"
printf '==============================================================================\n'

require_path "$ROBOT_PORT"
require_path "$TELEOP_PORT"
require_path "$TOP_CAM"
require_path "$WRIST_CAM"
require_path "$RUNTIME_CONFIG"

# Check GPU policy server reachability
"$PYTHON_BIN" - "$SERVER_ADDRESS" <<'PY'
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
        "Please start the GPU policy server on GPU PC first:\n"
        "  bash project/scripts/gpu/run_smolvla_red_policy_server.sh\n"
    ) from exc

print(f"✅ [OK] GPU Policy server reachable at {address}")
PY

if [[ "$SKIP_CONFIRM" != "true" ]]; then
  read -r -p "Workspace clear, 5 blocks placed on table, emergency stop ready? Type START: " answer
  [[ "$answer" == "START" ]] || fail "Cancelled by user"
fi

printf '\n[1/2] Moving follower and leader to observe pose...\n'
"$PYTHON_BIN" -m lerobot.grad_project.control.hybrid_goto_both_pose \
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

CAMERAS="{ top: {type: opencv, index_or_path: '$TOP_CAM', width: $WIDTH, height: $HEIGHT, fps: $FPS, fourcc: 'MJPG'}, wrist: {type: opencv, index_or_path: '$WRIST_CAM', width: $WIDTH, height: $HEIGHT, fps: $FPS, fourcc: 'MJPG'} }"

robot_safety_args=()
if [[ -n "$MAX_RELATIVE_TARGET" ]]; then
  robot_safety_args+=(
    --robot.max_relative_target="$MAX_RELATIVE_TARGET"
    --robot.max_tracking_error="$MAX_TRACKING_ERROR"
    --robot.tracking_error_grace_steps="$TRACKING_ERROR_GRACE_STEPS"
  )
fi

printf '\n[2/2] Starting real-time async inference loop...\n'
exec "$PYTHON_BIN" -m lerobot.async_inference.robot_client \
  --server_address="$SERVER_ADDRESS" \
  --policy_type=act \
  --pretrained_name_or_path="$MODEL_PATH" \
  --actions_per_chunk="$ACTIONS_PER_CHUNK" \
  --task="$TASK" \
  --device=cuda \
  --robot.type=so101_follower \
  --robot.port="$ROBOT_PORT" \
  --robot.id=follower \
  --robot.disable_torque_on_disconnect="$DISABLE_TORQUE_ON_DISCONNECT" \
  "${robot_safety_args[@]}" \
  --robot.cameras="$CAMERAS" \
  --teleop.type=so101_leader \
  --teleop.port="$TELEOP_PORT" \
  --teleop.id=leader \
  --teleop.disable_torque_on_disconnect=true \
  --fps="$FPS" \
  --chunk_size_threshold="$CHUNK_SIZE_THRESHOLD" \
  --aggregate_fn_name="$AGGREGATE_FN_NAME" \
  --inference_seconds="$INFERENCE_SECONDS"
