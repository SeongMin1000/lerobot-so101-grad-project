#!/usr/bin/env bash
#
# Unified Asynchronous Robot Client Launcher for SO-101 (ACT & SmolVLA):
#   1) Smoothly moves follower & leader arms to the saved "observe" pose (3.0s).
#   2) Connects to GPU Policy Server (e.g. 100.85.69.64:8080).
#   3) Executes autonomous 5-block task inference with coordinated safety limiting.
#
# Usage:
#   # 1. Run with active.env default model:
#   bash project/scripts/robot/run_async_inference.sh
#
#   # 2. Run ACT model:
#   POLICY_TYPE=act MODEL_PATH="eslab1234/task1_hybrid_5blocks_v3_223ep_merged_act_b16_150k_v2" bash project/scripts/robot/run_async_inference.sh
#
#   # 3. Run SmolVLA model:
#   POLICY_TYPE=smolvla MODEL_PATH="eslab1234/smolvla_task1_5blocks_v3_330ep_fullft_b16_150k_v1" bash project/scripts/robot/run_async_inference.sh
#

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}"

# Preserve user-provided CLI variables so they are not overwritten by active.env
CLI_MODEL_PATH="${MODEL_PATH:-}"
CLI_POLICY_TYPE="${POLICY_TYPE:-}"
CLI_TASK="${TASK:-}"
CLI_ACTIONS_PER_CHUNK="${ACTIONS_PER_CHUNK:-}"
CLI_CHUNK_SIZE_THRESHOLD="${CHUNK_SIZE_THRESHOLD:-}"
CLI_AGGREGATE_FN_NAME="${AGGREGATE_FN_NAME:-}"
CLI_MAX_RELATIVE_TARGET="${MAX_RELATIVE_TARGET:-}"
CLI_MAX_TRACKING_ERROR="${MAX_TRACKING_ERROR:-}"
CLI_TRACKING_ERROR_GRACE_STEPS="${TRACKING_ERROR_GRACE_STEPS:-}"
CLI_SERVER_ADDRESS="${SERVER_ADDRESS:-}"
CLI_CAMERA_KEY_MODE="${CAMERA_KEY_MODE:-}"

# Source active profile if present (only for fallbacks)
if [[ -f "$LEROBOT_ROOT/project/config/experiment-profiles/active.env" ]]; then
  source "$LEROBOT_ROOT/project/config/experiment-profiles/active.env"
fi

# Re-apply CLI variables if the user provided them explicitly
[[ -n "$CLI_MODEL_PATH" ]] && MODEL_PATH="$CLI_MODEL_PATH"
[[ -n "$CLI_POLICY_TYPE" ]] && POLICY_TYPE="$CLI_POLICY_TYPE"
[[ -n "$CLI_TASK" ]] && TASK="$CLI_TASK"
[[ -n "$CLI_ACTIONS_PER_CHUNK" ]] && ACTIONS_PER_CHUNK="$CLI_ACTIONS_PER_CHUNK"
[[ -n "$CLI_CHUNK_SIZE_THRESHOLD" ]] && CHUNK_SIZE_THRESHOLD="$CLI_CHUNK_SIZE_THRESHOLD"
[[ -n "$CLI_AGGREGATE_FN_NAME" ]] && AGGREGATE_FN_NAME="$CLI_AGGREGATE_FN_NAME"
[[ -n "$CLI_MAX_RELATIVE_TARGET" ]] && MAX_RELATIVE_TARGET="$CLI_MAX_RELATIVE_TARGET"
[[ -n "$CLI_MAX_TRACKING_ERROR" ]] && MAX_TRACKING_ERROR="$CLI_MAX_TRACKING_ERROR"
[[ -n "$CLI_TRACKING_ERROR_GRACE_STEPS" ]] && TRACKING_ERROR_GRACE_STEPS="$CLI_TRACKING_ERROR_GRACE_STEPS"
[[ -n "$CLI_SERVER_ADDRESS" ]] && SERVER_ADDRESS="$CLI_SERVER_ADDRESS"
[[ -n "$CLI_CAMERA_KEY_MODE" ]] && CAMERA_KEY_MODE="$CLI_CAMERA_KEY_MODE"

# Locate Python environment
if [[ -x "$HOME/miniforge3/envs/lerobot/bin/python" ]]; then
  PYTHON_BIN="$HOME/miniforge3/envs/lerobot/bin/python"
elif [[ -x "$HOME/miniconda3/envs/lerobot/bin/python" ]]; then
  PYTHON_BIN="$HOME/miniconda3/envs/lerobot/bin/python"
else
  PYTHON_BIN="python"
fi
export PYTHONPATH="$LEROBOT_ROOT/src:$LEROBOT_ROOT:${PYTHONPATH:-}"

RUNTIME_CONFIG="${LEROBOT_RUNTIME_CONFIG:-${RUNTIME_CONFIG:-$LEROBOT_ROOT/project/config/runtime.json}}"
if [[ ! -e "$RUNTIME_CONFIG" && -e "$LEROBOT_ROOT/project/config/runtime.json" ]]; then
  RUNTIME_CONFIG="$LEROBOT_ROOT/project/config/runtime.json"
fi
export LEROBOT_RUNTIME_CONFIG="$RUNTIME_CONFIG"

# Policy selection & smart defaults
POLICY_TYPE="${POLICY_TYPE:-act}"
SERVER_ADDRESS="${SERVER_ADDRESS:-100.85.69.64:8080}"
TASK="${TASK:-Pick and place 5 blocks in sequence (red, yellow, wood, green, blue).}"

if [[ "$POLICY_TYPE" == "act" ]]; then
  MODEL_PATH="${MODEL_PATH:-eslab1234/task1_hybrid_5blocks_v3_223ep_merged_act_b16_150k_v2}"
  ACTIONS_PER_CHUNK="${ACTIONS_PER_CHUNK:-30}"
  MAX_RELATIVE_TARGET="${MAX_RELATIVE_TARGET:-1.0}"
elif [[ "$POLICY_TYPE" == "smolvla" ]]; then
  MODEL_PATH="${MODEL_PATH:-eslab1234/smolvla_task1_5blocks_v3_330ep_fullft_b16_150k_v1}"
  ACTIONS_PER_CHUNK="${ACTIONS_PER_CHUNK:-20}"
  MAX_RELATIVE_TARGET="${MAX_RELATIVE_TARGET:-1.25}"
else
  MODEL_PATH="${MODEL_PATH:-}"
  ACTIONS_PER_CHUNK="${ACTIONS_PER_CHUNK:-30}"
  MAX_RELATIVE_TARGET="${MAX_RELATIVE_TARGET:-1.0}"
fi

# Ports & Cameras
ROBOT_PORT="${ROBOT_PORT:-/dev/so101_follower}"
TELEOP_PORT="${TELEOP_PORT:-/dev/so101_leader}"
TOP_CAM="${TOP_CAM:-/dev/cam_top}"
WRIST_CAM="${WRIST_CAM:-/dev/cam_wrist}"

FPS="${FPS:-30}"
WIDTH="${WIDTH:-640}"
HEIGHT="${HEIGHT:-480}"
OBSERVE_DURATION_S="${OBSERVE_DURATION_S:-3.0}"
OBSERVE_SETTLE_S="${OBSERVE_SETTLE_S:-0.5}"

# Chunking & Safety Watchdog
CHUNK_SIZE_THRESHOLD="${CHUNK_SIZE_THRESHOLD:-0.5}"
AGGREGATE_FN_NAME="${AGGREGATE_FN_NAME:-latest_only}"
MAX_TRACKING_ERROR="${MAX_TRACKING_ERROR:-35.0}"
TRACKING_ERROR_GRACE_STEPS="${TRACKING_ERROR_GRACE_STEPS:-10}"
DISABLE_TORQUE_ON_DISCONNECT="${DISABLE_TORQUE_ON_DISCONNECT:-false}"
INFERENCE_SECONDS="${INFERENCE_SECONDS:-0}"
SKIP_CONFIRM="${SKIP_CONFIRM:-false}"

CAMERA_KEY_MODE="${CAMERA_KEY_MODE:-policy}"
if [[ "$CAMERA_KEY_MODE" == "policy" ]]; then
  TOP_KEY="camera1"
  WRIST_KEY="camera2"
else
  TOP_KEY="top"
  WRIST_KEY="wrist"
fi

fail() {
  printf '[ERROR] %s\n' "$*" >&2
  exit 1
}

require_path() {
  local target_path="$1"
  [[ -e "$target_path" ]] || fail "Required path not found: $target_path"
}

printf '\n==============================================================================\n'
printf '🤖 [SO-101 UNIFIED ASYNC INFERENCE CLIENT: %s]\n' "${POLICY_TYPE^^}"
printf '==============================================================================\n'
printf 'Model:          %s\n' "$MODEL_PATH"
printf 'Policy Type:    %s\n' "$POLICY_TYPE"
printf 'Task:           %s\n' "$TASK"
printf 'Server:         %s\n' "$SERVER_ADDRESS"
printf 'Cameras:        top=%s (/dev/cam_top), wrist=%s (/dev/cam_wrist)\n' "$TOP_KEY" "$WRIST_KEY"
printf 'Chunk Settings: actions=%s, threshold=%s, agg=%s\n' "$ACTIONS_PER_CHUNK" "$CHUNK_SIZE_THRESHOLD" "$AGGREGATE_FN_NAME"
printf 'Safety Limiter: max_step=%s deg, tracking_error_max=%s deg (grace=%s steps)\n' \
  "$MAX_RELATIVE_TARGET" "$MAX_TRACKING_ERROR" "$TRACKING_ERROR_GRACE_STEPS"
printf '==============================================================================\n'

require_path "$ROBOT_PORT"
require_path "$TELEOP_PORT"
require_path "$TOP_CAM"
require_path "$WRIST_CAM"
require_path "$RUNTIME_CONFIG"

# GPU Policy server reachability check
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

CAMERAS="{ $TOP_KEY: {type: opencv, index_or_path: '$TOP_CAM', width: $WIDTH, height: $HEIGHT, fps: $FPS, fourcc: 'MJPG'}, $WRIST_KEY: {type: opencv, index_or_path: '$WRIST_CAM', width: $WIDTH, height: $HEIGHT, fps: $FPS, fourcc: 'MJPG'} }"

robot_safety_args=()
if [[ -n "$MAX_RELATIVE_TARGET" ]]; then
  robot_safety_args+=(
    --robot.max_relative_target="$MAX_RELATIVE_TARGET"
    --robot.max_tracking_error="$MAX_TRACKING_ERROR"
    --robot.tracking_error_grace_steps="$TRACKING_ERROR_GRACE_STEPS"
  )
fi

printf '\n[2/2] Starting real-time async inference loop (%s). Keep one hand on Ctrl+C.\n' "$POLICY_TYPE"
exec "$PYTHON_BIN" -m lerobot.async_inference.robot_client \
  --server_address="$SERVER_ADDRESS" \
  --policy_type="$POLICY_TYPE" \
  --pretrained_name_or_path="$MODEL_PATH" \
  --actions_per_chunk="$ACTIONS_PER_CHUNK" \
  --task="$TASK" \
  --policy_device=cuda \
  --client_device=cpu \
  --robot.type=so101_follower \
  --robot.port="$ROBOT_PORT" \
  --robot.id=follower \
  --robot.disable_torque_on_disconnect="$DISABLE_TORQUE_ON_DISCONNECT" \
  "${robot_safety_args[@]}" \
  --robot.cameras="$CAMERAS" \
  --fps="$FPS" \
  --chunk_size_threshold="$CHUNK_SIZE_THRESHOLD" \
  --aggregate_fn_name="$AGGREGATE_FN_NAME"

