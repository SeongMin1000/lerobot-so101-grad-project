#!/usr/bin/env bash
#
# Dedicated Interactive Inference Launcher for Multi-task 5-Blocks SmolVLA:
# Model: eslab1234/smolvla_multitask_5blocks_v1_444ep_fullft_b16_150k
#
# Key Features:
#   - PURE INFERENCE (NO recording, NO video encoding overhead).
#   - Model is loaded ONCE onto the remote GPU Policy Server.
#   - Interactive controls in terminal:
#       [SPACE] / [R] : Stop inference -> Return arm smoothly to OBSERVE pose (reset trial).
#       [SPACE] / [R] : Start inference fresh from OBSERVE pose.
#       [Q] / [Ctrl+C]: Stop and exit safely.
#
# Official Task Instructions:
#   - Task 1 (Placement):
#       "Pick up the 5 blocks in sequence (red, yellow, wood, green, blue), then place each block separately into its designated target position."
#   - Task 2 (Stacking):
#       "Pick up the 5 blocks in sequence (red, yellow, wood, green, blue), then hover over the target area and stack each block on top of the previous block."
#
# Usage:
#   # 1. Run Task 1 (Placement, default max_relative_target = 1.25)
#   bash project/scripts/robot/run_smolvla_multitask_5blocks_inference.sh
#
#   # 2. Adjust max_relative_target (e.g. 1.5 deg/step)
#   bash project/scripts/robot/run_smolvla_multitask_5blocks_inference.sh --max_relative_target=1.5
#
#   # 3. Run Task 2 (Stacking)
#   TASK_MODE=2 bash project/scripts/robot/run_smolvla_multitask_5blocks_inference.sh
#

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}"

# Model Configuration
export MODEL_PATH="${MODEL_PATH:-eslab1234/smolvla_multitask_5blocks_v1_444ep_fullft_b16_150k}"
export POLICY_TYPE="smolvla"

# Exact Official Prompts
TASK1_PROMPT="Pick up the 5 blocks in sequence (red, yellow, wood, green, blue), then place each block separately into its designated target position."
TASK2_PROMPT="Pick up the 5 blocks in sequence (red, yellow, wood, green, blue), then hover over the target area and stack each block on top of the previous block."

TASK_MODE="${TASK_MODE:-1}"

# Default max_relative_target (1.25 deg/step for SmolVLA)
MAX_REL_TARGET="${MAX_RELATIVE_TARGET:-1.25}"

# Device Ports & Cameras
ROBOT_PORT="${ROBOT_PORT:-/dev/so101_follower}"
TOP_CAM="${TOP_CAM:-/dev/cam_top}"
WRIST_CAM="${WRIST_CAM:-/dev/cam_wrist}"
RUNTIME_CONFIG="${LEROBOT_RUNTIME_CONFIG:-${RUNTIME_CONFIG:-$LEROBOT_ROOT/project/config/runtime.json}}"

# Server Address
SERVER_ADDRESS="${SERVER_ADDRESS:-100.85.69.64:8080}"

# Control Parameters
FPS="${FPS:-30}"
WIDTH="${WIDTH:-640}"
HEIGHT="${HEIGHT:-480}"
ACTIONS_PER_CHUNK="${ACTIONS_PER_CHUNK:-20}"
CHUNK_SIZE_THRESHOLD="${CHUNK_SIZE_THRESHOLD:-0.5}"
AGGREGATE_FN_NAME="${AGGREGATE_FN_NAME:-latest_only}"
MAX_TRACKING_ERROR="${MAX_TRACKING_ERROR:-35.0}"
TRACKING_ERROR_GRACE_STEPS="${TRACKING_ERROR_GRACE_STEPS:-10}"
CAMERA_KEY_MODE="${CAMERA_KEY_MODE:-policy}"
AUTO_START="${AUTO_START:-false}"

# Forwarded arguments array
PASSTHROUGH_ARGS=()

# Parse CLI options
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_path=*|--model-path=*)
      MODEL_PATH="${1#*=}"
      shift
      ;;
    --model_path|--model-path)
      MODEL_PATH="$2"
      shift 2
      ;;
    --max_relative_target=*|--max-relative-target=*)
      MAX_REL_TARGET="${1#*=}"
      shift
      ;;
    --max_relative_target|--max-relative-target|-m)
      MAX_REL_TARGET="$2"
      shift 2
      ;;
    --server_address=*|--server-address=*)
      SERVER_ADDRESS="${1#*=}"
      shift
      ;;
    --server_address|--server-address)
      SERVER_ADDRESS="$2"
      shift 2
      ;;
    --task_mode=*|--task-mode=*)
      TASK_MODE="${1#*=}"
      shift
      ;;
    --task=*)
      TASK="${1#*=}"
      shift
      ;;
    --auto_start|--auto-start)
      AUTO_START="true"
      shift
      ;;
    *)
      PASSTHROUGH_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "$MODEL_PATH" =~ /checkpoints/[0-9]+$ && ! "$MODEL_PATH" =~ /pretrained_model$ ]]; then
  printf 'ℹ️  Detected checkpoint step directory. Appending /pretrained_model: %s/pretrained_model\n' "$MODEL_PATH"
  MODEL_PATH="${MODEL_PATH}/pretrained_model"
fi
export MODEL_PATH

MAX_RELATIVE_TARGET="$MAX_REL_TARGET"

if [[ -z "${TASK:-}" ]]; then
  if [[ "$TASK_MODE" == "2" || "$TASK_MODE" =~ ^(stack|task2)$ ]]; then
    TASK="$TASK2_PROMPT"
    printf '👉 [Task Selection] Mode 2 (Stacking)\n'
  else
    TASK="$TASK1_PROMPT"
    printf '👉 [Task Selection] Mode 1 (Placement)\n'
  fi
else
  printf '👉 [Task Selection] Custom Task Prompt Provided: %s\n' "$TASK"
fi

# Locate Python environment
if [[ -x "$HOME/miniforge3/envs/lerobot/bin/python" ]]; then
  PYTHON_BIN="$HOME/miniforge3/envs/lerobot/bin/python"
elif [[ -x "$HOME/miniconda3/envs/lerobot/bin/python" ]]; then
  PYTHON_BIN="$HOME/miniconda3/envs/lerobot/bin/python"
else
  PYTHON_BIN="python3"
fi
export PYTHONPATH="$LEROBOT_ROOT/src:$LEROBOT_ROOT:${PYTHONPATH:-}"

fail() {
  printf '\e[1;31m[ERROR]\e[0m %s\n' "$*" >&2
  exit 1
}

# Verify physical devices
[[ -c "$ROBOT_PORT" ]] || fail "Follower arm port not found: $ROBOT_PORT"
[[ -e "$TOP_CAM" ]] || fail "Top camera not found: $TOP_CAM"
[[ -e "$WRIST_CAM" ]] || fail "Wrist camera not found: $WRIST_CAM"
[[ -f "$RUNTIME_CONFIG" ]] || fail "Runtime config not found: $RUNTIME_CONFIG"

# Check GPU policy server connectivity
printf '🔌 Checking connection to GPU Policy Server at %s...\n' "$SERVER_ADDRESS"
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
        f"[ERROR] Policy server unreachable at {address}: {exc}\n"
        "Please start the GPU policy server on your GPU machine first."
    ) from exc

print(f"✅ Connected to GPU Policy Server: {address}")
PY

# Camera mapping contract
if [[ "$CAMERA_KEY_MODE" == "policy" ]]; then
  TOP_KEY="camera1"
  WRIST_KEY="camera2"
else
  TOP_KEY="top"
  WRIST_KEY="wrist"
fi

CAMERAS="{ $TOP_KEY: {type: opencv, index_or_path: '$TOP_CAM', width: $WIDTH, height: $HEIGHT, fps: $FPS, fourcc: 'MJPG'}, $WRIST_KEY: {type: opencv, index_or_path: '$WRIST_CAM', width: $WIDTH, height: $HEIGHT, fps: $FPS, fourcc: 'MJPG'} }"

printf '\n==============================================================================\n'
printf '🤖 [SMOLVLA MULTITASK 5-BLOCKS INTERACTIVE INFERENCE]\n'
printf '==============================================================================\n'
printf 'Model:          %s\n' "$MODEL_PATH"
printf 'Policy Server:  %s\n' "$SERVER_ADDRESS"
printf 'Task:           %s\n' "$TASK"
printf 'Follower Port:  %s\n' "$ROBOT_PORT"
printf 'Cameras:        top=%s (%s), wrist=%s (%s)\n' "$TOP_KEY" "$TOP_CAM" "$WRIST_KEY" "$WRIST_CAM"
printf 'Chunk Settings: actions=%s, threshold=%s, agg=%s\n' "$ACTIONS_PER_CHUNK" "$CHUNK_SIZE_THRESHOLD" "$AGGREGATE_FN_NAME"
printf 'Safety Limiter: max_step=%s deg, tracking_error_max=%s deg (grace=%s steps)\n' \
  "$MAX_RELATIVE_TARGET" "$MAX_TRACKING_ERROR" "$TRACKING_ERROR_GRACE_STEPS"
printf 'Recording:      DISABLED (pure inference mode - no video/dataset overhead)\n'
printf '------------------------------------------------------------------------------\n'
printf 'Interactive Controls (Terminal focus required):\n'
printf '  [SPACE] / [R] : Stop inference -> Smooth return to OBSERVE pose (reset blocks)\n'
printf '  [SPACE] / [R] : Start inference fresh from OBSERVE pose\n'
printf '  [Q] / [Ctrl+C]: Stop and exit cleanly\n'
printf '==============================================================================\n\n'

export PYTHONUNBUFFERED=1

exec "$PYTHON_BIN" -u -m lerobot.grad_project.control.smolvla_interactive_inference \
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
  --robot.disable_torque_on_disconnect=false \
  --robot.max_relative_target="$MAX_RELATIVE_TARGET" \
  --robot.max_tracking_error="$MAX_TRACKING_ERROR" \
  --robot.tracking_error_grace_steps="$TRACKING_ERROR_GRACE_STEPS" \
  --robot.cameras="$CAMERAS" \
  --fps="$FPS" \
  --chunk_size_threshold="$CHUNK_SIZE_THRESHOLD" \
  --aggregate_fn_name="$AGGREGATE_FN_NAME" \
  --runtime_config="$RUNTIME_CONFIG" \
  --observe_duration_s=2.5 \
  --auto_start="$AUTO_START" \
  "${PASSTHROUGH_ARGS[@]}"
