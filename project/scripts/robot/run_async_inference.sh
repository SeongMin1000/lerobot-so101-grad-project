#!/usr/bin/env bash
#
# Unified Asynchronous Robot Client Launcher for SO-101 (ACT & SmolVLA):
#   1) Smoothly moves follower & leader arms to the saved "observe" pose (3.0s).
#   2) Connects to GPU Policy Server (e.g. 100.85.69.64:8080).
#   3) Runs in either:
#      - Autonomous Async Inference mode (default)
#      - Human-in-the-Loop (HIL / DAgger) mode with leader-arm correction recording
#
# Usage:
#   # 1. Autonomous inference (default):
#   bash project/scripts/robot/run_async_inference.sh
#   POLICY_TYPE=smolvla bash project/scripts/robot/run_async_inference.sh
#
#   # 2. HIL (DAgger) mode via CLI flags (mirroring standard LeRobot):
#   bash project/scripts/robot/run_async_inference.sh --hil
#   bash project/scripts/robot/run_async_inference.sh --strategy.type=dagger --dataset.repo_id="eslab1234/my_hil_dataset"
#
#   # 3. HIL mode via Environment Variables:
#   HIL=true bash project/scripts/robot/run_async_inference.sh
#   HIL=true POLICY_TYPE=smolvla DATASET_NAME="smolvla_hil_v1" NUM_CORRECTIONS=20 bash project/scripts/robot/run_async_inference.sh
#

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}"

# Preserve user-provided CLI variables so they are not overwritten by active.env
CLI_MODEL_PATH="${MODEL_PATH:-}"
CLI_POLICY_TYPE="${POLICY_TYPE:-}"
CLI_TASK="${TASK:-}"
CLI_TASK_MODE="${TASK_MODE:-1}"
CLI_ACTIONS_PER_CHUNK="${ACTIONS_PER_CHUNK:-}"
CLI_CHUNK_SIZE_THRESHOLD="${CHUNK_SIZE_THRESHOLD:-}"
CLI_AGGREGATE_FN_NAME="${AGGREGATE_FN_NAME:-}"
CLI_MAX_RELATIVE_TARGET="${MAX_RELATIVE_TARGET:-}"
CLI_MAX_TRACKING_ERROR="${MAX_TRACKING_ERROR:-}"
CLI_TRACKING_ERROR_GRACE_STEPS="${TRACKING_ERROR_GRACE_STEPS:-}"
CLI_SERVER_ADDRESS="${SERVER_ADDRESS:-}"
CLI_CAMERA_KEY_MODE="${CAMERA_KEY_MODE:-}"
CLI_HIL="${HIL:-${STRATEGY:-false}}"
CLI_DATASET_REPO_ID="${DATASET_REPO_ID:-}"
CLI_DATASET_NAME="${DATASET_NAME:-}"
CLI_NUM_CORRECTIONS="${NUM_CORRECTIONS:-${NUM_EPISODES:-20}}"
CLI_MAX_CORRECTION_SECONDS="${MAX_CORRECTION_SECONDS:-${EPISODE_TIME_S:-30}}"
CLI_PUSH_TO_HUB="${PUSH_TO_HUB:-false}"
CLI_RESUME="${RESUME:-false}"

CLI_RECORD_MODE="${RECORD_MODE:-full_on_intervention}"

# Parse command-line options
while [[ $# -gt 0 ]]; do
  case "$1" in
    --hil|--strategy.type=dagger|--strategy=dagger|-hil)
      CLI_HIL="true"
      shift
      ;;
    --strategy.type=*|--strategy=*)
      val="${1#*=}"
      if [[ "$val" == "dagger" ]]; then
        CLI_HIL="true"
      fi
      shift
      ;;
    --record_mode=*|--record-mode=*)
      CLI_RECORD_MODE="${1#*=}"
      shift
      ;;
    --dataset.repo_id=*|--dataset_repo_id=*)
      CLI_DATASET_REPO_ID="${1#*=}"
      shift
      ;;
    --dataset.single_task=*|--task=*)
      CLI_TASK="${1#*=}"
      shift
      ;;
    --task_mode=*|--task-mode=*)
      CLI_TASK_MODE="${1#*=}"
      shift
      ;;
    --dataset.num_episodes=*|--num_episodes=*|--num_corrections=*)
      CLI_NUM_CORRECTIONS="${1#*=}"
      shift
      ;;
    --dataset.episode_time_s=*|--max_correction_seconds=*)
      CLI_MAX_CORRECTION_SECONDS="${1#*=}"
      shift
      ;;
    --push_to_hub=*|--dataset.push_to_hub=*)
      CLI_PUSH_TO_HUB="${1#*=}"
      shift
      ;;
    --resume=*|--dataset.resume=*)
      CLI_RESUME="${1#*=}"
      shift
      ;;
    --policy_type=*|--policy.type=*)
      CLI_POLICY_TYPE="${1#*=}"
      shift
      ;;
    --model_path=*|--policy.path=*)
      CLI_MODEL_PATH="${1#*=}"
      shift
      ;;
    --server_address=*)
      CLI_SERVER_ADDRESS="${1#*=}"
      shift
      ;;
    *)
      shift
      ;;
  esac
done

# Source active profile only if explicitly requested
if [[ "${USE_ACTIVE_ENV:-false}" == "true" && -f "$LEROBOT_ROOT/project/config/experiment-profiles/active.env" ]]; then
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
HIL_ENABLED="$CLI_HIL"
[[ "$HIL_ENABLED" =~ ^(true|dagger|1|yes)$ ]] && HIL_ENABLED="true" || HIL_ENABLED="false"

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
if [[ -z "$CLI_POLICY_TYPE" && -n "${MODEL_PATH:-}" ]]; then
  if [[ "${MODEL_PATH:-}" =~ smolvla|smol_vla ]]; then
    POLICY_TYPE="smolvla"
  elif [[ "${MODEL_PATH:-}" =~ act ]]; then
    POLICY_TYPE="act"
  fi
fi
POLICY_TYPE="${POLICY_TYPE:-act}"
SERVER_ADDRESS="${SERVER_ADDRESS:-100.85.69.64:8080}"

TASK1_PROMPT="Pick up the 5 blocks in sequence (red, yellow, wood, green, blue), then place each at the target area."
TASK2_PROMPT="Pick up the 5 blocks in sequence (red, yellow, wood, green, blue), then hover over and stack each at the target area."

TASK_MODE="${CLI_TASK_MODE:-1}"
if [[ -n "$CLI_TASK" ]]; then
  TASK="$CLI_TASK"
elif [[ "$TASK_MODE" =~ ^(2|task2|stack)$ ]]; then
  TASK="$TASK2_PROMPT"
else
  TASK="$TASK1_PROMPT"
fi

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
export CAMERA_MAX_AGE_MS="${CAMERA_MAX_AGE_MS:-1500}"

FPS="${FPS:-30}"
WIDTH="${WIDTH:-640}"
HEIGHT="${HEIGHT:-480}"
OBSERVE_DURATION_S="${OBSERVE_DURATION_S:-3.0}"
OBSERVE_SETTLE_S="${OBSERVE_SETTLE_S:-0.5}"

# Chunking & Safety Watchdog
CHUNK_SIZE_THRESHOLD="${CHUNK_SIZE_THRESHOLD:-0.5}"
AGGREGATE_FN_NAME="${AGGREGATE_FN_NAME:-latest_only}"
MAX_TRACKING_ERROR="${MAX_TRACKING_ERROR:-45.0}"
TRACKING_ERROR_GRACE_STEPS="${TRACKING_ERROR_GRACE_STEPS:-15}"
DISABLE_TORQUE_ON_DISCONNECT="${DISABLE_TORQUE_ON_DISCONNECT:-false}"
INFERENCE_SECONDS="${INFERENCE_SECONDS:-0}"
SKIP_CONFIRM="${SKIP_CONFIRM:-false}"

# HIL Specific Configurations
HF_USER="${HF_USER:-eslab1234}"
if [[ -z "$CLI_DATASET_REPO_ID" ]]; then
  if [[ -n "$CLI_DATASET_NAME" ]]; then
    DATASET_REPO_ID="${HF_USER}/${CLI_DATASET_NAME}"
  else
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    DATASET_REPO_ID="${HF_USER}/${POLICY_TYPE}_hil_corrections_${TIMESTAMP}"
  fi
else
  DATASET_REPO_ID="$CLI_DATASET_REPO_ID"
fi
NUM_CORRECTIONS="$CLI_NUM_CORRECTIONS"
MAX_CORRECTION_SECONDS="$CLI_MAX_CORRECTION_SECONDS"
PUSH_TO_HUB="$CLI_PUSH_TO_HUB"
RESUME="$CLI_RESUME"
STREAMING_ENCODING="${STREAMING_ENCODING:-true}"
ENCODER_THREADS="${ENCODER_THREADS:-2}"
PLAY_SOUNDS="${PLAY_SOUNDS:-false}"
SERVER_RPC_TIMEOUT_S="${SERVER_RPC_TIMEOUT_S:-3.0}"
LEADER_HANDOVER_DURATION_S="${LEADER_HANDOVER_DURATION_S:-1.2}"
RECORD_MODE="${CLI_RECORD_MODE:-full_on_intervention}"

DEBUG_OBSERVATION_DIR="${DEBUG_OBSERVATION_DIR:-$LEROBOT_ROOT/var/debug/hil_client_camera_inputs}"
DEBUG_OBSERVATION_LIMIT="${DEBUG_OBSERVATION_LIMIT:-1}"
DEBUG_MOTOR_TRACE_DIR="${DEBUG_MOTOR_TRACE_DIR:-$LEROBOT_ROOT/var/debug/hil_motor_traces}"
DEBUG_MOTOR_TRACE_LIMIT="${DEBUG_MOTOR_TRACE_LIMIT:-300}"

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
if [[ "$HIL_ENABLED" == "true" ]]; then
  printf '🤖 [SO-101 UNIFIED ASYNC HIL RECORDER (DAGGER): %s]\n' "${POLICY_TYPE^^}"
else
  printf '🤖 [SO-101 UNIFIED ASYNC INFERENCE CLIENT: %s]\n' "${POLICY_TYPE^^}"
fi
printf '==============================================================================\n'
printf 'Mode:           %s\n' "$([[ "$HIL_ENABLED" == "true" ]] && printf 'Human-in-the-Loop (DAgger Data Collection)' || printf 'Autonomous Inference')"
printf 'Model:          %s\n' "$MODEL_PATH"
printf 'Policy Type:    %s\n' "$POLICY_TYPE"
printf 'Task Mode:      %s (%s)\n' "$TASK_MODE" "$([[ "$TASK_MODE" =~ ^(2|task2|stack)$ ]] && printf 'Task 2: Stacking' || printf 'Task 1: Placement')"
printf 'Task:           %s\n' "$TASK"
printf 'Server:         %s\n' "$SERVER_ADDRESS"
printf 'Cameras:        top=%s (/dev/cam_top), wrist=%s (/dev/cam_wrist)\n' "$TOP_KEY" "$WRIST_KEY"
printf 'Chunk Settings: actions=%s, threshold=%s, agg=%s\n' "$ACTIONS_PER_CHUNK" "$CHUNK_SIZE_THRESHOLD" "$AGGREGATE_FN_NAME"
printf 'Safety Limiter: max_step=%s deg, tracking_error_max=%s deg (grace=%s steps)\n' \
  "$MAX_RELATIVE_TARGET" "$MAX_TRACKING_ERROR" "$TRACKING_ERROR_GRACE_STEPS"

if [[ "$HIL_ENABLED" == "true" ]]; then
  printf '%s\n' '------------------------------------------------------------------------------'
  printf 'Dataset Repo:   %s%s\n' "$DATASET_REPO_ID" "$([[ "$RESUME" == "true" ]] && printf ' (resume)' || true)"
  printf 'Target Episodes:%s (max %ss per correction)\n' "$NUM_CORRECTIONS" "$MAX_CORRECTION_SECONDS"
  printf 'Push to Hub:    %s\n' "$PUSH_TO_HUB"
  printf 'HIL Controls:   [Space] Pause / Resume policy\n'
  printf '                [Enter / C] Start human correction\n'
  printf '                [Right Arrow] Save correction episode\n'
  printf '                [Left Arrow] Discard correction\n'
  printf '                [Q / Esc] Stop session and save dataset\n'
fi
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
  PROMPT_MSG="Workspace clear, emergency stop ready? Type START: "
  if [[ "$HIL_ENABLED" == "true" ]]; then
    PROMPT_MSG="Robot area clear, leader arm free, emergency stop ready? Type HIL: "
  fi
  read -r -p "$PROMPT_MSG" answer
  if [[ "$HIL_ENABLED" == "true" ]]; then
    [[ "$answer" == "HIL" || "$answer" == "START" ]] || fail "Cancelled by user"
  else
    [[ "$answer" == "START" ]] || fail "Cancelled by user"
  fi
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

if [[ "$HIL_ENABLED" == "true" ]]; then
  printf '\n[2/2] Starting real-time async HIL DAgger loop (%s). Use Space to pause & intervene.\n' "$POLICY_TYPE"
  mkdir -p logs "$DEBUG_OBSERVATION_DIR" "$DEBUG_MOTOR_TRACE_DIR"
  exec "$PYTHON_BIN" -m lerobot.grad_project.recording.smolvla_hil_record \
    --robot.type=so101_follower \
    --robot.port="$ROBOT_PORT" \
    --robot.id=follower \
    --robot.disable_torque_on_disconnect="$DISABLE_TORQUE_ON_DISCONNECT" \
    "${robot_safety_args[@]}" \
    --robot.cameras="$CAMERAS" \
    --teleop.type=so101_leader \
    --teleop.port="$TELEOP_PORT" \
    --teleop.id=leader \
    --server_address="$SERVER_ADDRESS" \
    --server_rpc_timeout_s="$SERVER_RPC_TIMEOUT_S" \
    --policy_type="$POLICY_TYPE" \
    --pretrained_name_or_path="$MODEL_PATH" \
    --policy_device=cuda \
    --client_device=cpu \
    --actions_per_chunk="$ACTIONS_PER_CHUNK" \
    --chunk_size_threshold="$CHUNK_SIZE_THRESHOLD" \
    --aggregate_fn_name="$AGGREGATE_FN_NAME" \
    --runtime_config="$RUNTIME_CONFIG" \
    --observe_pose_name=observe \
    --observe_duration_s="$OBSERVE_DURATION_S" \
    --observe_fps="$FPS" \
    --observe_settle_s="$OBSERVE_SETTLE_S" \
    --leader_handover_duration_s="$LEADER_HANDOVER_DURATION_S" \
    --leader_handover_fps="$FPS" \
    --dataset.repo_id="$DATASET_REPO_ID" \
    --record_mode="$RECORD_MODE" \
    --dataset.single_task="$TASK" \
    --dataset.num_episodes="$NUM_CORRECTIONS" \
    --dataset.episode_time_s="$MAX_CORRECTION_SECONDS" \
    --dataset.reset_time_s=0 \
    --dataset.fps="$FPS" \
    --dataset.video=true \
    --dataset.streaming_encoding="$STREAMING_ENCODING" \
    --dataset.encoder_threads="$ENCODER_THREADS" \
    --dataset.push_to_hub="$PUSH_TO_HUB" \
    --resume="$RESUME" \
    --display_data=false \
    --play_sounds="$PLAY_SOUNDS" \
    --debug_observation_dir="$DEBUG_OBSERVATION_DIR" \
    --debug_observation_limit="$DEBUG_OBSERVATION_LIMIT" \
    --debug_motor_trace_dir="$DEBUG_MOTOR_TRACE_DIR" \
    --debug_motor_trace_limit="$DEBUG_MOTOR_TRACE_LIMIT"
else
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
fi


