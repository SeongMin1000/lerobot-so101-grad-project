#!/usr/bin/env bash
# Robot PC: remote SmolVLA + SO-101 leader Human-in-the-Loop recorder.
#
# The policy runs on the existing GPU policy_server.  Only human recovery and
# correction windows are saved; autonomous policy mistakes are never written to
# the training dataset.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}"
RUNTIME_CONFIG="${LEROBOT_RUNTIME_CONFIG:-${RUNTIME_CONFIG:-$LEROBOT_ROOT/project/config/runtime.json}}"
export LEROBOT_RUNTIME_CONFIG="$RUNTIME_CONFIG"

HF_USER="${HF_USER:-eslab1234}"
ROBOT_PORT="${ROBOT_PORT:-/dev/so101_follower}"
TELEOP_PORT="${TELEOP_PORT:-/dev/so101_leader}"
TOP_CAM="${TOP_CAM:-/dev/cam_top}"
WRIST_CAM="${WRIST_CAM:-/dev/cam_wrist}"
BELLY_CAM="${BELLY_CAM:-/dev/cam_belly}"

SERVER_ADDRESS="${SERVER_ADDRESS:-100.85.69.64:8080}"
SERVER_RPC_TIMEOUT_S="${SERVER_RPC_TIMEOUT_S:-3.0}"
MODEL_PATH="${MODEL_PATH:-eslab1234/smolvla_red_full_138ep_recovery_lora_r64_lr1e3_20k_v1}"
TASK="${TASK:-Pick up the red block and place it in the red target slot.}"

DATASET_NAME="${DATASET_NAME:-red_smolvla_hil_corrections_r1}"
DATASET_REPO_ID="${DATASET_REPO_ID:-${HF_USER}/${DATASET_NAME}}"
NUM_CORRECTIONS="${NUM_CORRECTIONS:-20}"
MAX_CORRECTION_SECONDS="${MAX_CORRECTION_SECONDS:-30}"
RESUME="${RESUME:-false}"
PUSH_TO_HUB="${PUSH_TO_HUB:-true}"

FPS="${FPS:-30}"
WIDTH="${WIDTH:-640}"
HEIGHT="${HEIGHT:-480}"
ACTIONS_PER_CHUNK="${ACTIONS_PER_CHUNK:-30}"
CHUNK_SIZE_THRESHOLD="${CHUNK_SIZE_THRESHOLD:-0.6}"
AGGREGATE_FN_NAME="${AGGREGATE_FN_NAME:-latest_only}"

MAX_RELATIVE_TARGET="${MAX_RELATIVE_TARGET:-3.0}"
MAX_TRACKING_ERROR="${MAX_TRACKING_ERROR:-20.0}"
TRACKING_ERROR_GRACE_STEPS="${TRACKING_ERROR_GRACE_STEPS:-5}"

OBSERVE_DURATION_S="${OBSERVE_DURATION_S:-3.0}"
OBSERVE_SETTLE_S="${OBSERVE_SETTLE_S:-0.5}"
LEADER_HANDOVER_DURATION_S="${LEADER_HANDOVER_DURATION_S:-1.2}"
STREAMING_ENCODING="${STREAMING_ENCODING:-true}"
ENCODER_THREADS="${ENCODER_THREADS:-2}"
PLAY_SOUNDS="${PLAY_SOUNDS:-false}"

DEBUG_OBSERVATION_DIR="${DEBUG_OBSERVATION_DIR:-$LEROBOT_ROOT/var/debug/hil_client_camera_inputs}"
DEBUG_OBSERVATION_LIMIT="${DEBUG_OBSERVATION_LIMIT:-1}"
DEBUG_MOTOR_TRACE_DIR="${DEBUG_MOTOR_TRACE_DIR:-$LEROBOT_ROOT/var/debug/hil_motor_traces}"
DEBUG_MOTOR_TRACE_LIMIT="${DEBUG_MOTOR_TRACE_LIMIT:-600}"

fail() {
  printf '[ERROR] %s\n' "$*" >&2
  exit 1
}

require_path() {
  [[ -e "$1" ]] || fail "Required path not found: $1"
}

case "$RESUME" in true|false) ;; *) fail "RESUME must be true or false" ;; esac
case "$PUSH_TO_HUB" in true|false) ;; *) fail "PUSH_TO_HUB must be true or false" ;; esac
case "$STREAMING_ENCODING" in true|false) ;; *) fail "STREAMING_ENCODING must be true or false" ;; esac
case "$PLAY_SOUNDS" in true|false) ;; *) fail "PLAY_SOUNDS must be true or false" ;; esac

[[ "${CONDA_DEFAULT_ENV:-}" == "lerobot" ]] || fail "Activate first: conda activate lerobot"
[[ -t 0 ]] || fail "Run this robot-side HIL script in an interactive foreground terminal"
command -v python >/dev/null 2>&1 || fail "python is not available"

require_path "$LEROBOT_ROOT"
require_path "$RUNTIME_CONFIG"
require_path "$ROBOT_PORT"
require_path "$TELEOP_PORT"
require_path "$TOP_CAM"
require_path "$WRIST_CAM"
require_path "$BELLY_CAM"
require_path "$LEROBOT_ROOT/src/lerobot/grad_project/recording/smolvla_hil_record.py"

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
        "Start project/scripts/gpu/run_smolvla_red_policy_server.sh on the GPU PC first."
    ) from exc

print(f"[OK] policy server reachable: {address}")
PY

cd "$LEROBOT_ROOT"
mkdir -p logs "$DEBUG_OBSERVATION_DIR" "$DEBUG_MOTOR_TRACE_DIR"

# Keep top/wrist/belly here.  Those are the original dataset feature names;
# the checkpoint's saved rename processor maps them to camera1/camera2/camera3
# only on the GPU server.
CAMERAS="{ top: {type: opencv, index_or_path: '$TOP_CAM', width: $WIDTH, height: $HEIGHT, fps: $FPS, fourcc: 'MJPG'}, wrist: {type: opencv, index_or_path: '$WRIST_CAM', width: $WIDTH, height: $HEIGHT, fps: $FPS, fourcc: 'MJPG'}, belly: {type: opencv, index_or_path: '$BELLY_CAM', width: $WIDTH, height: $HEIGHT, fps: $FPS, fourcc: 'MJPG', rotation: 180} }"

printf '\n[HIL CONFIG]\n'
printf '  model:       %s\n' "$MODEL_PATH"
printf '  task:        %s\n' "$TASK"
printf '  dataset:     %s%s\n' "$DATASET_REPO_ID" "$([[ "$RESUME" == true ]] && printf ' (resume)' || true)"
printf '  corrections: %s new episode(s)\n' "$NUM_CORRECTIONS"
printf '  chunk:       %s / threshold=%s / %s\n' \
  "$ACTIONS_PER_CHUNK" "$CHUNK_SIZE_THRESHOLD" "$AGGREGATE_FN_NAME"
printf '  safety:      step=%s, tracking=%s x %s\n\n' \
  "$MAX_RELATIVE_TARGET" "$MAX_TRACKING_ERROR" "$TRACKING_ERROR_GRACE_STEPS"

read -r -p "Robot area clear, leader arm free, emergency stop ready? Type HIL: " answer
[[ "$answer" == "HIL" ]] || fail "Cancelled by user"

exec python -m lerobot.grad_project.recording.smolvla_hil_record \
  --robot.type=so101_follower \
  --robot.port="$ROBOT_PORT" \
  --robot.id=follower \
  --robot.disable_torque_on_disconnect=false \
  --robot.max_relative_target="$MAX_RELATIVE_TARGET" \
  --robot.max_tracking_error="$MAX_TRACKING_ERROR" \
  --robot.tracking_error_grace_steps="$TRACKING_ERROR_GRACE_STEPS" \
  --robot.cameras="$CAMERAS" \
  --teleop.type=so101_leader \
  --teleop.port="$TELEOP_PORT" \
  --teleop.id=leader \
  --server_address="$SERVER_ADDRESS" \
  --server_rpc_timeout_s="$SERVER_RPC_TIMEOUT_S" \
  --policy_type=smolvla \
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
  --debug_motor_trace_limit="$DEBUG_MOTOR_TRACE_LIMIT" \
  2>&1 | tee "logs/hil_${DATASET_NAME}_latest.log"
