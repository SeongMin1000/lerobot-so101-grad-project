#!/usr/bin/env bash
# Robot PC: SmolVLA RWFM Autonomous Rollout Recorder (Approach & Grasp Evaluation)
#
# Records autonomous SmolVLA approach and grasp rollouts with interactive
# post-movement semantic segment labeling (NORMAL / SELF_CORRECTION / FAILURE).
# Saves authoritative metadata to meta/rwfm_rollout_annotations.json.

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

SERVER_ADDRESS="${SERVER_ADDRESS:-100.85.69.64:8080}"
SERVER_RPC_TIMEOUT_S="${SERVER_RPC_TIMEOUT_S:-3.0}"

# Model Configuration: Remote GPU server path or Hugging Face Hub repo ID
MODEL_PATH="${MODEL_PATH:-outputs/train/smolvla_multitask_5blocks_v3_575ep_fullft_b16_300k/checkpoints/285000/pretrained_model}"

TASK1_PROMPT="Pick up the 5 blocks in sequence (red, yellow, wood, green, blue), then place each block separately into its designated target position."
TASK2_PROMPT="Pick up the 5 blocks in sequence (red, yellow, wood, green, blue), then hover over the target area and stack each block on top of the previous block."

TASK_MODE="${TASK_MODE:-1}"
if [[ "$TASK_MODE" =~ ^(2|task2|stack)$ ]]; then
  DEFAULT_TASK="$TASK2_PROMPT"
  DEFAULT_DATASET_NAME="smolvla_task2_rwfm_rollout_v1"
  DEFAULT_ACTIONS_PER_CHUNK=50
  DEFAULT_CHUNK_SIZE_THRESHOLD="0.75"
  DEFAULT_AGGREGATE_FN_NAME="weighted_average"
else
  DEFAULT_TASK="$TASK1_PROMPT"
  DEFAULT_DATASET_NAME="smolvla_task1_rwfm_rollout_v1"
  DEFAULT_ACTIONS_PER_CHUNK=30
  DEFAULT_CHUNK_SIZE_THRESHOLD="0.6"
  DEFAULT_AGGREGATE_FN_NAME="latest_only"
fi

TASK="${TASK:-$DEFAULT_TASK}"
DATASET_NAME="${DATASET_NAME:-$DEFAULT_DATASET_NAME}"
DATASET_REPO_ID="${DATASET_REPO_ID:-${HF_USER}/${DATASET_NAME}}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-${NUM_CORRECTIONS:-50}}"
MAX_ROLLOUT_SECONDS="${MAX_ROLLOUT_SECONDS:-${MAX_CORRECTION_SECONDS:-60}}"
RESUME="${RESUME:-false}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"
RECORD_MODE="rwfm_rollout"
FAILURE_TAIL_FRAMES="${FAILURE_TAIL_FRAMES:-0}"

FPS="${FPS:-30}"
WIDTH="${WIDTH:-640}"
HEIGHT="${HEIGHT:-480}"

ACTIONS_PER_CHUNK="${ACTIONS_PER_CHUNK:-$DEFAULT_ACTIONS_PER_CHUNK}"
CHUNK_SIZE_THRESHOLD="${CHUNK_SIZE_THRESHOLD:-$DEFAULT_CHUNK_SIZE_THRESHOLD}"
AGGREGATE_FN_NAME="${AGGREGATE_FN_NAME:-$DEFAULT_AGGREGATE_FN_NAME}"

MAX_RELATIVE_TARGET="${MAX_RELATIVE_TARGET:-1.75}"
MAX_TRACKING_ERROR="${MAX_TRACKING_ERROR:-45.0}"
TRACKING_ERROR_GRACE_STEPS="${TRACKING_ERROR_GRACE_STEPS:-15}"
export CAMERA_MAX_AGE_MS="${CAMERA_MAX_AGE_MS:-3000}"

PAN_BIAS_DIRECTION="${PAN_BIAS_DIRECTION:-left}"
PAN_BIAS_NEAR_DEG="${PAN_BIAS_NEAR_DEG:-1.0}"
PAN_BIAS_FAR_DEG="${PAN_BIAS_FAR_DEG:-3.5}"

OBSERVE_DURATION_S=2.0
OBSERVE_SETTLE_S=0.1
STREAMING_ENCODING=true
ENCODER_THREADS=4
PLAY_SOUNDS=false

# Parse optional command line flags
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
    --task=*|--task-prompt=*)
      TASK="${1#*=}"
      shift
      ;;
    --task_mode=*|--task-mode=*)
      TASK_MODE="${1#*=}"
      shift
      ;;
    --dataset_name=*|--dataset-name=*)
      DATASET_NAME="${1#*=}"
      DATASET_REPO_ID="${HF_USER}/${DATASET_NAME}"
      shift
      ;;
    --num_rollouts=*|--num-rollouts=*)
      NUM_ROLLOUTS="${1#*=}"
      shift
      ;;
    --failure_tail_frames=*|--failure-tail-frames=*)
      FAILURE_TAIL_FRAMES="${1#*=}"
      shift
      ;;
    --server_address=*|--server-address=*)
      SERVER_ADDRESS="${1#*=}"
      shift
      ;;
    --resume)
      RESUME=true
      shift
      ;;
    --push_to_hub|--push-to-hub)
      PUSH_TO_HUB=true
      shift
      ;;
    -h|--help)
      cat <<EOF
Usage: $0 [OPTIONS]

Options:
  --model_path=PATH           Path to deployed SmolVLA policy checkpoint
  --task_mode=1|2             Select Task 1 (separate) or Task 2 (stack)
  --task=PROMPT               Override task instruction prompt
  --dataset_name=NAME         Local/Hub dataset name
  --num_rollouts=N            Target number of rollouts to record (default: 50)
  --failure_tail_frames=N     Split last N frames as FAILURE (default: 0 = whole segment)
  --server_address=IP:PORT    GPU Policy Server address (default: 100.85.69.64:8080)
  --resume                    Resume recording into existing dataset
  --push_to_hub               Upload dataset to HF Hub after completion
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

fail() {
  printf '\e[1;31m[ERROR]\e[0m %s\n' "$*" >&2
  exit 1
}

# Preflight checks
[[ -c "$ROBOT_PORT" ]] || fail "Follower port $ROBOT_PORT not found"
[[ -c "$TELEOP_PORT" ]] || fail "Leader port $TELEOP_PORT not found"
[[ -e "$TOP_CAM" ]] || fail "Top camera $TOP_CAM not found"
[[ -e "$WRIST_CAM" ]] || fail "Wrist camera $WRIST_CAM not found"

mkdir -p "$LEROBOT_ROOT/logs"
cd "$LEROBOT_ROOT"

CAMERAS=$(cat <<EOF
{
  "top": {
    "type": "opencv",
    "index_or_path": "$TOP_CAM",
    "width": $WIDTH,
    "height": $HEIGHT,
    "fps": $FPS
  },
  "wrist": {
    "type": "opencv",
    "index_or_path": "$WRIST_CAM",
    "width": $WIDTH,
    "height": $HEIGHT,
    "fps": $FPS
  }
}
EOF
)

printf '\n\e[1;32m=== Starting SmolVLA RWFM Autonomous Rollout Recording ===\e[0m\n'
printf '  record mode: %s\n' "$RECORD_MODE"
printf '  model:       %s\n' "$MODEL_PATH"
printf '  server:      %s\n' "$SERVER_ADDRESS"
printf '  dataset:     %s%s\n' "$DATASET_REPO_ID" "$([[ "$RESUME" == true ]] && printf ' (resume)' || true)"
printf '  rollouts:    %s episode(s)\n' "$NUM_ROLLOUTS"
printf '  fail tail:   %s frame(s)\n' "$FAILURE_TAIL_FRAMES"
printf '  task mode:   %s\n\n' "$TASK_MODE"

export PYTHONUNBUFFERED=1

exec python -u -m lerobot.grad_project.recording.smolvla_hil_record \
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
  --record_mode="$RECORD_MODE" \
  --failure_tail_frames="$FAILURE_TAIL_FRAMES" \
  --runtime_config="$RUNTIME_CONFIG" \
  --observe_pose_name=observe \
  --pan_bias_direction="$PAN_BIAS_DIRECTION" \
  --pan_bias_near_deg="$PAN_BIAS_NEAR_DEG" \
  --pan_bias_far_deg="$PAN_BIAS_FAR_DEG" \
  --observe_duration_s="$OBSERVE_DURATION_S" \
  --macro_return_duration_s="$OBSERVE_DURATION_S" \
  --observe_fps="$FPS" \
  --observe_settle_s="$OBSERVE_SETTLE_S" \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --dataset.single_task="$TASK" \
  --dataset.num_episodes="$NUM_ROLLOUTS" \
  --dataset.episode_time_s="$MAX_ROLLOUT_SECONDS" \
  --dataset.reset_time_s=0 \
  --dataset.fps="$FPS" \
  --dataset.video=true \
  --dataset.streaming_encoding="$STREAMING_ENCODING" \
  --dataset.encoder_threads="$ENCODER_THREADS" \
  --dataset.push_to_hub="$PUSH_TO_HUB" \
  --resume="$RESUME" \
  --display_data=false \
  --play_sounds="$PLAY_SOUNDS" \
  2>&1 | tee "logs/rwfm_rollout_${DATASET_NAME}_latest.log"
