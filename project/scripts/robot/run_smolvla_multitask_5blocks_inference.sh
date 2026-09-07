#!/usr/bin/env bash
#
# Dedicated Inference Launcher for Multi-task 5-Blocks SmolVLA:
# Model: eslab1234/smolvla_multitask_5blocks_v1_444ep_fullft_b16_150k
#
# Official Task Instructions:
#   - Task 1 (Placement):
#       "Pick up the 5 blocks in sequence (red, yellow, wood, green, blue), then place each block separately into its designated target position."
#   - Task 2 (Stacking):
#       "Pick up the 5 blocks in sequence (red, yellow, wood, green, blue), then hover over the target area and stack each block on top of the previous block."
#
# Usage:
#   # 1. Run Task 1 (Default max_relative_target = 1.25)
#   bash project/scripts/robot/run_smolvla_multitask_5blocks_inference.sh
#
#   # 2. Adjust max_relative_target (e.g. 1.5 deg/step)
#   bash project/scripts/robot/run_smolvla_multitask_5blocks_inference.sh --max_relative_target=1.5
#
#   # 3. Run Task 2 with custom max_relative_target
#   TASK_MODE=2 bash project/scripts/robot/run_smolvla_multitask_5blocks_inference.sh --max_relative_target=1.5
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

export MAX_RELATIVE_TARGET="$MAX_REL_TARGET"

if [[ -z "${TASK:-}" ]]; then
  if [[ "$TASK_MODE" == "2" || "$TASK_MODE" =~ ^(stack|task2)$ ]]; then
    export TASK="$TASK2_PROMPT"
    printf '👉 [Task Selection] Mode 2 (Stacking)\n'
  else
    export TASK="$TASK1_PROMPT"
    printf '👉 [Task Selection] Mode 1 (Placement)\n'
  fi
else
  printf '👉 [Task Selection] Custom Task Prompt Provided\n'
fi

# Optimized Async Control Parameters for SmolVLA
export ACTIONS_PER_CHUNK="${ACTIONS_PER_CHUNK:-20}"
export CHUNK_SIZE_THRESHOLD="${CHUNK_SIZE_THRESHOLD:-0.5}"
export AGGREGATE_FN_NAME="${AGGREGATE_FN_NAME:-latest_only}"

# Physical Gripper Thickness Tolerance (Prevents False Stall Abort when grasping 2cm blocks)
export MAX_TRACKING_ERROR="${MAX_TRACKING_ERROR:-35.0}"
export TRACKING_ERROR_GRACE_STEPS="${TRACKING_ERROR_GRACE_STEPS:-10}"

# 2-Camera Key Contract (camera1: top, camera2: wrist)
export CAMERA_KEY_MODE="${CAMERA_KEY_MODE:-policy}"

# Server Address (Override with SERVER_ADDRESS="ip:port")
export SERVER_ADDRESS="${SERVER_ADDRESS:-100.85.69.64:8080}"

printf '⚙️  [Config] MAX_RELATIVE_TARGET: %s deg/step\n' "$MAX_RELATIVE_TARGET"

# Execute unified async launcher
exec bash "$LEROBOT_ROOT/project/scripts/robot/run_async_inference.sh" "${PASSTHROUGH_ARGS[@]}"
