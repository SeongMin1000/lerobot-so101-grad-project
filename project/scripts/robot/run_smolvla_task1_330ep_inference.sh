#!/usr/bin/env bash
#
# Dedicated Inference Launcher for Task 1 (330ep Full Fine-Tuning):
# Model: eslab1234/smolvla_task1_5blocks_v3_330ep_fullft_b16_150k_v1
#
# Task Instruction:
#   "Pick up the 5 blocks in sequence (red, yellow, wood, green, blue), then place each at the target area."
#
# Usage:
#   # 1. Default execution (max_relative_target = 1.25)
#   bash project/scripts/robot/run_smolvla_task1_330ep_inference.sh
#
#   # 2. Set max_relative_target via CLI flag
#   bash project/scripts/robot/run_smolvla_task1_330ep_inference.sh --max_relative_target=1.5
#   bash project/scripts/robot/run_smolvla_task1_330ep_inference.sh -m 2.0
#
#   # 3. Set via Environment Variable
#   MAX_RELATIVE_TARGET=1.5 bash project/scripts/robot/run_smolvla_task1_330ep_inference.sh
#

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}"

# Model Configuration
export MODEL_PATH="eslab1234/smolvla_task1_5blocks_v3_330ep_fullft_b16_150k_v1"
export POLICY_TYPE="smolvla"

# Exact Official Task 1 Prompt
export TASK="${TASK:-Pick up the 5 blocks in sequence (red, yellow, wood, green, blue), then place each at the target area.}"

# Default max_relative_target (1.25 deg/step for SmolVLA)
MAX_REL_TARGET="${MAX_RELATIVE_TARGET:-1.25}"

# Forwarded arguments array
PASSTHROUGH_ARGS=()

# Parse CLI options for max_relative_target
while [[ $# -gt 0 ]]; do
  case "$1" in
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

export MAX_RELATIVE_TARGET="$MAX_REL_TARGET"

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
