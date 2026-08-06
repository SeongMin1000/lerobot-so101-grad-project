#!/usr/bin/env bash

# One-command, bounded safety run for the latest red-block SmolVLA checkpoint.
# This wrapper intentionally uses fixed values so stale exported shell variables
# cannot silently turn the safety run into an unlimited or faster run.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}"
cd "$LEROBOT_ROOT"
mkdir -p logs

export LEROBOT_ROOT
export RUNTIME_CONFIG="$LEROBOT_ROOT/project/config/runtime.json"
export ROBOT_PORT="/dev/so101_follower"
export TELEOP_PORT="/dev/so101_leader"
export TOP_CAM="/dev/cam_top"
export WRIST_CAM="/dev/cam_wrist"
export BELLY_CAM="/dev/cam_belly"
export CAMERA_KEY_MODE="policy"
export SERVER_ADDRESS="100.85.69.64:8080"
export MODEL_PATH="eslab1234/smolvla_red_full_126ep_lora_r64_lr1e3_40k_v2"
export TASK="Pick up the red block and place it in the red target slot."
export ACTIONS_PER_CHUNK=10
export CHUNK_SIZE_THRESHOLD=0.6
export AGGREGATE_FN_NAME="latest_only"
export MAX_RELATIVE_TARGET=3
export MAX_TRACKING_ERROR=20
export TRACKING_ERROR_GRACE_STEPS=5
export FPS=30
export WIDTH=640
export HEIGHT=480
export OBSERVE_DURATION_S=3.0
export OBSERVE_SETTLE_S=0.5
export INFERENCE_SECONDS=0
export DISABLE_TORQUE_ON_DISCONNECT=false
export SKIP_CONFIRM=false
export DEBUG_OBSERVATION_DIR="$LEROBOT_ROOT/var/debug/client_camera_inputs"
export DEBUG_OBSERVATION_LIMIT=1
export DEBUG_MOTOR_TRACE_DIR="$LEROBOT_ROOT/var/debug/motor_traces"
export DEBUG_MOTOR_TRACE_LIMIT=300

bash "$SCRIPT_DIR/run_smolvla_red_observe_inference.sh" \
  2>&1 | tee logs/latest_126ep_safe_run.log
