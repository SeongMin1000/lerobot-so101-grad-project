#!/usr/bin/env bash
# Run the SmolVLA model immediately preceding the 126-episode model.
#
# This wrapper intentionally keeps the last comparison settings unchanged and
# only switches MODEL_PATH. The GPU policy server receives the model path from
# the robot client, so restart the server before starting this script.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_PATH="eslab1234/smolvla_red_H_full_30ep_lora_r64_10k_v1"
export TASK="${TASK:-Pick up the red block and place it in the red target slot.}"

# Keep these equal to the most recent 126-episode test so the model comparison
# is not confounded by a different client-side execution configuration.
export ACTIONS_PER_CHUNK="${ACTIONS_PER_CHUNK:-10}"
export CHUNK_SIZE_THRESHOLD="${CHUNK_SIZE_THRESHOLD:-0.5}"
export AGGREGATE_FN_NAME="${AGGREGATE_FN_NAME:-latest_only}"
export MAX_RELATIVE_TARGET="${MAX_RELATIVE_TARGET:-2.0}"
export MAX_TRACKING_ERROR="${MAX_TRACKING_ERROR:-20}"
export TRACKING_ERROR_GRACE_STEPS="${TRACKING_ERROR_GRACE_STEPS:-5}"
export INFERENCE_SECONDS="${INFERENCE_SECONDS:-0}"
export FPS="${FPS:-30}"

printf '[MODEL COMPARISON] Previous model: %s\n' "$MODEL_PATH"
exec bash "$SCRIPT_DIR/run_smolvla_red_observe_inference.sh"
