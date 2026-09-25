#!/usr/bin/env bash
# GPU PC: start the async SmolVLA policy server.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}"
POLICY_SERVER_HOST="${POLICY_SERVER_HOST:-0.0.0.0}"
POLICY_SERVER_PORT="${POLICY_SERVER_PORT:-8080}"
FPS="${FPS:-30}"
INFERENCE_LATENCY="${INFERENCE_LATENCY:-0.033}"
OBS_QUEUE_TIMEOUT="${OBS_QUEUE_TIMEOUT:-1.0}"
DEBUG_OBSERVATION_DIR="${DEBUG_OBSERVATION_DIR:-$LEROBOT_ROOT/var/debug/server_camera_inputs}"
DEBUG_OBSERVATION_LIMIT="${DEBUG_OBSERVATION_LIMIT:-1}"

fail() {
  printf '[ERROR] %s\n' "$*" >&2
  exit 1
}

[[ "${CONDA_DEFAULT_ENV:-}" == "lerobot" ]] || fail \
  "Activate the environment first: conda activate lerobot"
[[ -d "$LEROBOT_ROOT" ]] || fail "LeRobot directory not found: $LEROBOT_ROOT"
command -v python >/dev/null 2>&1 || fail "python is not available"

cd "$LEROBOT_ROOT"

python -c 'import lerobot.async_inference.policy_server as p; print("[OK]", p.__file__)'

printf 'Starting policy server on %s:%s (fps=%s)\n' \
  "$POLICY_SERVER_HOST" "$POLICY_SERVER_PORT" "$FPS"
printf 'The robot client will provide the LoRA model path during connection.\n'
printf 'Matched server camera stages will be saved under %s\n' "$DEBUG_OBSERVATION_DIR"

exec python -m lerobot.async_inference.policy_server \
  --host="$POLICY_SERVER_HOST" \
  --port="$POLICY_SERVER_PORT" \
  --fps="$FPS" \
  --inference_latency="$INFERENCE_LATENCY" \
  --obs_queue_timeout="$OBS_QUEUE_TIMEOUT" \
  --debug_observation_dir="$DEBUG_OBSERVATION_DIR" \
  --debug_observation_limit="$DEBUG_OBSERVATION_LIMIT"
