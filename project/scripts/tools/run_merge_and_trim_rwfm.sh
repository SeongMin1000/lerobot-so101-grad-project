#!/usr/bin/env bash
# Merge and trim 10 RWFM rollout recording sessions into Task 1 and Task 2 datasets.
# Total: 50 episodes for Task 1, 50 episodes for Task 2.
# Canonical color sequence: red (10) -> yellow (10) -> wood (10) -> green (10) -> blue (10).

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}"

TASK="${1:-both}"
BUFFER_FRAMES="${BUFFER_FRAMES:-2}"
ARM_TOL="${ARM_TOL:-0.35}"
CREATE_MULTITASK="${CREATE_MULTITASK:-true}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"

cd "$LEROBOT_ROOT"

PYTHON_BIN="/home/eslab/miniforge3/envs/lerobot/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi

EXTRA_FLAGS=()
if [[ "$CREATE_MULTITASK" == "true" ]]; then
  EXTRA_FLAGS+=(--create-multitask)
fi
if [[ "$PUSH_TO_HUB" == "true" ]]; then
  EXTRA_FLAGS+=(--push-to-hub)
fi

printf '====================================================================\n'
printf '🚀 [RUN MERGE & TRIM RWFM ROLLOUTS]\n'
printf '   Task:              %s\n' "$TASK"
printf '   Buffer frames:     %s (idle tail cushion)\n' "$BUFFER_FRAMES"
printf '   Arm motion tol:    %s deg/step\n' "$ARM_TOL"
printf '   Create multitask:  %s\n' "$CREATE_MULTITASK"
printf '   Push to Hub:       %s\n' "$PUSH_TO_HUB"
printf '====================================================================\n'

"$PYTHON_BIN" project/scripts/tools/merge_and_trim_rwfm_rollouts.py \
  --task="$TASK" \
  --buffer-frames="$BUFFER_FRAMES" \
  --arm-motion-tol="$ARM_TOL" \
  "${EXTRA_FLAGS[@]}"
