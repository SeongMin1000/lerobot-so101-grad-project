#!/usr/bin/env bash
# Read-only post-incident check for the three SmolVLA runtime regressions.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}"

if [[ "${CONDA_DEFAULT_ENV:-}" != "lerobot" ]]; then
  printf '[ERROR] Activate the environment first: conda activate lerobot\n' >&2
  exit 2
fi

cd "$LEROBOT_ROOT"
exec python -m lerobot.grad_project.tools.runtime_regression_check "$@"
