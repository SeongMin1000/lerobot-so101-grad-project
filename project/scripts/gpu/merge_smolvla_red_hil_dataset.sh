#!/usr/bin/env bash
# GPU PC: merge the existing demonstration/recovery dataset with HIL corrections.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}"

HF_USER="${HF_USER:-eslab1234}"
BASE_DATASET="${BASE_DATASET:-${HF_USER}/red_full_138ep_recovery_v1}"
HIL_DATASET="${HIL_DATASET:-}"
MERGED_DATASET="${MERGED_DATASET:-${HF_USER}/red_full_138ep_recovery_hil_r1_v1}"

# Correction episodes are much shorter than the original end-to-end episodes.
# Repeating the HIL source increases its frame weight without deleting or
# modifying either source dataset.  Start with 3; lower it if overfitting is seen.
HIL_REPEAT="${HIL_REPEAT:-3}"
PUSH_TO_HUB="${PUSH_TO_HUB:-true}"

fail() {
  printf '[ERROR] %s\n' "$*" >&2
  exit 1
}

[[ "${CONDA_DEFAULT_ENV:-}" == "lerobot" ]] || fail "Activate first: conda activate lerobot"
command -v lerobot-edit-dataset >/dev/null 2>&1 || fail "lerobot-edit-dataset is not installed"
[[ -n "$HIL_DATASET" ]] || fail \
  "Set the exact stamped dataset id, e.g. HIL_DATASET=eslab1234/red_smolvla_hil_corrections_r1_20260806_190000"
[[ "$HIL_REPEAT" =~ ^[1-9][0-9]*$ ]] || fail "HIL_REPEAT must be a positive integer"
case "$PUSH_TO_HUB" in true|false) ;; *) fail "PUSH_TO_HUB must be true or false" ;; esac

DATA_CACHE_ROOT="${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}"
MERGED_ROOT="$DATA_CACHE_ROOT/$MERGED_DATASET"
[[ ! -e "$MERGED_ROOT" ]] || fail \
  "Output already exists: $MERGED_ROOT (choose a new MERGED_DATASET; it will not be overwritten)"

REPO_IDS="$(python - "$BASE_DATASET" "$HIL_DATASET" "$HIL_REPEAT" <<'PY'
import json
import sys

base, hil, repeat_text = sys.argv[1:]
print(json.dumps([base, *([hil] * int(repeat_text))]))
PY
)"

cd "$LEROBOT_ROOT"
printf '[MERGE] source base: %s\n' "$BASE_DATASET"
printf '[MERGE] source HIL:  %s (repeat=%s)\n' "$HIL_DATASET" "$HIL_REPEAT"
printf '[MERGE] output:      %s\n' "$MERGED_DATASET"
printf '[MERGE] repo list:   %s\n' "$REPO_IDS"

lerobot-edit-dataset \
  --new_repo_id="$MERGED_DATASET" \
  --operation.type=merge \
  --operation.repo_ids="$REPO_IDS" \
  --operation.concatenate_videos=false \
  --operation.concatenate_data=false \
  --push_to_hub="$PUSH_TO_HUB"

printf '\n[DONE] merged dataset: %s\n' "$MERGED_DATASET"
printf '[NEXT] TRAIN_DATASET=%s bash project/scripts/gpu/train_smolvla_red_hil.sh\n' "$MERGED_DATASET"
