#!/usr/bin/env bash
#
# SmolVLA Same-Color Paired PCGrad Expert-Only Corrective Fine-Tuning Launcher
#
# Base Model: 822ep Expert-only checkpoint
# Trainable Scope: Action Expert only (freeze_vision_encoder=True, train_expert_only=True)
# Target Datasets: Task 1 branch (120ep) & Task 2 branch (127ep)
# Pairing: Same-color paired batches with 5-step cycle balance
# Standards: [CHANGE 18] LeRobot checkpoint layout, W&B project "lerobot-so101-grad", Hub push
#

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}"

RUN_NAME="${RUN_NAME:-smolvla_822base_branch247_samecolor_pcgrad_expertonly_lr2e6_15k}"
BASE_POLICY="${BASE_POLICY:-eslab1234/smolvla_multitask_5blocks_v3_822ep_from865_expertonly_b32_lr5e6_50k}"
TASK1_DATASET="${TASK1_DATASET:-eslab1234/smolvla_task1_hil_575_285k_v4_120ep_trimmed_merged}"
TASK2_DATASET="${TASK2_DATASET:-eslab1234/smolvla_task2_hil_575_285k_v4_127ep_trimmed_merged}"

OUTPUT_DIR="${OUTPUT_DIR:-$LEROBOT_ROOT/outputs/train/$RUN_NAME}"
STEPS="${STEPS:-15000}"
SAVE_FREQ="${SAVE_FREQ:-2500}"
LOG_FREQ="${LOG_FREQ:-20}"

LR="${LR:-2e-6}"
FINAL_LR="${FINAL_LR:-1e-6}"
WARMUP_STEPS="${WARMUP_STEPS:-300}"

MICROBATCH_PER_TASK="${MICROBATCH_PER_TASK:-16}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-8}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
USE_PCGRAD="${USE_PCGRAD:-true}"
PAIRED_FM="${PAIRED_FM:-true}"
USE_WANDB="${USE_WANDB:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-lerobot-so101-grad}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
PUSH_TO_HUB="${PUSH_TO_HUB:-true}"
HUB_REPO_ID="${HUB_REPO_ID:-eslab1234/$RUN_NAME}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "================================================================="
echo "🚀 SmolVLA Same-Color Paired PCGrad Expert-Only Fine-Tuning"
echo "================================================================="
echo "Run Name            : $RUN_NAME"
echo "Base Policy         : $BASE_POLICY"
echo "Task 1 Dataset      : $TASK1_DATASET"
echo "Task 2 Dataset      : $TASK2_DATASET"
echo "Output Directory    : $OUTPUT_DIR"
echo "Steps               : $STEPS"
echo "Save Freq           : $SAVE_FREQ"
echo "Learning Rate       : $LR -> $FINAL_LR (warmup: $WARMUP_STEPS steps)"
echo "Samples / Step      : $MICROBATCH_PER_TASK T1 + $MICROBATCH_PER_TASK T2 = $((MICROBATCH_PER_TASK * 2)) total"
echo "Micro-batch Size    : $MICRO_BATCH_SIZE (accumulated)"
echo "Apply PCGrad        : $USE_PCGRAD"
echo "Paired FM Randomness: $PAIRED_FM"
echo "Device              : $DEVICE"
echo "Seed                : $SEED"
echo "Use WandB           : $USE_WANDB (Project: $WANDB_PROJECT)"
echo "Push to Hub         : $PUSH_TO_HUB (Repo: $HUB_REPO_ID)"
echo "================================================================="

cd "$LEROBOT_ROOT"

if [ -f "/home/eslab/miniconda3/envs/lerobot/bin/python" ]; then
    PYTHON_BIN="/home/eslab/miniconda3/envs/lerobot/bin/python"
elif [ -f "/home/eslab/miniforge3/envs/lerobot/bin/python" ]; then
    PYTHON_BIN="/home/eslab/miniforge3/envs/lerobot/bin/python"
else
    PYTHON_BIN="python3"
fi

WANDB_ARGS=""
if [ "$USE_WANDB" = "true" ]; then
    WANDB_ARGS="--use-wandb true --wandb-project $WANDB_PROJECT"
    if [ -n "$WANDB_ENTITY" ]; then
        WANDB_ARGS="$WANDB_ARGS --wandb-entity $WANDB_ENTITY"
    fi
else
    WANDB_ARGS="--use-wandb false"
fi

"$PYTHON_BIN" project/scripts/train/train_smolvla_same_color_pcgrad.py \
    --job-name "$RUN_NAME" \
    --policy-path "$BASE_POLICY" \
    --task1-dataset "$TASK1_DATASET" \
    --task2-dataset "$TASK2_DATASET" \
    --output-dir "$OUTPUT_DIR" \
    --steps "$STEPS" \
    --save-freq "$SAVE_FREQ" \
    --log-freq "$LOG_FREQ" \
    --lr "$LR" \
    --final-lr "$FINAL_LR" \
    --warmup-steps "$WARMUP_STEPS" \
    --microbatch-per-task "$MICROBATCH_PER_TASK" \
    --micro-batch-size "$MICRO_BATCH_SIZE" \
    --use-pcgrad "$USE_PCGRAD" \
    --paired-fm-randomness "$PAIRED_FM" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --push-to-hub "$PUSH_TO_HUB" \
    --hub-repo-id "$HUB_REPO_ID" \
    $WANDB_ARGS
