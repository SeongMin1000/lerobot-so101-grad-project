#!/usr/bin/env bash
#
# Unified SmolVLA Trainer Launcher: General Expert-Only RWFM + Same-Color PCGrad
#
# Base Policy: 865 Full-Finetuning checkpoint (configured to Expert-Only 153 tensors)
# General: 725 episodes (575 clean + 100 rollout + 50 HIL)
# PCGrad: Same-color paired batches from Task 1 (120ep) & Task 2 (127ep)
# Multi-rate: General RWFM every step, Placement PCGrad every 6th step
# Single Optimizer Step: g_final = g_general + g_place, exactly 1 optimizer.step() per global step
#

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}"

RUN_NAME="${RUN_NAME:-smolvla_865base_rwfm725_samecolor_pcgrad_expertonly_30k}"
BASE_POLICY="${BASE_POLICY:-eslab1234/smolvla_multitask_5blocks_v3_865ep_trimmed_from575_285k_fullft_lr1e5_150k}"
GENERAL_DATASET_REPO="${GENERAL_DATASET_REPO:-eslab1234/smolvla_rwfm_general_725ep_v1}"
TASK1_BRANCH_DATASET="${TASK1_BRANCH_DATASET:-eslab1234/smolvla_task1_hil_575_285k_v4_120ep_trimmed_merged}"
TASK2_BRANCH_DATASET="${TASK2_BRANCH_DATASET:-eslab1234/smolvla_task2_hil_575_285k_v4_127ep_trimmed_merged}"

OUTPUT_DIR="${OUTPUT_DIR:-$LEROBOT_ROOT/outputs/train/$RUN_NAME}"
TOTAL_STEPS="${TOTAL_STEPS:-30000}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
LOG_FREQ="${LOG_FREQ:-20}"

LR="${LR:-1e-6}"
FINAL_LR="${FINAL_LR:-3e-7}"
WARMUP_STEPS="${WARMUP_STEPS:-300}"

GENERAL_BATCH_SIZE="${GENERAL_BATCH_SIZE:-32}"
CLEAN_BATCH_SIZE="${CLEAN_BATCH_SIZE:-24}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-8}"
HIL_BATCH_SIZE="${HIL_BATCH_SIZE:-2}"
HIL_INTERVAL="${HIL_INTERVAL:-3}"

PCGRAD_BATCH_SIZE_PER_TASK="${PCGRAD_BATCH_SIZE_PER_TASK:-16}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-8}"
PCGRAD_INTERVAL="${PCGRAD_INTERVAL:-6}"

GENERAL_GRADIENT_WEIGHT="${GENERAL_GRADIENT_WEIGHT:-1.0}"
PCGRAD_GRADIENT_WEIGHT="${PCGRAD_GRADIENT_WEIGHT:-1.0}"
GENERAL_ANCHOR_PROJECTION="${GENERAL_ANCHOR_PROJECTION:-false}"

TEMPERATURE="${TEMPERATURE:-1.0}"
CHUNK_SIZE="${CHUNK_SIZE:-50}"

DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
USE_WANDB="${USE_WANDB:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-lerobot-so101-grad}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
PUSH_TO_HUB="${PUSH_TO_HUB:-true}"
HUB_REPO_ID="${HUB_REPO_ID:-eslab1234/$RUN_NAME}"
RESUME="${RESUME:-false}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "================================================================="
echo "🚀 Unified SmolVLA: General Expert-Only RWFM + Same-Color PCGrad"
echo "================================================================="
echo "Run Name                  : $RUN_NAME"
echo "Base Policy               : $BASE_POLICY"
echo "General Dataset           : $GENERAL_DATASET_REPO"
echo "Task 1 Branch             : $TASK1_BRANCH_DATASET"
echo "Task 2 Branch             : $TASK2_BRANCH_DATASET"
echo "Output Directory          : $OUTPUT_DIR"
echo "Total Steps               : $TOTAL_STEPS"
echo "Save Freq                 : $SAVE_FREQ"
echo "Learning Rate             : $LR -> $FINAL_LR (warmup: $WARMUP_STEPS steps)"
echo "General Batch             : $GENERAL_BATCH_SIZE (Clean: $CLEAN_BATCH_SIZE / Rollout: $ROLLOUT_BATCH_SIZE)"
echo "HIL Cadence               : $HIL_BATCH_SIZE samples every $HIL_INTERVAL steps"
echo "PCGrad Cadence            : $PCGRAD_BATCH_SIZE_PER_TASK T1 + $PCGRAD_BATCH_SIZE_PER_TASK T2 every $PCGRAD_INTERVAL steps"
echo "Gradient Weights          : General=$GENERAL_GRADIENT_WEIGHT, PCGrad=$PCGRAD_GRADIENT_WEIGHT"
echo "General Anchor Projection : $GENERAL_ANCHOR_PROJECTION"
echo "Device                    : $DEVICE"
echo "Seed                      : $SEED"
echo "WandB                     : $USE_WANDB (Project: $WANDB_PROJECT)"
echo "Push to Hub               : $PUSH_TO_HUB (Repo: $HUB_REPO_ID)"
echo "Resume                    : $RESUME"
echo "================================================================="

cd "$LEROBOT_ROOT"

if [ -f "/home/eslab/miniforge3/envs/lerobot/bin/python" ]; then
    PYTHON_BIN="/home/eslab/miniforge3/envs/lerobot/bin/python"
elif [ -f "/home/eslab/miniconda3/envs/lerobot/bin/python" ]; then
    PYTHON_BIN="/home/eslab/miniconda3/envs/lerobot/bin/python"
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

"$PYTHON_BIN" project/scripts/train/train_smolvla_rwfm_pcgrad_combined.py \
    --job-name "$RUN_NAME" \
    --base-policy "$BASE_POLICY" \
    --general-dataset "$GENERAL_DATASET_REPO" \
    --task1-dataset "$TASK1_BRANCH_DATASET" \
    --task2-dataset "$TASK2_BRANCH_DATASET" \
    --output-dir "$OUTPUT_DIR" \
    --steps "$TOTAL_STEPS" \
    --save-freq "$SAVE_FREQ" \
    --log-freq "$LOG_FREQ" \
    --lr "$LR" \
    --final-lr "$FINAL_LR" \
    --warmup-steps "$WARMUP_STEPS" \
    --general-batch-size "$GENERAL_BATCH_SIZE" \
    --clean-batch-size "$CLEAN_BATCH_SIZE" \
    --rollout-batch-size "$ROLLOUT_BATCH_SIZE" \
    --hil-batch-size "$HIL_BATCH_SIZE" \
    --hil-interval "$HIL_INTERVAL" \
    --microbatch-per-task "$PCGRAD_BATCH_SIZE_PER_TASK" \
    --micro-batch-size "$MICRO_BATCH_SIZE" \
    --pcgrad-interval "$PCGRAD_INTERVAL" \
    --general-gradient-weight "$GENERAL_GRADIENT_WEIGHT" \
    --pcgrad-gradient-weight "$PCGRAD_GRADIENT_WEIGHT" \
    --general-anchor-projection "$GENERAL_ANCHOR_PROJECTION" \
    --temperature "$TEMPERATURE" \
    --chunk-size "$CHUNK_SIZE" \
    --device "$DEVICE" \
    --seed "$SEED" \
    $WANDB_ARGS \
    --push-to-hub "$PUSH_TO_HUB" \
    --hub-repo-id "$HUB_REPO_ID" \
    --resume "$RESUME"
