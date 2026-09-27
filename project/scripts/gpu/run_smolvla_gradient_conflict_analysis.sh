#!/usr/bin/env bash
#
# SmolVLA Multi-Task Gradient Conflict Diagnostic Launcher (865ep vs 822ep)
# Analyzes Action Expert gradient cosine similarity, gradient norms,
# within-task baselines, and delta comparisons across trajectory phases:
# EARLY, MID, LATE, WHOLE on SO-101 demonstrations.
#

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}"

PRIMARY_POLICY="${PRIMARY_POLICY:-eslab1234/smolvla_multitask_5blocks_v3_822ep_from865_expertonly_b32_lr5e6_50k}"
BASELINE_POLICY="${BASELINE_POLICY:-eslab1234/smolvla_multitask_5blocks_v3_865ep_trimmed_from575_285k_fullft_lr1e5_150k}"
TASK1_DATASET="${TASK1_DATASET:-eslab1234/smolvla_task1_hil_575_285k_v4_120ep_trimmed_merged}"
TASK2_DATASET="${TASK2_DATASET:-eslab1234/smolvla_task2_hil_575_285k_v4_127ep_trimmed_merged}"

PHASES="${PHASES:-EARLY MID LATE WHOLE}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"
N_REPEATS="${N_REPEATS:-15}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_DIR="${OUTPUT_DIR:-$LEROBOT_ROOT/outputs/gradient_conflict_comparison_865_vs_822}"
SEED="${SEED:-42}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "================================================================="
echo "🔬 SmolVLA Multi-Task Gradient Conflict: 865ep vs 822ep"
echo "================================================================="
echo "Primary Target (822) : $PRIMARY_POLICY"
echo "Baseline Target (865): $BASELINE_POLICY"
echo "Task 1 Dataset       : $TASK1_DATASET"
echo "Task 2 Dataset       : $TASK2_DATASET"
echo "Phases               : $PHASES"
echo "Batch Size           : $BATCH_SIZE (micro-batch: $MICRO_BATCH_SIZE)"
echo "Repeats (N)          : $N_REPEATS"
echo "Device               : $DEVICE"
echo "Output Dir           : $OUTPUT_DIR"
echo "Seed                 : $SEED"
echo "================================================================="

cd "$LEROBOT_ROOT"

if [ -f "/home/eslab/miniconda3/envs/lerobot/bin/python" ]; then
    PYTHON_BIN="/home/eslab/miniconda3/envs/lerobot/bin/python"
elif [ -f "/home/eslab/miniforge3/envs/lerobot/bin/python" ]; then
    PYTHON_BIN="/home/eslab/miniforge3/envs/lerobot/bin/python"
else
    PYTHON_BIN="python3"
fi

"$PYTHON_BIN" project/scripts/tools/analyze_smolvla_gradient_conflict.py \
    --primary-policy "$PRIMARY_POLICY" \
    --baseline-policy "$BASELINE_POLICY" \
    --task1-dataset "$TASK1_DATASET" \
    --task2-dataset "$TASK2_DATASET" \
    --phases $PHASES \
    --batch-size "$BATCH_SIZE" \
    --micro-batch-size "$MICRO_BATCH_SIZE" \
    --n-repeats "$N_REPEATS" \
    --device "$DEVICE" \
    --output-dir "$OUTPUT_DIR" \
    --seed "$SEED"
