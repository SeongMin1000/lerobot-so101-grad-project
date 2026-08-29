#!/usr/bin/env bash
# GPU PC: Run SmolVLA ReinFlow Online RL Training (Real-Robot / Mock Mode)

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}"

CONDA_ENV="${CONDA_ENV:-lerobot}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
TARGET_COLOR="${TARGET_COLOR:-red}"
NUM_EPISODES="${NUM_EPISODES:-20}"
STEPS_PER_EPISODE="${STEPS_PER_EPISODE:-25}"
BATCH_SIZE="${BATCH_SIZE:-8}"
PPO_EPOCHS="${PPO_EPOCHS:-4}"
ACTOR_LR="${ACTOR_LR:-3e-5}"
CRITIC_LR="${CRITIC_LR:-1e-4}"
SIGMA="${SIGMA:-0.05}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/reinflow_smolvla}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
MOCK_FLAG="${MOCK_FLAG:-}"

cd "$LEROBOT_ROOT"

printf '[REINFLOW RL] Starting SmolVLA ReinFlow Training Runner\n'
printf '[REINFLOW RL] Target Color: %s | Checkpoint: %s\n' "$TARGET_COLOR" "${CHECKPOINT_PATH:-None (Base SmolVLM2)}"
printf '[REINFLOW RL] Episodes: %s | Steps/Ep: %s | Batch Size: %s | PPO Epochs: %s\n' \
  "$NUM_EPISODES" "$STEPS_PER_EPISODE" "$BATCH_SIZE" "$PPO_EPOCHS"
printf '[REINFLOW RL] Actor LR: %s | Critic LR: %s | Sigma: %s | Output: %s\n' \
  "$ACTOR_LR" "$CRITIC_LR" "$SIGMA" "$OUTPUT_DIR"

export PYTHONUNBUFFERED=1
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \
python -m lerobot.grad_project.rl.real_robot_rollout \
  --checkpoint-path="$CHECKPOINT_PATH" \
  --target-color="$TARGET_COLOR" \
  --num-episodes="$NUM_EPISODES" \
  --steps-per-episode="$STEPS_PER_EPISODE" \
  --batch-size="$BATCH_SIZE" \
  --ppo-epochs="$PPO_EPOCHS" \
  --actor-lr="$ACTOR_LR" \
  --critic-lr="$CRITIC_LR" \
  --sigma="$SIGMA" \
  --output-dir="$OUTPUT_DIR" \
  $MOCK_FLAG
