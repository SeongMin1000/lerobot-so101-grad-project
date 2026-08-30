#!/usr/bin/env bash
# GPU PC: Run SmolVLA ReinFlow Online RL Policy Server (gRPC Port 8080)
#
# Listens for Robot Client connections, streams 4-step Flow-SDE actions,
# collects real-time rollouts, and performs asynchronous online PPO fine-tuning.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
FPS="${FPS:-30}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
SIGMA="${SIGMA:-0.05}"
RL_STEPS="${RL_STEPS:-4}"
ACTOR_LR="${ACTOR_LR:-3e-5}"
CRITIC_LR="${CRITIC_LR:-1e-4}"
UPDATE_BATCH_SIZE="${UPDATE_BATCH_SIZE:-16}"
PPO_EPOCHS="${PPO_EPOCHS:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/reinflow_smolvla_live}"

cd "$LEROBOT_ROOT"

printf '[REINFLOW POLICY SERVER] Starting ReinFlow Online RL Policy Server on %s:%s\n' "$HOST" "$PORT"
printf '[REINFLOW POLICY SERVER] Flow-SDE Steps: %s | Sigma: %s | FPS: %s\n' "$RL_STEPS" "$SIGMA" "$FPS"
printf '[REINFLOW POLICY SERVER] Update Batch Size: %s | PPO Epochs: %s | Output: %s\n' \
  "$UPDATE_BATCH_SIZE" "$PPO_EPOCHS" "$OUTPUT_DIR"

export PYTHONUNBUFFERED=1
CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \
python -m lerobot.grad_project.rl.reinflow_policy_server \
  --host="$HOST" \
  --port="$PORT" \
  --fps="$FPS" \
  --sigma="$SIGMA" \
  --rl_steps="$RL_STEPS" \
  --actor_lr="$ACTOR_LR" \
  --critic_lr="$CRITIC_LR" \
  --update_batch_size="$UPDATE_BATCH_SIZE" \
  --ppo_epochs="$PPO_EPOCHS" \
  --checkpoint_output_dir="$OUTPUT_DIR"
