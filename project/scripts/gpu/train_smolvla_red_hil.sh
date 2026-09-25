#!/usr/bin/env bash
# GPU PC: conservative iterative LoRA fine-tune after one HIL collection round.

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}"

HF_USER="${HF_USER:-eslab1234}"
TRAIN_DATASET="${TRAIN_DATASET:-}"
CURRENT_MODEL="${CURRENT_MODEL:-${HF_USER}/smolvla_red_full_138ep_recovery_lora_r64_lr1e3_20k_v1}"
RUN_NAME="${RUN_NAME:-smolvla_red_hil_r1_lora_r64_lr3e4_10k_v1}"

STEPS="${STEPS:-10000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LEARNING_RATE="${LEARNING_RATE:-3e-4}"
MIN_LEARNING_RATE="${MIN_LEARNING_RATE:-2.5e-6}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
SAVE_FREQ="${SAVE_FREQ:-2500}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"

DATA_CACHE_ROOT="${HF_LEROBOT_HOME:-$HOME/.cache/huggingface/lerobot}"
TRAIN_DATASET_ROOT="${TRAIN_DATASET_ROOT:-$DATA_CACHE_ROOT/$TRAIN_DATASET}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/$RUN_NAME}"

fail() {
  printf '[ERROR] %s\n' "$*" >&2
  exit 1
}

[[ "${CONDA_DEFAULT_ENV:-}" == "lerobot" ]] || fail "Activate first: conda activate lerobot"
command -v lerobot-train >/dev/null 2>&1 || fail "lerobot-train is not installed"
[[ -n "$TRAIN_DATASET" ]] || fail \
  "Set TRAIN_DATASET to the merged dataset, e.g. eslab1234/red_full_138ep_recovery_hil_r1_v1"
[[ -d "$TRAIN_DATASET_ROOT" ]] || fail \
  "Merged dataset not found at $TRAIN_DATASET_ROOT (run merge_smolvla_red_hil_dataset.sh first)"
[[ ! -e "$LEROBOT_ROOT/$OUTPUT_DIR" ]] || fail \
  "Output already exists: $LEROBOT_ROOT/$OUTPUT_DIR (choose a new RUN_NAME; it will not be overwritten)"

RENAME_MAP='{"observation.images.top":"observation.images.camera1","observation.images.wrist":"observation.images.camera2","observation.images.belly":"observation.images.camera3"}'
IMAGE_TRANSFORMS='{"brightness":{"weight":1.0,"type":"ColorJitter","kwargs":{"brightness":[0.9,1.1]}},"contrast":{"weight":1.0,"type":"ColorJitter","kwargs":{"contrast":[0.9,1.1]}},"sharpness":{"weight":0.5,"type":"SharpnessJitter","kwargs":{"sharpness":[0.8,1.2]}}}'

cd "$LEROBOT_ROOT"
mkdir -p logs

printf '[TRAIN] dataset: %s\n' "$TRAIN_DATASET"
printf '[TRAIN] continue adapter: %s\n' "$CURRENT_MODEL"
printf '[TRAIN] output model: %s/%s\n' "$HF_USER" "$RUN_NAME"
printf '[TRAIN] steps=%s batch=%s peak_lr=%s warmup=%s\n' \
  "$STEPS" "$BATCH_SIZE" "$LEARNING_RATE" "$WARMUP_STEPS"

CUDA_VISIBLE_DEVICES="$CUDA_DEVICE" \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
lerobot-train \
  --dataset.repo_id="$TRAIN_DATASET" \
  --dataset.root="$TRAIN_DATASET_ROOT" \
  --dataset.use_imagenet_stats=true \
  --dataset.image_transforms.enable=true \
  --dataset.image_transforms.max_num_transforms=2 \
  --dataset.image_transforms.tfs="$IMAGE_TRANSFORMS" \
  --rename_map="$RENAME_MAP" \
  --policy.path="$CURRENT_MODEL" \
  --policy.device=cuda \
  --policy.use_amp=false \
  --policy.use_peft=true \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.train_state_proj=true \
  --policy.optimizer_lr="$LEARNING_RATE" \
  --policy.optimizer_betas='[0.9,0.95]' \
  --policy.optimizer_weight_decay=0.01 \
  --policy.scheduler_warmup_steps="$WARMUP_STEPS" \
  --policy.scheduler_decay_steps="$STEPS" \
  --policy.scheduler_decay_lr="$MIN_LEARNING_RATE" \
  --peft.method_type=LORA \
  --peft.r=64 \
  --peft.lora_alpha=64 \
  --batch_size="$BATCH_SIZE" \
  --steps="$STEPS" \
  --seed=1000 \
  --num_workers=4 \
  --prefetch_factor=4 \
  --persistent_workers=true \
  --save_checkpoint=true \
  --save_freq="$SAVE_FREQ" \
  --log_freq=100 \
  --output_dir="$OUTPUT_DIR" \
  --job_name="$RUN_NAME" \
  --policy.push_to_hub=true \
  --policy.repo_id="${HF_USER}/${RUN_NAME}" \
  --wandb.enable=true \
  --wandb.project=lerobot \
  2>&1 | tee "logs/${RUN_NAME}_latest.log"
