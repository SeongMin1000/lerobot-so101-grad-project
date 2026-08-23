#!/usr/bin/env bash
#
# Hybrid 5-block one-take dataset recorder launcher (v3) for SO-101.
# Features C2-continuous Parabolic/Arc Spline trajectory for smooth block transitions.
#

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}"
RUNTIME_CONFIG="${LEROBOT_RUNTIME_CONFIG:-${RUNTIME_CONFIG:-$LEROBOT_ROOT/project/config/runtime.json}}"
DETECTOR_CALIB="${LEROBOT_DETECTOR_CALIB:-${DETECTOR_CALIB:-$LEROBOT_ROOT/project/config/detector.json}}"

export LEROBOT_RUNTIME_CONFIG="$RUNTIME_CONFIG"
export LEROBOT_DETECTOR_CALIB="$DETECTOR_CALIB"

HF_USER="${HF_USER:-eslab1234}"
DATASET_NAME="${DATASET_NAME:-task1_hybrid_5blocks_v3}"
DATASET_REPO_ID="${DATASET_REPO_ID:-${HF_USER}/${DATASET_NAME}}"
TASK="${TASK:-Pick and place 5 blocks in sequence (red, yellow, wood, green, blue).}"
COLOR_SEQUENCE="${COLOR_SEQUENCE:-red,yellow,wood,green,blue}"
NUM_EPISODES="${NUM_EPISODES:-10}"
MAX_BLOCKS="${MAX_BLOCKS:-5}"

ROBOT_PORT="${ROBOT_PORT:-/dev/so101_follower}"
TELEOP_PORT="${TELEOP_PORT:-/dev/so101_leader}"
TOP_CAM="${TOP_CAM:-/dev/cam_top}"
WRIST_CAM="${WRIST_CAM:-/dev/cam_wrist}"

FPS="${FPS:-30}"
WIDTH="${WIDTH:-640}"
HEIGHT="${HEIGHT:-480}"

MACRO_GOTO_DURATION_S="${MACRO_GOTO_DURATION_S:-2.0}"
MACRO_RETURN_DURATION_S="${MACRO_RETURN_DURATION_S:-2.0}"
HOVER_Z_OFFSET_M="${HOVER_Z_OFFSET_M:-0.10}"

YOLO_MODEL_PATH="${YOLO_MODEL_PATH:-$LEROBOT_ROOT/project/models/yolo_block_detector/best.pt}"
GRASP_CALIBRATION="${GRASP_CALIBRATION:-$LEROBOT_ROOT/project/config/grasp_pixel_to_robot_record.json}"

STREAMING_ENCODING="${STREAMING_ENCODING:-true}"
ENCODER_THREADS="${ENCODER_THREADS:-2}"
PLAY_SOUNDS="${PLAY_SOUNDS:-true}"
RESUME="${RESUME:-false}"

echo "================================================================="
echo "🎬 SO-101 Hybrid 5-Block One-Take Recorder (v3: Parabolic Arc)"
echo "================================================================="
echo "Dataset Repo ID : $DATASET_REPO_ID"
echo "Task Prompt     : $TASK"
echo "Color Sequence  : $COLOR_SEQUENCE"
echo "Episodes        : $NUM_EPISODES"
echo "Robot Port      : $ROBOT_PORT"
echo "Teleop Port     : $TELEOP_PORT"
echo "Top Camera      : $TOP_CAM"
echo "Wrist Camera    : $WRIST_CAM"
echo "================================================================="

cd "$LEROBOT_ROOT"

python -m lerobot.grad_project.recording.hybrid_record_5blocks_onetake_v3 \
  --robot.type=so101_follower \
  --robot.port="$ROBOT_PORT" \
  --robot.id=follower \
  --robot.disable_torque_on_disconnect=false \
  --teleop.type=so101_leader \
  --teleop.port="$TELEOP_PORT" \
  --teleop.id=leader \
  --robot.cameras="{ top: {type: opencv, index_or_path: '$TOP_CAM', width: $WIDTH, height: $HEIGHT, fps: $FPS, fourcc: 'MJPG'}, wrist: {type: opencv, index_or_path: '$WRIST_CAM', width: $WIDTH, height: $HEIGHT, fps: $FPS, fourcc: 'MJPG'} }" \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --dataset.single_task="$TASK" \
  --dataset.num_episodes="$NUM_EPISODES" \
  --dataset.fps="$FPS" \
  --dataset.video=true \
  --dataset.streaming_encoding="$STREAMING_ENCODING" \
  --dataset.encoder_threads="$ENCODER_THREADS" \
  --color_sequence="$COLOR_SEQUENCE" \
  --max_blocks="$MAX_BLOCKS" \
  --macro_goto_duration_s="$MACRO_GOTO_DURATION_S" \
  --macro_return_duration_s="$MACRO_RETURN_DURATION_S" \
  --hover_z_offset_m="$HOVER_Z_OFFSET_M" \
  --yolo_model_path="$YOLO_MODEL_PATH" \
  --grasp_calibration="$GRASP_CALIBRATION" \
  --runtime_config="$RUNTIME_CONFIG" \
  --play_sounds="$PLAY_SOUNDS" \
  --resume="$RESUME"
