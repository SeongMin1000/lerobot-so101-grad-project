#!/usr/bin/env bash
#
# Hybrid Task 2 Vertical Stacking 5-block dataset recorder launcher (v1) for SO-101.
# Features Far-to-Near Spatial Prioritization, 2-Phase Teleoperation, High-Arc Transit, and Vertical Escape.
#

set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}"
RUNTIME_CONFIG="${LEROBOT_RUNTIME_CONFIG:-${RUNTIME_CONFIG:-$LEROBOT_ROOT/project/config/runtime.json}}"
DETECTOR_CALIB="${LEROBOT_DETECTOR_CALIB:-${DETECTOR_CALIB:-$LEROBOT_ROOT/project/config/detector.json}}"

export LEROBOT_RUNTIME_CONFIG="$RUNTIME_CONFIG"
export LEROBOT_DETECTOR_CALIB="$DETECTOR_CALIB"

HF_USER="${HF_USER:-eslab1234}"
DATASET_NAME="${DATASET_NAME:-task2_stack_5blocks_v1}"
DATASET_REPO_ID="${DATASET_REPO_ID:-${HF_USER}/${DATASET_NAME}}"
TASK="${TASK:-Stack 5 blocks vertically in sequence.}"
COLOR_SEQUENCE="${COLOR_SEQUENCE:-red,yellow,wood,green,blue}"
SORT_ORDER="${SORT_ORDER:-zone_color_priority}"
ZONE_SPLIT_X_M="${ZONE_SPLIT_X_M:-0.32}"
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
MACRO_TRANSIT_DURATION_S="${MACRO_TRANSIT_DURATION_S:-2.2}"
MACRO_RETURN_DURATION_S="${MACRO_RETURN_DURATION_S:-2.0}"

# Distance-adaptive direction compensation ("none", "right", or "left")
PAN_BIAS_DIRECTION="${PAN_BIAS_DIRECTION:-none}"
PAN_BIAS_NEAR_DEG="${PAN_BIAS_NEAR_DEG:-0.0}"
PAN_BIAS_FAR_DEG="${PAN_BIAS_FAR_DEG:-0.0}"

YOLO_MODEL_PATH="${YOLO_MODEL_PATH:-$LEROBOT_ROOT/project/models/yolo_block_detector/best.pt}"
GRASP_CALIBRATION="${GRASP_CALIBRATION:-$LEROBOT_ROOT/project/config/grasp_pixel_to_robot_record.json}"

DISPLAY_DATA="${DISPLAY_DATA:-false}"
STREAMING_ENCODING="${STREAMING_ENCODING:-true}"
ENCODER_THREADS="${ENCODER_THREADS:-2}"
PLAY_SOUNDS="${PLAY_SOUNDS:-true}"
RESUME="${RESUME:-false}"

echo "================================================================="
echo "🏗️  SO-101 Hybrid Task 2 Vertical Stacking Recorder (v1)"
echo "================================================================="
echo "Dataset Repo ID : $DATASET_REPO_ID"
echo "Task Prompt     : $TASK"
echo "Sort Order      : $SORT_ORDER (2-Zone Color Priority)"
echo "Color Sequence  : $COLOR_SEQUENCE"
echo "Episodes        : $NUM_EPISODES"
echo "Display Data    : $DISPLAY_DATA"
echo "Robot Port      : $ROBOT_PORT"
echo "Teleop Port     : $TELEOP_PORT"
echo "Top Camera      : $TOP_CAM"
echo "Wrist Camera    : $WRIST_CAM"
echo "================================================================="

cd "$LEROBOT_ROOT"

python -m lerobot.grad_project.recording.hybrid_record_task2_stack_v1 \
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
  --sort_order="$SORT_ORDER" \
  --zone_split_x_m="$ZONE_SPLIT_X_M" \
  --max_blocks="$MAX_BLOCKS" \
  --macro_goto_duration_s="$MACRO_GOTO_DURATION_S" \
  --macro_transit_duration_s="$MACRO_TRANSIT_DURATION_S" \
  --macro_return_duration_s="$MACRO_RETURN_DURATION_S" \
  --pan_bias_direction="$PAN_BIAS_DIRECTION" \
  --pan_bias_near_deg="$PAN_BIAS_NEAR_DEG" \
  --pan_bias_far_deg="$PAN_BIAS_FAR_DEG" \
  --yolo_model_path="$YOLO_MODEL_PATH" \
  --grasp_calibration="$GRASP_CALIBRATION" \
  --runtime_config="$RUNTIME_CONFIG" \
  --play_sounds="$PLAY_SOUNDS" \
  --display_data="$DISPLAY_DATA" \
  --resume="$RESUME"
