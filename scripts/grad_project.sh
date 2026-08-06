#!/usr/bin/env bash
set -euo pipefail

ROOT="${LEROBOT_ROOT:-$HOME/lerobot}"
CONFIG="${LEROBOT_DETECTOR_CALIB:-$ROOT/configs/opencv_detector.json}"
RUNTIME="${LEROBOT_RUNTIME_CONFIG:-$ROOT/hybrid_runtime.json}"
TOP_CAM="${TOP_CAM:-/dev/cam_top}"
WRIST_CAM="${WRIST_CAM:-/dev/cam_wrist}"
ROBOT_PORT="${ROBOT_PORT:-/dev/so101_follower}"
TELEOP_PORT="${TELEOP_PORT:-/dev/so101_leader}"
FPS="${FPS:-30}"
WIDTH="${WIDTH:-640}"
HEIGHT="${HEIGHT:-480}"
SERVER_ADDRESS="${SERVER_ADDRESS:-100.85.69.64:8080}"
HF_USER="${HF_USER:-eslab1234}"

cd "$ROOT"

usage() {
  cat <<'USAGE'
사용법: bash ~/lerobot/scripts/grad_project.sh <명령>

OpenCV
  opencv-debug         화면만 표시. 사진 저장 안 함
  opencv-debug-save    최대 50쌍만 순환 저장
  opencv-background    빈 배경 다시 촬영
  opencv-target        목표 구역 4점 지정
  opencv-workspace     작업 영역 점 지정
  opencv-ignore        로봇팔 제외 영역 지정
  opencv-clear-ignore  제외 영역 초기화
  opencv-colors        5색 + 단일 블록 크기 보정
  opencv-test          코드/JSON/self-test 검사

실행
  act-one              ACT로 블록 1개 실행
  act-five             ACT로 최대 5개 실행
  record               색상 순서 데이터셋 촬영
  record-zone          지정 구역 근접 집기 데이터셋 촬영
  server               GPU 서버 policy_server 실행
  check                장치·설정 경로 확인
  camera               Top 카메라 재현성 설정 적용(60Hz/WB 고정)
  clean-generated      로그·디버그 사진만 즉시 삭제

ACT 실행 전 POLICY_PATH 환경변수를 지정해야 함.
예: POLICY_PATH=eslab1234/모델명 bash ~/lerobot/scripts/grad_project.sh act-one
USAGE
}

require_file() {
  [[ -f "$1" ]] || { echo "[ERROR] 파일 없음: $1" >&2; exit 1; }
}

control_exists() {
  local device="$1"
  local control="$2"
  v4l2-ctl -d "$device" --list-ctrls 2>/dev/null | grep -qE "^[[:space:]]*${control}[[:space:]]"
}

set_control() {
  local device="$1"
  local control="$2"
  local value="$3"
  if control_exists "$device" "$control"; then
    v4l2-ctl -d "$device" --set-ctrl="${control}=${value}"
  else
    echo "[SKIP] $device does not expose control: $control"
  fi
}

apply_top_camera_profile() {
  v4l2-ctl -d "$TOP_CAM" \
    --set-fmt-video="width=${WIDTH},height=${HEIGHT},pixelformat=MJPG"
  v4l2-ctl -d "$TOP_CAM" --set-parm="$FPS"
  set_control "$TOP_CAM" brightness 0
  set_control "$TOP_CAM" contrast 40
  set_control "$TOP_CAM" saturation 64
  set_control "$TOP_CAM" hue 0
  set_control "$TOP_CAM" white_balance_automatic 0
  set_control "$TOP_CAM" white_balance_temperature 4600
  set_control "$TOP_CAM" gamma 300
  set_control "$TOP_CAM" sharpness 50
  set_control "$TOP_CAM" power_line_frequency 2
  set_control "$TOP_CAM" backlight_compensation 0
  set_control "$TOP_CAM" auto_exposure 3
  set_control "$TOP_CAM" exposure_dynamic_framerate 0
}

opencv_base=(
  python -m lerobot.async_inference.opencv_block_detector
  --device "$TOP_CAM" --width "$WIDTH" --height "$HEIGHT"
  --fps "$FPS" --fourcc MJPG --config "$CONFIG"
)

run_act() {
  local max_blocks="$1"
  : "${POLICY_PATH:?POLICY_PATH를 지정하세요.}"
  require_file "$RUNTIME"
  require_file "$CONFIG"
  python -m lerobot.async_inference.hybrid_cv_act_client \
    --robot.type=so101_follower \
    --robot.port="$ROBOT_PORT" \
    --robot.id=follower \
    --robot.disable_torque_on_disconnect=false \
    --robot.cameras="{ top: {type: opencv, index_or_path: '$TOP_CAM', width: $WIDTH, height: $HEIGHT, fps: $FPS, fourcc: 'MJPG'}, wrist: {type: opencv, index_or_path: '$WRIST_CAM', width: $WIDTH, height: $HEIGHT, fps: $FPS, fourcc: 'MJPG'} }" \
    --task="${TASK:-pick the selected block}" \
    --server_address="$SERVER_ADDRESS" \
    --policy_type=act \
    --pretrained_name_or_path="$POLICY_PATH" \
    --policy_device=cuda \
    --client_device=cpu \
    --actions_per_chunk="${ACTIONS_PER_CHUNK:-10}" \
    --chunk_size_threshold="${CHUNK_SIZE_THRESHOLD:-0.95}" \
    --aggregate_fn_name="${AGGREGATE_FN_NAME:-latest_only}" \
    --fps="$FPS" \
    --runtime_config="$RUNTIME" \
    --detector_calib="$CONFIG" \
    --top_key=top \
    --max_blocks="$max_blocks" \
    --act_grasp_seconds="${ACT_GRASP_SECONDS:-7}"
}

case "${1:-}" in
  opencv-debug)
    require_file "$CONFIG"
    "${opencv_base[@]}" --debug --show-masks
    ;;
  opencv-debug-save)
    require_file "$CONFIG"
    mkdir -p "$ROOT/runtime/opencv_debug"
    "${opencv_base[@]}" --debug --show-masks \
      --save-debug-dir "$ROOT/runtime/opencv_debug" \
      --max-debug-frames "${MAX_DEBUG_FRAMES:-50}"
    ;;
  opencv-background)
    require_file "$CONFIG"
    "${opencv_base[@]}" --capture-background \
      --countdown 5 --warmup-frames 60 --background-frames 31
    ;;
  opencv-target)
    require_file "$CONFIG"
    "${opencv_base[@]}" --calibrate-target --exit-after-calibration
    ;;
  opencv-workspace)
    require_file "$CONFIG"
    "${opencv_base[@]}" --calibrate-workspace --exit-after-calibration
    ;;
  opencv-ignore)
    require_file "$CONFIG"
    "${opencv_base[@]}" --calibrate-ignore \
      --ignore-points "${IGNORE_POINTS:-6}" --exit-after-calibration
    ;;
  opencv-clear-ignore)
    require_file "$CONFIG"
    "${opencv_base[@]}" --clear-ignore-polygons --exit-after-calibration
    ;;
  opencv-colors)
    require_file "$CONFIG"
    python -m lerobot.async_inference.opencv_calibrate \
      --device "$TOP_CAM" --width "$WIDTH" --height "$HEIGHT" \
      --fps "$FPS" --fourcc MJPG --config "$CONFIG" \
      --colors red,yellow,wood,green,blue \
      --samples-per-color "${SAMPLES_PER_COLOR:-5}" \
      --frames-per-sample "${FRAMES_PER_SAMPLE:-15}" \
      --preview-dir "$ROOT/assets/opencv/calibration"
    ;;
  opencv-test)
    require_file "$CONFIG"
    python -m json.tool "$CONFIG" >/dev/null
    python -m py_compile \
      src/lerobot/async_inference/opencv_block_detector.py \
      src/lerobot/async_inference/opencv_calibrate.py
    python -m lerobot.async_inference.opencv_block_detector --self-test
    echo "[OK] unified OpenCV detector"
    ;;
  act-one)
    run_act 1
    ;;
  act-five)
    run_act 5
    ;;
  record-zone)
    require_file "$RUNTIME"
    ZONE="${ZONE:-A1}"
    NUM_EPISODES="${NUM_EPISODES:-20}"
    EPISODE_TIME_S="${EPISODE_TIME_S:-5}"
    DATASET_NAME="${DATASET_NAME:-grasp_${ZONE}_5colors_pregrasp_v1}"
    BASE_REPO_ID="${HF_USER}/${DATASET_NAME}"
    ACTUAL_REPO_ID=""
    for EP in $(seq 1 "$NUM_EPISODES"); do
      echo "[$EP/$NUM_EPISODES] 블록을 $ZONE 구역에 놓고 Enter"
      read -r
      python -m lerobot.async_inference.hybrid_goto_both_pose \
        --robot.type=so101_follower --robot.port="$ROBOT_PORT" \
        --robot.id=follower --robot.disable_torque_on_disconnect=false \
        --teleop.type=so101_leader --teleop.port="$TELEOP_PORT" \
        --teleop.id=leader --runtime_config="$RUNTIME" \
        --pregrasp_label="$ZONE" --duration_s=3.0 \
        --keep_torque_on_disconnect=true
      if [[ "$EP" -eq 1 ]]; then
        lerobot-record \
          --robot.type=so101_follower --robot.port="$ROBOT_PORT" \
          --robot.id=follower --robot.disable_torque_on_disconnect=false \
          --teleop.type=so101_leader --teleop.port="$TELEOP_PORT" \
          --teleop.id=leader \
          --robot.cameras="{ top: {type: opencv, index_or_path: '$TOP_CAM', width: $WIDTH, height: $HEIGHT, fps: $FPS, fourcc: 'MJPG'}, wrist: {type: opencv, index_or_path: '$WRIST_CAM', width: $WIDTH, height: $HEIGHT, fps: $FPS, fourcc: 'MJPG'} }" \
          --dataset.repo_id="$BASE_REPO_ID" --dataset.num_episodes=1 \
          --dataset.episode_time_s="$EPISODE_TIME_S" --dataset.reset_time_s=0 \
          --dataset.fps="$FPS" --dataset.single_task="grasp one block from pregrasp and lift" \
          --dataset.push_to_hub=false --display_data=true --resume=false
        ACTUAL_DATASET_NAME=$(find "$HOME/.cache/huggingface/lerobot/$HF_USER" \
          -maxdepth 1 -type d -name "${DATASET_NAME}_*" \
          -printf "%T@ %f\n" | sort -nr | head -1 | awk '{print $2}')
        ACTUAL_REPO_ID="${HF_USER}/${ACTUAL_DATASET_NAME}"
      else
        lerobot-record \
          --robot.type=so101_follower --robot.port="$ROBOT_PORT" \
          --robot.id=follower --robot.disable_torque_on_disconnect=false \
          --teleop.type=so101_leader --teleop.port="$TELEOP_PORT" \
          --teleop.id=leader \
          --robot.cameras="{ top: {type: opencv, index_or_path: '$TOP_CAM', width: $WIDTH, height: $HEIGHT, fps: $FPS, fourcc: 'MJPG'}, wrist: {type: opencv, index_or_path: '$WRIST_CAM', width: $WIDTH, height: $HEIGHT, fps: $FPS, fourcc: 'MJPG'} }" \
          --dataset.repo_id="$ACTUAL_REPO_ID" --dataset.num_episodes=1 \
          --dataset.episode_time_s="$EPISODE_TIME_S" --dataset.reset_time_s=0 \
          --dataset.fps="$FPS" --dataset.single_task="grasp one block from pregrasp and lift" \
          --dataset.push_to_hub=false --display_data=true --resume=true
      fi
    done
    echo "[DONE] $ACTUAL_REPO_ID"
    ;;
  record)
    require_file "$RUNTIME"
    require_file "$CONFIG"
    DATASET_NAME="${DATASET_NAME:-pick_place_5colors_fixed_slots_v1}"
    NUM_SCENES="${NUM_SCENES:-40}"
    NUM_EPISODES=$((NUM_SCENES * 5))
    python -m lerobot.async_inference.hybrid_record_color_sequence \
      --robot.type=so101_follower \
      --robot.port="$ROBOT_PORT" \
      --robot.id=follower \
      --robot.disable_torque_on_disconnect=false \
      --teleop.type=so101_leader \
      --teleop.port="$TELEOP_PORT" \
      --teleop.id=leader \
      --robot.cameras="{ top: {type: opencv, index_or_path: '$TOP_CAM', width: $WIDTH, height: $HEIGHT, fps: $FPS, fourcc: 'MJPG'}, wrist: {type: opencv, index_or_path: '$WRIST_CAM', width: $WIDTH, height: $HEIGHT, fps: $FPS, fourcc: 'MJPG'} }" \
      --runtime_config="$RUNTIME" \
      --detector_calib="$CONFIG" \
      --top_key=top \
      --color_sequence=red,yellow,wood,green,blue \
      --target_slot_sequence=S1,S2,S3,S4,S5 \
      --observe_pose_name=observe \
      --observe_duration_s=2.5 \
      --goto_duration_s=3.0 \
      --detect_samples=7 \
      --detect_min_hits=4 \
      --detect_sample_interval_s=0.08 \
      --verify_scene_state=true \
      --state_samples=9 \
      --state_min_hits=5 \
      --state_sample_interval_s=0.08 \
      --auto_goto_before_each_episode=true \
      --wait_enter_before_episode=true \
      --wait_scene_reset=true \
      --dataset.repo_id="${HF_USER}/${DATASET_NAME}" \
      --dataset.num_episodes="$NUM_EPISODES" \
      --dataset.episode_time_s=30 \
      --dataset.reset_time_s=0 \
      --dataset.fps="$FPS" \
      --dataset.single_task="pick the selected block and place it in its designated target slot" \
      --dataset.push_to_hub=false \
      --display_data=true
    ;;
  server)
    python -m lerobot.async_inference.policy_server \
      --host=0.0.0.0 --port=8080 --fps=30 \
      --inference_latency=0.033 --obs_queue_timeout=1
    ;;
  check)
    echo "ROOT=$ROOT"
    echo "RUNTIME=$RUNTIME"
    echo "DETECTOR=$CONFIG"
    ls -l "$ROBOT_PORT" "$TELEOP_PORT" "$TOP_CAM" "$WRIST_CAM" 2>/dev/null || true
    python -m lerobot.async_inference.hybrid_paths
    ;;
  camera)
    apply_top_camera_profile
    v4l2-ctl -d "$WRIST_CAM" \
      --set-fmt-video="width=${WIDTH},height=${HEIGHT},pixelformat=MJPG"
    v4l2-ctl -d "$WRIST_CAM" --set-parm="$FPS"
    echo "[OK] camera settings applied"
    v4l2-ctl -d "$TOP_CAM" --list-ctrls
    ;;
  clean-generated)
    rm -rf \
      "$ROOT/logs" \
      "$ROOT/hybrid_debug" \
      "$ROOT/hybrid_debug_diffusion" \
      "$ROOT/hybrid_debug_record" \
      "$ROOT/runtime/opencv_debug"
    mkdir -p "$ROOT/runtime"
    echo "[OK] generated logs/debug images removed; calibration data preserved"
    ;;
  *)
    usage
    exit 2
    ;;
esac
