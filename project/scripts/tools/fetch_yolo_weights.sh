#!/usr/bin/env bash
# Pull the freshly trained weights back from the GPU box and sanity-check them.
#
# Lands them as best_v2.pt rather than overwriting best.pt: the old model is
# the only working detector until the new one is shown to be at least as good,
# and swapping it out before that check leaves nothing to fall back to.
set -euo pipefail

GPU_HOST="${GPU_HOST:-100.85.69.64}"
GPU_PORT="${GPU_PORT:-5555}"
REMOTE_DIR="${REMOTE_DIR:-\$HOME/yolo_train_v2}"
DEST="project/models/yolo_block_detector/best_v2.pt"

echo "==> 학습 결과 위치 확인"
REMOTE_BEST=$(ssh -p "$GPU_PORT" "$GPU_HOST" "find $REMOTE_DIR -name best.pt -newermt '-1 day' | head -1")
[ -n "$REMOTE_BEST" ] || { echo "best.pt를 못 찾음. 학습이 아직 안 끝났을 수 있음:" >&2
  ssh -p "$GPU_PORT" "$GPU_HOST" "tail -3 $REMOTE_DIR/train.log" >&2; exit 1; }

echo "==> 성능 요약"
ssh -p "$GPU_PORT" "$GPU_HOST" "grep -E 'all|epochs completed' $REMOTE_DIR/train.log | tail -8" || true

mkdir -p "$(dirname "$DEST")"
scp -P "$GPU_PORT" "$GPU_HOST:$REMOTE_BEST" "$DEST"
echo "==> 저장됨: $DEST"

echo "==> 불러오기 확인"
~/miniforge3/envs/lerobot/bin/python3 -c "
import sys; sys.path.insert(0, 'src')
from lerobot.grad_project.perception.yolo_block_detector import YoloBlockDetector
d = YoloBlockDetector.load('$DEST', 'project/config/detector.json', frame_color='bgr')
print('클래스:', d.model.names)
"
echo
echo "다음: 새 모델로 현재 화면 탐지 비교"
echo "  python project/scripts/tools/compare_yolo_models.py"
