#!/usr/bin/env bash
# Train the block detector on the GPU box and bring the weights back.
#
# The dataset is built on the robot PC (that is where the camera is) but the
# only GPU is on the other machine, so this ships the reviewed dataset over,
# trains there, and copies best.pt back into the path the pipeline already
# looks at. Run it from the robot PC.
#
#   bash project/scripts/tools/train_yolo_on_gpu.sh <dataset.zip> [epochs]
#
# <dataset.zip> is what you download back from Roboflow after reviewing the
# auto-labels -- not the pre-review zip, unless you are deliberately training
# on unreviewed labels.
set -euo pipefail

GPU_HOST="${GPU_HOST:-100.85.69.64}"
GPU_PORT="${GPU_PORT:-5555}"
GPU_PY="${GPU_PY:-\$HOME/yolo_env/bin/python}"
REMOTE_DIR="${REMOTE_DIR:-\$HOME/yolo_train_v2}"
LOCAL_WEIGHTS="project/models/yolo_block_detector/best_v2.pt"

ZIP="${1:?사용법: train_yolo_on_gpu.sh <dataset.zip> [epochs]}"
EPOCHS="${2:-150}"
[ -f "$ZIP" ] || { echo "파일 없음: $ZIP" >&2; exit 1; }

echo "==> 데이터셋 전송: $ZIP"
ssh -p "$GPU_PORT" "$GPU_HOST" "rm -rf $REMOTE_DIR && mkdir -p $REMOTE_DIR"
scp -P "$GPU_PORT" "$ZIP" "$GPU_HOST:$REMOTE_DIR/dataset.zip"

echo "==> 압축 해제 + data.yaml 경로 확인"
ssh -p "$GPU_PORT" "$GPU_HOST" "cd $REMOTE_DIR && unzip -q dataset.zip && find . -name data.yaml | head -3"

echo "==> 학습 시작 (epochs=$EPOCHS). 로그: $REMOTE_DIR/train.log"
# Roboflow exports put data.yaml at different depths depending on the format,
# so locate it rather than assuming, and make its relative paths absolute --
# they are written relative to the yaml and break when run from elsewhere.
ssh -p "$GPU_PORT" "$GPU_HOST" "cd $REMOTE_DIR && \
  YAML=\$(find . -name data.yaml | head -1) && \
  DATA_DIR=\$(dirname \$(readlink -f \$YAML)) && \
  nohup $GPU_PY -c \"
from ultralytics import YOLO
import yaml, os
d = yaml.safe_load(open('\$YAML'))
for k in ('train','val'):
    if k in d and not os.path.isabs(d[k]):
        d[k] = os.path.normpath(os.path.join('\$DATA_DIR', d[k]))
yaml.safe_dump(d, open('resolved.yaml','w'))
print('data:', d)
YOLO('yolo11n.pt').train(data='resolved.yaml', epochs=$EPOCHS, imgsz=640, batch=16, device=0)
\" > train.log 2>&1 &
  echo started"

echo
echo "학습이 백그라운드로 돌고 있음. 진행 확인:"
echo "  ssh -p $GPU_PORT $GPU_HOST 'tail -5 $REMOTE_DIR/train.log'"
echo
echo "끝나면 가중치 가져오기:"
echo "  bash project/scripts/tools/fetch_yolo_weights.sh"
echo "  (가져온 파일: $LOCAL_WEIGHTS)"
