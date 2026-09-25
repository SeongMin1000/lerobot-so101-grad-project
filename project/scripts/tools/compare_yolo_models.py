#!/usr/bin/env python3
"""Run the old and new detectors on the same live frame, side by side.

Validation numbers from training only say the new model fits its own
validation split -- which came from the same session as its training data.
The question that actually matters is whether it reads the board better than
the model already in use, right now, through the camera in its current
position. So: one frame, both models, same picture.

    python project/scripts/tools/compare_yolo_models.py
"""

import sys

import cv2
import numpy as np

from lerobot.grad_project.paths import lerobot_root
from lerobot.grad_project.perception.yolo_block_detector import YoloBlockDetector

OLD = "project/models/yolo_block_detector/best.pt"
NEW = "project/models/yolo_block_detector/best_v2.pt"
OUT = "var/dryrun_debug/model_compare.jpg"
EXPECTED = 5


def grab() -> np.ndarray:
    capture = cv2.VideoCapture("/dev/cam_top", cv2.CAP_V4L2)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    frame = None
    for _ in range(10):
        ok, frame = capture.read()
    capture.release()
    if frame is None:
        raise SystemExit("/dev/cam_top 에서 프레임을 못 읽음")
    return frame


def report(tag: str, detector: YoloBlockDetector, frame: np.ndarray):
    result = detector.detect(frame)
    print(f"\n[{tag}] {len(result.blocks)}개 탐지 (기대: {EXPECTED})")
    for block in sorted(result.blocks, key=lambda b: b.color):
        where = "구역안" if block.in_target else "구역밖"
        print(f"   {block.color:7s} ({block.cx:6.1f},{block.cy:6.1f})  {where}")
    missing = {"red", "green", "blue", "yellow", "wood"} - {b.color for b in result.blocks}
    if missing:
        print(f"   못 찾은 색: {', '.join(sorted(missing))}")
    return result


def main() -> None:
    root = lerobot_root()
    new_path = root / NEW
    if not new_path.is_file():
        raise SystemExit(f"새 모델이 없음: {new_path}\n먼저 fetch_yolo_weights.sh 를 실행하세요")

    frame = grab()
    panels = []
    for tag, rel in (("기존 모델", OLD), ("새 모델", NEW)):
        detector = YoloBlockDetector.load(
            str(root / rel), root / "project/config/detector.json", frame_color="bgr"
        )
        result = report(tag, detector, frame)
        panel = detector.draw(frame, result)
        cv2.putText(panel, tag, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        panels.append(panel)

    out = root / OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.hstack(panels))
    print(f"\n비교 이미지: {out}")
    print("새 모델이 같거나 더 잘 잡으면 교체:")
    print(f"  cp {NEW} {OLD}")


if __name__ == "__main__":
    sys.exit(main())
