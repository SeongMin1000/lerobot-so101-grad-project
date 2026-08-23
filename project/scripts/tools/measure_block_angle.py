#!/usr/bin/env python3
"""Measure how far each block is rotated, without touching the live pipeline.

Standalone on purpose: the pick-and-place path is working and this is still an
experiment, so nothing here is imported by it. It reuses the trained detector
read-only to locate the blocks, then measures rotation inside each box.

Why classic vision rather than the network: the network already answers the
hard question -- WHERE the block is, among shadows, the arm and the desk. What
is left is a ~40x40 crop that is mostly one solid-coloured block against a
plain board, and a min-area rect over that recovers the rotation. Teaching the
network angles instead would mean re-labelling every box as an oriented one.

Angles are reported in [-45, 45): a square block is symmetric every 90 degrees,
so that range already covers every distinct way the jaws can meet it.

Usage: measure_block_angle.py
Writes an overlay to var/dryrun_debug/block_angles.jpg -- red line along the
measured face direction, magenta across it.
"""

import cv2
import numpy as np

from lerobot.grad_project.paths import lerobot_root
from lerobot.grad_project.perception.yolo_block_detector import YoloBlockDetector

MODEL = "project/models/yolo_block_detector/best.pt"
OUT = "var/dryrun_debug/block_angles.jpg"
MIN_FILL = 0.15  # contour must cover this much of the crop to be the block
MAX_ASPECT = 1.6  # more elongated than this is a shadow or two merged blocks
COLOUR_TOLERANCE = 34.0  # LAB distance still counted as the same block


def block_angle_deg(frame_bgr: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[float, str]:
    """Rotation of the block inside `bbox`, plus why it was rejected if it was."""
    height, width = frame_bgr.shape[:2]
    x, y, w, h = bbox
    pad = 4
    left, top = max(0, x - pad), max(0, y - pad)
    right, bottom = min(width, x + w + pad), min(height, y + h + pad)
    if right - left < 8 or bottom - top < 8:
        return 0.0, "box too small"

    crop = frame_bgr[top:bottom, left:right]
    # Match on COLOUR, not brightness. A brightness split lumps the block in
    # with anything else dark in the crop -- the navy block merged with a
    # shadow above it and the pair measured as one axis-aligned blob, angle 0.
    # Every block is a solid distinct colour, so distance from the colour at
    # the box's centre picks out that block and rejects its neighbours.
    lab = cv2.cvtColor(cv2.GaussianBlur(crop, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.int16)
    mid_y, mid_x = lab.shape[0] // 2, lab.shape[1] // 2
    patch = lab[max(0, mid_y - 3) : mid_y + 4, max(0, mid_x - 3) : mid_x + 4]
    reference = np.median(patch.reshape(-1, 3), axis=0)
    distance = np.linalg.norm(lab - reference, axis=2)
    mask = (distance < COLOUR_TOLERANCE).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, "no contour"
    # The block is the region under the centre, not merely the biggest one.
    holding = [c for c in contours if cv2.pointPolygonTest(c, (float(mid_x), float(mid_y)), False) >= 0]
    largest = max(holding or contours, key=cv2.contourArea)
    fill = cv2.contourArea(largest) / max((right - left) * (bottom - top), 1)
    if fill < MIN_FILL:
        return 0.0, f"contour only {fill:.0%} of crop"

    (_, _), (rect_w, rect_h), angle = cv2.minAreaRect(largest)
    short, long = min(rect_w, rect_h), max(rect_w, rect_h)
    if short < 4:
        return 0.0, "rect too thin"
    if long / max(short, 1e-6) > MAX_ASPECT:
        return 0.0, f"aspect {long / short:.1f} not square"
    return float(((angle + 45.0) % 90.0) - 45.0), "ok"


def main() -> None:
    root = lerobot_root()
    capture = cv2.VideoCapture("/dev/cam_top", cv2.CAP_V4L2)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    frame = None
    for _ in range(10):
        _, frame = capture.read()
    capture.release()
    if frame is None:
        raise SystemExit("/dev/cam_top 에서 프레임을 못 읽음")

    detector = YoloBlockDetector.load(
        str(root / MODEL), root / "project/config/detector.json", frame_color="bgr"
    )
    result = detector.detect(frame)
    overlay = detector.draw(frame, result)

    print(f"{'색상':<8}{'각도':>8}   판정")
    for block in sorted(result.blocks, key=lambda b: b.color):
        angle, note = block_angle_deg(frame, block.bbox)
        print(f"  {block.color:<7}{angle:+7.1f}도   {note}")
        radians = np.radians(angle)
        cx, cy, length = block.cx, block.cy, 20
        cv2.line(
            overlay,
            (int(cx - length * np.cos(radians)), int(cy - length * np.sin(radians))),
            (int(cx + length * np.cos(radians)), int(cy + length * np.sin(radians))),
            (0, 0, 255),
            2,
        )
        cv2.line(
            overlay,
            (int(cx + length * np.sin(radians)), int(cy - length * np.cos(radians))),
            (int(cx - length * np.sin(radians)), int(cy + length * np.cos(radians))),
            (255, 0, 255),
            2,
        )

    out = root / OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), overlay)
    print(f"\n오버레이: {out}")


if __name__ == "__main__":
    main()
