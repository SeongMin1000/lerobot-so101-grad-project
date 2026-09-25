#!/usr/bin/env python3
"""YOLO-based drop-in replacement for TopBlockDetector.

Same output shape (`BlockDetection` / `DetectionResult`) as the OpenCV
detector in `opencv_block_detector.py`, so any caller that already knows how
to consume that result (target-zone membership, `chosen` block selection,
debug overlay) works unchanged with either detector.

Requires `ultralytics` (not installed by default -- `pip install ultralytics`
or `uv add ultralytics` on whichever machine runs inference). Class names are
read directly from the trained checkpoint (`model.names`), not hardcoded, so
this does not depend on remembering Roboflow's export class order.

NOTE: YOLO gives axis-aligned boxes with no rotation estimate, so
`BlockDetection.angle_deg` is always 0.0 here (the OpenCV detector's
`angle_deg` came from a rotated min-area-rect; nothing downstream in this
project currently reads it).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from lerobot.grad_project.perception.opencv_block_detector import (
    BlockDetection,
    DetectionResult,
    _ensure_bgr,
    _point_in_polygon,
)

DEFAULT_CONF = 0.5
DEFAULT_IOU = 0.5


def _as_polygon(points: Any) -> np.ndarray | None:
    if not points:
        return None
    return np.array(points, dtype=np.float32)


def _shrink_polygon(polygon: np.ndarray | None, inset_px: float) -> np.ndarray | None:
    """Pull every vertex `inset_px` toward the polygon's centroid."""
    if polygon is None or inset_px <= 0:
        return polygon
    centre = polygon.mean(axis=0)
    shrunk = []
    for vertex in polygon:
        direction = centre - vertex
        distance = float(np.linalg.norm(direction))
        if distance <= inset_px:
            return polygon  # inset larger than the zone; leave it alone
        shrunk.append(vertex + direction / distance * inset_px)
    return np.array(shrunk, dtype=np.float32)


class YoloBlockDetector:
    """Runs a trained Ultralytics YOLO checkpoint and maps results onto the
    same `DetectionResult` shape the OpenCV pipeline produces."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        detector_config: dict[str, Any] | None = None,
        frame_color: str = "bgr",
        conf: float = DEFAULT_CONF,
        iou: float = DEFAULT_IOU,
        device: str | None = None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "ultralytics is not installed. Run `pip install ultralytics` "
                "(or `uv add ultralytics`) in this environment first."
            ) from e

        model_path = Path(model_path).expanduser()
        if not model_path.is_file():
            raise FileNotFoundError(
                f"YOLO weights not found: {model_path}. Train locally first "
                "(see project/scripts/tools/prep_yolo_dataset.py) or download "
                "the .pt from Roboflow, then point --yolo_model_path at it."
            )

        self.model = YOLO(str(model_path))
        self.conf = conf
        self.iou = iou
        self.device = device
        self.frame_color = frame_color

        cfg = detector_config or {}
        self.cfg = cfg
        self.target_polygon = _as_polygon(cfg.get("target_polygon"))
        self.workspace_polygon = _as_polygon(cfg.get("workspace_polygon"))
        self.prefer_colors: list[str] = list(cfg.get("prefer_colors", []))
        # A block straddling the boundary has its centre just inside, so a
        # plain point-in-polygon test calls it placed and nothing ever tidies
        # it. Shrink the polygon so only properly-seated blocks count.
        # Keep this small: too large and a block the arm just placed near the
        # front row reads as outside, so the arm picks up its own work and
        # puts it back forever.
        self.target_inset_px = float(cfg.get("target_inset_px", 9.0))
        self._inset_polygon = _shrink_polygon(self.target_polygon, self.target_inset_px)

    @classmethod
    def load(
        cls,
        model_path: str | Path,
        detector_config_path: str | Path,
        *,
        frame_color: str = "bgr",
        conf: float = DEFAULT_CONF,
        iou: float = DEFAULT_IOU,
        device: str | None = None,
    ) -> "YoloBlockDetector":
        resolved = Path(detector_config_path).expanduser().resolve()
        with resolved.open("r", encoding="utf-8") as file:
            cfg = json.load(file)
        return cls(
            model_path,
            detector_config=cfg,
            frame_color=frame_color,
            conf=conf,
            iou=iou,
            device=device,
        )

    def detect(self, frame: np.ndarray) -> DetectionResult:
        frame_bgr = _ensure_bgr(frame, self.frame_color)
        results = self.model.predict(
            source=frame_bgr,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            verbose=False,
        )
        result = results[0]
        names = result.names

        # The task always uses exactly one block of each colour, so a second
        # box of a colour we already have is by definition spurious -- a
        # shadow, a checkerboard square, part of the arm. Keeping only the
        # most confident box per colour makes those impossible to act on.
        best_box: dict[str, tuple[float, Any]] = {}
        for box in result.boxes if result.boxes is not None else []:
            color = str(names[int(box.cls[0].item())])
            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            # Anything outside the board is furniture, cabling or the robot's
            # own base -- acting on one would drive the arm off the table.
            if self.workspace_polygon is not None and not _point_in_polygon(cx, cy, self.workspace_polygon):
                continue
            if color in best_box and best_box[color][0] >= confidence:
                continue
            best_box[color] = (confidence, (x1, y1, x2, y2))

        blocks: list[BlockDetection] = []
        for color, (_confidence, (x1, y1, x2, y2)) in best_box.items():
            w, h = x2 - x1, y2 - y1
            cx, cy = x1 + w / 2.0, y1 + h / 2.0
            in_target = _point_in_polygon(cx, cy, self._inset_polygon)
            blocks.append(
                BlockDetection(
                    color=color,
                    cx=cx,
                    cy=cy,
                    area=w * h,
                    angle_deg=0.0,
                    in_target=in_target,
                    bbox=(int(round(x1)), int(round(y1)), int(round(w)), int(round(h))),
                )
            )

        outside = [block for block in blocks if not block.in_target]
        target_count = sum(1 for block in blocks if block.in_target)
        chosen = self.choose_next(outside)
        return DetectionResult(blocks, outside, target_count, chosen)

    def choose_next(self, outside: list[BlockDetection]) -> BlockDetection | None:
        if not outside:
            return None
        priority = {color: index for index, color in enumerate(self.prefer_colors)}
        return sorted(
            outside,
            key=lambda block: (priority.get(block.color, 999), -block.cy, block.cx),
        )[0]

    def draw(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        output = _ensure_bgr(frame, self.frame_color).copy()
        chosen = result.chosen
        for block in result.blocks:
            x, y, w, h = block.bbox
            is_chosen = (
                chosen is not None and abs(block.cx - chosen.cx) < 1.0 and abs(block.cy - chosen.cy) < 1.0
            )
            draw_color = (180, 180, 180) if block.in_target else (0, 255, 0)
            if is_chosen:
                draw_color = (0, 165, 255)
            cv2.rectangle(output, (x, y), (x + w, y + h), draw_color, 2)
            cv2.putText(
                output,
                f"{block.color} {'IN' if block.in_target else 'OUT'}",
                (x, max(15, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                draw_color,
                1,
            )
        cv2.putText(
            output,
            f"YOLO  target={result.target_count} outside={len(result.outside_blocks)}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )
        return output
