"""Pixel (top camera) -> real-world table coordinate (cm) conversion.

Calibrated from the official target-zone rectangle, which the task rules
document as exactly 20cm (width) x 10cm (height) measured at its interior
boundary. The four corners of that rectangle in pixel space are already
recorded in project/config/detector.json as `target_polygon`.

World frame: origin at the target zone's top-left corner (as traversed in
the pixel polygon, clockwise), +x along the 20cm edge, +y along the 10cm
edge, both in centimeters. This is NOT the robot's own base frame -- a
separate robot-to-table alignment (done physically, with the real arm) is
still required to relate this table frame to robot joint space.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from lerobot.grad_project.paths import lerobot_root

TARGET_WIDTH_CM = 20.0
TARGET_HEIGHT_CM = 10.0

DEFAULT_CALIBRATION_PATH = "project/config/camera_calibration.json"
DEFAULT_DETECTOR_CONFIG_PATH = "project/config/detector.json"


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else lerobot_root() / path


def compute_homography_from_target_polygon(
    detector_config_path: str | Path = DEFAULT_DETECTOR_CONFIG_PATH,
) -> np.ndarray:
    """Compute the pixel->table(cm) homography from detector.json's target_polygon.

    target_polygon is [[x,y], ...] for 4 corners in clockwise order starting
    near top-left (matches how it's laid out in detector.json today: the
    first two points span the 20cm edge, the next two close the 10cm edge).
    """
    with _resolve(detector_config_path).open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    polygon = cfg.get("target_polygon")
    if not polygon or len(polygon) != 4:
        raise ValueError(f"target_polygon missing or not 4 points in {detector_config_path}: {polygon}")

    pixel_pts = np.array(polygon, dtype=np.float32)
    world_pts = np.array(
        [
            [0.0, 0.0],
            [TARGET_WIDTH_CM, 0.0],
            [TARGET_WIDTH_CM, TARGET_HEIGHT_CM],
            [0.0, TARGET_HEIGHT_CM],
        ],
        dtype=np.float32,
    )

    homography = cv2.getPerspectiveTransform(pixel_pts, world_pts)
    return homography


def save_calibration(
    homography: np.ndarray,
    out_path: str | Path = DEFAULT_CALIBRATION_PATH,
    source: str = "target_polygon (detector.json), 20x10cm official target zone",
) -> Path:
    out_path = _resolve(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "homography_pixel_to_table_cm": homography.tolist(),
        "world_frame_origin": "target zone top-left corner (per target_polygon clockwise order)",
        "world_frame_axes": "+x along 20cm edge, +y along 10cm edge, units cm",
        "source": source,
        "note": "This maps camera pixels to a TABLE frame, not the robot base frame. "
        "A separate robot-to-table alignment (done physically) is required before use in IK.",
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return out_path


def load_homography(calibration_path: str | Path = DEFAULT_CALIBRATION_PATH) -> np.ndarray:
    with _resolve(calibration_path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    return np.array(data["homography_pixel_to_table_cm"], dtype=np.float64)


def pixel_to_table_xy(px: float, py: float, homography: np.ndarray) -> tuple[float, float]:
    """Convert a single pixel (px, py) to (x_cm, y_cm) in the table frame."""
    pt = np.array([[[px, py]]], dtype=np.float64)
    world = cv2.perspectiveTransform(pt, homography)
    x, y = world[0, 0]
    return float(x), float(y)


def _self_test() -> None:
    h = compute_homography_from_target_polygon()
    print("homography:\n", h)

    with _resolve(DEFAULT_DETECTOR_CONFIG_PATH).open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    polygon = cfg["target_polygon"]

    print("\nsanity check: transforming the 4 known corners should give ~(0,0) (20,0) (20,10) (0,10)")
    for px, py in polygon:
        x, y = pixel_to_table_xy(px, py, h)
        print(f"  pixel ({px:.0f},{py:.0f}) -> table ({x:.2f}, {y:.2f}) cm")

    saved_path = save_calibration(h)
    print(f"\nsaved calibration to {saved_path}")


if __name__ == "__main__":
    _self_test()
