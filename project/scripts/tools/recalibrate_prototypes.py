#!/usr/bin/env python3
"""Recalibrate detector.json color prototypes using our own current-lighting
dataset images instead of the stale 2026-07-23 calibration, by sampling boxes
from the "blocks5" (cleanest, least-occluded) frames using the v2 auto-labels
as approximate locations, then taking robust median/MAD per color -- same
method as opencv_calibrate.py's _robust_prototype.
"""
import json
from pathlib import Path

import cv2
import numpy as np

FRAMES_DIR = Path("/home/eslab/lerobot/var/yolo_frames")
LABELS_DIR = Path("/home/eslab/lerobot/var/yolo_frames_labels")
DETECTOR_CONFIG = Path("/home/eslab/lerobot/project/config/detector.json")

CLASS_NAMES = ["red", "blue", "green", "yellow", "wood"]
ERODE_FRAC = 0.2  # shrink box by this fraction per side to avoid edge/shadow pixels


def box_feature(img_bgr, cx, cy, w, h):
    img_h, img_w = img_bgr.shape[:2]
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    bw, bh = x2 - x1, y2 - y1
    ex, ey = int(bw * ERODE_FRAC), int(bh * ERODE_FRAC)
    x1, y1 = max(0, x1 + ex), max(0, y1 + ey)
    x2, y2 = min(img_w, x2 - ex), min(img_h, y2 - ey)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = img_bgr[y1:y2, x1:x2]
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return np.array([np.median(lab[:, :, 1]), np.median(lab[:, :, 2]), np.median(hsv[:, :, 1])], dtype=np.float32)


def robust_prototype(samples):
    array = np.stack(samples, axis=0).astype(np.float32)
    center = np.median(array, axis=0)
    mad = np.median(np.abs(array - center), axis=0)
    scale_floor = np.array([4.0, 4.0, 8.0], dtype=np.float32)
    scale = np.maximum(1.4826 * mad, scale_floor)
    rc = [round(float(v), 4) for v in center]
    rs = [round(float(v), 4) for v in scale]
    return {
        "feature_order": ["lab_a", "lab_b", "hsv_s"],
        "feature_center": rc,
        "feature_scale": rs,
        "lab_ab_center": rc[:2],
        "lab_ab_scale": rs[:2],
        "hsv_s_center": rc[2],
        "hsv_s_scale": rs[2],
        "samples": [[round(float(v), 4) for v in item] for item in array],
        "sample_count": int(len(samples)),
    }


def main():
    samples_by_class = {name: [] for name in CLASS_NAMES}

    frame_paths = sorted(FRAMES_DIR.glob("*_blocks5.jpg"))
    print(f"using {len(frame_paths)} 'blocks5' frames as calibration source")

    for img_path in frame_paths:
        label_path = LABELS_DIR / (img_path.stem + ".txt")
        if not label_path.exists():
            continue
        lines = label_path.read_text().strip().splitlines()
        if not lines:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        for line in lines:
            parts = line.split()
            class_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:5])
            color = CLASS_NAMES[class_id]
            feat = box_feature(img, cx, cy, w, h)
            if feat is not None:
                samples_by_class[color].append(feat)

    for color, samples in samples_by_class.items():
        print(f"{color}: {len(samples)} samples")

    with DETECTOR_CONFIG.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    old_prototypes = cfg["color_classification"]["prototypes"]
    new_prototypes = {}
    for color in CLASS_NAMES:
        samples = samples_by_class[color]
        if len(samples) < 10:
            print(f"WARNING: too few samples for {color} ({len(samples)}), keeping old prototype")
            new_prototypes[color] = old_prototypes[color]
            continue
        new_prototypes[color] = robust_prototype(samples)
        old_center = old_prototypes[color]["feature_center"]
        new_center = new_prototypes[color]["feature_center"]
        print(f"{color}: old_center={old_center} -> new_center={new_center}")

    cfg["color_classification"]["prototypes"] = new_prototypes
    cfg["color_classification"]["calibrated_at"] = "2026-08-19T00:00:00 (recalibrated from dataset)"

    backup_path = DETECTOR_CONFIG.with_suffix(".json.bak_20260723")
    if not backup_path.exists():
        with backup_path.open("w", encoding="utf-8") as f:
            json.dump(json.load(DETECTOR_CONFIG.open()), f, indent=2)
        print(f"backed up old config to {backup_path}")

    with DETECTOR_CONFIG.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)
    print(f"wrote recalibrated prototypes to {DETECTOR_CONFIG}")


if __name__ == "__main__":
    main()
