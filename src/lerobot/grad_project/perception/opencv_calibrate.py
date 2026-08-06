#!/usr/bin/env python3
"""Calibrate LAB a/b + HSV saturation prototypes for the unified OpenCV detector.

Place only one requested color block in the workspace. For each sample, move the
block to the suggested board region and press Enter. The script detects the
largest valid shape candidate without using any color label, then stores the
median [LAB a, LAB b, HSV saturation] feature. Multiple positions are summarized with a robust median
and MAD scale so corner lighting variations are represented without overlapping
HSV ranges.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from lerobot.grad_project.paths import lerobot_root
from lerobot.grad_project.perception.opencv_block_detector import (
    TopBlockDetector,
    _odd_kernel,
    _resolve_project_path,
    open_camera,
)
from lerobot.grad_project.config_io import save_json_atomic

DEFAULT_COLORS = ["red", "yellow", "wood", "green", "blue"]
SUGGESTED_POSITIONS = [
    "가운데",
    "왼쪽 아래",
    "오른쪽 아래",
    "왼쪽 위",
    "오른쪽 위",
]


def _candidate_feature(
    detector: TopBlockDetector,
    frame_bgr: np.ndarray,
) -> tuple[np.ndarray, Any, np.ndarray, np.ndarray]:
    debug_data = detector._build_foreground_mask(frame_bgr)
    foreground = debug_data["foreground_mask"]
    aligned_bgr = debug_data["aligned_bgr"]
    contours, _ = cv2.findContours(
        foreground,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    candidates = []
    for contour in contours:
        candidate = detector._shape_candidate(contour)
        if candidate is not None:
            candidates.append(candidate)
    if not candidates:
        raise RuntimeError(
            "형상 후보를 찾지 못했습니다. foreground threshold mask와 배경/조명을 확인하세요."
        )
    candidate = max(candidates, key=lambda item: item.area)
    inner_mask = detector._candidate_inner_mask(aligned_bgr.shape[:2], candidate)
    lab = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2HSV)
    lab_pixels = lab[inner_mask > 0]
    hsv_pixels = hsv[inner_mask > 0]
    if len(lab_pixels) < 25 or len(hsv_pixels) < 25:
        raise RuntimeError(
            f"후보 내부 픽셀이 너무 적습니다: lab={len(lab_pixels)} hsv={len(hsv_pixels)}"
        )
    feature = np.asarray(
        [
            float(np.median(lab_pixels[:, 1])),
            float(np.median(lab_pixels[:, 2])),
            float(np.median(hsv_pixels[:, 1])),
        ],
        dtype=np.float32,
    )
    return feature, candidate, foreground, aligned_bgr


def _capture_one_sample(
    capture: cv2.VideoCapture,
    detector: TopBlockDetector,
    frames_per_sample: int,
) -> tuple[np.ndarray, float, tuple[int, int], np.ndarray]:
    features: list[np.ndarray] = []
    areas: list[float] = []
    widths: list[int] = []
    heights: list[int] = []
    last_overlay: np.ndarray | None = None
    for _ in range(max(3, frames_per_sample)):
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError("카메라 프레임 읽기 실패")
        try:
            feature, candidate, _, aligned_bgr = _candidate_feature(detector, frame)
        except RuntimeError:
            time.sleep(0.03)
            continue
        features.append(feature)
        areas.append(float(candidate.area))
        x, y, width, height = candidate.bbox
        widths.append(int(width))
        heights.append(int(height))
        overlay = aligned_bgr.copy()
        cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 255, 255), 2)
        cv2.putText(
            overlay,
            f"LAB ab=({feature[0]:.1f},{feature[1]:.1f}) S={feature[2]:.1f}",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
        )
        last_overlay = overlay
        time.sleep(0.03)
    if len(features) < max(2, frames_per_sample // 3):
        raise RuntimeError(
            f"안정적인 후보 프레임이 부족합니다: {len(features)}/{frames_per_sample}"
        )
    sample = np.median(np.stack(features, axis=0), axis=0)
    area = float(np.median(np.asarray(areas, dtype=np.float32)))
    width = int(round(float(np.median(np.asarray(widths, dtype=np.float32)))))
    height = int(round(float(np.median(np.asarray(heights, dtype=np.float32)))))
    assert last_overlay is not None
    return sample, area, (width, height), last_overlay


def _robust_prototype(samples: list[np.ndarray]) -> dict[str, Any]:
    array = np.stack(samples, axis=0).astype(np.float32)
    center = np.median(array, axis=0)
    mad = np.median(np.abs(array - center), axis=0)
    # Saturation is intentionally retained: yellow is normally much more
    # saturated than wood even when their LAB a/b values overlap.
    scale_floor = np.asarray([4.0, 4.0, 8.0], dtype=np.float32)
    scale = np.maximum(1.4826 * mad, scale_floor)
    rounded_center = [round(float(value), 4) for value in center]
    rounded_scale = [round(float(value), 4) for value in scale]
    return {
        "feature_order": ["lab_a", "lab_b", "hsv_s"],
        "feature_center": rounded_center,
        "feature_scale": rounded_scale,
        # Legacy keys are retained for easy inspection/backward compatibility.
        "lab_ab_center": rounded_center[:2],
        "lab_ab_scale": rounded_scale[:2],
        "hsv_s_center": rounded_center[2],
        "hsv_s_scale": rounded_scale[2],
        "samples": [[round(float(value), 4) for value in item] for item in array],
        "sample_count": int(len(samples)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="/dev/cam_top")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--fourcc", default="MJPG")
    parser.add_argument(
        "--config",
        default="project/config/detector.json",
    )
    parser.add_argument(
        "--colors",
        default=",".join(DEFAULT_COLORS),
        help="comma-separated color names",
    )
    parser.add_argument("--samples-per-color", type=int, default=5)
    parser.add_argument("--frames-per-sample", type=int, default=15)
    parser.add_argument("--warmup-frames", type=int, default=30)
    parser.add_argument("--preview-dir", default="var/calibration/opencv")
    args = parser.parse_args()

    config_path = _resolve_project_path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    detector = TopBlockDetector(config, frame_color="bgr", config_path=config_path)
    colors = [item.strip() for item in args.colors.split(",") if item.strip()]
    invalid = [color for color in colors if color not in DEFAULT_COLORS]
    if invalid:
        raise SystemExit(f"지원하지 않는 색: {invalid}")

    preview_dir = _resolve_project_path(args.preview_dir)
    preview_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_preview_dir: Path | None = Path(
        tempfile.mkdtemp(
            prefix=".opencv_calibration_tmp_",
            dir=preview_dir.parent,
        )
    )

    capture = open_camera(args.device, args.width, args.height, args.fps, args.fourcc)
    try:
        for _ in range(max(0, args.warmup_frames)):
            ok, _ = capture.read()
            if not ok:
                raise RuntimeError("카메라 warmup 실패")

        classification = config.setdefault("color_classification", {})
        prototypes: dict[str, Any] = {}
        classification["prototypes"] = prototypes
        classification["feature_order"] = ["lab_a", "lab_b", "hsv_s"]
        classification["requires_recalibration"] = True
        all_area_samples: list[float] = []
        all_width_samples: list[int] = []
        all_height_samples: list[int] = []

        print("\n[중요] 현재 캘리브레이션할 블록 하나만 종이 위에 두세요.")
        print("다른 4개 블록은 반드시 치우고, 카메라/조명/observe 상태를 고정하세요.")

        for color in colors:
            samples: list[np.ndarray] = []
            print("\n" + "=" * 72)
            print(f"[{color.upper()}] 캘리브레이션 시작")
            print("=" * 72)
            for index in range(max(1, args.samples_per_color)):
                position = SUGGESTED_POSITIONS[index % len(SUGGESTED_POSITIONS)]
                while True:
                    input(
                        f"{color} 블록만 '{position}'에 두고 Enter "
                        f"({index + 1}/{args.samples_per_color})..."
                    )
                    try:
                        sample, area, size, overlay = _capture_one_sample(
                            capture,
                            detector,
                            args.frames_per_sample,
                        )
                    except RuntimeError as error:
                        print(f"[RETRY] {error}")
                        continue
                    samples.append(sample)
                    all_area_samples.append(area)
                    all_width_samples.append(size[0])
                    all_height_samples.append(size[1])
                    if temp_preview_dir is None:
                        raise RuntimeError("temporary calibration directory is unavailable")
                    preview_path = temp_preview_dir / f"{color}_{index + 1}.jpg"
                    cv2.imwrite(str(preview_path), overlay)
                    print(
                        f"[OK] sample {index + 1}: "
                        f"a={sample[0]:.2f}, b={sample[1]:.2f}, S={sample[2]:.2f}, "
                        f"area={area:.1f}, bbox={size[0]}x{size[1]} "
                        f"preview={preview_path}"
                    )
                    break

            prototypes[color] = _robust_prototype(samples)
            print(f"[PROTOTYPE] {color}: {json.dumps(prototypes[color], ensure_ascii=False)}")

        area_array = np.asarray(all_area_samples, dtype=np.float32)
        width_array = np.asarray(all_width_samples, dtype=np.float32)
        height_array = np.asarray(all_height_samples, dtype=np.float32)
        if area_array.size:
            config["shape_calibration"] = {
                "single_block_area_samples": [round(float(v), 3) for v in area_array],
                "single_block_area_median": round(float(np.median(area_array)), 3),
                "single_block_area_p95": round(float(np.percentile(area_array, 95)), 3),
                "single_block_area_max": round(float(np.max(area_array)), 3),
                "single_block_width_median": round(float(np.median(width_array)), 3),
                "single_block_height_median": round(float(np.median(height_array)), 3),
                "sample_count": int(area_array.size),
            }
            print("[SHAPE MODEL]", json.dumps(config["shape_calibration"], ensure_ascii=False))

        classification["requires_recalibration"] = False
        classification["calibrated_colors"] = colors
        classification["calibrated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        save_json_atomic(config_path, config)

        if preview_dir.exists():
            shutil.rmtree(preview_dir)
        if temp_preview_dir is None:
            raise RuntimeError("temporary calibration directory is unavailable")
        os.replace(temp_preview_dir, preview_dir)
        temp_preview_dir = None

        print("\n[OK] prototype calibration saved:", config_path)
        print("[OK] calibration previews committed:", preview_dir)
        print("\n다음 실행:")
        print("bash ~/lerobot/project/scripts/tools/grad_project.sh opencv-debug")
        return 0
    finally:
        capture.release()
        cv2.destroyAllWindows()
        if temp_preview_dir is not None and temp_preview_dir.exists():
            shutil.rmtree(temp_preview_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
