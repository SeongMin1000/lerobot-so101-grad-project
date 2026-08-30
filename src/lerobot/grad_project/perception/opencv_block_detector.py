#!/usr/bin/env python3
"""Unified OpenCV top-camera block detector for SO-101.

Detection order:
    empty-scene background subtraction in LAB chroma
    -> pixel-level workspace and ignore masks
    -> contour and shape filtering
    -> color classification only inside each surviving candidate

This is the single active detector used by debug, record, ACT, and Diffusion.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from lerobot.grad_project.paths import lerobot_root
from lerobot.grad_project.config_io import save_json_atomic


DEFAULT_CONFIG: dict[str, Any] = {
    "version": "4.0",
    "detector_type": "checkerboard_aligned_contour_lab_bidir_luma_watershed",
    "frame_color": "bgr",
    "background_image": "project/assets/opencv/top_background.png",
    "hsv_ranges": {
        "yellow": [[16, 120, 80, 45, 255, 255]],
        "green": [[45, 35, 35, 92, 255, 255]],
        "blue": [[88, 45, 25, 132, 255, 240]],
        "red": [[0, 80, 45, 10, 255, 255], [170, 80, 45, 179, 255, 255]],
        "wood": [[8, 15, 60, 38, 150, 255]],
    },
    "target_polygon": None,
    "workspace_polygon": None,
    "ignore_polygons": [],
    "prefer_colors": ["red", "yellow", "wood", "green", "blue"],
    "background": {
        "blur_kernel": 5,
        "alignment": {
            "enabled": True,
            "motion": "translation",
            "iterations": 60,
            "epsilon": 0.0001,
            "gauss_filter_size": 5,
        },
        "lab_chroma_threshold": 8.5,
        "chroma_min_saturation": 8,
        "use_luma_difference": True,
        "white_background_l_min": 150.0,
        "black_background_l_max": 105.0,
        "lab_dark_threshold": 28.0,
        "lab_bright_threshold": 28.0,
        "luma_min_saturation": 18,
        "luma_min_chroma_difference": 4.0,
    },
    "morphology": {
        "open_kernel": 3,
        "close_kernel": 1,
        "dilate_iterations": 0,
    },
    "shape_filter": {
        "min_contour_area": 300.0,
        "max_contour_area": 12000.0,
        "max_aspect": 2.5,
        "min_solidity": 0.76,
        "min_extent": 0.45,
        "min_rectangularity": 0.42,
    },
    "outer_candidate": {
        "min_component_area": 280.0,
        "max_component_area": 18000.0,
        "min_component_width": 8,
        "min_component_height": 8,
        "max_component_aspect": 7.0,
        "min_component_solidity": 0.25,
        "min_component_extent": 0.16,
        "min_component_rectangularity": 0.14,
    },
    "color_classification": {
        "method": "lab_ab_s_topk",
        "feature_order": ["lab_a", "lab_b", "hsv_s"],
        "feature_weights": [1.0, 1.0, 1.35],
        "erode_kernel": 5,
        "prototypes": {},
        "require_calibrated_prototypes": True,
        "use_sample_prototypes": True,
        "sample_top_k": 3,
        "sample_scale_multiplier": 1.20,
        "max_normalized_distance": 4.5,
        "min_distance_margin": 0.25,
        "min_candidate_pixels": 25,
        "fallback_to_hsv_when_uncalibrated": False,
        "hue_margin": 0,
        "saturation_margin": 0,
        "value_margin": 0,
        "min_match_fraction": 0.12,
        "max_fallback_distance": 1.8,
    },
    "shape_calibration": {
        "single_block_area_samples": [],
        "single_block_area_median": None,
        "single_block_area_p95": None,
        "single_block_area_max": None,
        "single_block_width_median": None,
        "single_block_height_median": None,
        "sample_count": 0,
    },
    "split_touching": {
        "enabled": True,
        "strategy": "watershed_then_color",
        "watershed_enabled": True,
        "watershed_peak_ratio": 0.34,
        "watershed_marker_min_distance_factor": 0.38,
        "watershed_seed_radius_factor": 0.24,
        "watershed_max_blocks": 5,
        "watershed_min_coverage": 0.50,
        "watershed_blur_kernel": 3,
        "color_fallback_enabled": True,
        "split_pixel_top_k": 3,
        "gate_by_single_block_area": True,
        "fallback_min_outer_area": 1500.0,
        "single_block_split_factor": 1.25,
        "split_min_outer_aspect": 1.40,
        "minimum_colors": 2,
        "pixel_max_normalized_distance": 5.0,
        "pixel_min_distance_margin": 0.10,
        "open_kernel": 3,
        "close_kernel": 1,
        "min_component_area": 100.0,
        "max_component_area": 3000.0,
        "min_component_width": 7,
        "min_component_height": 7,
        "max_component_aspect": 3.5,
        "min_component_solidity": 0.48,
        "min_component_extent": 0.25,
        "min_component_rectangularity": 0.22,
        "min_component_fraction_of_single": 0.18,
        "max_component_fraction_of_single": 1.75,
        "min_component_center_distance": 10.0,
        "min_total_component_coverage": 0.42,
        "reject_large_unsplit_factor": 1.75,
        "reject_large_unsplit_above": 2800.0,
    },
}



@dataclass
class BlockDetection:
    color: str
    cx: float
    cy: float
    area: float
    angle_deg: float
    in_target: bool
    bbox: tuple[int, int, int, int]

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["bbox"] = list(self.bbox)
        return data


@dataclass
class DetectionResult:
    blocks: list[BlockDetection]
    outside_blocks: list[BlockDetection]
    target_count: int
    chosen: BlockDetection | None

    def to_json(self) -> dict[str, Any]:
        return {
            "blocks": [block.to_json() for block in self.blocks],
            "outside_blocks": [block.to_json() for block in self.outside_blocks],
            "target_count": self.target_count,
            "chosen": None if self.chosen is None else self.chosen.to_json(),
        }


@dataclass
class ShapeCandidate:
    contour: np.ndarray
    bbox: tuple[int, int, int, int]
    cx: float
    cy: float
    area: float
    angle_deg: float
    aspect: float
    solidity: float
    extent: float
    rectangularity: float
    color: str | None = None
    color_score: float = 0.0
    debug_reason: str = ""


def _deep_copy(data: Any) -> Any:
    return json.loads(json.dumps(data))


def _odd_kernel(value: int, *, allow_one: bool = True) -> int:
    value = max(1, int(value))
    if value % 2 == 0:
        value += 1
    if not allow_one:
        value = max(3, value)
    return value


def _as_polygon(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.int32)
    if array.ndim != 2 or array.shape[1] != 2 or len(array) < 3:
        return None
    return array


def _resolve_project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = lerobot_root() / path
    return path.resolve()


def _ensure_bgr(frame: np.ndarray, frame_color: str) -> np.ndarray:
    mode = frame_color.lower()
    if mode == "bgr":
        return frame
    if mode == "rgb":
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    raise ValueError(f"frame_color must be bgr or rgb, got {frame_color!r}")


def _point_in_polygon(x: float, y: float, polygon: np.ndarray | None) -> bool:
    if polygon is None:
        return False
    return cv2.pointPolygonTest(
        polygon.astype(np.float32),
        (float(x), float(y)),
        False,
    ) >= 0


def _polygon_mask(shape: tuple[int, int], polygon: np.ndarray | None) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if polygon is None:
        mask.fill(255)
    else:
        cv2.fillPoly(mask, [polygon.astype(np.int32)], 255)
    return mask


def _hsv_mask(hsv: np.ndarray, ranges: list[list[int]]) -> np.ndarray:
    result = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for values in ranges:
        if len(values) != 6:
            raise ValueError(f"HSV range must have six integers, got {values}")
        low = np.asarray(values[:3], dtype=np.uint8)
        high = np.asarray(values[3:], dtype=np.uint8)
        result = cv2.bitwise_or(result, cv2.inRange(hsv, low, high))
    return result


def _expand_hsv_ranges(
    ranges: list[list[int]],
    *,
    hue_margin: int,
    saturation_margin: int,
    value_margin: int,
) -> list[list[int]]:
    expanded: list[list[int]] = []
    for h1, s1, v1, h2, s2, v2 in ranges:
        expanded.append(
            [
                max(0, h1 - hue_margin),
                max(0, s1 - saturation_margin),
                max(0, v1 - value_margin),
                min(179, h2 + hue_margin),
                min(255, s2 + saturation_margin),
                min(255, v2 + value_margin),
            ]
        )
    return expanded


def _distance_to_interval(value: float, low: float, high: float) -> float:
    if low <= value <= high:
        return 0.0
    return min(abs(value - low), abs(value - high))


def _median_hsv_distance(
    median_hsv: tuple[float, float, float],
    ranges: list[list[int]],
) -> float:
    h, s, v = median_hsv
    best = float("inf")
    for h1, s1, v1, h2, s2, v2 in ranges:
        hue_distance = _distance_to_interval(h, h1, h2) / 30.0
        saturation_distance = _distance_to_interval(s, s1, s2) / 100.0
        value_distance = _distance_to_interval(v, v1, v2) / 100.0
        best = min(
            best,
            math.sqrt(
                hue_distance**2
                + saturation_distance**2
                + value_distance**2
            ),
        )
    return best


def _save_image_atomic(path: Path, image: np.ndarray) -> None:
    """Atomically replace an image without creating timestamped backups."""
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    try:
        if not cv2.imwrite(str(temporary), image):
            raise RuntimeError(f"failed to write image: {temporary}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class TopBlockDetector:
    """Background/contour detector with color classification after shape filtering."""

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        frame_color: str | None = None,
        config_path: str | Path | None = None,
    ) -> None:
        cfg = _deep_copy(DEFAULT_CONFIG)
        if config:
            cfg.update(config)
        if frame_color is not None:
            cfg["frame_color"] = frame_color

        self.cfg = cfg
        self.config_path = None if config_path is None else Path(config_path).resolve()
        self.target_polygon = _as_polygon(cfg.get("target_polygon"))
        self.workspace_polygon = _as_polygon(cfg.get("workspace_polygon"))
        self.ignore_polygons = [
            polygon
            for polygon in (
                _as_polygon(item) for item in cfg.get("ignore_polygons", [])
            )
            if polygon is not None
        ]
        self.background_path = _resolve_project_path(cfg["background_image"])
        self._background_bgr: np.ndarray | None = None
        self.last_debug: dict[str, Any] = {}

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        frame_color: str | None = None,
    ) -> "TopBlockDetector":
        resolved = Path(path).expanduser().resolve()
        with resolved.open("r", encoding="utf-8") as file:
            config = json.load(file)
        if not isinstance(config, dict):
            raise TypeError(f"OpenCV config root must be a JSON object: {resolved}")
        return cls(config, frame_color=frame_color, config_path=resolved)

    def save(self, path: str | Path | None = None) -> Path | None:
        target = self.config_path if path is None else Path(path).expanduser().resolve()
        if target is None:
            raise ValueError("config path is required for save")
        return save_json_atomic(target, self.cfg)

    def _load_background(self, frame_shape: tuple[int, int, int]) -> np.ndarray:
        if self._background_bgr is None:
            background = cv2.imread(str(self.background_path), cv2.IMREAD_COLOR)
            if background is None:
                raise FileNotFoundError(
                    f"OpenCV Unified background image not found: {self.background_path}\n"
                    "Run: bash ~/lerobot/project/scripts/tools/grad_project.sh opencv-background"
                )
            self._background_bgr = background

        if self._background_bgr.shape != frame_shape:
            raise ValueError(
                "background/current frame shape mismatch: "
                f"background={self._background_bgr.shape}, current={frame_shape}. "
                "Recapture the background with: bash ~/lerobot/project/scripts/tools/grad_project.sh opencv-background"
            )
        return self._background_bgr

    def _alignment_mask(self, shape: tuple[int, int]) -> np.ndarray:
        mask = _polygon_mask(shape, self.workspace_polygon)
        for polygon in self.ignore_polygons:
            cv2.fillPoly(mask, [polygon.astype(np.int32)], 0)
        return mask

    def _align_current_to_background(
        self,
        current_bgr: np.ndarray,
        background_bgr: np.ndarray,
        alignment_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float | None, str]:
        background_cfg = self.cfg.get("background", {})
        alignment_cfg = background_cfg.get("alignment", {})
        if not bool(alignment_cfg.get("enabled", True)):
            return current_bgr, np.eye(2, 3, dtype=np.float32), None, "disabled"

        motion_name = str(alignment_cfg.get("motion", "translation")).lower()
        motion_map = {
            "translation": cv2.MOTION_TRANSLATION,
            "euclidean": cv2.MOTION_EUCLIDEAN,
            "affine": cv2.MOTION_AFFINE,
        }
        motion_type = motion_map.get(motion_name, cv2.MOTION_TRANSLATION)
        warp = np.eye(2, 3, dtype=np.float32)
        iterations = max(10, int(alignment_cfg.get("iterations", 60)))
        epsilon = max(1e-7, float(alignment_cfg.get("epsilon", 0.0001)))
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            iterations,
            epsilon,
        )
        gauss_size = _odd_kernel(
            int(alignment_cfg.get("gauss_filter_size", 5)),
            allow_one=True,
        )

        template = cv2.cvtColor(background_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        moving = cv2.cvtColor(current_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        try:
            try:
                score, warp = cv2.findTransformECC(
                    template,
                    moving,
                    warp,
                    motion_type,
                    criteria,
                    alignment_mask,
                    gauss_size,
                )
            except (TypeError, cv2.error):
                score, warp = cv2.findTransformECC(
                    template,
                    moving,
                    warp,
                    motion_type,
                    criteria,
                    alignment_mask,
                )
            aligned = cv2.warpAffine(
                current_bgr,
                warp,
                (current_bgr.shape[1], current_bgr.shape[0]),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REPLICATE,
            )
            return aligned, warp, float(score), "ok"
        except cv2.error as error:
            return current_bgr, warp, None, f"failed:{error.code}"

    def _build_foreground_mask(
        self,
        current_bgr: np.ndarray,
    ) -> dict[str, Any]:
        """Create the actual contour input mask for a fixed checkerboard arena.

        The current frame is first aligned to the stored empty background. A
        pixel is kept when it has a strong chroma change, when it became darker
        on a bright checker square, or when it became brighter on a dark checker
        square. All evidence also requires enough current saturation so neutral
        shadows and shifted black/white grid edges are less likely to survive.
        """
        background_bgr = self._load_background(current_bgr.shape)
        background_cfg = self.cfg.get("background", {})
        blur_kernel = _odd_kernel(background_cfg.get("blur_kernel", 5))

        alignment_mask = self._alignment_mask(current_bgr.shape[:2])
        aligned_bgr, warp, alignment_score, alignment_status = (
            self._align_current_to_background(
                current_bgr,
                background_bgr,
                alignment_mask,
            )
        )

        current_blurred = cv2.GaussianBlur(
            aligned_bgr,
            (blur_kernel, blur_kernel),
            0,
        )
        background_blurred = cv2.GaussianBlur(
            background_bgr,
            (blur_kernel, blur_kernel),
            0,
        )

        current_lab = cv2.cvtColor(current_blurred, cv2.COLOR_BGR2LAB).astype(np.float32)
        background_lab = cv2.cvtColor(background_blurred, cv2.COLOR_BGR2LAB).astype(np.float32)
        current_hsv = cv2.cvtColor(current_blurred, cv2.COLOR_BGR2HSV)
        saturation = current_hsv[:, :, 1]

        delta_a = current_lab[:, :, 1] - background_lab[:, :, 1]
        delta_b = current_lab[:, :, 2] - background_lab[:, :, 2]
        chroma_difference = cv2.magnitude(delta_a, delta_b)

        chroma_threshold = float(background_cfg.get("lab_chroma_threshold", 8.5))
        chroma_min_saturation = int(background_cfg.get("chroma_min_saturation", 8))
        chroma_boolean = (
            (chroma_difference >= chroma_threshold)
            & (saturation >= chroma_min_saturation)
        )

        background_l = background_lab[:, :, 0]
        current_l = current_lab[:, :, 0]
        dark_difference = background_l - current_l
        bright_difference = current_l - background_l
        dark_boolean = np.zeros(current_l.shape, dtype=bool)
        bright_boolean = np.zeros(current_l.shape, dtype=bool)

        use_luma = bool(
            background_cfg.get(
                "use_luma_difference",
                background_cfg.get("use_dark_difference", True),
            )
        )
        if use_luma:
            white_background = background_l >= float(
                background_cfg.get("white_background_l_min", 150.0)
            )
            black_background = background_l <= float(
                background_cfg.get("black_background_l_max", 105.0)
            )
            minimum_saturation = int(
                background_cfg.get(
                    "luma_min_saturation",
                    background_cfg.get("dark_min_saturation", 18),
                )
            )
            minimum_chroma = float(
                background_cfg.get(
                    "luma_min_chroma_difference",
                    background_cfg.get("dark_min_chroma_difference", 4.0),
                )
            )
            common_color_gate = (
                (saturation >= minimum_saturation)
                & (chroma_difference >= minimum_chroma)
            )
            dark_boolean = (
                white_background
                & (dark_difference >= float(background_cfg.get("lab_dark_threshold", 28.0)))
                & common_color_gate
            )
            bright_boolean = (
                black_background
                & (bright_difference >= float(background_cfg.get("lab_bright_threshold", 28.0)))
                & common_color_gate
            )

        workspace_mask = self._alignment_mask(current_l.shape)
        workspace_boolean = workspace_mask > 0
        chroma_boolean &= workspace_boolean
        dark_boolean &= workspace_boolean
        bright_boolean &= workspace_boolean

        chroma_mask = chroma_boolean.astype(np.uint8) * 255
        dark_mask = dark_boolean.astype(np.uint8) * 255
        bright_mask = bright_boolean.astype(np.uint8) * 255
        foreground_pre_morph = (
            chroma_boolean | dark_boolean | bright_boolean
        ).astype(np.uint8) * 255

        morphology = self.cfg.get("morphology", {})
        foreground = foreground_pre_morph.copy()
        open_kernel = _odd_kernel(morphology.get("open_kernel", 3))
        close_kernel = _odd_kernel(morphology.get("close_kernel", 1))
        if open_kernel > 1:
            foreground = cv2.morphologyEx(
                foreground,
                cv2.MORPH_OPEN,
                np.ones((open_kernel, open_kernel), dtype=np.uint8),
            )
        if close_kernel > 1:
            foreground = cv2.morphologyEx(
                foreground,
                cv2.MORPH_CLOSE,
                np.ones((close_kernel, close_kernel), dtype=np.uint8),
            )

        dilate_iterations = int(morphology.get("dilate_iterations", 0))
        if dilate_iterations > 0:
            foreground = cv2.dilate(
                foreground,
                np.ones((3, 3), dtype=np.uint8),
                iterations=dilate_iterations,
            )

        return {
            "aligned_bgr": aligned_bgr,
            "foreground_mask": foreground,
            "foreground_pre_morph": foreground_pre_morph,
            "chroma_difference": chroma_difference,
            "dark_difference": dark_difference,
            "bright_difference": bright_difference,
            "chroma_mask": chroma_mask,
            "dark_mask": dark_mask,
            "bright_mask": bright_mask,
            "workspace_mask": workspace_mask,
            "alignment_warp": warp,
            "alignment_score": alignment_score,
            "alignment_status": alignment_status,
        }

    def _shape_candidate(self, contour: np.ndarray) -> ShapeCandidate | None:
        filters = self.cfg.get("shape_filter", {})
        area = float(cv2.contourArea(contour))
        if area < float(filters.get("min_contour_area", 300.0)):
            return None
        if area > float(filters.get("max_contour_area", 4000.0)):
            return None

        x, y, width, height = cv2.boundingRect(contour)
        if width <= 1 or height <= 1:
            return None

        aspect = max(width / height, height / width)
        if aspect > float(filters.get("max_aspect", 2.5)):
            return None

        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = area / hull_area if hull_area > 1e-6 else 0.0
        if solidity < float(filters.get("min_solidity", 0.76)):
            return None

        extent = area / float(width * height)
        if extent < float(filters.get("min_extent", 0.45)):
            return None

        rotated_rect = cv2.minAreaRect(contour)
        rect_width, rect_height = rotated_rect[1]
        rotated_area = float(rect_width * rect_height)
        rectangularity = area / rotated_area if rotated_area > 1e-6 else 0.0
        if rectangularity < float(filters.get("min_rectangularity", 0.42)):
            return None

        moments = cv2.moments(contour)
        if abs(moments["m00"]) < 1e-6:
            return None
        cx = float(moments["m10"] / moments["m00"])
        cy = float(moments["m01"] / moments["m00"])

        return ShapeCandidate(
            contour=contour,
            bbox=(int(x), int(y), int(width), int(height)),
            cx=cx,
            cy=cy,
            area=area,
            angle_deg=float(rotated_rect[2]),
            aspect=aspect,
            solidity=solidity,
            extent=extent,
            rectangularity=rectangularity,
        )

    def _candidate_inner_mask(
        self,
        image_shape: tuple[int, int],
        candidate: ShapeCandidate,
    ) -> np.ndarray:
        classification = self.cfg.get("color_classification", {})
        candidate_mask = np.zeros(image_shape, dtype=np.uint8)
        cv2.drawContours(candidate_mask, [candidate.contour], -1, 255, thickness=-1)
        erode_kernel = _odd_kernel(classification.get("erode_kernel", 5))
        eroded = cv2.erode(
            candidate_mask,
            np.ones((erode_kernel, erode_kernel), dtype=np.uint8),
            iterations=1,
        )
        minimum_pixels = int(classification.get("min_candidate_pixels", 25))
        if cv2.countNonZero(eroded) < minimum_pixels:
            return candidate_mask
        return eroded

    def _classify_color_hsv_fallback(
        self,
        hsv: np.ndarray,
        inner_mask: np.ndarray,
    ) -> tuple[str | None, float]:
        """Legacy HSV fallback used only before prototypes are calibrated."""
        classification = self.cfg.get("color_classification", {})
        pixel_count = max(1, cv2.countNonZero(inner_mask))
        hue_margin = int(classification.get("hue_margin", 0))
        saturation_margin = int(classification.get("saturation_margin", 0))
        value_margin = int(classification.get("value_margin", 0))

        scores: dict[str, float] = {}
        hsv_ranges: dict[str, list[list[int]]] = self.cfg.get("hsv_ranges", {})
        for color, ranges in hsv_ranges.items():
            expanded = _expand_hsv_ranges(
                ranges,
                hue_margin=hue_margin,
                saturation_margin=saturation_margin,
                value_margin=value_margin,
            )
            color_mask = _hsv_mask(hsv, expanded)
            overlap = cv2.bitwise_and(color_mask, inner_mask)
            scores[color] = cv2.countNonZero(overlap) / pixel_count

        if not scores:
            return None, 0.0

        priority = {
            color: index
            for index, color in enumerate(self.cfg.get("prefer_colors", []))
        }
        best_color, best_score = sorted(
            scores.items(),
            key=lambda item: (-item[1], priority.get(item[0], 999), item[0]),
        )[0]
        minimum_fraction = float(classification.get("min_match_fraction", 0.12))
        if best_score >= minimum_fraction:
            return best_color, best_score

        pixels = hsv[inner_mask > 0]
        if pixels.size == 0:
            return None, best_score
        median = tuple(float(value) for value in np.median(pixels, axis=0))
        distances = {
            color: _median_hsv_distance(median, ranges)
            for color, ranges in hsv_ranges.items()
        }
        fallback_color, fallback_distance = sorted(
            distances.items(),
            key=lambda item: (item[1], priority.get(item[0], 999), item[0]),
        )[0]
        maximum_distance = float(classification.get("max_fallback_distance", 1.8))
        if fallback_distance <= maximum_distance:
            return fallback_color, 1.0 / (1.0 + fallback_distance)
        return None, best_score

    def _prototype_feature_arrays(
        self,
        prototype: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Return only the current [LAB a, LAB b, HSV S] prototype format."""
        center = np.asarray(prototype.get("feature_center", []), dtype=np.float32)
        scale = np.asarray(prototype.get("feature_scale", []), dtype=np.float32)
        samples = np.asarray(prototype.get("samples", []), dtype=np.float32)
        if center.shape != (3,) or scale.shape != (3,):
            return None
        if samples.ndim != 2 or samples.shape[1:] != (3,) or len(samples) == 0:
            return None
        return center, np.maximum(scale, 1.0), samples

    def _validate_color_prototypes(self) -> None:
        classification = self.cfg.get("color_classification", {})
        if not bool(classification.get("require_calibrated_prototypes", True)):
            return
        prototypes = classification.get("prototypes", {})
        required = list(self.cfg.get("prefer_colors", []))
        missing: list[str] = []
        invalid: list[str] = []
        for color in required:
            prototype = prototypes.get(color)
            if not isinstance(prototype, dict):
                missing.append(color)
                continue
            if self._prototype_feature_arrays(prototype) is None:
                invalid.append(color)
        if missing or invalid:
            details = []
            if missing:
                details.append("missing=" + ",".join(missing))
            if invalid:
                details.append("legacy_or_invalid=" + ",".join(invalid))
            raise RuntimeError(
                "OpenCV color calibration is incomplete (" + "; ".join(details) + "). "
                "Run: bash ~/lerobot/project/scripts/tools/grad_project.sh opencv-colors"
            )

    @staticmethod
    def _top_k_mean(values: np.ndarray, top_k: int) -> float:
        if values.size == 0:
            return float("inf")
        count = min(max(1, int(top_k)), int(values.size))
        nearest = np.partition(values, count - 1)[:count]
        return float(np.mean(nearest))

    def _candidate_feature(
        self,
        hsv: np.ndarray,
        lab: np.ndarray,
        inner_mask: np.ndarray,
    ) -> np.ndarray | None:
        pixels_lab = lab[inner_mask > 0]
        pixels_hsv = hsv[inner_mask > 0]
        if pixels_lab.size == 0 or pixels_hsv.size == 0:
            return None
        return np.asarray(
            [
                float(np.median(pixels_lab[:, 1])),
                float(np.median(pixels_lab[:, 2])),
                float(np.median(pixels_hsv[:, 1])),
            ],
            dtype=np.float32,
        )

    def _classify_color(
        self,
        hsv: np.ndarray,
        lab: np.ndarray,
        candidate: ShapeCandidate,
    ) -> tuple[str | None, float, str]:
        """Classify with LAB a/b + HSV saturation and top-k sample distance."""
        classification = self.cfg.get("color_classification", {})
        inner_mask = self._candidate_inner_mask(hsv.shape[:2], candidate)
        feature3 = self._candidate_feature(hsv, lab, inner_mask)
        if feature3 is None:
            return None, 0.0, "no-candidate-pixels"

        prototypes = classification.get("prototypes", {})
        if not prototypes:
            if bool(classification.get("fallback_to_hsv_when_uncalibrated", True)):
                color, score = self._classify_color_hsv_fallback(hsv, inner_mask)
                return color, score, "hsv-fallback" if color else "hsv-no-match"
            return None, 0.0, "no-prototypes"

        use_samples = bool(classification.get("use_sample_prototypes", True))
        top_k = int(classification.get("sample_top_k", 3))
        scale_multiplier = max(
            0.1,
            float(classification.get("sample_scale_multiplier", 1.25)),
        )

        distances: dict[str, float] = {}
        for color, prototype in prototypes.items():
            arrays = self._prototype_feature_arrays(prototype)
            if arrays is None:
                continue
            center, scale, samples = arrays
            scale = np.maximum(scale * scale_multiplier, 1.0)
            feature = feature3
            weights = np.asarray(
                classification.get("feature_weights", [1.0, 1.0, 1.35]),
                dtype=np.float32,
            )
            if weights.shape != (3,):
                weights = np.asarray([1.0, 1.0, 1.35], dtype=np.float32)

            if use_samples and samples.ndim == 2 and len(samples) > 0:
                sample_distances = np.linalg.norm(
                    ((samples - feature.reshape(1, -1)) / scale.reshape(1, -1))
                    * weights.reshape(1, -1),
                    axis=1,
                )
                distances[str(color)] = self._top_k_mean(sample_distances, top_k)
            else:
                distances[str(color)] = float(
                    np.linalg.norm(((feature - center) / scale) * weights)
                )

        if not distances:
            return None, 0.0, "invalid-prototypes"

        ranked = sorted(distances.items(), key=lambda item: item[1])
        best_color, best_distance = ranked[0]
        second_distance = ranked[1][1] if len(ranked) > 1 else float("inf")
        maximum_distance = float(classification.get("max_normalized_distance", 4.5))
        minimum_margin = float(classification.get("min_distance_margin", 0.25))
        margin = second_distance - best_distance
        diagnostics = (
            f"best={best_color}:{best_distance:.2f} "
            f"second={second_distance:.2f} margin={margin:.2f} "
            f"a={feature3[0]:.0f} b={feature3[1]:.0f} s={feature3[2]:.0f}"
        )

        if best_distance > maximum_distance:
            return None, 1.0 / (1.0 + best_distance), "far-color " + diagnostics
        if margin < minimum_margin:
            return None, 1.0 / (1.0 + best_distance), "ambiguous-color " + diagnostics
        return best_color, 1.0 / (1.0 + best_distance), diagnostics


    def _candidate_from_contour_with_limits(
        self,
        contour: np.ndarray,
        limits: dict[str, Any],
    ) -> ShapeCandidate | None:
        """Build a candidate with caller-provided relaxed limits.

        Touching blocks form one outer contour. After prototype-based pixel
        assignment, each color component can be partially clipped or have a
        slightly imperfect boundary, so split components use relaxed shape
        limits rather than the stricter outer-contour limits.
        """
        area = float(cv2.contourArea(contour))
        if area < float(limits.get("min_component_area", 120.0)):
            return None
        if area > float(limits.get("max_component_area", 2600.0)):
            return None

        x, y, width, height = cv2.boundingRect(contour)
        if width < int(limits.get("min_component_width", 8)):
            return None
        if height < int(limits.get("min_component_height", 8)):
            return None

        aspect = max(width / max(height, 1), height / max(width, 1))
        if aspect > float(limits.get("max_component_aspect", 3.2)):
            return None

        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = area / hull_area if hull_area > 1e-6 else 0.0
        if solidity < float(limits.get("min_component_solidity", 0.55)):
            return None

        extent = area / float(max(1, width * height))
        if extent < float(limits.get("min_component_extent", 0.30)):
            return None

        rotated_rect = cv2.minAreaRect(contour)
        rect_width, rect_height = rotated_rect[1]
        rotated_area = float(rect_width * rect_height)
        rectangularity = area / rotated_area if rotated_area > 1e-6 else 0.0
        if rectangularity < float(limits.get("min_component_rectangularity", 0.28)):
            return None

        moments = cv2.moments(contour)
        if abs(moments["m00"]) < 1e-6:
            return None

        return ShapeCandidate(
            contour=contour,
            bbox=(int(x), int(y), int(width), int(height)),
            cx=float(moments["m10"] / moments["m00"]),
            cy=float(moments["m01"] / moments["m00"]),
            area=area,
            angle_deg=float(rotated_rect[2]),
            aspect=aspect,
            solidity=solidity,
            extent=extent,
            rectangularity=rectangularity,
        )

    def _single_block_area_stats(self) -> tuple[float | None, float | None, float | None]:
        shape = self.cfg.get("shape_calibration", {})
        values = []
        for value in shape.get("single_block_area_samples", []):
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if number > 0:
                values.append(number)
        median = shape.get("single_block_area_median")
        p95 = shape.get("single_block_area_p95")
        maximum = shape.get("single_block_area_max")
        if values:
            array = np.asarray(values, dtype=np.float32)
            median = float(np.median(array))
            p95 = float(np.percentile(array, 95))
            maximum = float(np.max(array))
        def valid(value: Any) -> float | None:
            try:
                result = float(value)
            except (TypeError, ValueError):
                return None
            return result if result > 0 else None
        return valid(median), valid(p95), valid(maximum)

    def _should_attempt_split(self, candidate: ShapeCandidate) -> tuple[bool, float]:
        split_cfg = self.cfg.get("split_touching", {})
        if not bool(split_cfg.get("enabled", True)):
            return False, float("inf")
        if not bool(split_cfg.get("gate_by_single_block_area", True)):
            return True, 0.0

        aspect_threshold = float(split_cfg.get("split_min_outer_aspect", 1.55))
        if candidate.aspect >= aspect_threshold:
            return True, -aspect_threshold

        _, p95, maximum = self._single_block_area_stats()
        factor = float(split_cfg.get("single_block_split_factor", 1.35))
        if p95 is not None or maximum is not None:
            reference = max(value for value in (p95, maximum) if value is not None)
            threshold = reference * factor
        else:
            threshold = float(split_cfg.get("fallback_min_outer_area", 1700.0))
        return candidate.area >= threshold, threshold

    def _estimate_touching_count(self, candidate: ShapeCandidate) -> int:
        split_cfg = self.cfg.get("split_touching", {})
        maximum = max(2, int(split_cfg.get("watershed_max_blocks", 5)))
        single_median, _, _ = self._single_block_area_stats()
        if single_median is not None:
            estimate = int(round(candidate.area / max(single_median, 1.0)))
        else:
            estimate = int(round(max(2.0, candidate.aspect)))
        return int(np.clip(estimate, 2, maximum))

    @staticmethod
    def _nearest_mask_point(mask: np.ndarray, point_xy: np.ndarray) -> tuple[int, int] | None:
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return None
        distances = (xs.astype(np.float32) - point_xy[0]) ** 2 + (
            ys.astype(np.float32) - point_xy[1]
        ) ** 2
        index = int(np.argmin(distances))
        return int(xs[index]), int(ys[index])

    def _watershed_seed_points(
        self,
        mask: np.ndarray,
        candidate: ShapeCandidate,
        count: int,
    ) -> list[tuple[int, int]]:
        split_cfg = self.cfg.get("split_touching", {})
        distance = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        maximum = float(distance.max())
        if maximum <= 0:
            return []

        shape = self.cfg.get("shape_calibration", {})
        width_ref = float(shape.get("single_block_width_median") or 0.0)
        height_ref = float(shape.get("single_block_height_median") or 0.0)
        reference_size = min(value for value in (width_ref, height_ref) if value > 0) \
            if width_ref > 0 and height_ref > 0 else math.sqrt(max(candidate.area / count, 1.0))
        minimum_distance = max(6.0, reference_size * float(
            split_cfg.get("watershed_marker_min_distance_factor", 0.42)
        ))
        peak_ratio = float(split_cfg.get("watershed_peak_ratio", 0.38))
        dilated = cv2.dilate(distance, np.ones((3, 3), dtype=np.float32))
        peak_mask = (distance >= dilated - 1e-6) & (distance >= maximum * peak_ratio)
        peak_y, peak_x = np.nonzero(peak_mask)
        ranked = sorted(
            ((float(distance[y, x]), int(x), int(y)) for x, y in zip(peak_x, peak_y)),
            reverse=True,
        )

        points: list[tuple[int, int]] = []
        for _, x, y in ranked:
            if all(math.hypot(x - px, y - py) >= minimum_distance for px, py in points):
                points.append((x, y))
                if len(points) >= count:
                    return points

        # Full-edge contact can create one long distance-transform ridge. In that
        # case, place geometry-only seeds along the contour's principal axis.
        contour_points = candidate.contour.reshape(-1, 2).astype(np.float32)
        x0, y0, _, _ = candidate.bbox
        contour_points[:, 0] -= float(x0)
        contour_points[:, 1] -= float(y0)
        mean, eigenvectors, _ = cv2.PCACompute2(contour_points, mean=None)
        center = mean.reshape(2)
        axis = eigenvectors[0].reshape(2)
        projections = (contour_points - center.reshape(1, 2)) @ axis
        low, high = float(projections.min()), float(projections.max())
        for fraction in np.linspace(0.5 / count, 1.0 - 0.5 / count, count):
            target = center + axis * (low + fraction * (high - low))
            snapped = self._nearest_mask_point(mask, target)
            if snapped is None:
                continue
            if all(
                math.hypot(snapped[0] - px, snapped[1] - py) >= minimum_distance * 0.65
                for px, py in points
            ):
                points.append(snapped)
            if len(points) >= count:
                break
        return points[:count]

    def _split_candidate_by_watershed(
        self,
        current_bgr: np.ndarray,
        candidate: ShapeCandidate,
    ) -> list[ShapeCandidate]:
        split_cfg = dict(self.cfg.get("split_touching", {}))
        if not bool(split_cfg.get("watershed_enabled", True)):
            return []

        count = self._estimate_touching_count(candidate)
        x, y, width, height = candidate.bbox
        full_mask = np.zeros(current_bgr.shape[:2], dtype=np.uint8)
        cv2.drawContours(full_mask, [candidate.contour], -1, 255, thickness=-1)
        roi_mask = full_mask[y : y + height, x : x + width]
        roi_bgr = current_bgr[y : y + height, x : x + width]
        if roi_mask.size == 0 or roi_bgr.size == 0:
            return []

        seed_points = self._watershed_seed_points(roi_mask, candidate, count)
        if len(seed_points) < 2:
            return []

        markers = np.zeros(roi_mask.shape, dtype=np.int32)
        markers[roi_mask == 0] = 1
        distance = cv2.distanceTransform(roi_mask, cv2.DIST_L2, 5)
        radius_factor = float(split_cfg.get("watershed_seed_radius_factor", 0.22))
        for index, (seed_x, seed_y) in enumerate(seed_points, start=2):
            radius = max(2, int(round(float(distance[seed_y, seed_x]) * radius_factor)))
            seed_mask = np.zeros_like(roi_mask)
            cv2.circle(seed_mask, (seed_x, seed_y), radius, 255, thickness=-1)
            seed_mask = cv2.bitwise_and(seed_mask, roi_mask)
            markers[seed_mask > 0] = index

        blur_kernel = _odd_kernel(split_cfg.get("watershed_blur_kernel", 3))
        watershed_image = cv2.GaussianBlur(
            roi_bgr, (blur_kernel, blur_kernel), 0
        )
        cv2.watershed(watershed_image, markers)

        single_median, _, single_maximum = self._single_block_area_stats()
        if single_median is not None:
            split_cfg["min_component_area"] = max(
                float(split_cfg.get("min_component_area", 120.0)),
                single_median * float(split_cfg.get("min_component_fraction_of_single", 0.22)),
            )
        if single_maximum is not None:
            split_cfg["max_component_area"] = min(
                float(split_cfg.get("max_component_area", 2600.0)),
                single_maximum * float(split_cfg.get("max_component_fraction_of_single", 1.60)),
            )

        parts: list[ShapeCandidate] = []
        total_area = 0.0
        for label in range(2, 2 + len(seed_points)):
            component = ((markers == label) & (roi_mask > 0)).astype(np.uint8) * 255
            component_contours, _ = cv2.findContours(
                component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not component_contours:
                continue
            contour_roi = max(component_contours, key=cv2.contourArea)
            contour_full = contour_roi.copy()
            contour_full[:, 0, 0] += x
            contour_full[:, 0, 1] += y
            subcandidate = self._candidate_from_contour_with_limits(
                contour_full, split_cfg
            )
            if subcandidate is None:
                continue
            parts.append(subcandidate)
            total_area += subcandidate.area

        if len(parts) < 2:
            return []
        coverage = total_area / max(candidate.area, 1.0)
        if coverage < float(split_cfg.get("watershed_min_coverage", 0.62)):
            return []
        return parts

    def _split_candidate_by_color_prototypes(
        self,
        hsv: np.ndarray,
        lab: np.ndarray,
        candidate: ShapeCandidate,
    ) -> list[tuple[str, ShapeCandidate, float]]:
        """Fallback split using calibrated [LAB a, LAB b, HSV S] prototypes."""
        split_cfg = dict(self.cfg.get("split_touching", {}))
        if not bool(split_cfg.get("enabled", True)):
            return []

        single_median, _, single_maximum = self._single_block_area_stats()
        if single_median is not None:
            split_cfg["min_component_area"] = max(
                float(split_cfg.get("min_component_area", 120.0)),
                single_median * float(split_cfg.get("min_component_fraction_of_single", 0.22)),
            )
        if single_maximum is not None:
            split_cfg["max_component_area"] = min(
                float(split_cfg.get("max_component_area", 2600.0)),
                single_maximum * float(split_cfg.get("max_component_fraction_of_single", 1.60)),
            )

        prototypes = self.cfg.get("color_classification", {}).get("prototypes", {})
        top_k = int(split_cfg.get("split_pixel_top_k", 2))
        usable: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
        for color, prototype in prototypes.items():
            arrays = self._prototype_feature_arrays(prototype)
            if arrays is None:
                continue
            center, scale, samples = arrays
            usable.append((str(color), center, np.maximum(scale, 1.0), samples))
        if len(usable) < 2:
            return []

        x, y, width, height = candidate.bbox
        roi_lab = lab[y : y + height, x : x + width]
        roi_hsv = hsv[y : y + height, x : x + width]
        if roi_lab.size == 0 or roi_hsv.size == 0:
            return []
        feature3 = np.dstack(
            [roi_lab[:, :, 1], roi_lab[:, :, 2], roi_hsv[:, :, 1]]
        ).astype(np.float32)

        full_mask = np.zeros(lab.shape[:2], dtype=np.uint8)
        cv2.drawContours(full_mask, [candidate.contour], -1, 255, thickness=-1)
        roi_candidate = full_mask[y : y + height, x : x + width] > 0

        distance_maps: list[np.ndarray] = []
        for _, center, scale, samples in usable:
            feature = feature3
            if samples.ndim == 2 and len(samples) > 0:
                weights = np.asarray(
                    self.cfg.get("color_classification", {}).get(
                        "feature_weights", [1.0, 1.0, 1.35]
                    ),
                    dtype=np.float32,
                )
                if weights.shape != (3,):
                    weights = np.asarray([1.0, 1.0, 1.35], dtype=np.float32)
                maps = np.linalg.norm(
                    ((feature[:, :, None, :] - samples[None, None, :, :])
                    / scale[None, None, None, :])
                    * weights[None, None, None, :],
                    axis=3,
                )
                count = min(max(1, top_k), maps.shape[2])
                nearest = np.partition(maps, count - 1, axis=2)[:, :, :count]
                distance_maps.append(np.mean(nearest, axis=2))
            else:
                distance_maps.append(
                    np.linalg.norm(
                        ((feature - center.reshape(1, 1, -1))
                        / scale.reshape(1, 1, -1))
                        * weights.reshape(1, 1, -1),
                        axis=2,
                    )
                )
        distance_stack = np.stack(distance_maps, axis=2)
        ranked = np.sort(distance_stack, axis=2)
        best_index = np.argmin(distance_stack, axis=2)
        best_distance = ranked[:, :, 0]
        second_distance = ranked[:, :, 1]
        valid = (
            roi_candidate
            & (best_distance <= float(split_cfg.get("pixel_max_normalized_distance", 5.0)))
            & ((second_distance - best_distance) >= float(
                split_cfg.get("pixel_min_distance_margin", 0.10)
            ))
        )

        open_kernel = _odd_kernel(split_cfg.get("open_kernel", 3))
        close_kernel = _odd_kernel(split_cfg.get("close_kernel", 3))
        parts: list[tuple[str, ShapeCandidate, float]] = []
        for color_index, (color, _, _, _) in enumerate(usable):
            color_roi = ((valid & (best_index == color_index)).astype(np.uint8) * 255)
            color_roi = cv2.morphologyEx(
                color_roi, cv2.MORPH_OPEN,
                np.ones((open_kernel, open_kernel), dtype=np.uint8),
            )
            color_roi = cv2.morphologyEx(
                color_roi, cv2.MORPH_CLOSE,
                np.ones((close_kernel, close_kernel), dtype=np.uint8),
            )
            component_contours, _ = cv2.findContours(
                color_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            accepted = None
            accepted_roi = None
            for contour_roi in sorted(component_contours, key=cv2.contourArea, reverse=True):
                contour_full = contour_roi.copy()
                contour_full[:, 0, 0] += x
                contour_full[:, 0, 1] += y
                subcandidate = self._candidate_from_contour_with_limits(
                    contour_full, split_cfg
                )
                if subcandidate is not None:
                    accepted = subcandidate
                    accepted_roi = contour_roi
                    break
            if accepted is None or accepted_roi is None:
                continue
            component_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.drawContours(component_mask, [accepted_roi], -1, 255, thickness=-1)
            values = best_distance[component_mask > 0]
            median_distance = float(np.median(values)) if values.size else float("inf")
            score = 1.0 / (1.0 + median_distance)
            accepted.color = color
            accepted.color_score = score
            parts.append((color, accepted, score))

        distinct_colors = {color for color, _, _ in parts}
        if len(distinct_colors) < int(split_cfg.get("minimum_colors", 2)):
            return []
        total_component_area = sum(part.area for _, part, _ in parts)
        if total_component_area / max(candidate.area, 1.0) < float(
            split_cfg.get("min_total_component_coverage", 0.50)
        ):
            return []
        return parts

    def detect(self, frame: np.ndarray) -> DetectionResult:
        self._validate_color_prototypes()
        current_bgr = _ensure_bgr(frame, self.cfg.get("frame_color", "bgr"))
        debug_data = self._build_foreground_mask(current_bgr)
        aligned_bgr = debug_data["aligned_bgr"]
        foreground = debug_data["foreground_mask"]
        hsv = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2LAB)

        contours, _ = cv2.findContours(
            foreground,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        blocks: list[BlockDetection] = []
        accepted_candidates: list[ShapeCandidate] = []
        unknown_candidates: list[ShapeCandidate] = []
        split_groups: list[list[ShapeCandidate]] = []
        outer_limits = dict(self.cfg.get("outer_candidate", {}))

        def add_block(color: str, candidate: ShapeCandidate, score: float) -> None:
            candidate.color = color
            candidate.color_score = score
            in_target = _point_in_polygon(
                candidate.cx,
                candidate.cy,
                self.target_polygon,
            )
            blocks.append(
                BlockDetection(
                    color=color,
                    cx=candidate.cx,
                    cy=candidate.cy,
                    area=candidate.area,
                    angle_deg=candidate.angle_deg,
                    in_target=in_target,
                    bbox=candidate.bbox,
                )
            )
            accepted_candidates.append(candidate)

        for contour in contours:
            outer = self._candidate_from_contour_with_limits(contour, outer_limits)
            if outer is None:
                continue

            should_split, split_threshold = self._should_attempt_split(outer)
            if should_split:
                watershed_parts = self._split_candidate_by_watershed(
                    aligned_bgr,
                    outer,
                )
                if len(watershed_parts) >= 2:
                    classified: list[tuple[ShapeCandidate, str | None, float, str]] = []
                    for part in watershed_parts:
                        color, score, reason = self._classify_color(hsv, lab, part)
                        classified.append((part, color, score, reason))

                    counts: dict[str, int] = {}
                    for _, color, _, _ in classified:
                        if color is not None:
                            counts[color] = counts.get(color, 0) + 1

                    group: list[ShapeCandidate] = []
                    for part, color, score, reason in classified:
                        if color is not None and counts.get(color, 0) > 1:
                            color = None
                            reason = "duplicate-color-after-watershed " + reason
                        part.debug_reason = "watershed " + reason
                        group.append(part)
                        if color is None:
                            part.color = None
                            part.color_score = score
                            unknown_candidates.append(part)
                        else:
                            add_block(color, part, score)
                    split_groups.append(group)
                    # A successful physical split is never collapsed back into
                    # one ambiguous blob merely because one color is uncertain.
                    continue

                split_cfg = self.cfg.get("split_touching", {})
                split_parts = (
                    self._split_candidate_by_color_prototypes(hsv, lab, outer)
                    if bool(split_cfg.get("color_fallback_enabled", True))
                    else []
                )
                if split_parts:
                    group = []
                    for color, part, score in split_parts:
                        part.debug_reason = "color-fallback-split"
                        add_block(color, part, score)
                        group.append(part)
                    split_groups.append(group)
                    continue

            candidate = self._shape_candidate(contour)
            if candidate is None:
                if should_split:
                    outer.debug_reason = "shape-filter-rejected-after-split-failure"
                    unknown_candidates.append(outer)
                continue

            split_cfg = self.cfg.get("split_touching", {})
            _, _, single_maximum = self._single_block_area_stats()
            if single_maximum is not None:
                reject_large_above = single_maximum * float(
                    split_cfg.get("reject_large_unsplit_factor", 1.75)
                )
            else:
                reject_large_above = float(
                    split_cfg.get("reject_large_unsplit_above", 2800.0)
                )
            if should_split and candidate.area > reject_large_above:
                candidate.debug_reason = (
                    f"large-unsplit area={candidate.area:.0f} "
                    f"split_gate={split_threshold:.0f}"
                )
                unknown_candidates.append(candidate)
                continue

            color, score, reason = self._classify_color(hsv, lab, candidate)
            candidate.debug_reason = reason
            if color is None:
                candidate.color = None
                candidate.color_score = score
                unknown_candidates.append(candidate)
                continue
            add_block(color, candidate, score)

        outside = [block for block in blocks if not block.in_target]
        target_count = sum(1 for block in blocks if block.in_target)
        chosen = self.choose_next(outside)

        self.last_debug = {
            **debug_data,
            "accepted_candidates": accepted_candidates,
            "unknown_candidates": unknown_candidates,
            "split_groups": split_groups,
        }
        return DetectionResult(blocks, outside, target_count, chosen)

    def choose_next(self, outside: list[BlockDetection]) -> BlockDetection | None:
        if not outside:
            return None
        priority = {
            color: index
            for index, color in enumerate(self.cfg.get("prefer_colors", []))
        }
        return sorted(
            outside,
            key=lambda block: (
                priority.get(block.color, 999),
                -block.cy,
                block.cx,
            ),
        )[0]

    def draw(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        aligned = self.last_debug.get("aligned_bgr")
        if isinstance(aligned, np.ndarray):
            output = aligned.copy()
        else:
            output = _ensure_bgr(frame, self.cfg.get("frame_color", "bgr")).copy()

        if self.target_polygon is not None:
            cv2.polylines(
                output,
                [self.target_polygon.reshape((-1, 1, 2))],
                True,
                (0, 0, 255),
                2,
            )
        if self.workspace_polygon is not None:
            cv2.polylines(
                output,
                [self.workspace_polygon.reshape((-1, 1, 2))],
                True,
                (255, 255, 255),
                1,
            )
        for polygon in self.ignore_polygons:
            overlay = output.copy()
            cv2.fillPoly(overlay, [polygon], (30, 30, 30))
            output = cv2.addWeighted(overlay, 0.45, output, 0.55, 0)
            cv2.polylines(output, [polygon.reshape((-1, 1, 2))], True, (0, 0, 0), 2)

        chosen = result.chosen
        for block in result.blocks:
            x, y, width, height = block.bbox
            draw_color = (180, 180, 180) if block.in_target else (0, 255, 0)
            if (
                chosen is not None
                and abs(block.cx - chosen.cx) < 1.0
                and abs(block.cy - chosen.cy) < 1.0
            ):
                draw_color = (0, 255, 255)
            cv2.rectangle(
                output,
                (x, y),
                (x + width, y + height),
                draw_color,
                2,
            )
            cv2.circle(output, (int(block.cx), int(block.cy)), 4, draw_color, -1)
            cv2.putText(
                output,
                f"{block.color} {'IN' if block.in_target else 'OUT'}",
                (x, max(15, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                draw_color,
                1,
                cv2.LINE_AA,
            )

        for candidate in self.last_debug.get("unknown_candidates", []):
            x, y, width, height = candidate.bbox
            cv2.rectangle(
                output,
                (x, y),
                (x + width, y + height),
                (255, 0, 255),
                1,
            )
            reason = candidate.debug_reason or "unknown-color"
            short_reason = reason if len(reason) <= 44 else reason[:41] + "..."
            cv2.putText(
                output,
                short_reason,
                (x, max(15, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 0, 255),
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            output,
            f"OpenCV unified  target={result.target_count} outside={len(result.outside_blocks)}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return output

    def debug_images(self) -> dict[str, np.ndarray]:
        images: dict[str, np.ndarray] = {}
        keys = [
            ("foreground_mask", "foreground after morphology"),
            ("foreground_pre_morph", "foreground before morphology"),
            ("chroma_mask", "chroma threshold mask"),
            ("dark_mask", "dark-on-white threshold mask"),
            ("bright_mask", "bright-on-black threshold mask"),
            ("workspace_mask", "workspace mask"),
        ]
        for key, title in keys:
            image = self.last_debug.get(key)
            if isinstance(image, np.ndarray):
                images[title] = image

        chroma = self.last_debug.get("chroma_difference")
        if isinstance(chroma, np.ndarray):
            heat = np.clip(chroma * 8.0, 0, 255).astype(np.uint8)
            images["chroma difference heatmap"] = cv2.applyColorMap(
                heat,
                cv2.COLORMAP_TURBO,
            )
        dark = self.last_debug.get("dark_difference")
        if isinstance(dark, np.ndarray):
            heat = np.clip(np.maximum(dark, 0.0) * 5.0, 0, 255).astype(np.uint8)
            images["dark difference heatmap"] = cv2.applyColorMap(
                heat,
                cv2.COLORMAP_INFERNO,
            )
        bright = self.last_debug.get("bright_difference")
        if isinstance(bright, np.ndarray):
            heat = np.clip(np.maximum(bright, 0.0) * 5.0, 0, 255).astype(np.uint8)
            images["bright difference heatmap"] = cv2.applyColorMap(
                heat,
                cv2.COLORMAP_VIRIDIS,
            )
        return images


def click_polygon(
    frame_bgr: np.ndarray,
    title: str,
    number_of_points: int,
) -> list[list[int]]:
    points: list[list[int]] = []
    view = frame_bgr.copy()

    def redraw() -> None:
        nonlocal view
        view = frame_bgr.copy()
        for index, point in enumerate(points):
            cv2.circle(view, tuple(point), 5, (0, 255, 255), -1)
            cv2.putText(
                view,
                str(index + 1),
                tuple(point),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
        if len(points) >= 2:
            cv2.polylines(
                view,
                [np.asarray(points, dtype=np.int32)],
                False,
                (0, 255, 255),
                2,
            )

    def on_mouse(event: int, x: int, y: int, flags: int, param: Any) -> None:
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < number_of_points:
            points.append([int(x), int(y)])
            redraw()

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(title, on_mouse)
    print(
        f"Click {number_of_points} points clockwise. "
        "Enter=save, r=reset, q/Esc=cancel."
    )
    try:
        while True:
            cv2.imshow(title, view)
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 10) and len(points) == number_of_points:
                return points
            if key == ord("r"):
                points.clear()
                redraw()
            if key in (27, ord("q")):
                raise KeyboardInterrupt("polygon calibration cancelled")
    finally:
        cv2.destroyWindow(title)


def open_camera(
    device: str | int,
    width: int,
    height: int,
    fps: int,
    fourcc: str,
) -> cv2.VideoCapture:
    dev_target = int(device) if str(device).isdigit() else str(device)
    capture = cv2.VideoCapture(dev_target, cv2.CAP_V4L2)
    if not capture.isOpened():
        raise RuntimeError(f"failed to open camera: {device}")
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_FPS, fps)
    return capture


def capture_median_background(
    capture: cv2.VideoCapture,
    *,
    warmup_frames: int,
    sample_frames: int,
) -> np.ndarray:
    for _ in range(max(0, warmup_frames)):
        ok, _ = capture.read()
        if not ok:
            raise RuntimeError("camera read failed during background warmup")

    samples: list[np.ndarray] = []
    for index in range(max(3, sample_frames)):
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError("camera read failed during background capture")
        samples.append(frame)
        if index % 10 == 0:
            print(f"[CAPTURE] background frame {index + 1}/{sample_frames}")

    return np.median(np.stack(samples, axis=0), axis=0).astype(np.uint8)


def _self_test() -> int:
    import tempfile

    height, width = 480, 640
    background = np.empty((height, width, 3), dtype=np.uint8)
    square = 40
    for row in range(0, height, square):
        for column in range(0, width, square):
            value = 225 if ((row // square) + (column // square)) % 2 == 0 else 35
            background[row : row + square, column : column + square] = value
    cv2.rectangle(background, (245, 130), (387, 193), (25, 25, 25), 12)
    cv2.rectangle(background, (270, 360), (370, 479), (120, 35, 145), -1)
    current = background.copy()

    colors_hsv = {
        "red": (3, 210, 180),
        "yellow": (28, 210, 210),
        "wood": (23, 80, 170),
        "green": (68, 190, 150),
        "blue": (115, 200, 100),
    }
    positions = [(70, 80), (540, 80), (70, 300), (540, 300), (420, 230)]
    for (color, hsv_value), (x, y) in zip(colors_hsv.items(), positions, strict=True):
        hsv_patch = np.uint8([[hsv_value]])
        bgr_value = tuple(int(value) for value in cv2.cvtColor(hsv_patch, cv2.COLOR_HSV2BGR)[0, 0])
        cv2.rectangle(current, (x, y), (x + 34, y + 34), bgr_value, -1)

    # Luminance-only shadow: should be ignored by chroma-only background subtraction.
    cv2.rectangle(current, (120, 180), (210, 250), (175, 175, 175), -1)

    with tempfile.TemporaryDirectory(prefix="opencv_detector_selftest_") as directory:
        background_path = Path(directory) / "background.png"
        cv2.imwrite(str(background_path), background)
        config = _deep_copy(DEFAULT_CONFIG)
        config["background_image"] = str(background_path)
        config["workspace_polygon"] = [[0, 0], [639, 0], [639, 479], [0, 479]]
        config["ignore_polygons"] = [[[250, 350], [390, 350], [390, 479], [250, 479]]]
        config["color_classification"]["require_calibrated_prototypes"] = False
        config["color_classification"]["fallback_to_hsv_when_uncalibrated"] = True
        detector = TopBlockDetector(config, frame_color="bgr")
        result = detector.detect(current)
        detected = {block.color for block in result.blocks}
        expected = set(colors_hsv)
        if detected != expected:
            print("[SELF-TEST ERROR] expected:", sorted(expected))
            print("[SELF-TEST ERROR] detected:", sorted(detected))
            print(json.dumps(result.to_json(), indent=2))
            return 2

        # Touching red/yellow pair must be split into two detections although
        # RETR_EXTERNAL sees one outer foreground contour.
        touching = background.copy()
        touching_specs = {
            "red": ((430, 160), (3, 210, 180)),
            "yellow": ((464, 160), (28, 210, 210)),
        }
        for _, ((x, y), hsv_value) in touching_specs.items():
            hsv_patch = np.uint8([[hsv_value]])
            bgr_value = tuple(
                int(value)
                for value in cv2.cvtColor(hsv_patch, cv2.COLOR_HSV2BGR)[0, 0]
            )
            cv2.rectangle(touching, (x, y), (x + 34, y + 34), bgr_value, -1)

        # Calibrated prototypes for the synthetic colors.
        touching_config = _deep_copy(config)
        touching_config["color_classification"]["prototypes"] = {}
        touching_config["shape_calibration"] = {
            "single_block_area_samples": [1156.0, 1156.0, 1156.0],
            "single_block_area_median": 1156.0,
            "single_block_area_p95": 1156.0,
            "single_block_area_max": 1156.0,
            "sample_count": 3,
        }
        for color, (_, hsv_value) in touching_specs.items():
            hsv_patch = np.uint8([[hsv_value]])
            bgr_patch = cv2.cvtColor(hsv_patch, cv2.COLOR_HSV2BGR)
            lab_patch = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2LAB)[0, 0]
            feature = [
                float(lab_patch[1]),
                float(lab_patch[2]),
                float(hsv_value[1]),
            ]
            touching_config["color_classification"]["prototypes"][color] = {
                "feature_order": ["lab_a", "lab_b", "hsv_s"],
                "feature_center": feature,
                "feature_scale": [3.0, 3.0, 8.0],
                "lab_ab_center": feature[:2],
                "lab_ab_scale": [3.0, 3.0],
                "samples": [feature, feature, feature],
                "sample_count": 3,
            }

        touching_detector = TopBlockDetector(
            touching_config,
            frame_color="bgr",
        )
        touching_result = touching_detector.detect(touching)
        touching_colors = {block.color for block in touching_result.blocks}
        if touching_colors != {"red", "yellow"}:
            print("[SELF-TEST ERROR] touching split expected red+yellow")
            print("[SELF-TEST ERROR] detected:", sorted(touching_colors))
            print(json.dumps(touching_result.to_json(), indent=2))
            return 3

        print("[OK] synthetic self-test detected:", sorted(detected))
        print("[OK] watershed touching split detected:", sorted(touching_colors))
        return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="/dev/cam_top")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--fourcc", default="MJPG")
    parser.add_argument("--config", required=False)
    parser.add_argument("--frame-color", choices=["rgb", "bgr"], default="bgr")
    parser.add_argument("--capture-background", action="store_true")
    parser.add_argument("--background-frames", type=int, default=31)
    parser.add_argument("--warmup-frames", type=int, default=45)
    parser.add_argument("--countdown", type=int, default=3)
    parser.add_argument("--calibrate-target", action="store_true")
    parser.add_argument("--calibrate-workspace", action="store_true")
    parser.add_argument("--calibrate-ignore", action="store_true")
    parser.add_argument("--ignore-points", type=int, default=4)
    parser.add_argument("--clear-ignore-polygons", action="store_true")
    parser.add_argument("--exit-after-calibration", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--show-masks", action="store_true")
    parser.add_argument("--save-debug-dir")
    parser.add_argument("--max-debug-frames", type=int, default=50)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        return _self_test()

    config_path = (
        _resolve_project_path(args.config)
        if args.config
        else _resolve_project_path("project/config/detector.json")
    )
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    if not isinstance(config, dict):
        raise TypeError(f"detector config root must be an object: {config_path}")
    config["frame_color"] = args.frame_color

    print(f"[CONFIG] detector_config={config_path}")
    print(f"[CONFIG] background={_resolve_project_path(config['background_image'])}")

    capture = open_camera(args.device, args.width, args.height, args.fps, args.fourcc)
    try:
        for _ in range(15):
            ok, initial_frame = capture.read()
            if not ok:
                raise RuntimeError("failed to read initial frame")

        if args.capture_background:
            print("[IMPORTANT] Remove all five blocks and keep the robot in the observe pose.")
            for remaining in range(max(0, args.countdown), 0, -1):
                print(f"[CAPTURE] starting in {remaining}...")
                time.sleep(1.0)
            background = capture_median_background(
                capture,
                warmup_frames=args.warmup_frames,
                sample_frames=args.background_frames,
            )
            background_path = _resolve_project_path(config["background_image"])
            _save_image_atomic(background_path, background)
            config["background_captured_at"] = datetime.now().isoformat(timespec="seconds")
            config["background_shape"] = list(background.shape)
            save_json_atomic(config_path, config)
            print(f"[OK] background saved: {background_path}")
            return 0

        changed = False
        if args.calibrate_target:
            config["target_polygon"] = click_polygon(
                initial_frame,
                "OpenCV: click INNER target corners",
                4,
            )
            changed = True
        if args.calibrate_workspace:
            config["workspace_polygon"] = click_polygon(
                initial_frame,
                "OpenCV: click workspace polygon",
                4,
            )
            changed = True
        if args.clear_ignore_polygons:
            config["ignore_polygons"] = []
            changed = True
        if args.calibrate_ignore:
            polygon = click_polygon(
                initial_frame,
                "OpenCV: click robot/ignore polygon",
                max(3, args.ignore_points),
            )
            config["ignore_polygons"] = [polygon]
            changed = True
        if changed:
            save_json_atomic(config_path, config)
            print(f"[OK] detector calibration saved: {config_path}")
            if args.calibrate_target:
                tv_path = config_path.parent / "target_verifier.json"
                if tv_path.exists() and "target_polygon" in config:
                    try:
                        with tv_path.open("r", encoding="utf-8") as f:
                            tv_cfg = json.load(f)
                        tv_cfg["target_polygon"] = config["target_polygon"]
                        save_json_atomic(tv_path, tv_cfg)
                        print(f"[OK] synced target_polygon to: {tv_path}")
                    except Exception as e:
                        print(f"[WARN] failed to sync target_verifier.json: {e}")
            if args.exit_after_calibration:
                return 0

        detector = TopBlockDetector(
            config,
            frame_color=args.frame_color,
            config_path=config_path,
        )
        output_directory = None
        if args.save_debug_dir:
            output_directory = _resolve_project_path(args.save_debug_dir)
            output_directory.mkdir(parents=True, exist_ok=True)

        last_print = 0.0
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                print("camera read failed", file=sys.stderr)
                time.sleep(0.05)
                continue

            result = detector.detect(frame)
            debug = detector.draw(frame, result)

            now = time.time()
            if now - last_print >= 0.5:
                print(json.dumps(result.to_json(), ensure_ascii=False))
                last_print = now

            if output_directory is not None and frame_index % 15 == 0:
                ring_size = max(1, int(args.max_debug_frames))
                slot = (frame_index // 15) % ring_size
                cv2.imwrite(str(output_directory / f"overlay_{slot:03d}.jpg"), debug)
                foreground = detector.last_debug.get("foreground_mask")
                if foreground is not None:
                    cv2.imwrite(
                        str(output_directory / f"mask_{slot:03d}.png"),
                        foreground,
                    )
            frame_index += 1

            if args.debug:
                cv2.imshow("top detector", debug)
                if args.show_masks:
                    for title, image in detector.debug_images().items():
                        cv2.imshow(title, image)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
            else:
                time.sleep(1.0 / max(1, args.fps))
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
