#!/usr/bin/env python3
"""
Minimal OpenCV verifier for SO-101 + end-to-end policy (SmolVLA/ACT).

OpenCV responsibilities in this file
------------------------------------
1. Keep fixed geometry: workspace/target/5 target slots.
2. Compare the live top-camera image with an empty-board reference.
3. Decide only whether each fixed target slot is occupied.
4. Provide a stable success/failure signal to the FSM.

OpenCV intentionally DOES NOT:
- classify block colors,
- choose the next block,
- estimate a grasp pose,
- control the robot.

The policy handles perception + manipulation. The FSM keeps the fixed color order
(red -> yellow -> wood -> green -> blue), and this verifier checks whether the
corresponding destination slot became occupied.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import cv2
import numpy as np

from lerobot.grad_project.paths import lerobot_root


DEFAULT_CONFIG: dict[str, Any] = {
    # "rgb" when called with LeRobot camera observations, "bgr" for cv2.VideoCapture.
    "frame_color": "bgr",

    # Four target inner corners in TL, TR, BR, BL order.
    "target_polygon": None,

    # Five fixed destination slots inside the 20 cm x 10 cm target.
    # Physical-board layout:
    #   red | yellow | wood
    #     green  |  blue
    # The lower two-slot row uses the full target width (two equal half-width
    # cells) so its horizontal detection regions are wider.
    "slot_layout": "two_rows_3_2",
    "slot_labels": ["blue", "wood", "green", "yellow", "red"],
    "slot_inner_margin": 0.15,

    # Empty-board reference image. Capture it with the robot at observe pose and
    # all blocks outside the target.
    "reference_image": "../assets/opencv/target_empty_reference.png",

    # Difference segmentation. Chroma is weighted more than brightness so that
    # ordinary shadows do not dominate the result.
    "chroma_threshold": 16.0,
    "luma_threshold": 34.0,
    "luma_weight": 0.35,

    # Binary-mask cleanup.
    "blur_kernel": 5,
    "morph_kernel": 5,

    # Slot occupancy test. Tune these using --debug.
    "min_foreground_ratio": 0.08,
    "min_largest_contour_area": 350.0,
    "min_component_slot_overlap": 0.35,

    # A slot must show the same occupied state for this many frames before the
    # stable-check helper accepts it.
    "stable_frames": 6,
}


@dataclass(frozen=True)
class SlotStatus:
    index: int
    label: str
    occupied: bool
    foreground_ratio: float
    largest_contour_area: float
    polygon: list[list[float]]


@dataclass(frozen=True)
class VerificationResult:
    occupied_count: int
    occupied_labels: list[str]
    slots: list[SlotStatus]

    def to_json(self) -> dict[str, Any]:
        return {
            "occupied_count": self.occupied_count,
            "occupied_labels": self.occupied_labels,
            "slots": [asdict(slot) for slot in self.slots],
        }


def _deep_copy_jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _ensure_bgr(frame: np.ndarray, frame_color: str) -> np.ndarray:
    if frame is None or frame.size == 0:
        raise ValueError("Empty camera frame")
    frame_color = frame_color.lower()
    if frame_color == "bgr":
        return frame
    if frame_color == "rgb":
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    raise ValueError(f"frame_color must be 'rgb' or 'bgr', got {frame_color!r}")


def _as_quad(points: Any) -> np.ndarray:
    quad = np.asarray(points, dtype=np.float32)
    if quad.shape != (4, 2):
        raise ValueError("target_polygon must contain exactly four [x, y] points")
    return quad


def _interpolate(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return (1.0 - t) * a + t * b


def _quad_point(quad: np.ndarray, u: float, v: float) -> np.ndarray:
    """Map normalized target coordinates (u, v) into a perspective quad."""
    tl, tr, br, bl = quad
    top = _interpolate(tl, tr, u)
    bottom = _interpolate(bl, br, u)
    return _interpolate(top, bottom, v)


def _normalized_rect_to_quad(
    quad: np.ndarray,
    rect: tuple[float, float, float, float],
    inner_margin: float,
) -> np.ndarray:
    """Convert a normalized [u0, v0, u1, v1] cell into an image-space quad."""
    if not 0.0 <= inner_margin < 0.45:
        raise ValueError("slot_inner_margin must be in [0, 0.45)")

    u0, v0, u1, v1 = rect
    if not (0.0 <= u0 < u1 <= 1.0 and 0.0 <= v0 < v1 <= 1.0):
        raise ValueError(f"Invalid normalized slot rectangle: {rect}")

    du = (u1 - u0) * inner_margin
    dv = (v1 - v0) * inner_margin
    u0 += du
    u1 -= du
    v0 += dv
    v1 -= dv

    return np.asarray(
        [
            _quad_point(quad, u0, v0),
            _quad_point(quad, u1, v0),
            _quad_point(quad, u1, v1),
            _quad_point(quad, u0, v1),
        ],
        dtype=np.float32,
    )


def build_slot_polygons(
    quad: np.ndarray,
    labels: list[str],
    layout: str,
    inner_margin: float,
) -> list[np.ndarray]:
    """Build fixed slot polygons for the selected target layout."""
    count = len(labels)
    if count <= 0:
        raise ValueError("slot count must be positive")

    if layout == "two_rows_3_2":
        if count != 5:
            raise ValueError("two_rows_3_2 layout requires exactly five slot labels")

        # Normalized coordinates inside the target quadrilateral.
        # Physical board:
        # - Top: three equal cells across the full target width.
        # - Bottom: two equal half-width cells across the full target width.
        #
        # Because the camera observes the board from the opposite direction,
        # the physical lower two-slot row appears at the top of the camera
        # screen. Therefore the first two camera-view rectangles are widened;
        # the three-slot row remains unchanged.
        # Camera-view layout:
        #
        #        blue        wood
        #   green    yellow     red
        #
        # The physical board appears reversed because the camera
        # observes it from the opposite direction.
        rects = [
            # Camera screen top row: physical lower two-slot row.
            # Before: u=1/6..5/6 (each slot was only 1/3 of target width).
            # Now:    u=0..1     (each slot uses 1/2 of target width).
            (0.0, 0.0, 0.5, 0.5),  # two-slot row: camera left
            (0.5, 0.0, 1.0, 0.5),  # two-slot row: camera right

            # Camera screen bottom row: three slots
            (0.0,       0.5, 1.0 / 3.0, 1.0),  # green
            (1.0 / 3.0, 0.5, 2.0 / 3.0, 1.0),  # yellow
            (2.0 / 3.0, 0.5, 1.0,       1.0),  # red
        ]
    elif layout == "horizontal":
        rects = [(i / count, 0.0, (i + 1) / count, 1.0) for i in range(count)]
    elif layout == "vertical":
        rects = [(0.0, i / count, 1.0, (i + 1) / count) for i in range(count)]
    else:
        raise ValueError(
            "slot_layout must be 'two_rows_3_2', 'horizontal', or 'vertical'"
        )

    return [
        _normalized_rect_to_quad(quad, rect, inner_margin)
        for rect in rects
    ]


def polygon_mask(shape: tuple[int, int], polygon: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [polygon.astype(np.int32)], 255)
    return mask


def click_target_quad(frame_bgr: np.ndarray) -> list[list[int]]:
    """Click TL, TR, BR, BL target inner corners."""
    title = "Click target: TL -> TR -> BR -> BL, then ENTER (r: reset, q: quit)"
    points: list[list[int]] = []
    view = frame_bgr.copy()

    def redraw() -> None:
        nonlocal view
        view = frame_bgr.copy()
        for idx, p in enumerate(points):
            cv2.circle(view, tuple(p), 6, (0, 255, 255), -1)
            cv2.putText(
                view,
                str(idx + 1),
                (p[0] + 7, p[1] - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        if len(points) >= 2:
            cv2.polylines(
                view,
                [np.asarray(points, dtype=np.int32)],
                isClosed=(len(points) == 4),
                color=(0, 255, 255),
                thickness=2,
            )

    def on_mouse(event: int, x: int, y: int, flags: int, param: Any) -> None:
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([int(x), int(y)])
            redraw()

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(title, on_mouse)
    redraw()

    while True:
        cv2.imshow(title, view)
        key = cv2.waitKey(20) & 0xFF
        if key in (13, 10) and len(points) == 4:
            break
        if key == ord("r"):
            points.clear()
            redraw()
        if key in (27, ord("q")):
            cv2.destroyWindow(title)
            raise KeyboardInterrupt("Target calibration cancelled")

    cv2.destroyWindow(title)
    return points


class TargetOccupancyVerifier:
    def __init__(
        self,
        config: dict[str, Any] | None = None,
        *,
        config_path: str | Path | None = None,
        frame_color: str | None = None,
    ) -> None:
        cfg = _deep_copy_jsonable(DEFAULT_CONFIG)
        if config:
            cfg.update(config)
        if frame_color is not None:
            cfg["frame_color"] = frame_color

        self.cfg = cfg
        self.config_path = Path(config_path) if config_path is not None else None

        target = cfg.get("target_polygon")
        if target is None:
            raise ValueError("target_polygon is missing. Run with --calibrate-target first.")
        self.target_quad = _as_quad(target)

        labels = list(cfg.get("slot_labels", []))
        if not labels:
            raise ValueError("slot_labels must not be empty")
        self.slot_labels = labels
        # Backward compatibility: old configs may contain split_axis only.
        layout = str(cfg.get("slot_layout") or cfg.get("split_axis") or "two_rows_3_2")
        self.slot_polygons = build_slot_polygons(
            self.target_quad,
            labels=labels,
            layout=layout,
            inner_margin=float(cfg.get("slot_inner_margin", 0.08)),
        )

        reference_path = Path(str(cfg.get("reference_image", "../assets/opencv/target_empty_reference.png")))
        if not reference_path.is_absolute() and self.config_path is not None:
            reference_path = self.config_path.parent / reference_path
        self.reference_path = reference_path

        reference = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
        if reference is None:
            raise FileNotFoundError(
                f"Reference image not found: {reference_path}. "
                "Run with --capture-reference while the target is empty."
            )
        self.reference_bgr = reference
        self.reference_lab = self._preprocess_lab(reference)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        frame_color: str | None = None,
    ) -> "TargetOccupancyVerifier":
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cls(cfg, config_path=path, frame_color=frame_color)

    def _preprocess_lab(self, bgr: np.ndarray) -> np.ndarray:
        blur_kernel = int(self.cfg.get("blur_kernel", 5))
        if blur_kernel < 1:
            blur_kernel = 1
        if blur_kernel % 2 == 0:
            blur_kernel += 1
        blurred = cv2.GaussianBlur(bgr, (blur_kernel, blur_kernel), 0)
        return cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB).astype(np.float32)

    def _foreground_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        if frame_bgr.shape[:2] != self.reference_bgr.shape[:2]:
            raise ValueError(
                "Live frame size does not match reference: "
                f"live={frame_bgr.shape[:2]} reference={self.reference_bgr.shape[:2]}"
            )

        current_lab = self._preprocess_lab(frame_bgr)
        delta = np.abs(current_lab - self.reference_lab)
        luma = delta[:, :, 0]
        chroma = np.maximum(delta[:, :, 1], delta[:, :, 2])

        chroma_threshold = float(self.cfg.get("chroma_threshold", 16.0))
        luma_threshold = float(self.cfg.get("luma_threshold", 34.0))
        luma_weight = float(self.cfg.get("luma_weight", 0.35))

        # Color/material changes are trusted directly. Brightness-only changes must
        # be stronger because ordinary shadows mostly affect L.
        # Weighted LAB difference: chroma dominates, while luminance contributes
        # only partially. This keeps colored/wooden blocks while suppressing most
        # soft shadows. Very large luminance changes are accepted separately.
        score = chroma + luma * luma_weight
        foreground = (score >= chroma_threshold).astype(np.uint8) * 255

        # Also accept very large luma changes (useful for the wooden block on some
        # checkerboard squares), while rejecting small soft shadows.
        foreground[luma >= luma_threshold] = 255

        # Do not run morphology across the whole target here. If blocks in the
        # upper and lower rows are close, a global CLOSE operation can connect
        # them into one component. Cleanup is performed after clipping the mask
        # to each individual slot in verify().
        return foreground

    def verify(self, frame: np.ndarray) -> VerificationResult:
        """Evaluate every fixed slot independently.

        A foreground component is intentionally allowed to intersect multiple
        slots. Each slot receives only the pixels inside its own polygon before
        morphology and contour analysis. Therefore two vertically aligned
        blocks remain independently detectable even when their raw difference
        masks or shadows touch across the row boundary.
        """
        frame_bgr = _ensure_bgr(
            frame,
            str(self.cfg.get("frame_color", "bgr")),
        )
        foreground = self._foreground_mask(frame_bgr)

        min_ratio = float(
            self.cfg.get("min_foreground_ratio", 0.055)
        )
        min_area = float(
            self.cfg.get("min_largest_contour_area", 220.0)
        )

        slot_masks = [
            polygon_mask(foreground.shape, polygon)
            for polygon in self.slot_polygons
        ]

        # 목표 구역 밖의 변화는 제거한다.
        target_mask = polygon_mask(
            foreground.shape,
            self.target_quad,
        )
        target_foreground = cv2.bitwise_and(
            foreground,
            target_mask,
        )

        kernel_size = max(1, int(self.cfg.get("morph_kernel", 5)))
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

        # 1. Clean foreground inside target area
        cleaned_target = cv2.morphologyEx(target_foreground, cv2.MORPH_OPEN, kernel)
        cleaned_target = cv2.morphologyEx(cleaned_target, cv2.MORPH_CLOSE, kernel)

        # 2. Extract connected components (each candidate block)
        num_labels, labels_im, stats, _ = cv2.connectedComponentsWithStats(
            cleaned_target, connectivity=8
        )

        min_component_overlap = float(self.cfg.get("min_component_slot_overlap", 0.35))
        exclusive_slot_fgs = [np.zeros_like(cleaned_target) for _ in self.slot_polygons]

        # 3. Assign each block exclusively to the single slot with maximum overlap
        for k in range(1, num_labels):
            comp_area = stats[k, cv2.CC_STAT_AREA]
            if comp_area < min_area * 0.4:
                continue

            comp_mask = (labels_im == k).astype(np.uint8) * 255
            overlaps = [
                int(cv2.countNonZero(cv2.bitwise_and(comp_mask, s_mask)))
                for s_mask in slot_masks
            ]
            best_idx = int(np.argmax(overlaps))
            best_overlap = overlaps[best_idx]

            # Dominant assignment: only assign if largest overlap meets criteria
            if best_overlap >= min_area and (best_overlap / max(1, comp_area)) >= min_component_overlap:
                exclusive_slot_fgs[best_idx] = cv2.bitwise_or(
                    exclusive_slot_fgs[best_idx],
                    cv2.bitwise_and(comp_mask, slot_masks[best_idx])
                )

        statuses = []

        for index, (
            label,
            polygon,
            slot_mask,
            slot_fg,
        ) in enumerate(
            zip(
                self.slot_labels,
                self.slot_polygons,
                slot_masks,
                exclusive_slot_fgs,
                strict=True,
            )
        ):
            slot_pixels = int(cv2.countNonZero(slot_mask))
            changed_pixels = int(cv2.countNonZero(slot_fg))
            ratio = changed_pixels / max(1, slot_pixels)

            slot_contours, _ = cv2.findContours(
                slot_fg,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            largest = max(
                (
                    float(cv2.contourArea(c))
                    for c in slot_contours
                ),
                default=0.0,
            )

            occupied = (
                ratio >= min_ratio
                and largest >= min_area
            )

            statuses.append(
                SlotStatus(
                    index=index,
                    label=label,
                    occupied=occupied,
                    foreground_ratio=float(ratio),
                    largest_contour_area=float(largest),
                    polygon=polygon.tolist(),
                )
            )

        occupied_labels = [
            slot.label
            for slot in statuses
            if slot.occupied
        ]

        return VerificationResult(
            occupied_count=len(occupied_labels),
            occupied_labels=occupied_labels,
            slots=statuses,
        )

    def draw(self, frame: np.ndarray, result: VerificationResult) -> np.ndarray:
        bgr = _ensure_bgr(frame, str(self.cfg.get("frame_color", "bgr"))).copy()
        target = self.target_quad.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(bgr, [target], True, (255, 255, 255), 2)

        for slot, polygon in zip(result.slots, self.slot_polygons, strict=True):
            pts = polygon.astype(np.int32).reshape((-1, 1, 2))
            # Magenta means OCCUPIED; it is not a detected block color.\n            line_color = (255, 0, 255) if slot.occupied else (0, 165, 255)
            # 자주색: OCCUPIED / 주황색: EMPTY
            line_color = (
                (255, 0, 255)
                if slot.occupied
                else (0, 165, 255)
            )
            cv2.polylines(bgr, [pts], True, line_color, 2)

            center = np.mean(polygon, axis=0).astype(int)
            text = (
                f"{slot.index + 1}:{slot.label} "
                f"{'OCC' if slot.occupied else 'EMPTY'} "
                f"r={slot.foreground_ratio:.2f} a={slot.largest_contour_area:.0f}"
            )
            cv2.putText(
                bgr,
                text,
                (int(center[0]) - 55, int(center[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                line_color,
                1,
                cv2.LINE_AA,
            )

        cv2.putText(
            bgr,
            f"occupied={result.occupied_count}/{len(result.slots)}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return bgr

    def wait_for_stable_slot(
        self,
        frame_getter: Callable[[], np.ndarray],
        slot_label: str,
        *,
        expected_occupied: bool = True,
        timeout_s: float = 4.0,
        stable_frames: int | None = None,
        poll_interval_s: float = 0.05,
    ) -> tuple[bool, VerificationResult | None]:
        """Wait until one slot has the requested state for consecutive frames."""
        if slot_label not in self.slot_labels:
            raise KeyError(f"Unknown slot label {slot_label!r}; expected one of {self.slot_labels}")

        required = int(stable_frames or self.cfg.get("stable_frames", 6))
        deadline = time.monotonic() + timeout_s
        streak = 0
        last_result: VerificationResult | None = None

        while time.monotonic() < deadline:
            frame = frame_getter()
            last_result = self.verify(frame)
            status = next(slot for slot in last_result.slots if slot.label == slot_label)
            if status.occupied == expected_occupied:
                streak += 1
                if streak >= required:
                    return True, last_result
            else:
                streak = 0
            time.sleep(max(0.0, poll_interval_s))

        return False, last_result


def open_camera(device: str | int, width: int, height: int, fps: int, fourcc: str) -> cv2.VideoCapture:
    dev_target = int(device) if str(device).isdigit() else str(device)
    cap = cv2.VideoCapture(dev_target, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera: {device}")
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def read_frame(cap: cv2.VideoCapture) -> np.ndarray:
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError("Camera frame read failed")
    return frame


def load_or_default_config(path: Path) -> dict[str, Any]:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
    else:
        cfg = _deep_copy_jsonable(DEFAULT_CONFIG)
    return cfg


def save_config(path: Path, cfg: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="/dev/cam_top")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--fourcc", default="MJPG")
    parser.add_argument(
        "--config",
        default=str(lerobot_root() / "project/config/target_verifier.json"),
    )
    parser.add_argument(
        "--layout",
        choices=["two_rows_3_2", "horizontal", "vertical"],
        default=None,
        help="Target slot layout. Default: two_rows_3_2",
    )
    # Kept only so older commands do not break. Prefer --layout.
    parser.add_argument("--split-axis", choices=["horizontal", "vertical"], default=None)
    parser.add_argument("--calibrate-target", action="store_true")
    parser.add_argument("--capture-reference", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = load_or_default_config(config_path)
    cfg["frame_color"] = "bgr"
    if args.layout is not None:
        cfg["slot_layout"] = args.layout
    elif args.split_axis is not None:
        cfg["slot_layout"] = args.split_axis
    else:
        cfg.setdefault("slot_layout", "two_rows_3_2")

    # Remove the legacy key when saving a newly calibrated config.
    cfg.pop("split_axis", None)

    cap = open_camera(args.device, args.width, args.height, args.fps, args.fourcc)
    try:
        # Flush a few startup frames after camera exposure settles.
        for _ in range(12):
            frame = read_frame(cap)
            time.sleep(0.02)

        if args.calibrate_target:
            cfg["target_polygon"] = click_target_quad(frame)
            save_config(config_path, cfg)
            print(f"Saved target geometry to {config_path}")
            det_path = config_path.parent / "detector.json"
            if det_path.exists():
                try:
                    with det_path.open("r", encoding="utf-8") as f:
                        det_cfg = json.load(f)
                    det_cfg["target_polygon"] = cfg["target_polygon"]
                    save_json_atomic(det_path, det_cfg)
                    print(f"[OK] synced target_polygon to: {det_path}")
                except Exception as e:
                    print(f"[WARN] failed to sync detector.json: {e}")

        if args.capture_reference:
            if cfg.get("target_polygon") is None:
                raise ValueError("Calibrate the target before capturing the reference")
            reference_path = Path(str(cfg.get("reference_image", "../assets/opencv/target_empty_reference.png")))
            if not reference_path.is_absolute():
                reference_path = config_path.parent / reference_path
            reference_path.parent.mkdir(parents=True, exist_ok=True)
            frame = read_frame(cap)
            if not cv2.imwrite(str(reference_path), frame):
                raise RuntimeError(f"Failed to save reference image: {reference_path}")
            save_config(config_path, cfg)
            print(f"Saved empty-target reference to {reference_path}")

        if args.debug:
            verifier = TargetOccupancyVerifier(cfg, config_path=config_path, frame_color="bgr")
            while True:
                frame = read_frame(cap)
                result = verifier.verify(frame)
                overlay = verifier.draw(frame, result)
                cv2.imshow("SO-101 target occupancy verifier", overlay)
                print(json.dumps(result.to_json(), ensure_ascii=False), flush=True)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("b"):
                    # Refresh the empty-board reference without restarting.
                    if not cv2.imwrite(str(verifier.reference_path), frame):
                        raise RuntimeError(f"Failed to save reference image: {verifier.reference_path}")
                    verifier = TargetOccupancyVerifier(cfg, config_path=config_path, frame_color="bgr")
                    print(f"Updated reference: {verifier.reference_path}")

        if not (args.calibrate_target or args.capture_reference or args.debug):
            parser.print_help()
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
