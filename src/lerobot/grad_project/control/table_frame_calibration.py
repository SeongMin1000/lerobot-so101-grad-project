#!/usr/bin/env python3
"""Table(cm) -> robot base frame(m) alignment.

`pixel_to_table.py` gets you from a camera pixel to a point on the table in
centimeters. `table_ik.py` needs a target in the robot's own base frame in
meters. Neither knows about the other -- this module is the missing link,
fit from a handful of physically-measured correspondence points (jog the
real arm's gripper tip to touch a known table point, save it).

The table surface is treated as a plane; the fit is a 2D->3D affine map
robot_xyz = M @ [x_cm/100, y_cm/100, 1]^T (M is 3x3: one row per output
axis, columns are [x_coeff, y_coeff, offset]). This absorbs the table's tilt
and the camera-to-robot rotation/translation in one shot -- no need to
measure them separately. 3 non-collinear points determine M exactly; more
points (recommended: the 4 target-zone corners + center) least-squares
average out measurement noise.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lerobot.grad_project.paths import lerobot_root

DEFAULT_CALIBRATION_PATH = "project/config/table_robot_calibration.json"


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else lerobot_root() / path


@dataclass
class CalibrationPoint:
    label: str
    table_x_cm: float
    table_y_cm: float
    robot_xyz_m: tuple[float, float, float]


def fit_table_to_robot_affine(points: list[CalibrationPoint]) -> np.ndarray:
    if len(points) < 3:
        raise ValueError(f"Need at least 3 calibration points, got {len(points)}")

    a = np.array(
        [[p.table_x_cm / 100.0, p.table_y_cm / 100.0, 1.0] for p in points],
        dtype=np.float64,
    )
    m = np.zeros((3, 3), dtype=np.float64)
    for axis in range(3):
        b = np.array([p.robot_xyz_m[axis] for p in points], dtype=np.float64)
        coeffs, *_ = np.linalg.lstsq(a, b, rcond=None)
        m[axis] = coeffs

    residuals = []
    for p in points:
        predicted = apply_table_to_robot(p.table_x_cm, p.table_y_cm, m)
        residuals.append(float(np.linalg.norm(predicted - np.array(p.robot_xyz_m))))
    max_residual_m = max(residuals) if residuals else 0.0
    if max_residual_m > 0.01:
        import logging

        logging.getLogger(__name__).warning(
            "Table->robot affine fit residual is %.1fmm (>10mm) for at least one "
            "point -- re-check the physically-measured points, table flatness, "
            "or add more points.",
            max_residual_m * 1000.0,
        )

    return m


def apply_table_to_robot(x_cm: float, y_cm: float, m: np.ndarray) -> np.ndarray:
    """Return [x, y, z] in the robot base frame (meters) for a table-frame (cm) point."""
    v = np.array([x_cm / 100.0, y_cm / 100.0, 1.0], dtype=np.float64)
    return m @ v


def save_calibration(
    m: np.ndarray,
    points: list[CalibrationPoint],
    out_path: str | Path = DEFAULT_CALIBRATION_PATH,
) -> Path:
    out_path = _resolve(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "affine_table_cm_to_robot_m": m.tolist(),
        "note": (
            "robot_xyz_m = M @ [x_cm/100, y_cm/100, 1]. Fit from physically "
            "measured correspondence points below; re-run the calibration "
            "wizard if the camera mount, table, or robot base is moved."
        ),
        "source_points": [
            {
                "label": p.label,
                "table_x_cm": p.table_x_cm,
                "table_y_cm": p.table_y_cm,
                "robot_xyz_m": list(p.robot_xyz_m),
            }
            for p in points
        ],
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return out_path


def load_calibration(path: str | Path = DEFAULT_CALIBRATION_PATH) -> np.ndarray:
    path = _resolve(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Table->robot calibration not found: {path}. Run "
            "`python -m lerobot.grad_project.tools.save_table_calibration_point` "
            "with the real arm first (need >=3 points, see its docstring)."
        )
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return np.array(data["affine_table_cm_to_robot_m"], dtype=np.float64)


def load_raw_points(path: str | Path) -> list[dict[str, Any]]:
    path = _resolve(path)
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as f:
        return json.load(f).get("points", [])
