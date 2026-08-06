#!/usr/bin/env python3
"""Canonical path handling for hybrid OpenCV/FSM configuration files.

Relative config paths are always resolved from the LeRobot repository root,
not from the shell's current working directory. Absolute paths remain valid.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

ROOT_ENV = "LEROBOT_ROOT"
RUNTIME_ENV = "LEROBOT_RUNTIME_CONFIG"
DETECTOR_ENV = "LEROBOT_DETECTOR_CALIB"

DEFAULT_RUNTIME_NAME = "project/config/runtime.json"
DEFAULT_DETECTOR_NAME = "project/config/detector.json"


def lerobot_root() -> Path:
    """Return the canonical LeRobot repository root."""
    configured = os.environ.get(ROOT_ENV)
    if configured:
        root = Path(configured).expanduser()
    else:
        # .../lerobot/src/lerobot/grad_project/paths.py
        root = Path(__file__).resolve().parents[3]

    root = root.resolve()
    if not (root / "pyproject.toml").is_file():
        raise FileNotFoundError(
            f"Invalid LeRobot root: {root}. "
            f"Set {ROOT_ENV}=/absolute/path/to/lerobot."
        )
    return root


def _resolve_project_file(
    value: str | Path | None,
    *,
    env_name: str,
    default_name: str,
    must_exist: bool,
    label: str,
) -> Path:
    raw: str | Path

    # Environment override is used only when the caller kept the default name
    # or passed an empty value. An explicit custom path always wins.
    if value is None or str(value).strip() in {"", default_name}:
        raw = os.environ.get(env_name, default_name)
    else:
        raw = value

    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = lerobot_root() / path
    path = path.resolve()

    if must_exist and not path.is_file():
        raise FileNotFoundError(
            f"{label} not found: {path}. "
            f"Canonical LeRobot root: {lerobot_root()}"
        )
    return path


def runtime_config_path(
    value: str | Path | None = DEFAULT_RUNTIME_NAME,
    *,
    must_exist: bool = True,
) -> Path:
    return _resolve_project_file(
        value,
        env_name=RUNTIME_ENV,
        default_name=DEFAULT_RUNTIME_NAME,
        must_exist=must_exist,
        label="runtime config",
    )


def detector_calib_path(
    value: str | Path | None = DEFAULT_DETECTOR_NAME,
    *,
    must_exist: bool = True,
) -> Path:
    return _resolve_project_file(
        value,
        env_name=DETECTOR_ENV,
        default_name=DEFAULT_DETECTOR_NAME,
        must_exist=must_exist,
        label="detector calibration",
    )


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return data


def check_paths() -> int:
    root = lerobot_root()
    runtime = runtime_config_path(must_exist=True)
    detector = detector_calib_path(must_exist=True)
    duplicate = lerobot_root() / "hybrid_runtime.json"

    print(f"LEROBOT_ROOT={root}")
    print(f"RUNTIME_CONFIG={runtime}")
    print(f"DETECTOR_CALIB={detector}")

    runtime_data = _load_json_object(runtime)
    detector_data = _load_json_object(detector)

    print("runtime poses:", list(runtime_data.get("poses", {})))
    print(
        "runtime pregrasp:",
        [item.get("label") for item in runtime_data.get("pregrasp_points", [])],
    )
    print("runtime drop_slots:", runtime_data.get("drop_slots", []))
    print("detector target_polygon:", detector_data.get("target_polygon"))
    print("detector workspace_polygon:", detector_data.get("workspace_polygon"))

    if duplicate.exists() and duplicate.resolve() != runtime:
        print(f"[ERROR] duplicate runtime config still exists: {duplicate}")
        return 2

    print("[OK] canonical config paths are valid and no package-local duplicate exists")
    return 0


if __name__ == "__main__":
    raise SystemExit(check_paths())
