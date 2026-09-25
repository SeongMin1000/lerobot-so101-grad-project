#!/usr/bin/env python3
"""Atomic JSON persistence without accumulating backup files."""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any


def _absolute_path(path: str | Path) -> Path:
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()


def save_json_atomic(
    path: str | Path,
    data: Any,
    *,
    backup_root: str | Path | None = None,
    indent: int = 2,
) -> None:
    """Validate, fsync, and atomically replace JSON. No backup files are created."""
    del backup_root
    target = _absolute_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=indent) + "\n"
    json.loads(payload)
    previous_mode = (target.stat().st_mode & 0o777) if target.exists() else 0o644
    temp_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=target.parent,
            prefix=f".{target.name}.", suffix=".tmp", delete=False,
        ) as tmp:
            tmp.write(payload)
            tmp.flush()
            os.fsync(tmp.fileno())
            temp_path = Path(tmp.name)
        os.chmod(temp_path, previous_mode)
        with temp_path.open("r", encoding="utf-8") as file:
            json.load(file)
        os.replace(temp_path, target)
        temp_path = None
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()
