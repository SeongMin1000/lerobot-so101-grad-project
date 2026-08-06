# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Small, lossless debug captures for the async robot-to-policy image path."""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        if value.dtype == torch.bfloat16:
            value = value.float()
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _is_image_array(array: np.ndarray) -> bool:
    if array.ndim == 3:
        return array.shape[-1] in (1, 3, 4) or array.shape[0] in (1, 3, 4)
    if array.ndim == 4:
        return array.shape[0] == 1 and (array.shape[-1] in (1, 3, 4) or array.shape[1] in (1, 3, 4))
    if array.ndim == 5:
        return array.shape[0] == 1 and (array.shape[-1] in (1, 3, 4) or array.shape[2] in (1, 3, 4))
    return False


def _to_hwc(array: np.ndarray) -> np.ndarray:
    """Select the first batch/latest time item and return an HWC array."""
    if array.ndim == 5:
        array = array[0, -1]
    elif array.ndim == 4:
        array = array[0]

    if array.ndim != 3:
        raise ValueError(f"Expected an image array after removing batch dimensions, got {array.shape}")

    if array.shape[-1] in (1, 3, 4):
        return array
    if array.shape[0] in (1, 3, 4):
        return np.moveaxis(array, 0, -1)
    raise ValueError(f"Cannot determine channel dimension for image shape {array.shape}")


def _preview_uint8(array_hwc: np.ndarray) -> tuple[np.ndarray, str]:
    """Make a viewable image without changing the array used for hashes/stats."""
    array = array_hwc.astype(np.float32, copy=False)
    min_value = float(np.nanmin(array))
    max_value = float(np.nanmax(array))

    if np.issubdtype(array_hwc.dtype, np.integer) or max_value > 1.0:
        preview = np.clip(array, 0, 255)
        preview_scale = "0..255"
    elif min_value >= 0.0:
        preview = np.clip(array, 0, 1) * 255.0
        preview_scale = "0..1"
    else:
        preview = (np.clip(array, -1, 1) + 1.0) * 127.5
        preview_scale = "-1..1"

    preview = np.rint(preview).astype(np.uint8)
    if preview.shape[-1] == 1:
        preview = np.repeat(preview, 3, axis=-1)
    elif preview.shape[-1] == 4:
        preview = preview[..., :3]
    return preview, preview_scale


def _safe_image_name(key: str) -> str:
    name = key.rsplit(".", 1)[-1]
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in name)


def save_image_mapping(
    values: dict[str, Any],
    capture_dir: str | Path,
    *,
    stage: str | None = None,
) -> Path:
    """Save every image-like value and exact numeric metadata.

    ``stage=None`` stores files directly in ``capture_dir`` and is used by the
    robot client. Server stages use subdirectories such as ``raw``, ``helper``
    and ``policy`` so one matched capture can show the complete path.
    """
    capture_dir = Path(capture_dir).expanduser()
    output_dir = capture_dir if stage is None else capture_dir / stage
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, dict[str, Any]] = {}
    for key, value in values.items():
        try:
            array = _as_numpy(value)
        except (TypeError, RuntimeError):
            continue
        if not _is_image_array(array):
            continue

        array_hwc = _to_hwc(array)
        if not np.isfinite(array_hwc).all():
            finite_min = finite_max = None
        else:
            finite_min = float(array_hwc.min())
            finite_max = float(array_hwc.max())

        contiguous = np.ascontiguousarray(array)
        digest = hashlib.sha256()
        digest.update(str(contiguous.dtype).encode())
        digest.update(str(tuple(contiguous.shape)).encode())
        digest.update(contiguous.tobytes())

        preview, preview_scale = _preview_uint8(array_hwc)
        image_name = _safe_image_name(key)
        Image.fromarray(preview).save(output_dir / f"{image_name}.png")

        rgb = preview[..., :3].astype(np.float64)
        metadata[key] = {
            "file": f"{image_name}.png",
            "shape": list(array.shape),
            "display_shape": list(preview.shape),
            "dtype": str(array.dtype),
            "min": finite_min,
            "max": finite_max,
            "channel_mean_rgb": [float(value) for value in rgb.mean(axis=(0, 1))],
            "channel_std_rgb": [float(value) for value in rgb.std(axis=(0, 1))],
            "preview_scale": preview_scale,
            "sha256": digest.hexdigest(),
        }

    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_dir
