"""Learn the grasp wrist angle from teleoperated demonstrations.

The pipeline already learns *where* a block is from demonstrations (the
pixel->robot homography in collect_grasp_corrections.py). What it never
learned is *how to hold the gripper* getting there: that came from taking one
old hand-saved pose ("A2"), reading its tilt, and reusing that single number
everywhere, nudged by a hand-written distance ramp. The operator's own report
is that a near-vertical descent grasps most reliably except right at the far
corners -- which is nothing like the ~57 deg-from-vertical that A2 encodes.

So learn the tilt the same way the positions were learned. Each taught grasp
now stores the full joint vector, and the angle between the gripper's approach
axis and straight down is read back out of it. That angle is mostly a function
of how far out the block is (close in the arm can come almost straight down;
at full stretch it cannot), so a fit over reach generalises from a handful of
demonstrations rather than needing one per position.

Orientation is then rebuilt parametrically from (azimuth, tilt):

    approach = sin(tilt) * radial - cos(tilt) * up      # tool z
    jaw_axis = up x radial                              # tool y, tangential
    tool_x   = jaw_axis x approach

At tilt=0 that is a pure overhead grasp; at tilt=90 it is horizontal. Azimuth
comes from the target itself and needs no learning -- the arm must face the
block regardless.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lerobot.grad_project.paths import lerobot_root

SAMPLES_PATH = "project/config/grasp_correction_samples.json"


def orientation_from_tilt(target_xyz_m: np.ndarray, tilt_deg: float) -> np.ndarray:
    """Gripper rotation facing `target_xyz_m`, tilted `tilt_deg` off vertical."""
    radial = np.array([target_xyz_m[0], target_xyz_m[1], 0.0], dtype=float)
    radial /= np.linalg.norm(radial)
    up = np.array([0.0, 0.0, 1.0])

    tilt = np.deg2rad(tilt_deg)
    approach = np.sin(tilt) * radial - np.cos(tilt) * up
    approach /= np.linalg.norm(approach)

    jaw_axis = np.cross(up, radial)
    jaw_axis /= np.linalg.norm(jaw_axis)

    tool_x = np.cross(jaw_axis, approach)
    tool_x /= np.linalg.norm(tool_x)

    return np.column_stack([tool_x, jaw_axis, approach])


class GraspPitchModel:
    """Tilt-vs-reach fitted from taught grasps, with linear extrapolation."""

    def __init__(self, reaches_m: np.ndarray, tilts_deg: np.ndarray):
        order = np.argsort(reaches_m)
        self.reaches_m = np.asarray(reaches_m, dtype=float)[order]
        self.tilts_deg = np.asarray(tilts_deg, dtype=float)[order]
        # A straight line through the samples is the extrapolation rule outside
        # the taught span; inside it, interpolate the samples directly so the
        # model reproduces what was actually demonstrated.
        self.slope, self.intercept = np.polyfit(self.reaches_m, self.tilts_deg, 1)

    @classmethod
    def from_samples(cls, samples: list[dict]) -> "GraspPitchModel | None":
        usable = [s for s in samples if "tilt_from_vertical_deg" in s and "reach_m" in s]
        if len(usable) < 2:
            return None
        return cls(
            np.array([s["reach_m"] for s in usable]),
            np.array([s["tilt_from_vertical_deg"] for s in usable]),
        )

    @classmethod
    def load(cls, path: str | Path = SAMPLES_PATH) -> "GraspPitchModel | None":
        path = Path(path)
        if not path.is_absolute():
            path = lerobot_root() / path
        if not path.is_file():
            return None
        return cls.from_samples(json.loads(path.read_text()).get("samples", []))

    def tilt_for(self, reach_m: float) -> float:
        if reach_m < self.reaches_m[0] or reach_m > self.reaches_m[-1]:
            tilt = self.slope * reach_m + self.intercept
        else:
            tilt = float(np.interp(reach_m, self.reaches_m, self.tilts_deg))
        return float(np.clip(tilt, 0.0, 89.0))

    def describe(self) -> str:
        pairs = ", ".join(
            f"{r:.2f}m->{t:.0f}deg" for r, t in zip(self.reaches_m, self.tilts_deg, strict=True)
        )
        return f"{len(self.reaches_m)} taught grasps [{pairs}]"


def _self_test() -> None:
    # A tilt of 0 must point the approach axis straight down, 90 straight out.
    target = np.array([0.30, 0.10, 0.03])
    for tilt, expected_z in ((0.0, -1.0), (90.0, 0.0)):
        rotation = orientation_from_tilt(target, tilt)
        assert abs(rotation[2, 2] - expected_z) < 1e-9, (tilt, rotation[:, 2])
        assert abs(np.linalg.det(rotation) - 1.0) < 1e-9, "not a rotation matrix"

    model = GraspPitchModel(np.array([0.18, 0.28, 0.40]), np.array([10.0, 35.0, 60.0]))
    assert abs(model.tilt_for(0.28) - 35.0) < 1e-9, "must reproduce a taught point"
    assert 10.0 < model.tilt_for(0.23) < 35.0, "must interpolate between them"
    assert model.tilt_for(0.45) > 60.0, "must keep leaning over past the last point"
    assert model.tilt_for(0.05) >= 0.0, "must never go past vertical"
    print("self-test OK:", model.describe())


if __name__ == "__main__":
    _self_test()
