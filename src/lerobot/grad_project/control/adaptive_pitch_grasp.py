"""Adaptive per-target grasp-pitch search -- alternative to the fixed pitch
ramp in `pick_and_place_yolo.py`'s `grasp_orientation_for`.

Not wired into the live pipeline. Swap it in by replacing
`PickAndPlaceYolo.grasp_orientation_for` with `adaptive_grasp_orientation_for`
below (it needs the same `self`: `self._ref_azimuth_rad`, `self._ref_orientation`,
`self._ref_joints`, `self.solve_ik`, `self.cfg`), and adding the three config
fields from `AdaptivePitchConfig` onto `PickAndPlaceYoloConfig`.

## Why this exists

The fixed ramp approaches every block at a hand-picked tilt (shallower than
horizontal by a fixed amount depending on reach), which is why the gripper was
observed sliding into blocks and sometimes pushing them instead of closing
around them from above.

A scan across the board found:
  - A fully vertical (straight overhead) approach is NOT reachable everywhere
    on this 5-DOF arm -- IK error was 14-290mm depending on position when
    orientation was forced to point straight down.
  - But *some* extra tilt toward vertical almost always is reachable, and how
    much varies a lot by position: ~0 deg at the farthest corner (arm near
    full extension) up to ~60-65 deg (effectively vertical) for close-in
    blocks.

Rather than hand-tune a fixed curve to approximate that, this searches for it
directly per target: step outward from the baseline tilt in `step_deg`
increments, testing each candidate orientation with a real IK solve, and keep
the steepest one still under `tolerance_m` error. Each step is one IK solve
(~1ms), so a 13-step scan costs nothing next to the seconds-long arm move it's
steering.

Verified in simulation (against the 8 taught grasp points in
project/config/grasp_correction_samples.json) but NOT YET TESTED on the real
robot -- do that before switching this in for real runs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class AdaptivePitchConfig:
    # How far the search scans toward vertical, in what step size, and the
    # IK-error tolerance that decides "still reachable."
    pitch_search_max_deg: float = 65.0
    pitch_search_step_deg: float = 5.0
    pitch_search_tolerance_m: float = 0.005


def adaptive_grasp_orientation_for(self, target_xyz_m: np.ndarray) -> np.ndarray:
    """The steepest (closest to straight-down) reachable grasp orientation.

    A shallow approach comes in nearly parallel to the table, so a small
    position error slides the jaws sideways into the block instead of
    closing around it from above -- that's the "pushes the block" failure.
    Straight overhead isn't reachable everywhere on this 5-DOF arm (a scan
    found ~14-290mm IK error for a fixed vertical orientation depending on
    reach), but *some* amount of extra tilt-toward-vertical almost always
    is, and how much varies a lot by position (0 deg at the far corner,
    60 deg -- i.e. effectively vertical -- for close-in blocks). Rather
    than hand-tune a fixed curve for that, search for it directly: each
    search step is one IK solve (~1ms), so scanning up to ~13 steps here
    costs nothing next to the seconds-long moves it's steering.
    """
    from lerobot.grad_project.control.pick_and_place_yolo import _rotation_about

    delta = float(np.arctan2(target_xyz_m[1], target_xyz_m[0])) - self._ref_azimuth_rad
    cos_d, sin_d = np.cos(delta), np.sin(delta)
    rot_z = np.array([[cos_d, -sin_d, 0.0], [sin_d, cos_d, 0.0], [0.0, 0.0, 1.0]])
    base_orientation = rot_z @ self._ref_orientation

    radial = np.array([target_xyz_m[0], target_xyz_m[1], 0.0])
    radial /= np.linalg.norm(radial)
    pitch_axis = np.cross(np.array([0.0, 0.0, 1.0]), radial)

    best_orientation = base_orientation
    best_deg = 0.0
    step_deg = self.cfg.pitch_search_step_deg
    steps = int(round(self.cfg.pitch_search_max_deg / step_deg))
    for i in range(1, steps + 1):
        candidate_deg = i * step_deg
        candidate = _rotation_about(pitch_axis, np.deg2rad(candidate_deg)) @ base_orientation
        _, err = self.solve_ik(self._ref_joints, target_xyz_m, candidate)
        if err > self.cfg.pitch_search_tolerance_m:
            break
        best_orientation, best_deg = candidate, candidate_deg

    if best_deg > 0:
        import logging

        logging.getLogger(__name__).info(
            "reach %.0fmm -> steepened the approach %.0f deg toward vertical",
            float(np.hypot(target_xyz_m[0], target_xyz_m[1])) * 1000,
            best_deg,
        )
    return best_orientation
