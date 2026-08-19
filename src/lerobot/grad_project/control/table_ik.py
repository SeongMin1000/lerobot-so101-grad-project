"""IK helper wired to the official SO-101 URDF (TheRobotStudio/SO-ARM100).

`RobotKinematics.inverse_kinematics` is a single solver step, not a
converged solve -- it must be called repeatedly (confirmed empirically:
~10 iterations fully converges for typical reachable targets). This module
wraps that into a single call that iterates until convergence or a max
iteration count, for use by the hardcoded pick-and-place motion planner.

NOTE: `solve_to_position` takes an end-effector position in the ROBOT'S OWN
base frame (meters), NOT the table/camera frame from pixel_to_table.py.
Converting a table-frame (cm) target into this frame still requires the
robot-to-table alignment step, which must be done physically with the real
arm and is not part of this module.
"""

from __future__ import annotations

import numpy as np

from lerobot.model.kinematics import RobotKinematics

DEFAULT_URDF_PATH = "project/assets/urdf/so101/so101_new_calib.urdf"


def load_kinematics(urdf_path: str = DEFAULT_URDF_PATH) -> RobotKinematics:
    return RobotKinematics(urdf_path)


def solve_to_position(
    kin: RobotKinematics,
    current_joint_deg: np.ndarray,
    target_xyz_m: np.ndarray,
    keep_orientation: np.ndarray | None = None,
    max_iters: int = 30,
    tol_m: float = 1e-4,
    orientation_weight: float = 0.01,
) -> tuple[np.ndarray, float]:
    """Solve IK to reach `target_xyz_m` (3,), iterating to convergence.

    keep_orientation: optional 3x3 rotation to hold while solving for
    position (defaults to the current end-effector orientation from FK at
    current_joint_deg, so the wrist doesn't spin arbitrarily).

    Returns (joint_deg, final_position_error_m).
    """
    current = np.array(current_joint_deg, dtype=float).copy()

    fk_now = kin.forward_kinematics(current)
    target_pose = fk_now.copy()
    target_pose[:3, 3] = target_xyz_m
    if keep_orientation is not None:
        target_pose[:3, :3] = keep_orientation

    err = float("inf")
    for _ in range(max_iters):
        current = kin.inverse_kinematics(
            current, target_pose, position_weight=1.0, orientation_weight=orientation_weight
        )
        check = kin.forward_kinematics(current)
        err = float(np.linalg.norm(check[:3, 3] - target_xyz_m))
        if err < tol_m:
            break

    return current, err


def _self_test() -> None:
    kin = load_kinematics()
    true_joints = np.array([10.0, -20.0, 30.0, 15.0, 5.0, 20.0])
    target_pose = kin.forward_kinematics(true_joints)
    target_xyz = target_pose[:3, 3]

    solved, err = solve_to_position(kin, np.zeros(6), target_xyz, keep_orientation=target_pose[:3, :3])
    print(f"solved joints: {np.round(solved, 2)}, position error (m): {err:.6f}")
    assert err < 1e-3, "IK did not converge in self-test"
    print("self-test OK")


if __name__ == "__main__":
    _self_test()
