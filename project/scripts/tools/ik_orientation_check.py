#!/usr/bin/env python3
"""Answer one question, with no robot and no risk: when we ask the wrist to
rotate by a block's angle, does the IK actually deliver that rotation?

WHY THIS EXISTS: table_ik.py's solve_to_position reports
    err = norm(forward_kinematics(solved)[:3,3] - target_xyz)
-- POSITION ONLY. Orientation never enters the returned error, and it is solved
as a soft constraint (orientation_weight=0.01 vs position_weight=1.0). So a
solution whose wrist points nowhere near the requested direction still comes
back as "converged", and every caller (including new_pick_and_place.py's
checked_solve, until 2026-09-01) accepted it.

That makes the pending ANGLE_SIGN test unreliable in a specific, nasty way: if
the solver quietly drops the requested wrist rotation, the wrist won't turn no
matter which sign is used -- and watching it not turn would look exactly like
"the sign is wrong", sending us to flip a sign that was never the problem. So
measure here first, on the desktop, before spending robot time.

Extra context that makes this worth measuring rather than assuming: this arm has
only 5 joints that move the TCP (the 6th, gripper, opens the jaw on a separate
URDF branch and does not move gripper_frame_link). 3 position + 3 orientation =
6 constraints cannot all be met with 5 DOF, so orientation being the soft one
may well be a deliberate, correct tradeoff -- not a bug to "fix" by cranking
orientation_weight up and losing position accuracy instead.

Reads: nothing but config/URDF. Connects to: nothing. Moves: nothing.

Usage (on the robot PC, which is where placo/the URDF live):
    python project/scripts/tools/ik_orientation_check.py
    python project/scripts/tools/ik_orientation_check.py --orientation-weight 0.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "robot"))
import new_pick_and_place as npp  # noqa: E402  (constants + orientation helpers only; never calls its main())

from lerobot.grad_project.control.grasp_pitch_model import GraspPitchModel, orientation_from_tilt
from lerobot.grad_project.control.table_frame_calibration import apply_table_to_robot, load_calibration
from lerobot.grad_project.control.table_ik import load_kinematics, solve_to_position

CALIB_PATH = "project/config/table_robot_calibration.json"

# A near, a middle, and a far spot inside the table, to see whether the answer
# changes with reach (it plausibly does -- the further out, the more the arm is
# stretched and the less orientation freedom is left over).
TEST_POINTS_CM = [("near", 10.0, 2.0), ("mid", 10.0, 5.0), ("far", 10.0, 9.0)]
TEST_ANGLES_DEG = [0.0, 15.0, 30.0, 45.0, -30.0]


def try_alternate_seeds(kin, base_seed, target_xyz, requested, weight) -> bool:
    """Would a different IK starting pose have solved this? solve_to_position
    iterates from a seed, so a rejected solve can mean 'this seed fell into a
    local minimum', not 'the arm cannot do this'. Answering that decides whether
    a seed-retry is worth adding to checked_solve, or whether the point really
    is out of reach. Pure computation -- nothing here is ever sent to the arm."""
    for perturb in ([0, -25, 0, 0, 0, 0], [0, 25, 0, 0, 0, 0], [0, 0, -25, 0, 0, 0],
                    [0, 0, 25, 0, 0, 0], [0, 0, 0, 0, 45, 0], [0, 0, 0, 0, -45, 0]):
        seed = np.array(base_seed, dtype=float) + np.array(perturb, dtype=float)
        solved, pos_err = solve_to_position(
            kin, seed, target_xyz, keep_orientation=requested,
            max_iters=60, orientation_weight=weight,
        )
        if pos_err <= npp.MAX_IK_ERR_M and npp.orientation_error_deg(kin, solved, requested) <= npp.MAX_IK_ORIENT_ERR_DEG:
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--orientation-weight",
        type=float,
        default=npp.IK_ORIENTATION_WEIGHT,
        help=f"IK orientation weight to test (default {npp.IK_ORIENTATION_WEIGHT}, table_ik.py's own default). "
        "Re-run with a larger value to see what raising it would buy -- and cost.",
    )
    args = parser.parse_args()

    kin = load_kinematics()
    calib = load_calibration(CALIB_PATH)
    pitch_model = GraspPitchModel.load()
    seed = npp.load_observe_pose()

    floor_z = npp.floor_z_from_calibration(calib)

    print(f"orientation_weight = {args.orientation_weight}  (position_weight is 1.0)")
    print(f"floor clamp = {floor_z * 1000:.1f}mm, path check tolerance = {npp.PATH_FLOOR_TOLERANCE_M * 1000:.0f}mm")
    print("\n'delivered' is the wrist rotation actually achieved -- if asked grows but")
    print("delivered stays near 0, the solver is silently dropping the rotation.")
    print("'path low' is the LOWEST tool height along the ramp from the observe pose to")
    print("this solution -- the transit sweep that nothing used to check. It must stay")
    print("above the floor clamp, and the margin here shows how close that gate runs.\n")
    print(f"{'point':>6} {'reach':>7} {'asked':>7} {'pos err':>9} {'orient err':>11} {'delivered':>10} {'path low':>10}")
    print("-" * 68)

    for label, x_cm, y_cm in TEST_POINTS_CM:
        target_xyz = apply_table_to_robot(x_cm, y_cm, calib)
        reach = float(np.linalg.norm(target_xyz[:2]))
        tilt_deg = pitch_model.tilt_for(reach) if pitch_model else 0.0
        base_orientation = orientation_from_tilt(target_xyz, tilt_deg)

        # Unrotated reference solve: what the wrist does with no block angle at all.
        base_solved, _ = solve_to_position(
            kin, seed, target_xyz, keep_orientation=base_orientation,
            max_iters=60, orientation_weight=args.orientation_weight,
        )
        base_achieved = kin.forward_kinematics(base_solved)[:3, :3]

        for angle in TEST_ANGLES_DEG:
            requested = npp.rotate_about_local_z(base_orientation, np.deg2rad(angle))
            solved, pos_err = solve_to_position(
                kin, seed, target_xyz, keep_orientation=requested,
                max_iters=60, orientation_weight=args.orientation_weight,
            )
            achieved = kin.forward_kinematics(solved)[:3, :3]

            ori_err = npp.orientation_error_deg(kin, solved, requested)
            # How far the wrist ACTUALLY moved away from the no-rotation solve.
            # Near |angle| => the rotation was delivered. Near 0 => it was dropped.
            r_delivered = base_achieved.T @ achieved
            cos = (np.trace(r_delivered) - 1.0) / 2.0
            delivered = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

            low_z = npp.path_lowest_z(kin, seed, solved)

            flag = ""
            if pos_err > npp.MAX_IK_ERR_M or ori_err > npp.MAX_IK_ORIENT_ERR_DEG:
                # Would be REJECTED by checked_solve. Before concluding the point
                # is unreachable, see whether it's just this one seed landing in a
                # local minimum -- the solver is iterative and seeded, so a
                # different starting pose can converge where this one didn't.
                rescued = try_alternate_seeds(kin, seed, target_xyz, requested, args.orientation_weight)
                flag = f"  <-- REJECTED{'; retry seed solves it' if rescued else '; no retry seed solves it'}"
            elif low_z < floor_z - npp.PATH_FLOOR_TOLERANCE_M:
                flag = f"  <-- TRANSIT PATH DIPS BELOW FLOOR ({floor_z * 1000:.1f}mm)"

            print(
                f"{label:>6} {reach*100:6.1f}cm {angle:+6.0f}° "
                f"{pos_err*1000:7.2f}mm {ori_err:10.1f}° {delivered:9.1f}° {low_z*1000:8.1f}mm{flag}"
            )
        print()

    print("Read it like this:")
    print("  delivered tracks |asked|  -> rotation works; ANGLE_SIGN test on the robot is meaningful.")
    print("  delivered stays near 0    -> rotation is being dropped; testing ANGLE_SIGN would just")
    print("                               show a wrist that never turns, for either sign. Fix this first.")
    print("  pos err grows with weight -> the tradeoff is real; do not raise the weight blindly.")


if __name__ == "__main__":
    main()
