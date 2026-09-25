#!/usr/bin/env python3
"""Turn taught grasp points into the calibration the pipeline runs on.

Reads what teach_grasp_points.py recorded -- (block as seen by the detector,
follower pose that actually held it) -- and fits pixel -> robot XY as a
homography, plus a table-frame Z row. Same shapes the pipeline already loads,
so nothing downstream changes.

A homography is the right model for XY and not merely a convenient one: the
camera images a plane, the arm works in a plane, and the map between two planes
under perspective IS a homography, 8 degrees of freedom. It also absorbs the
block's height exactly -- every block's top face lies on one plane parallel to
the table, so parallax is a fixed projective effect, not a residual.

Reports held-out error so the result can be compared against the demo-mined
calibration (12.5mm median) rather than trusted on faith. Refuses to write if
the fit is worse, and warns when coverage is thin: a homography extrapolates
badly, and the mined calibration went ~4cm wrong in exactly the region where it
had no samples.

Usage:
    python project/scripts/tools/fit_taught_points.py            # 검증만
    python project/scripts/tools/fit_taught_points.py --apply    # 적용
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "robot"))
import new_pick_and_place as npp  # noqa: E402

from lerobot.grad_project.paths import lerobot_root
from lerobot.grad_project.perception.pixel_to_table import load_homography, pixel_to_table_xy

POINTS_PATH = "project/config/taught_grasp_points.json"
PIX2ROBOT_PATH = "project/config/pixel_to_robot_homography.json"
CALIB_PATH = "project/config/table_robot_calibration.json"
MIN_POINTS = 12


def dlt(px: np.ndarray, xy: np.ndarray) -> np.ndarray:
    a = []
    for (u, v), (x, y) in zip(px, xy):
        a.append([-u, -v, -1, 0, 0, 0, u * x, v * x, x])
        a.append([0, 0, 0, -u, -v, -1, u * y, v * y, y])
    _, _, vt = np.linalg.svd(np.array(a))
    return vt[-1].reshape(3, 3)


def apply_h(h: np.ndarray, px: np.ndarray) -> np.ndarray:
    q = np.c_[px, np.ones(len(px))] @ h.T
    return q[:, :2] / q[:, 2:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="검증만 하지 말고 실제로 적용합니다.")
    ap.add_argument("--points", default=POINTS_PATH)
    args = ap.parse_args()

    root = lerobot_root()
    pts_path = Path(args.points)
    if not pts_path.is_absolute():
        pts_path = root / pts_path
    if not pts_path.is_file():
        print(f"!! 교시 파일이 없습니다: {pts_path}")
        print("   먼저 teach_grasp_points.py 로 점을 모으세요.")
        sys.exit(1)

    pts = json.loads(pts_path.read_text())["points"]
    print(f"교시된 점 {len(pts)}개\n")
    if len(pts) < MIN_POINTS:
        print(f"!! {MIN_POINTS}개 미만입니다. 맞출 수 없습니다.")
        sys.exit(1)

    px = np.array([p["pixel"] for p in pts])
    xy = np.array([p["tcp_xyz_m"][:2] for p in pts])
    z = np.array([p["tcp_xyz_m"][2] for p in pts])

    # Hold out every 5th point, spread through the set rather than a block at the
    # end, so the held-out number reflects the whole working area.
    va = np.zeros(len(pts), bool)
    va[::5] = True
    tr = ~va
    if tr.sum() < MIN_POINTS:
        print("!! 학습용 점이 너무 적습니다.")
        sys.exit(1)

    h = dlt(px[tr], xy[tr])
    err = np.linalg.norm(apply_h(h, px) - xy, axis=1) * 1000
    print(f"픽셀 -> 로봇 XY (교시 {tr.sum()}개로 학습, {va.sum()}개로 검증)")
    print(f"  학습 오차 중간값 {np.median(err[tr]):.1f}mm")
    print(f"  검증 오차 중간값 {np.median(err[va]):.1f}mm   최대 {err[va].max():.1f}mm")

    # Same held-out points, scored by whatever the pipeline uses today.
    cur = np.array(json.loads((root / PIX2ROBOT_PATH).read_text())["homography_pixel_to_robot_xy_m"])
    cur_err = np.linalg.norm(apply_h(cur, px) - xy, axis=1) * 1000
    print(f"  (지금 쓰는 시연 마이닝 캘리브레이션: 같은 점들에서 {np.median(cur_err[va]):.1f}mm)\n")

    # Z stays in the table frame, because that is the shape the pipeline loads.
    cam_h = load_homography(npp.CAMERA_CALIBRATION_PATH)
    tab = np.array([pixel_to_table_xy(u, v, cam_h) for u, v in px])
    design = np.c_[tab / 100.0, np.ones(len(tab))]
    zrow, *_ = np.linalg.lstsq(design[tr], z[tr], rcond=None)
    zerr = np.abs(design @ zrow - z) * 1000
    print(f"높이(Z): 검증 오차 중간값 {np.median(zerr[va]):.1f}mm\n")

    span_x, span_y = px[:, 0].ptp(), px[:, 1].ptp()
    print(f"교시 범위: 픽셀 x {px[:,0].min():.0f}~{px[:,0].max():.0f} ({span_x:.0f}), "
          f"y {px[:,1].min():.0f}~{px[:,1].max():.0f} ({span_y:.0f})")
    if span_x < 350 or span_y < 250:
        print("  !! 한쪽으로 몰려 있습니다. 빈 구역은 외삽이 되고 거기서 크게 틀어집니다.")

    better = np.median(err[va]) < np.median(cur_err[va])
    if not args.apply:
        print(f"\n{'적용할 만합니다' if better else '지금 것보다 나쁩니다'}. "
              f"적용하려면 --apply 를 붙여 다시 실행하세요.")
        return
    if not better:
        print("\n!! 지금 캘리브레이션보다 나쁘므로 적용하지 않습니다. 점을 더 모으세요.")
        sys.exit(1)

    stamp = date.today().strftime("%Y%m%d")
    for rel in (PIX2ROBOT_PATH, CALIB_PATH):
        src = root / rel
        shutil.copy(src, src.with_name(f"{src.stem}_backup_before_taught_{stamp}{src.suffix}"))

    note = (f"Fit {date.today()} from {len(pts)} grasp points taught by hand with the leader arm "
            f"(teach_grasp_points.py). Each point is a pose the operator confirmed was holding the "
            f"block, so nothing here infers when a grasp happened -- which is what limited the "
            f"demo-mined calibration to about 11mm however carefully it was re-analysed. Held-out "
            f"XY median {np.median(err[va]):.1f}mm against {np.median(cur_err[va]):.1f}mm for the "
            f"mined fit on these same points.")

    (root / PIX2ROBOT_PATH).write_text(json.dumps(
        {"homography_pixel_to_robot_xy_m": h.tolist(), "note": note}, indent=2, ensure_ascii=False))

    cal = json.loads((root / CALIB_PATH).read_text())
    m = np.array(cal["affine_table_cm_to_robot_m"])
    m[2] = zrow
    cal["affine_table_cm_to_robot_m"] = m.tolist()
    cal["note"] = cal.get("note", "") + " | " + note + " (Z row only; XY comes from the homography.)"
    (root / CALIB_PATH).write_text(json.dumps(cal, indent=2, ensure_ascii=False))

    print(f"\n적용했습니다. 백업은 *_backup_before_taught_{stamp}.json")
    print("\n다음 두 가지를 반드시 하세요:")
    print("  1. TCP_JAW_OFFSET_M / TCP_FORWARD_OFFSET_M 을 0 으로 되돌리기")
    print("     (교시 자세가 이미 그 편차를 포함하고 있어, 그대로 두면 두 번 적용됩니다)")
    print("  2. --dry-run 으로 floor_z 와 IK 수렴 확인 후 --max-blocks 1")


if __name__ == "__main__":
    main()
