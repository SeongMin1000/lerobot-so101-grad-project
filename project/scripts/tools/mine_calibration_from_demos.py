#!/usr/bin/env python3
"""Mine (block position -> real successful-grasp robot XYZ) pairs directly out of the
already-recorded teleop dataset (eslab1234/task1_hybrid_5blocks_v3_444ep_merged), instead
of spending more physical robot time on ruler measurements.

WHY THIS EXISTS: tcp_offset_probe.py's ruler calibration only gives 5 points in a narrow
reach band (24.5-33.1cm), with the gripper held nearly-closed the whole time. Real picks
in new_pick_and_place.py use a much wider reach range and hold the gripper OPEN through
the whole descent -- both known-unaccounted-for gaps (see YOLO_하드코딩_개발_이슈정리.md,
"그리퍼 열림/닫힘 상태에 따른 접촉점 편차"). This script gets a MUCH bigger, wider-coverage
calibration set for free, and -- because every sample comes from a REAL open->close teleop
grasp -- the gripper-state effect is already baked into every point, with no separate
open-gripper recalibration needed.

Each episode's fixed task ("Pick up the 5 blocks in sequence (red, yellow, wood, green,
blue), then place each at the target area.") gives up to 5 real grasp events per episode:
  1. Detect each grasp MOMENT from the recorded joint trajectory: the point where the
     gripper, while actively closing, stalls well above fully-closed and holds for a
     while before re-opening to drop the block off -- the exact same "something is being
     held" signature close_gripper_until_resistance() checks for live in
     new_pick_and_place.py, just read after the fact from logged data instead of live
     motor feedback.
  2. Forward-kinematics that moment's REAL recorded joint state -> the REAL robot_xyz the
     arm was physically at when it grasped that block. This is ground truth; no ruler
     needed, the robot's own encoders already measured it.
  3. Pair that with where the OBB model saw that block, in the episode's first frame
     (before anything has been touched), converted to table cm via the existing pixel/
     homography pipeline.
  Since the task's block order is FIXED, we don't need to guess which detected box is
  "the one that got picked" -- grasp event #1 is always red, #2 yellow, #3 wood, #4 green,
  #5 blue.

SIMPLIFYING ASSUMPTION: block positions are read once, from the episode's FIRST frame,
for all 5 blocks. This assumes an unpicked block doesn't get bumped while an earlier one
is being handled. If that's wrong for some episodes, those samples show up as residual
outliers in the fit/validation printout below, not as a silent systematic error -- look at
the validation numbers before trusting the output.

Run this on the ROBOT PC (needs table_ik.py / placo / the URDF -- same environment
tcp_offset_probe.py and new_pick_and_place.py already run in). No robot hardware
connection required, this only reads the dataset and does pure computation. Needs
pandas + pyarrow (`pip install pandas pyarrow` if missing) and network access to
huggingface.co (or a local dataset cache, see --local-dataset-root).

Usage:
    python project/scripts/tools/mine_calibration_from_demos.py --episodes 20   # quick check
    python project/scripts/tools/mine_calibration_from_demos.py                # all 444
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "robot"))  # new_pick_and_place.py lives there
import new_pick_and_place as npp  # noqa: E402  (reuses ObbBlockDetector -- NOT the robot-connecting main())

from lerobot.grad_project.control.table_frame_calibration import apply_table_to_robot, load_calibration
from lerobot.grad_project.control.table_ik import load_kinematics
from lerobot.grad_project.paths import lerobot_root
from lerobot.grad_project.perception.pixel_to_table import load_homography, pixel_to_table_xy

REPO_ID = "eslab1234/task1_hybrid_5blocks_v3_444ep_merged"
HF_BASE = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main"
# Fixed task order (see docstring). Imported rather than re-declared so the
# mined calibration and the running pipeline can never disagree about it --
# new_pick_and_place.py sorts its pick candidates by this exact list.
COLOR_SEQUENCE = npp.COLOR_SEQUENCE

# Grasp-moment detection: a hysteresis state machine over gripper.pos, not fixed windows.
# Tuned empirically against this exact dataset (see chat history / dev notes): real teleop
# closes/reopens take anywhere from ~1 to ~5 seconds depending on the demonstrator, far more
# variable than the robotic RAMP_STEPS-driven motion new_pick_and_place.py itself uses, so a
# fixed-duration window (what an earlier version of this script tried) missed most events.
# Also: gmin/gmax are computed as the 5th/95th percentile of the WHOLE episode, not the true
# min/max -- some episodes end with the gripper driven fully closed to ~0-2 (a rest/idle
# state, not a grasp) for a couple seconds, which would otherwise drag the true min far below
# every real grasp-hold value (~26 in most episodes) and break the relative thresholds below.
LOW_PCTL, HIGH_PCTL = 5, 95
CLOSED_FRAC = 0.35   # crossing below gmin + this fraction of span = "gripper is closing on something"
OPEN_FRAC = 0.65      # crossing back above gmin + this fraction of span = "released it"
MIN_DWELL_FRAMES = 70  # ~2.3s at 30fps -- filters brief noise dips from genuine holds
                        # (validated: 325/444 episodes yield exactly 5 events at this value)

CALIB_PATH = "project/config/table_robot_calibration.json"
OUT_PATH = "project/config/table_robot_calibration_from_demos.json"


def find_grasp_events(gripper: np.ndarray) -> list[int]:
    """Return frame indices of candidate grasp moments, in time order.

    Hysteresis state machine: track OPEN vs CLOSED state using two thresholds derived from
    this episode's own (robust, percentile-based) gripper range, so it self-calibrates per
    episode instead of assuming a fixed open/closed value. A CLOSED dwell that lasts at
    least MIN_DWELL_FRAMES and eventually crosses back into OPEN counts as one grasp event,
    reported at the dwell's own minimum (deepest close) frame. See module-level constants
    for the tuning and why percentiles instead of true min/max.
    """
    gmin, gmax = np.percentile(gripper, LOW_PCTL), np.percentile(gripper, HIGH_PCTL)
    span = gmax - gmin
    if span < 3:
        return []
    closed_th = gmin + CLOSED_FRAC * span
    open_th = gmin + OPEN_FRAC * span

    events = []
    state = "open"
    dwell_start = dwell_min_idx = None
    dwell_min_val = None
    for i, v in enumerate(gripper):
        if state == "open":
            if v < closed_th:
                state = "closed"
                dwell_start = dwell_min_idx = i
                dwell_min_val = v
        else:
            if v < dwell_min_val:
                dwell_min_val, dwell_min_idx = v, i
            if v > open_th:
                if dwell_start > 0 and (i - dwell_start) >= MIN_DWELL_FRAMES:
                    events.append(dwell_min_idx)
                state = "open"
    return events


def video_frame_url_and_time(ep_row) -> tuple[str, float]:
    chunk = int(ep_row["videos/observation.images.top/chunk_index"])
    file_idx = int(ep_row["videos/observation.images.top/file_index"])
    from_ts = float(ep_row["videos/observation.images.top/from_timestamp"])
    url = f"{HF_BASE}/videos/observation.images.top/chunk-{chunk:03d}/file-{file_idx:03d}.mp4"
    return url, from_ts


def extract_frame(video_url: str, time_s: float, out_path: Path) -> bool:
    cmd = ["ffmpeg", "-y", "-ss", f"{time_s:.3f}", "-i", video_url, "-frames:v", "1", str(out_path)]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0 and out_path.is_file()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=None, help="Limit to the first N episodes (for a quick check).")
    parser.add_argument("--val-frac", type=float, default=0.15)
    args = parser.parse_args()

    import pandas as pd  # local import: only needed here, not for anything importing this module

    root = lerobot_root()
    tmp_dir = Path(tempfile.mkdtemp(prefix="mine_calib_"))

    print("다운로드: data.parquet, episodes.parquet ...")
    data_path = tmp_dir / "data.parquet"
    ep_path = tmp_dir / "episodes.parquet"
    for url, dst in [
        (f"{HF_BASE}/data/chunk-000/file-000.parquet", data_path),
        (f"{HF_BASE}/meta/episodes/chunk-000/file-000.parquet", ep_path),
    ]:
        subprocess.run(["curl", "-sL", url, "-o", str(dst)], check=True)

    data = pd.read_parquet(data_path).sort_values(["episode_index", "frame_index"])
    data_by_episode = dict(tuple(data.groupby("episode_index")))  # avoid re-filtering 625k rows per episode
    episodes = pd.read_parquet(ep_path)
    if args.episodes:
        episodes = episodes.iloc[: args.episodes]
    print(f"{len(episodes)}개 에피소드 처리 시작 (전체 {len(data_by_episode)}개 중)")

    kin = load_kinematics()
    calib = load_calibration(CALIB_PATH)
    homography = load_homography(npp.CAMERA_CALIBRATION_PATH)
    detector = npp.ObbBlockDetector(npp.OBB_MODEL_PATH, npp.DETECTOR_CONFIG_PATH)

    samples = []  # each: dict(episode, color, table_x_cm, table_y_cm, angle_deg, grasp_xyz, reach_m)
    skipped_wrong_count = 0
    skipped_missing_color = 0

    for _, ep_row in episodes.iterrows():
        ep_idx = int(ep_row["episode_index"])
        ep_data = data_by_episode.get(ep_idx)
        if ep_data is None:
            continue
        gripper = np.stack(ep_data["observation.state"].to_numpy())[:, npp.GRIPPER_IDX]

        grasp_frames = find_grasp_events(gripper)
        if len(grasp_frames) != 5:
            skipped_wrong_count += 1
            continue

        video_url, frame_time = video_frame_url_and_time(ep_row)
        frame_path = tmp_dir / f"ep{ep_idx:04d}_first.jpg"
        if not extract_frame(video_url, frame_time, frame_path):
            print(f"  ep{ep_idx}: 첫 프레임 추출 실패, 스킵")
            continue

        frame_bgr = cv2.imread(str(frame_path))
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        detections = {b.color: b for b in detector.detect(frame_rgb)}

        # forward_kinematics's placo binding wants Python float (float64), not the
        # dataset's native float32 -- Boost.Python's set_joint rejects numpy.float32.
        joint_states = np.stack(ep_data["observation.state"].to_numpy()).astype(np.float64)
        for color, frame_idx in zip(COLOR_SEQUENCE, grasp_frames):
            if color not in detections:
                skipped_missing_color += 1
                continue
            block = detections[color]
            table_x_cm, table_y_cm = pixel_to_table_xy(block.cx, block.cy, homography)

            joint_state = joint_states[frame_idx]
            grasp_xyz = kin.forward_kinematics(joint_state)[:3, 3]

            samples.append(
                {
                    "episode": ep_idx,
                    "color": color,
                    "table_x_cm": float(table_x_cm),
                    "table_y_cm": float(table_y_cm),
                    "angle_deg": float(block.angle_deg),
                    "grasp_xyz": grasp_xyz.tolist(),
                    "reach_m": float(np.linalg.norm(grasp_xyz[:2])),
                }
            )

        if ep_idx % 20 == 0:
            print(f"  ep{ep_idx}: 누적 샘플 {len(samples)}개")

    print(
        f"\n총 {len(samples)}개 샘플 수집 완료. "
        f"(그립 이벤트 5개 아니라서 스킵된 에피소드 {skipped_wrong_count}개, "
        f"첫 프레임에서 특정 색 감지 실패 {skipped_missing_color}건)"
    )
    if len(samples) < 30:
        print("!! 샘플이 너무 적습니다 (30개 미만) -- 그립 감지 임계값을 다시 봐야 할 수 있습니다. 여기서 중단합니다.")
        sys.exit(1)

    # Split by EPISODE (not by sample) so train/val don't share within-episode correlation.
    rng_episodes = sorted({s["episode"] for s in samples})
    n_val_ep = max(1, int(len(rng_episodes) * args.val_frac))
    val_episodes = set(rng_episodes[:: len(rng_episodes) // n_val_ep][:n_val_ep])
    train = [s for s in samples if s["episode"] not in val_episodes]
    val = [s for s in samples if s["episode"] in val_episodes]
    print(f"train {len(train)}개 샘플 / val {len(val)}개 샘플 (val은 {len(val_episodes)}개 에피소드)")

    # Fit robot_xyz_m = M @ [x_m, y_m, 1], same convention as table_robot_calibration.json.
    design = np.array([[s["table_x_cm"] / 100.0, s["table_y_cm"] / 100.0, 1.0] for s in train])
    targets = np.array([s["grasp_xyz"] for s in train])
    M, *_ = np.linalg.lstsq(design, targets, rcond=None)
    M = M.T  # (3,3): rows = robot x/y/z, cols = [x_m coeff, y_m coeff, const]

    def predict(M, table_x_cm, table_y_cm):
        return M @ np.array([table_x_cm / 100.0, table_y_cm / 100.0, 1.0])

    def residual_report(label, M_or_none):
        errs_mm = []
        for s in val:
            if M_or_none is not None:
                pred = predict(M_or_none, s["table_x_cm"], s["table_y_cm"])
            else:
                pred = apply_table_to_robot(s["table_x_cm"], s["table_y_cm"], calib)
            err_mm = float(np.linalg.norm(pred - np.array(s["grasp_xyz"]))) * 1000
            errs_mm.append(err_mm)
        errs_mm = np.array(errs_mm)
        print(
            f"[{label}] val 오차(mm): mean={errs_mm.mean():.1f} median={np.median(errs_mm):.1f} "
            f"p90={np.percentile(errs_mm, 90):.1f} max={errs_mm.max():.1f}"
        )
        return errs_mm

    print("\n=== 검증 (val set, 학습에 안 쓴 에피소드) ===")
    residual_report("기존 캘리브레이션(자로 잰 5점)", None)
    residual_report("새 캘리브레이션(시연 데이터 마이닝)", M)

    out = {
        "affine_table_cm_to_robot_m": M.tolist(),
        "note": (
            f"Fit {__import__('datetime').date.today()} from {len(train)} real grasp events mined out of "
            f"{REPO_ID} (see mine_calibration_from_demos.py). Each point is where the OBB detector saw a "
            f"block in an episode's first frame, paired with the REAL forward-kinematics position the arm "
            f"was physically at when it grasped that exact block during teleop -- not a ruler measurement. "
            f"Held-out validation on {len(val_episodes)} episodes not used in the fit: see stdout at fit time. "
            f"Do NOT trust this file until the validation numbers above have been checked."
        ),
        "source_points": [
            {
                "label": f"ep{s['episode']}_{s['color']}",
                "table_x_cm": s["table_x_cm"],
                "table_y_cm": s["table_y_cm"],
                "robot_xyz_m": s["grasp_xyz"],
            }
            for s in train
        ],
    }
    out_path = root / OUT_PATH
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n새 캘리브레이션 후보 저장: {out_path}")
    print("주의: 위 검증 오차가 기존보다 확실히 낮을 때만 table_robot_calibration.json으로 바꿔치기하세요.")


if __name__ == "__main__":
    main()
