#!/usr/bin/env python3
"""Interactive 1-block teleoperation demonstration & joint mapping tool.

2-Step Workflow:
[STEP 1: Snapshot] Place ONE block on workspace with arm parked/clear. Press ENTER.
  -> Top camera captures crystal-clear block pixel (cx, cy) and computes (X, Y).
[STEP 2: Teach] Move leader arm to align follower gripper above the block. Press ENTER.
  -> Follower joints are captured and paired with the Step 1 coordinates.
  -> No occlusion/robot-arm misdetection ever occurs!

Usage:
  python project/scripts/tools/teach_block_hover_joints.py
"""

from __future__ import annotations

import json
import select
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.grad_project.paths import lerobot_root
from lerobot.grad_project.perception.yolo_block_detector import YoloBlockDetector
from lerobot.robots import make_robot_from_config
from lerobot.robots.so_follower import SO101FollowerConfig
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.teleoperators.so_leader import SO101LeaderConfig

JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
SAMPLES_SAVE_PATH = "project/config/hover_demonstration_samples_record.json"
MODEL_SAVE_PATH = "project/config/hover_joint_model_record.json"
CALIB_PATH = "project/config/grasp_pixel_to_robot_record.json"


def fit_and_save_rbf_model(samples: list[dict[str, Any]], root: Path) -> dict[str, Any]:
    """Fits Multiquadric RBF model for all 5 joints and saves to JSON."""
    if len(samples) < 6:
        raise ValueError(f"Need at least 6 samples to fit model (have {len(samples)}).")

    xy = np.array([s["robot_xy_m"] for s in samples], dtype=np.float64)
    Y = np.array([[s["joints"][n] for n in JOINT_NAMES] for s in samples], dtype=np.float64)

    # Multiquadric RBF Kernel: phi(r) = sqrt(r^2 + eps^2)
    eps = 0.08  # 8cm kernel bandwidth
    dists = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=-1)
    Phi = np.sqrt(dists ** 2 + eps ** 2)
    reg = 1e-4 * np.eye(len(xy))
    weights = np.linalg.solve(Phi + reg, Y)

    # Evaluate self-reconstruction error
    Y_pred = Phi @ weights
    errors_deg = np.abs(Y_pred - Y)
    mean_err_per_joint = np.mean(errors_deg, axis=0)
    max_err_per_joint = np.max(errors_deg, axis=0)

    print("\n" + "=" * 70)
    print("📊 [RBF JOINT MODEL FITTING RESULTS]")
    print("=" * 70)
    print(f"{'Joint':<15} | {'Mean Error (deg)':<18} | {'Max Error (deg)':<18}")
    print("-" * 70)
    for i, name in enumerate(JOINT_NAMES):
        print(f"{name:<15} | {mean_err_per_joint[i]:8.2f} deg       | {max_err_per_joint[i]:8.2f} deg")

    overall_mean = float(np.mean(mean_err_per_joint))
    overall_max = float(np.max(max_err_per_joint))
    print("-" * 70)
    print(f"📈 Overall Mean Joint Error : {overall_mean:.2f} degrees")
    print(f"📈 Overall Max  Joint Error : {overall_max:.2f} degrees")
    print("=" * 70)

    model_data = {
        "type": "rbf_multiquadric",
        "eps": eps,
        "joint_names": JOINT_NAMES,
        "centers_xy": xy.tolist(),
        "weights": weights.tolist(),
        "sample_count": len(samples),
        "overall_mean_err_deg": overall_mean,
        "overall_max_err_deg": overall_max,
        "mean_err_per_joint": {n: float(mean_err_per_joint[i]) for i, n in enumerate(JOINT_NAMES)},
        "note": "2-Step Taught demonstration RBF model (XY -> Joints).",
    }

    out_file = root / MODEL_SAVE_PATH
    out_file.write_text(json.dumps(model_data, indent=2), encoding="utf-8")
    print(f"💾 Model saved successfully to: {out_file}\n")
    return model_data


def check_stdin():
    """Non-blocking check if user hit Enter or typed in terminal."""
    if select.select([sys.stdin], [], [], 0.0)[0]:
        return sys.stdin.readline().strip()
    return None


def main():
    root = lerobot_root()
    yolo_model_path = root / "project/models/yolo_block_detector/best.pt"
    det_cfg_path = root / "project/config/detector.json"
    calib_file = root / CALIB_PATH

    if not calib_file.is_file():
        print(f"[ERROR] Calibration file not found: {calib_file}")
        sys.exit(1)

    g_data = json.loads(calib_file.read_text())
    H = np.array(g_data["homography_pixel_to_robot_xy_m"], dtype=np.float64)

    print("Loading YOLO detector...")
    detector = YoloBlockDetector.load(
        str(yolo_model_path),
        str(det_cfg_path) if det_cfg_path.is_file() else None,
        frame_color="rgb",
    )

    print("Connecting Robot & Teleoperator...")
    follower = make_robot_from_config(
        SO101FollowerConfig(
            port="/dev/so101_follower",
            id="follower",
            disable_torque_on_disconnect=False,
            cameras={"top": OpenCVCameraConfig(index_or_path="/dev/cam_top", width=640, height=480, fps=30)},
        )
    )
    teleop = make_teleoperator_from_config(
        SO101LeaderConfig(
            port="/dev/so101_leader",
            id="leader",
        )
    )

    follower.connect()
    teleop.connect()

    for motor in JOINT_NAMES:
        follower.bus.write("P_Coefficient", motor, 32)

    samples: list[dict[str, Any]] = []
    samples_file = root / SAMPLES_SAVE_PATH

    # Fresh start or append
    if samples_file.is_file():
        ans = input(f"기존 {SAMPLES_SAVE_PATH} 파일이 있습니다. 새로 시작하시겠습니까? [Y/n]: ").strip().lower()
        if ans not in ("", "y", "yes"):
            try:
                existing = json.loads(samples_file.read_text()).get("samples", [])
                samples.extend(existing)
                print(f"기존 {len(samples)}개 샘플 로드됨.")
            except Exception:
                pass

    print()
    print("=" * 70)
    print("🎮 2-Step 1-Block Joint Teaching Mode (Snapshot -> Teach)")
    print("=" * 70)
    print("2단계 진행 방식:")
    print("  [1단계: Snapshot] 블록 1개를 놓고 로봇이 뒤로 빠진 상태에서 [ENTER]")
    print("                    -> 깨끗한 시야에서 블록 (X, Y) 좌표 캡처!")
    print("  [2단계: Teach]    리더암으로 팔로워암을 블록 상공에 맞춘 뒤 [ENTER]")
    print("                    -> 1단계 좌표에 현재 관절값을 완벽 페어링 저장!")
    print("  • 'u' : 마지막 샘플 삭제 (Undo)")
    print("  • 'f' : 수집 완료 및 RBF 모델 피팅/저장 (Fit & Save)")
    print("  • 'q' : 종료")
    print("=" * 70)

    state = "SNAPSHOT"  # "SNAPSHOT" or "TEACH"
    pending_block_info: dict[str, Any] | None = None

    sample_target_count = 45
    print(f"\n📸 [샘플 #{len(samples)+1}/{sample_target_count}] 1단계: 블록을 놓고 [ENTER]를 누르세요...")

    fps = 30.0
    dt = 1.0 / fps

    try:
        while True:
            t0 = time.perf_counter()

            # 1. Real-time Teleoperation
            action = teleop.get_action()
            follower.send_action(action)

            # 2. Handle Terminal Commands
            user_cmd = check_stdin()
            if user_cmd is not None:
                cmd = user_cmd.lower().strip()
                if cmd in ("f", "fit"):
                    if len(samples) < 6:
                        print(f"⚠️ 최소 6개 이상 샘플이 필요합니다. (현재 {len(samples)}개)")
                    else:
                        fit_and_save_rbf_model(samples, root)
                        break
                elif cmd in ("q", "quit"):
                    print("종료합니다.")
                    break
                elif cmd in ("u", "undo"):
                    if samples:
                        removed = samples.pop()
                        samples_file.write_text(json.dumps({"samples": samples}, indent=2), encoding="utf-8")
                        print(f"🗑️ 마지막 샘플 #{len(samples)+1} 삭제됨! (남은 샘플: {len(samples)}개)")
                        state = "SNAPSHOT"
                        pending_block_info = None
                        print(f"\n📸 [샘플 #{len(samples)+1}/{sample_target_count}] 1단계: 블록을 놓고 [ENTER]를 누르세요...")
                    else:
                        print("⚠️ 삭제할 샘플이 없습니다.")
                else:
                    # ENTER pressed
                    if state == "SNAPSHOT":
                        # STEP 1: Capture top camera and locate the block cleanly
                        obs = follower.get_observation()
                        frame = obs.get("top")
                        if frame is None:
                            print("❌ 카메라 영상 획득 실패! 다시 시도하세요.")
                            continue

                        res = detector.detect(frame)
                        if not res.blocks:
                            print("⚠️ 블록이 검출되지 않았습니다! 카메라 시야 및 조명을 확인하세요.")
                            continue

                        # Pick block (if multiple, pick largest or red)
                        block = max(res.blocks, key=lambda b: b.bbox[2] * b.bbox[3])
                        pt = np.array([[[block.cx, block.cy]]], dtype=np.float64)
                        rx, ry = cv2.perspectiveTransform(pt, H)[0, 0]
                        rx_cm, ry_cm = float(rx * 100.0), float(ry * 100.0)
                        r_dist_cm = float(np.hypot(rx_cm, ry_cm))
                        azimuth_deg = float(np.degrees(np.arctan2(ry, rx)))

                        # Save preview image
                        try:
                            preview_dir = root / "var"
                            preview_dir.mkdir(parents=True, exist_ok=True)
                            preview_img = detector.draw(frame, res)
                            # Convert RGB to BGR for cv2.imwrite
                            cv2.imwrite(str(preview_dir / "teach_preview.jpg"), cv2.cvtColor(preview_img, cv2.COLOR_RGB2BGR))
                        except Exception:
                            pass

                        pending_block_info = {
                            "color": block.color,
                            "pixel": [float(block.cx), float(block.cy)],
                            "robot_xy_m": [float(rx), float(ry)],
                            "r_cm": r_dist_cm,
                            "azimuth_deg": azimuth_deg,
                        }

                        print(f"📸 [STEP 1 완료] {block.color.upper()} 검출 -> X={rx_cm:5.1f}cm, Y={ry_cm:5.1f}cm (R={r_dist_cm:4.1f}cm, Az={azimuth_deg:+5.1f}°)")
                        print("👉 [STEP 2] 리더암을 조작하여 블록 상공에 맞춘 뒤 [ENTER]를 누르세요...")
                        state = "TEACH"

                    elif state == "TEACH":
                        # STEP 2: Capture current follower joints and pair with STEP 1 coordinates
                        obs = follower.get_observation()
                        cur_joints = {n: float(obs[f"{n}.pos"]) for n in JOINT_NAMES}
                        cur_joints["gripper"] = float(obs.get("gripper.pos", 45.0))

                        sample_idx = len(samples) + 1
                        sample = {
                            "id": sample_idx,
                            "color": pending_block_info["color"],
                            "pixel": pending_block_info["pixel"],
                            "robot_xy_m": pending_block_info["robot_xy_m"],
                            "r_cm": pending_block_info["r_cm"],
                            "azimuth_deg": pending_block_info["azimuth_deg"],
                            "joints": cur_joints,
                        }
                        samples.append(sample)
                        samples_file.write_text(json.dumps({"samples": samples}, indent=2), encoding="utf-8")

                        rx_cm, ry_cm = sample["robot_xy_m"][0] * 100, sample["robot_xy_m"][1] * 100
                        print(
                            f"✅ [SAMPLE #{sample_idx:02d}/{sample_target_count} 등록] "
                            f"(X={rx_cm:5.1f}cm, Y={ry_cm:5.1f}cm) "
                            f"-> Pan={cur_joints['shoulder_pan']:+5.1f}°, Lift={cur_joints['shoulder_lift']:+5.1f}°, "
                            f"Elb={cur_joints['elbow_flex']:+5.1f}°, Flx={cur_joints['wrist_flex']:+5.1f}°"
                        )

                        if len(samples) >= sample_target_count:
                            print(f"\n🎉 목표 {sample_target_count}개 샘플 수집 완료! RBF 모델을 학습하고 저장합니다...")
                            fit_and_save_rbf_model(samples, root)
                            break

                        state = "SNAPSHOT"
                        pending_block_info = None
                        print(f"\n📸 [샘플 #{len(samples)+1}/{sample_target_count}] 1단계: 블록을 다음 위치에 놓고 [ENTER]를 누르세요...")

            # 30 FPS rate
            elapsed = time.perf_counter() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

    finally:
        follower.disconnect()
        teleop.disconnect()
        print("연결 해제 완료.")


if __name__ == "__main__":
    main()
