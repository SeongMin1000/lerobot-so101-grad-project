#!/usr/bin/env python3
"""Live YOLO block detector visualization and testing tool.

Captures frames from /dev/cam_top, runs YOLO detection, overlays bounding boxes
and class names, displays a live preview window, and saves snapshot images.

Usage:
  python project/scripts/tools/test_yolo_live.py
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

from lerobot.grad_project.paths import detector_calib_path, lerobot_root
from lerobot.grad_project.perception.yolo_block_detector import YoloBlockDetector


def main():
    root = lerobot_root()
    yolo_model_path = root / "project/models/yolo_block_detector/best.pt"
    det_cfg_path = detector_calib_path("project/config/detector.json", must_exist=False)

    if not yolo_model_path.is_file():
        print(f"[ERROR] YOLO model weights not found at: {yolo_model_path}")
        sys.exit(1)

    print(f"Loading YOLO detector from: {yolo_model_path}")
    detector = YoloBlockDetector.load(
        str(yolo_model_path),
        det_cfg_path if det_cfg_path.is_file() else None,
        frame_color="bgr",
    )

    # Load grasp homography and Z-plane for pixel -> robot cm conversion
    grasp_calib_path = root / "project/config/grasp_pixel_to_robot_record.json"
    if len(sys.argv) > 1:
        grasp_calib_path = Path(sys.argv[1]) if Path(sys.argv[1]).is_absolute() else root / sys.argv[1]
    elif not grasp_calib_path.is_file():
        grasp_calib_path = root / "project/config/grasp_pixel_to_robot.json"

    grasp_homography = None
    grasp_z_plane = None
    if grasp_calib_path.is_file():
        import json
        g_data = json.loads(grasp_calib_path.read_text())
        grasp_homography = np.array(g_data["homography_pixel_to_robot_xy_m"], dtype=np.float64)
        grasp_z_plane = np.array(g_data["grasp_z_plane_abc"], dtype=np.float64)
        print(f"Loaded grasp calibration: {grasp_calib_path}")

    cam_path = "/dev/cam_top"
    print(f"Opening camera: {cam_path}")
    cap = cv2.VideoCapture(cam_path, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print(f"[ERROR] Failed to open camera at {cam_path}")
        sys.exit(1)

    output_snapshot = root / "var/yolo_live_snapshot.jpg"
    print("=" * 60)
    print("🎥 Live YOLO Detection & Robot Coordinate Monitor Started!")
    print(f"📸 Snapshot will be saved to: {output_snapshot}")
    print("Press 'q' in the window or Ctrl+C in terminal to exit.")
    print("=" * 60)

    has_gui = True
    try:
        cv2.namedWindow("YOLO Block Detection", cv2.WINDOW_NORMAL)
    except Exception:
        has_gui = False
        print("[INFO] Headless environment: running in console/snapshot mode.")

    last_save_time = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            result = detector.detect(frame)
            annotated = detector.draw(frame, result)

            # Draw details
            detected_str_list = []
            for b in result.blocks:
                status = "IN" if b.in_target else "OUT"
                if grasp_homography is not None:
                    pt = np.array([[[b.cx, b.cy]]], dtype=np.float64)
                    rx, ry = cv2.perspectiveTransform(pt, grasp_homography)[0, 0]
                    # Convert to cm (record calibration is already ground-truth from ruler)
                    offset = 0.0 if "record" in str(grasp_calib_path) else 0.015
                    rx_cm, ry_cm = (rx + offset) * 100.0, (ry + offset) * 100.0
                    r_dist_cm = float(np.hypot(rx_cm, ry_cm))
                    too_close = " ⚠️[TOO CLOSE]" if r_dist_cm < 18.0 else ""
                    detected_str_list.append(f"{b.color.upper()}({status}, X={rx_cm:.1f}cm, Y={ry_cm:.1f}cm, R={r_dist_cm:.1f}cm{too_close})")

                    # Overlay distance on image
                    bx, by, bw, bh = b.bbox
                    cv2.putText(
                        annotated,
                        f"R={r_dist_cm:.1f}cm",
                        (bx, by + bh + 16),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 0, 255) if r_dist_cm < 18.0 else (0, 255, 255),
                        1,
                    )
                else:
                    detected_str_list.append(f"{b.color.upper()}({status}, cx={b.cx:.0f}, cy={b.cy:.0f})")

            current_time = time.time()
            if current_time - last_save_time >= 1.0:
                cv2.imwrite(str(output_snapshot), annotated)
                last_save_time = current_time
                print(f"[{time.strftime('%H:%M:%S')}] Found {len(result.blocks)} blocks: {', '.join(detected_str_list) if detected_str_list else 'None'}")

            if has_gui:
                cv2.imshow("YOLO Block Detection", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
            else:
                time.sleep(0.033)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        cap.release()
        if has_gui:
            cv2.destroyAllWindows()
        print("Camera released. Done.")


if __name__ == "__main__":
    main()
