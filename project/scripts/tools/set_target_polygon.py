#!/usr/bin/env python3
"""Click the target zone's four inner corners to (re)define it.

`target_polygon` in detector.json decides two things: whether a block counts
as already placed, and -- through pixel_to_table.py -- the origin and axes of
the table coordinate frame the drop slots are laid out in. The value shipped
here came from an earlier setup and no longer matches the tray, which is why
blocks sitting outside the white area were being reported as placed.

Auto-detecting the tray is unreliable: the checkerboard is full of bright
rectangles and the blocks break the white region up. Clicking is unambiguous.

Click the four corners of the WHITE INTERIOR, inside the dark border, in this
order (as seen on screen, not as seen from where you stand):

    1. top-left      2. top-right      3. bottom-right      4. bottom-left

Keys:  r = start over    s = save    q = quit without saving

NOTE: saving moves the table frame, so the drop slots move with it. Re-run
save_table_calibration_point.py afterwards so the robot-side mapping agrees.
"""

import json
import sys

import cv2
import numpy as np

from lerobot.grad_project.paths import lerobot_root

WINDOW = "click 4 inner corners: TL, TR, BR, BL   (r=reset  s=save  q=quit)"
LABELS = ["TL", "TR", "BR", "BL"]


def grab_frame() -> np.ndarray:
    capture = cv2.VideoCapture("/dev/cam_top", cv2.CAP_V4L2)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    frame = None
    for _ in range(10):
        ok, frame = capture.read()
    capture.release()
    if frame is None:
        raise SystemExit("could not read from /dev/cam_top")
    return frame


def render(frame: np.ndarray, points: list[tuple[int, int]], current: list) -> np.ndarray:
    view = frame.copy()
    if current:
        cv2.polylines(view, [np.array(current, dtype=np.int32)], True, (0, 0, 255), 1)
    for i, (x, y) in enumerate(points):
        cv2.circle(view, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(view, LABELS[i], (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    if len(points) == 4:
        cv2.polylines(view, [np.array(points, dtype=np.int32)], True, (0, 255, 0), 2)
        cv2.putText(view, "press s to save", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(
            view, f"click {LABELS[len(points)]}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
        )
    return view


def main() -> None:
    config_path = lerobot_root() / "project/config/detector.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    current = config.get("target_polygon", [])

    frame = grab_frame()
    points: list[tuple[int, int]] = []

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))

    cv2.namedWindow(WINDOW)
    cv2.setMouseCallback(WINDOW, on_mouse)
    print(__doc__)
    print(f"current target_polygon: {current}")

    while True:
        cv2.imshow(WINDOW, render(frame, points, current))
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            print("cancelled, nothing written")
            break
        if key == ord("r"):
            points.clear()
        if key == ord("s"):
            if len(points) != 4:
                print(f"need 4 corners, have {len(points)}")
                continue
            config["target_polygon"] = [[int(x), int(y)] for x, y in points]
            config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"saved target_polygon = {config['target_polygon']}")
            print("now re-run save_table_calibration_point.py so the drop slots follow the new frame")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
