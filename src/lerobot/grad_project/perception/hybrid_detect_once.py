#!/usr/bin/env python3
"""
Read one top camera frame through LeRobot robot.get_observation(), run OpenCV block detector,
and print the chosen block pixel. This avoids opening /dev/cam_top separately.

Use this when you want a clean pixel coordinate for grid calibration:
  1) Move arm to observe pose or keep it out of the top view.
  2) Place one block at a calibration grid point.
  3) Run this script and copy chosen.cx/chosen.cy.
"""


import json
import logging
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat

import cv2
import draccus

from lerobot.cameras.opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    bi_so_follower,
    koch_follower,
    make_robot_from_config,
    omx_follower,
    so_follower,
)
from lerobot.utils.import_utils import register_third_party_plugins

from lerobot.grad_project.perception.opencv_block_detector import TopBlockDetector
from lerobot.grad_project.paths import detector_calib_path


@dataclass
class DetectOnceConfig:
    robot: RobotConfig
    detector_calib: str = str(detector_calib_path())
    top_key: str = "top"
    save_debug_path: str = ""
    print_full_json: bool = True


@draccus.wrap()
def main(cfg: DetectOnceConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    logging.info("Config:\n%s", pformat(cfg))
    detector_path = detector_calib_path(cfg.detector_calib)
    print(f"[CONFIG] detector_calib={detector_path}")

    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    try:
        obs = robot.get_observation()
        if cfg.top_key not in obs:
            raise KeyError(f"top_key '{cfg.top_key}' not found. obs keys={list(obs)}")
        detector = TopBlockDetector.load(detector_path, frame_color="rgb")
        result = detector.detect(obs[cfg.top_key])
        data = result.to_json()
        if cfg.print_full_json:
            print(json.dumps(data, ensure_ascii=False, indent=2))
        if result.chosen is None:
            print("CHOSEN: NONE")
        else:
            print(f"CHOSEN_PIXEL u={result.chosen.cx:.1f} v={result.chosen.cy:.1f} color={result.chosen.color}")
        if cfg.save_debug_path:
            out = Path(cfg.save_debug_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            debug_bgr = detector.draw(obs[cfg.top_key], result)
            cv2.imwrite(str(out), debug_bgr)
            print(f"[OK] saved debug overlay to {out}")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    register_third_party_plugins()
    main()
