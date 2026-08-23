#!/usr/bin/env python3
"""Task 2: stack all 5 blocks at one point instead of spreading them in a grid.

Reuses every motion primitive from `pick_and_place_yolo.py` (detection,
pixel->robot conversion, orientation, the pick/retry FSM, safety limits) --
only "where does a block go, and how do we know how many are already there"
change. Two things are overridden:

  1. `drop_xyz_for_count` -- instead of one of 5 fixed floor slots, every
     block goes to the SAME (x, y) and the z climbs by one block's height
     per level.
  2. `refresh_queue` -- re-derives the current stack height from the camera
     every time it looks, instead of trusting a counter that only ever goes
     up. A block sitting in the stack zone counts toward the height; a block
     seen OUTSIDE it needs picking up -- including one that fell off, which
     just reappears outside the zone and gets queued again automatically. No
     separate "did it fall" detection needed.

`is_placed()` / `slot_pixels()` come from the base class unchanged: they
already just measure "how close is this block to a slot", and here there is
only one slot (the stack point), which is why the same proximity-with-margin
logic works instead of the target-polygon edge-case that bit Task 1.

CAUTION (not yet tested on the real robot -- verify with --dry_run=true and
then a 2-3 block stack before trusting it for all 5):
  - `drop_xyz_for_count`'s z already includes the current stack height, so
    the clearance moves (`transit_height_m` added on top of it) scale with
    stack height automatically. But the move immediately before that --
    lifting off the picked block, elsewhere on the board -- clears relative
    to where the block was picked, not the stack. Once the stack is a few
    blocks tall, the arm's joint-space path from that lift point over to
    above the stack is not guaranteed to stay above the stack the whole way
    (only the two endpoints are height-safe, not necessarily what's between
    them). Watch this specifically once you're past 2-3 blocks.

Example:
  python -m lerobot.grad_project.control.pick_and_stack_yolo \\
    --robot.type=so101_follower --robot.port=/dev/so101_follower \\
    --robot.id=follower --robot.disable_torque_on_disconnect=false \\
    --robot.max_relative_target=15 --robot.max_tracking_error=150 \\
    --yolo_model_path=project/models/yolo_block_detector/best.pt \\
    --dry_run=true
"""

import logging
from dataclasses import dataclass
from pprint import pformat

import cv2
import draccus
import numpy as np

from lerobot.grad_project.control.pick_and_place_yolo import (
    PickAndPlaceYolo,
    PickAndPlaceYoloConfig,
)
from lerobot.grad_project.control.table_frame_calibration import apply_table_to_robot
from lerobot.grad_project.perception.pixel_to_table import TARGET_HEIGHT_CM, TARGET_WIDTH_CM
from lerobot.utils.import_utils import register_third_party_plugins

logger = logging.getLogger(__name__)


@dataclass
class PickAndStackYoloConfig(PickAndPlaceYoloConfig):
    # Table-frame (cm) point every block stacks on top of. Defaults to the
    # zone centre; move it if that collides with where blocks get detected
    # (e.g. under the arm's resting shadow).
    stack_x_cm: float = TARGET_WIDTH_CM / 2
    stack_y_cm: float = TARGET_HEIGHT_CM / 2

    # Real block thickness -- each stacked level adds this much to the
    # release height. Measure your actual blocks; 2.5cm was this project's
    # nominal size but re-check it, stack error compounds with height.
    block_height_m: float = 0.025
    # Fine-tune knob if the first block doesn't land flush on the table.
    stack_z_offset_m: float = 0.0

    # A bit more headroom than the floor-spread default (0.12m): the transit
    # move has to clear whatever height the stack has already reached.
    transit_height_m: float = 0.15


class PickAndStackYolo(PickAndPlaceYolo):
    cfg: PickAndStackYoloConfig

    def drop_xyz_for_count(self, placed_count: int) -> tuple[str, np.ndarray]:
        xyz = apply_table_to_robot(self.cfg.stack_x_cm, self.cfg.stack_y_cm, self.table_to_robot)
        if self.grasp_z_plane is not None:
            xyz[2] = float(self.grasp_z_plane @ np.array([xyz[0], xyz[1], 1.0]))
        xyz[2] += self.cfg.block_height_m * placed_count + self.cfg.stack_z_offset_m
        xyz = xyz + np.array([self.cfg.grasp_offset_x_m, self.cfg.grasp_offset_y_m, 0.0])
        return f"stack_level{placed_count + 1}", xyz

    def slot_pixels(self) -> list[np.ndarray]:
        inverse = np.linalg.inv(self.homography)
        point = np.array([[[self.cfg.stack_x_cm, self.cfg.stack_y_cm]]], dtype=np.float64)
        return [cv2.perspectiveTransform(point, inverse)[0, 0]]

    def refresh_queue(self) -> int:
        """Look once; re-derive the stack height from what's actually there.

        Trusting a counter that only increments would drift the moment a
        block falls off mid-run (still tracked as "stacked" until the FSM
        crashes into a gap that isn't there). Recomputing placed_count from
        the camera every look means a fallen block just shows up back in the
        queue on the next cycle -- the same recovery path a totally fresh
        block gets, no special case needed.
        """
        self.move_to_pose(self.pose(self.cfg.observe_pose_name), self.cfg.hover_duration_s, self.cfg.observe_pose_name)
        result = self.detect_top()
        stacked = [b for b in result.blocks if self.is_placed(b)]
        self.placed_count = len(stacked)
        self._queue = [b for b in result.blocks if not self.is_placed(b) and b.color not in self._unreachable]
        self._queue.sort(key=lambda b: (self.detector.prefer_colors.index(b.color) if b.color in self.detector.prefer_colors else 999, -b.cy, b.cx))
        logger.info("stack height (from vision): %d, %d block(s) still to stack", self.placed_count, len(self._queue))
        return len(self._queue)


@draccus.wrap()
def main(cfg: PickAndStackYoloConfig) -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    logging.info("Config:\n%s", pformat(cfg))
    fsm = PickAndStackYolo(cfg)
    fsm.run()


if __name__ == "__main__":
    register_third_party_plugins()
    main()
