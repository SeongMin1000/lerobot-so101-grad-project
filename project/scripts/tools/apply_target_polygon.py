#!/usr/bin/env python3
"""Write a clicked target polygon into detector.json.

Takes the JSON line printed by the picker page, e.g.

    python project/scripts/tools/apply_target_polygon.py '[[233,193],[395,190],[397,271],[234,273]]'

Order must be top-left, top-right, bottom-right, bottom-left as seen in the
camera image -- pixel_to_table.py maps those to (0,0), (20,0), (20,10) and
(0,10) cm, so a different order silently rotates or mirrors the table frame.
"""

import json
import sys

from lerobot.grad_project.paths import lerobot_root


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    points = json.loads(sys.argv[1])
    if len(points) != 4 or any(len(p) != 2 for p in points):
        raise SystemExit(f"expected 4 [x, y] pairs, got: {points}")

    path = lerobot_root() / "project/config/detector.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    previous = config.get("target_polygon")
    config["target_polygon"] = [[int(x), int(y)] for x, y in points]
    path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"was: {previous}")
    print(f"now: {config['target_polygon']}")
    print("\nThe table frame moved with it, so the drop slots did too -- redo")
    print("save_table_calibration_point.py (tl, tr, br, bl, center, then --fit=true).")


if __name__ == "__main__":
    main()
