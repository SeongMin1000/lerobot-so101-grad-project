#!/usr/bin/env python3
"""Filter the Roboflow export down to the verified-clean subset:
episodes 0-48 (all 6 states) + episodes 49-124 (blocks5 only).
Roboflow renamed files to `epNNN_blocksN_jpg.rf.<hash>.jpg`; we match the
epNNN_blocksN prefix to decide keep/drop, and preserve Roboflow's own
train/valid split and class order (alphabetical: blue,green,red,wood,yellow).
"""
import re
import shutil
from pathlib import Path

SRC = Path("/home/eslab/lerobot/var/roboflow_export")
OUT = Path("/home/eslab/lerobot/var/final_dataset")

PATTERN = re.compile(r"^(ep(\d+)_blocks(\d))_jpg\.rf\.")

EPISODE_CUTOFF = 48  # episodes 0-48 inclusive: keep all states
# episodes 49+: keep only blocks5


def should_keep(ep: int, state: int) -> bool:
    if ep <= EPISODE_CUTOFF:
        return True
    return state == 5


def main():
    shutil.rmtree(OUT, ignore_errors=True)
    kept = 0
    dropped = 0
    for split in ("train", "valid"):
        (OUT / split / "images").mkdir(parents=True, exist_ok=True)
        (OUT / split / "labels").mkdir(parents=True, exist_ok=True)
        img_dir = SRC / split / "images"
        lbl_dir = SRC / split / "labels"
        for img_path in sorted(img_dir.glob("*.jpg")):
            m = PATTERN.match(img_path.name)
            if not m:
                print(f"WARNING: filename didn't match pattern: {img_path.name}")
                continue
            ep = int(m.group(2))
            state = int(m.group(3))
            if not should_keep(ep, state):
                dropped += 1
                continue
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            shutil.copy(img_path, OUT / split / "images" / img_path.name)
            if lbl_path.exists():
                shutil.copy(lbl_path, OUT / split / "labels" / lbl_path.name)
            else:
                (OUT / split / "labels" / (img_path.stem + ".txt")).write_text("")
            kept += 1

    yaml_content = """train: ../train/images
val: ../valid/images

nc: 5
names: ['blue', 'green', 'red', 'wood', 'yellow']
"""
    (OUT / "data.yaml").write_text(yaml_content)

    print(f"kept: {kept}, dropped: {dropped}")
    print(f"final dataset at: {OUT}")


if __name__ == "__main__":
    main()
