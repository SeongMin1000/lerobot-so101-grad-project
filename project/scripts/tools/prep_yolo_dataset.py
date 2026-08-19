#!/usr/bin/env python3
"""Split extracted frames + auto-generated labels into YOLO train/val folders
and write the dataset yaml config, ready for `yolo detect train`."""

import argparse
import random
import shutil
from pathlib import Path

CLASS_NAMES = ["red", "blue", "green", "yellow", "wood"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", default="var/yolo_frames")
    parser.add_argument("--labels-dir", default="var/yolo_frames_labels")
    parser.add_argument("--out-dir", default="var/yolo_dataset")
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=1000)
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    out_dir = Path(args.out_dir)

    image_paths = sorted(images_dir.glob("*.jpg"))
    rng = random.Random(args.seed)
    rng.shuffle(image_paths)

    n_val = max(1, int(len(image_paths) * args.val_fraction))
    val_set = set(p.name for p in image_paths[:n_val])

    for split in ("train", "val"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    n_train = n_val_actual = 0
    for img_path in image_paths:
        split = "val" if img_path.name in val_set else "train"
        label_path = labels_dir / (img_path.stem + ".txt")

        shutil.copy(img_path, out_dir / "images" / split / img_path.name)
        if label_path.exists():
            shutil.copy(label_path, out_dir / "labels" / split / label_path.name)
        else:
            (out_dir / "labels" / split / label_path.name).write_text("")

        if split == "train":
            n_train += 1
        else:
            n_val_actual += 1

    yaml_content = f"""path: {out_dir.resolve()}
train: images/train
val: images/val
names:
"""
    for i, name in enumerate(CLASS_NAMES):
        yaml_content += f"  {i}: {name}\n"

    yaml_path = out_dir / "blocks.yaml"
    yaml_path.write_text(yaml_content)

    print(f"train: {n_train} images, val: {n_val_actual} images")
    print(f"dataset yaml written to: {yaml_path}")
    print(f"\nready to train:\n  yolo detect train data={yaml_path} model=yolov8n.pt epochs=100 imgsz=640")


if __name__ == "__main__":
    main()
