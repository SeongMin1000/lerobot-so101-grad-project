#!/usr/bin/env python3
"""Auto-generate YOLO-format bbox labels for extracted frames using the
existing OpenCV TopBlockDetector, so a human only needs to correct mistakes
instead of drawing every box from scratch.
"""

import argparse
from pathlib import Path

import cv2

from lerobot.grad_project.perception.opencv_block_detector import TopBlockDetector

CLASS_NAMES = ["red", "blue", "green", "yellow", "wood"]
CLASS_ID = {name: i for i, name in enumerate(CLASS_NAMES)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images-dir", default="var/yolo_frames")
    parser.add_argument("--detector-config", default="project/config/detector.json")
    parser.add_argument("--out-labels-dir", default="var/yolo_frames_labels")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    out_dir = Path(args.out_labels_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    detector = TopBlockDetector.load(args.detector_config)

    image_paths = sorted(images_dir.glob("*.jpg"))
    print(f"found {len(image_paths)} images in {images_dir}")

    n_boxes_total = 0
    n_unknown_color = 0
    empty_images = []

    for img_path in image_paths:
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"  WARN: could not read {img_path}")
            continue

        result = detector.detect(frame)
        # NOTE: result.outside_blocks is a SUBSET of result.blocks (blocks not
        # yet in the target zone) -- do not concatenate them, that double-counts
        # every block still outside the target.
        all_blocks = list(result.blocks)

        h, w = frame.shape[:2]
        lines = []
        for block in all_blocks:
            if block.color not in CLASS_ID:
                n_unknown_color += 1
                continue
            x, y, bw, bh = block.bbox
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            nw = bw / w
            nh = bh / h
            lines.append(f"{CLASS_ID[block.color]} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
            n_boxes_total += 1

        if not lines:
            empty_images.append(img_path.name)

        label_path = out_dir / (img_path.stem + ".txt")
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""))

    print(f"\nDone. {n_boxes_total} boxes written across {len(image_paths)} images.")
    print(f"average boxes/image: {n_boxes_total / max(1, len(image_paths)):.2f}")
    if n_unknown_color:
        print(f"WARNING: {n_unknown_color} detections had unrecognized color, skipped")
    if empty_images:
        print(f"\n{len(empty_images)} images got ZERO boxes (check these first):")
        for name in empty_images[:30]:
            print(f"  {name}")
        if len(empty_images) > 30:
            print(f"  ... and {len(empty_images) - 30} more")


if __name__ == "__main__":
    main()
