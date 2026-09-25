#!/usr/bin/env python3
"""Guided capture -> auto-label -> Roboflow-ready dataset, for a moved camera.

Moving the top camera invalidates the training set: every image the current
model learned from was shot from the old pose. Rather than hand-drawing boxes
again, the existing model does the first pass -- it is wrong about *where the
camera is*, not about *what a red block looks like*, so its labels only need
nudging rather than authoring. That turns a multi-day labelling job into a
review pass.

Three subcommands, run in order:

  plan      Print the shot list -- which blocks go roughly where, in words.
            No absolute coordinates: the point is to spread the blocks over
            the board and over orientations, and eyeballed placement does
            that better than trying to hit numeric targets by hand.

  shoot     Capture one arrangement. Takes several frames spaced a few
            seconds apart -- nudge or rotate a block between shots and one
            arrangement yields several genuinely different images.

  finish    Auto-label everything with the current model, split train/valid,
            and zip it in the layout Roboflow imports directly (labels come
            in already attached, so you land straight on the review screen).

Example:
    python project/scripts/tools/collect_yolo_dataset.py plan
    python project/scripts/tools/collect_yolo_dataset.py shoot --tag=a01
    ... repeat for each arrangement ...
    python project/scripts/tools/collect_yolo_dataset.py finish
"""

import argparse
import json
import random
import shutil
import time
from pathlib import Path

import cv2

from lerobot.grad_project.paths import lerobot_root

OUT_DIR = "var/yolo_dataset_v2"
RAW_DIR = f"{OUT_DIR}/raw"
MODEL_PATH = "project/models/yolo_block_detector/best.pt"
# Roboflow's export order for this project; keep it or the class ids shift.
CLASS_NAMES = ["blue", "green", "red", "wood", "yellow"]

COLORS_KO = {"red": "빨강", "green": "초록", "blue": "파랑", "yellow": "노랑", "wood": "나무색"}

# Spots named by landmark rather than by grid cell. "로봇 가까이 왼쪽" leaves
# the operator guessing where the boundary between near and middle is; "목표
# 구역 왼쪽 옆 10cm" does not. Distances are rough on purpose -- the point is
# to cover the board, not to hit a coordinate.
SPOTS = [
    "로봇팔 바로 앞 (목표구역까지 가기 전), 왼쪽",
    "로봇팔 바로 앞 (목표구역까지 가기 전), 가운데",
    "로봇팔 바로 앞 (목표구역까지 가기 전), 오른쪽",
    "목표구역 왼쪽 옆으로 10cm쯤",
    "목표구역 오른쪽 옆으로 10cm쯤",
    "목표구역 왼쪽 옆에 딱 붙여서",
    "목표구역 오른쪽 옆에 딱 붙여서",
    "목표구역 너머(로봇에서 먼 쪽) 왼쪽",
    "목표구역 너머(로봇에서 먼 쪽) 가운데",
    "목표구역 너머(로봇에서 먼 쪽) 오른쪽",
    "판 왼쪽 먼 구석",
    "판 오른쪽 먼 구석",
]
POSES = [
    "반듯하게",
    "살짝 비스듬히",
    "45도쯤 돌려서",
    "많이 돌려서",
]
EXTRAS = [
    "두 개를 서로 붙여놓기",
    "하나는 목표구역 경계선에 살짝 걸치게",
    "하나는 목표구역 안에 넣어두기",
    "전부 최대한 넓게 흩어놓기",
    "세 개를 한쪽에 몰아놓기",
    "하나는 판 제일 끝 구석에",
]


def _paths() -> tuple[Path, Path]:
    root = lerobot_root()
    raw = root / RAW_DIR
    raw.mkdir(parents=True, exist_ok=True)
    return root, raw


def cmd_plan(args) -> None:
    rng = random.Random(args.seed)
    print(f"총 {args.arrangements}개 배치 · 배치당 {args.shots}장 = 약 {args.arrangements * args.shots}장\n")
    # A run walks the board from "all five scattered" to "all five in the
    # zone", so the model has to read every stage in between -- including the
    # crowded ones, where placed blocks sit a few cm apart and partly hide
    # each other. Scattered-only arrangements never produce that, so a third
    # of them are staged mid-run instead.
    staged_every = 3
    in_zone_cycle = [1, 2, 3, 4, 5]
    staged_index = 0

    for i in range(1, args.arrangements + 1):
        colors = CLASS_NAMES[:]
        rng.shuffle(colors)
        print(f"[a{i:02d}]")

        if i % staged_every == 0:
            placed = in_zone_cycle[staged_index % len(in_zone_cycle)]
            staged_index += 1
            print(f"    ** 목표구역 안에 {placed}개, 나머지는 밖에 ** (실제 작업 중간 상태)")
            for color in colors[:placed]:
                print(f"    {COLORS_KO[color]:4s} -> 목표구역 안, {rng.choice(POSES)}")
            for color, spot in zip(colors[placed:], rng.sample(SPOTS, len(colors) - placed), strict=True):
                print(f"    {COLORS_KO[color]:4s} -> {spot}, {rng.choice(POSES)}")
            if placed >= 2:
                print("    (+ 구역 안 블록끼리 5cm쯤 간격으로, 서로 살짝 가리게도 해보기)")
            continue

        # Distinct spots per arrangement. Sampling each block's spot
        # independently clumps them -- half the shots came out with four
        # blocks in the far row, which teaches the model nothing about the
        # near half of the board.
        spots = rng.sample(SPOTS, len(colors))
        for color, spot in zip(colors, spots, strict=True):
            print(f"    {COLORS_KO[color]:4s} -> {spot}, {rng.choice(POSES)}")
        print(f"    (+ {rng.choice(EXTRAS)})")
    print("\n배치할 때마다:  shoot --tag=aNN")
    print("촬영은 10초 간격으로 여러 장 -- 그 사이에 블록 5개를 전부 돌려주세요.")
    print("블록 하나하나가 각각 학습 예제라, 5개 다 돌리면 한 배치에서 얻는 예제가 몇 배로 늘어남.")
    print("각도를 정확히 맞출 필요는 없고, 서로 다른 방향이 되게 툭툭 돌리면 됨.")


def cmd_shoot(args) -> None:
    _, raw = _paths()
    capture = cv2.VideoCapture("/dev/cam_top", cv2.CAP_V4L2)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    try:
        for _ in range(10):  # let exposure settle
            capture.read()
        for shot in range(1, args.shots + 1):
            if shot > 1:
                print(f"  ... {args.interval:.0f}초 후 다음 장 — 블록 5개 전부 아무렇게나 돌려주세요")
                time.sleep(args.interval)
            ok, frame = capture.read()
            if not ok:
                raise SystemExit("카메라 프레임 읽기 실패")
            out = raw / f"{args.tag}_{shot:02d}.jpg"
            cv2.imwrite(str(out), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            print(f"  [{shot}/{args.shots}] 저장: {out.name}")
    finally:
        capture.release()
    print(f"총 {len(list(raw.glob('*.jpg')))}장 모임")


def cmd_finish(args) -> None:
    from lerobot.grad_project.perception.yolo_block_detector import YoloBlockDetector

    root, raw = _paths()
    images = sorted(raw.glob("*.jpg"))
    if not images:
        raise SystemExit(f"{raw} 에 사진이 없음 -- shoot 부터 하세요")

    detector = YoloBlockDetector.load(
        str(root / MODEL_PATH), root / "project/config/detector.json", frame_color="bgr", conf=args.conf
    )
    class_id = {name: i for i, name in enumerate(CLASS_NAMES)}

    out = root / OUT_DIR / "roboflow_upload"
    shutil.rmtree(out, ignore_errors=True)
    rng = random.Random(args.seed)
    counts = {"train": 0, "valid": 0}
    total_boxes = 0
    thin = []

    for image_path in images:
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        # detect() also filters to the workspace and one box per colour, which
        # is exactly the cleanup a hand-labelled set would need anyway.
        result = detector.detect(frame)
        height, width = frame.shape[:2]

        lines = []
        for block in result.blocks:
            if block.color not in class_id:
                continue
            x, y, w, h = block.bbox
            lines.append(
                f"{class_id[block.color]} {(x + w / 2) / width:.6f} {(y + h / 2) / height:.6f} "
                f"{w / width:.6f} {h / height:.6f}"
            )
        if len(lines) < args.expect_blocks:
            thin.append((image_path.name, len(lines)))

        split = "valid" if rng.random() < args.valid_frac else "train"
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)
        shutil.copy(image_path, out / split / "images" / image_path.name)
        (out / split / "labels" / f"{image_path.stem}.txt").write_text("\n".join(lines) + "\n")
        counts[split] += 1
        total_boxes += len(lines)

    (out / "data.yaml").write_text(
        "train: ../train/images\nval: ../valid/images\n\n"
        f"nc: {len(CLASS_NAMES)}\nnames: {CLASS_NAMES}\n"
    )

    archive = shutil.make_archive(str(root / OUT_DIR / "roboflow_upload"), "zip", root_dir=out)
    print(f"\n{len(images)}장 라벨링 완료 (train {counts['train']} / valid {counts['valid']})")
    print(f"박스 총 {total_boxes}개, 장당 평균 {total_boxes / max(1, len(images)):.2f}개")
    if thin:
        print(f"\n블록이 {args.expect_blocks}개 미만으로 잡힌 사진 {len(thin)}장 -- 검수 때 우선 확인:")
        for name, n in thin[:15]:
            print(f"  {name}: {n}개")
        if len(thin) > 15:
            print(f"  ... 외 {len(thin) - 15}장")
    print(f"\nRoboflow 업로드용 zip: {archive}")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p_plan = sub.add_parser("plan")
    p_plan.add_argument("--arrangements", type=int, default=18)
    p_plan.add_argument("--shots", type=int, default=4)
    p_plan.add_argument("--seed", type=int, default=0)
    p_plan.set_defaults(func=cmd_plan)

    p_shoot = sub.add_parser("shoot")
    p_shoot.add_argument("--tag", required=True)
    p_shoot.add_argument("--shots", type=int, default=4)
    p_shoot.add_argument("--interval", type=float, default=10.0)
    p_shoot.set_defaults(func=cmd_shoot)

    p_finish = sub.add_parser("finish")
    p_finish.add_argument("--conf", type=float, default=0.35)
    p_finish.add_argument("--valid-frac", type=float, default=0.15)
    p_finish.add_argument("--expect-blocks", type=int, default=5)
    p_finish.add_argument("--seed", type=int, default=0)
    p_finish.set_defaults(func=cmd_finish)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
