from pathlib import Path
from datetime import datetime
import os
import shutil

from lerobot.datasets.lerobot_dataset import LeRobotDataset

HF_USER = os.environ.get("HF_USER", "eslab1234")
ROOT = Path.home() / ".cache/huggingface/lerobot" / HF_USER

OUT_NAME = os.environ.get(
    "OUT_NAME",
    "grasp_oneblock_8zones_5colors_pregrasp_v2_merged",
)
OUT_REPO_ID = f"{HF_USER}/{OUT_NAME}"
OUT_ROOT = ROOT / OUT_NAME

SOURCE_NAMES = [
    "grasp_A1_5colors_pregrasp_v2_20260709_221238",
    "grasp_A2_5colors_pregrasp_v2_20260709_230638",
    "grasp_A3_5colors_pregrasp_v2_20260709_231738",
    "grasp_A3_5colors_pregrasp_v2_20260709_234747",
    "grasp_B1_5colors_pregrasp_v2_20260709_234957",
    "grasp_B2_5colors_pregrasp_v2_20260709_235810",
    "grasp_B2_5colors_pregrasp_v2_20260709_235956",
    "grasp_C1_5colors_pregrasp_v2_20260710_000736",
    "grasp_C2_5colors_pregrasp_v2_20260710_001311",
    "grasp_C3_5colors_pregrasp_v2_20260710_002049",
]

AUTO_KEYS = {
    "index",
    "episode_index",
    "frame_index",
    "timestamp",
    "task_index",
}

print("=" * 80)
print("[MERGE TARGET]")
print("OUT_REPO_ID:", OUT_REPO_ID)
print("OUT_ROOT:", OUT_ROOT)
print("=" * 80)

if OUT_ROOT.exists():
    raise SystemExit(f"이미 output 폴더가 있음. 삭제 후 다시 실행해야 함:\nrm -rf {OUT_ROOT}")

# 기준 dataset 로드
base_name = SOURCE_NAMES[0]
base_root = ROOT / base_name
base = LeRobotDataset(
    repo_id=f"{HF_USER}/{base_name}",
    root=base_root,
    return_uint8=True,
)

frame_keys = [k for k in base.features.keys() if k not in AUTO_KEYS]

print("[BASE]")
print("repo:", f"{HF_USER}/{base_name}")
print("fps:", base.fps)
print("episodes:", base.num_episodes)
print("frames:", base.num_frames)
print("frame_keys:", frame_keys)

# output dataset 생성
dst = LeRobotDataset.create(
    repo_id=OUT_REPO_ID,
    root=OUT_ROOT,
    fps=base.fps,
    features=base.features,
    robot_type=base.meta.robot_type,
    use_videos=True,
    image_writer_processes=0,
    image_writer_threads=4,
    encoder_threads=2,
)

merged_episodes = 0
merged_frames = 0

try:
    for src_name in SOURCE_NAMES:
        src_root = ROOT / src_name
        src_repo_id = f"{HF_USER}/{src_name}"

        print("\n" + "=" * 80)
        print("[SOURCE]", src_repo_id)
        print("=" * 80)

        if not src_root.exists():
            raise FileNotFoundError(f"source folder not found: {src_root}")

        src = LeRobotDataset(
            repo_id=src_repo_id,
            root=src_root,
            return_uint8=True,
        )

        if src.fps != base.fps:
            raise ValueError(f"FPS mismatch: {src_repo_id} fps={src.fps}, base={base.fps}")

        if src.features != base.features:
            raise ValueError(f"Feature mismatch: {src_repo_id}")

        print("episodes:", src.num_episodes)
        print("frames:", src.num_frames)

        for ep_idx in range(src.num_episodes):
            ep_ds = LeRobotDataset(
                repo_id=src_repo_id,
                root=src_root,
                episodes=[ep_idx],
                return_uint8=True,
            )

            print(f"[MERGE] {src_name} episode {ep_idx} -> merged episode {merged_episodes} / frames={len(ep_ds)}")

            for i in range(len(ep_ds)):
                item = ep_ds[i]

                frame = {}
                for k in frame_keys:
                    if k not in item:
                        raise KeyError(f"missing key '{k}' in {src_repo_id} episode {ep_idx} frame {i}")
                    frame[k] = item[k]

                frame["task"] = item.get("task", "grasp one block from pregrasp and lift")
                dst.add_frame(frame)

            dst.save_episode()
            merged_episodes += 1
            merged_frames += len(ep_ds)

finally:
    print("\n[FINALIZE]")
    dst.finalize()

print("\n" + "=" * 80)
print("[MERGE DONE]")
print("OUT_REPO_ID:", OUT_REPO_ID)
print("OUT_ROOT:", OUT_ROOT)
print("merged_episodes:", merged_episodes)
print("merged_frames:", merged_frames)
print("=" * 80)
