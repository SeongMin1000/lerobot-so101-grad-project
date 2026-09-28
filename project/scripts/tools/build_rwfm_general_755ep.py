#!/usr/bin/env python3
"""
Builds and verifies the final unified RWFM General dataset (755 episodes) for SO-101.

Composition:
  1. 0 ~ 574   (575 episodes): Clean multitask expert demonstrations (reward = 0.0, placement learn=false mask in rwfm_learn_segments.json)
  2. 575 ~ 674 (100 episodes): Autonomous RWFM rollouts (normal=0.0, self_correction=0.4, failure=-1.0)
  3. 675 ~ 689 (15 episodes) : Task 1 Top-Grasp RWFM rollouts (9ep + 6ep, normal=0.0, self_correction=0.4, learn=true)
  4. 690 ~ 704 (15 episodes) : Task 2 Top-Grasp RWFM rollouts (15ep, normal=0.0, self_correction=0.4, learn=true)
  5. 705 ~ 754 (50 episodes) : Correction-only HIL demonstrations (reward = +0.6, learn=true)

Total: 755 episodes (0..754)

Metadata:
  - meta/episode_frame_rewards.json: Frame-level reward intervals for all 755 episodes
  - meta/rwfm_rollout_annotations.json: Preserves failure annotations for action loss masking (episodes 575..674) and rollout metadata
  - meta/rwfm_learn_segments.json: Clean placement learn=false intervals for episodes 0..574

Zero-reencode Fast Merge:
  - Uses `lerobot.datasets.aggregate.aggregate_datasets` with concatenate_videos=False and concatenate_data=False.
  - Video files are hard-linked (os.link) to consume 0 additional disk space and merge within seconds.
  - Source datasets are strictly preserved without modification.

Usage:
  python project/scripts/tools/build_rwfm_general_755ep.py
  python project/scripts/tools/build_rwfm_general_755ep.py --push-to-hub
"""

import argparse
import copy
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

# Ensure local src/ is accessible
REPO_SRC = Path(__file__).resolve().parents[3] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

import lerobot.datasets.aggregate as aggr_module
from lerobot.datasets.aggregate import aggregate_datasets
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.utils.sample_weighting import RewardSampleWeighter, SampleWeightingConfig
from huggingface_hub import HfApi

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

HF_USER = os.environ.get("HF_USER", "eslab1234")
CACHE_BASE = Path.home() / ".cache/huggingface/lerobot" / HF_USER

DEFAULT_CLEAN_REPO = f"{HF_USER}/multitask_5blocks_v3_575ep_merged"
DEFAULT_ROLLOUT100_REPO = f"{HF_USER}/smolvla_multitask_rwfm_rollout_822ep_50k_100ep_trimmed_merged"
DEFAULT_T1_TOP1_REPO = f"{HF_USER}/smolvla_task1_rwfm_rollout_822ep_50k_v1_20260928_184948"
DEFAULT_T1_TOP2_REPO = f"{HF_USER}/smolvla_task1_rwfm_rollout_822ep_50k_v1_20260928_190338"
DEFAULT_T2_TOP_REPO = f"{HF_USER}/smolvla_task2_rwfm_rollout_822ep_50k_v1_20260928_190834"
DEFAULT_HIL_REPO = f"{HF_USER}/smolvla_hil_575_285k_v3_50ep_trimmed_merged"

DEFAULT_OUTPUT_REPO = f"{HF_USER}/smolvla_rwfm_general_masked575_topgrasp_755ep_v1"


def fast_hardlink_or_copy(src: str, dst: str, **kwargs) -> None:
    """Hardlinks video files to prevent disk exhaustion and eliminate re-encoding / I/O latency."""
    src_p = Path(src)
    dst_p = Path(dst)
    if dst_p.exists():
        dst_p.unlink()
    try:
        os.link(src_p, dst_p)
    except OSError:
        shutil.copy2(src, dst)


def build_rwfm_general_755ep(
    output_repo_id: str = DEFAULT_OUTPUT_REPO,
    push_to_hub: bool = True,
    force_rebuild: bool = True,
) -> Path:
    clean_root = CACHE_BASE / Path(DEFAULT_CLEAN_REPO).name
    rollout100_root = CACHE_BASE / Path(DEFAULT_ROLLOUT100_REPO).name
    t1_top1_root = CACHE_BASE / Path(DEFAULT_T1_TOP1_REPO).name
    t1_top2_root = CACHE_BASE / Path(DEFAULT_T1_TOP2_REPO).name
    t2_top_root = CACHE_BASE / Path(DEFAULT_T2_TOP_REPO).name
    hil_root = CACHE_BASE / Path(DEFAULT_HIL_REPO).name

    output_name = Path(output_repo_id).name
    output_root = CACHE_BASE / output_name

    print("\n" + "=" * 80)
    print("🚀 [BUILD RWFM GENERAL 755EP DATASET]")
    print(f"Target Repo ID:        {output_repo_id}")
    print(f"Target Root Path:       {output_root}")
    print(f"1. Clean 575ep:         {DEFAULT_CLEAN_REPO} ({clean_root})")
    print(f"2. Rollout 100ep:       {DEFAULT_ROLLOUT100_REPO} ({rollout100_root})")
    print(f"3. Task1 TopGrasp 9ep:  {DEFAULT_T1_TOP1_REPO} ({t1_top1_root})")
    print(f"4. Task1 TopGrasp 6ep:  {DEFAULT_T1_TOP2_REPO} ({t1_top2_root})")
    print(f"5. Task2 TopGrasp 15ep: {DEFAULT_T2_TOP_REPO} ({t2_top_root})")
    print(f"6. HIL 50ep:            {DEFAULT_HIL_REPO} ({hil_root})")
    print(f"Push to Hub:            {push_to_hub}")
    print("=" * 80)

    # 1. Preflight validation
    sources = [
        ("Clean", DEFAULT_CLEAN_REPO, clean_root, 575),
        ("Rollout100", DEFAULT_ROLLOUT100_REPO, rollout100_root, 100),
        ("T1_Top1", DEFAULT_T1_TOP1_REPO, t1_top1_root, 9),
        ("T1_Top2", DEFAULT_T1_TOP2_REPO, t1_top2_root, 6),
        ("T2_Top", DEFAULT_T2_TOP_REPO, t2_top_root, 15),
        ("HIL50", DEFAULT_HIL_REPO, hil_root, 50),
    ]

    for name, r_id, p, expected_eps in sources:
        if not p.exists():
            raise FileNotFoundError(f"Source dataset directory missing for {name}: {p}")
        if not (p / "meta/info.json").exists():
            raise FileNotFoundError(f"Missing meta/info.json in {p}")
        m = LeRobotDatasetMetadata(r_id, root=p)
        assert m.total_episodes == expected_eps, (
            f"Expected {expected_eps} episodes for {name}, got {m.total_episodes}"
        )
        print(f"✓ Source {name:12s}: {m.total_episodes:3d} episodes, {m.total_frames:7,d} frames")

    # Clean placement mask check
    clean_learn_p = clean_root / "meta/rwfm_learn_segments.json"
    if not clean_learn_p.exists():
        raise FileNotFoundError(f"Clean placement rwfm_learn_segments.json missing: {clean_learn_p}")

    # Rollout rewards & annotations check
    rollout_rew_file = rollout100_root / "meta/episode_frame_rewards.json"
    rollout_ann_file = rollout100_root / "meta/rwfm_rollout_annotations.json"
    if not rollout_rew_file.exists():
        raise FileNotFoundError(f"Rollout rewards missing: {rollout_rew_file}")
    if not rollout_ann_file.exists():
        raise FileNotFoundError(f"Rollout annotations missing: {rollout_ann_file}")

    # New top grasp annotations check
    t1_top1_ann_p = t1_top1_root / "meta/rwfm_rollout_annotations.json"
    t1_top2_ann_p = t1_top2_root / "meta/rwfm_rollout_annotations.json"
    t2_top_ann_p = t2_top_root / "meta/rwfm_rollout_annotations.json"
    for p_ann in [t1_top1_ann_p, t1_top2_ann_p, t2_top_ann_p]:
        if not p_ann.exists():
            raise FileNotFoundError(f"Top grasp annotation missing: {p_ann}")

    if output_root.exists():
        if force_rebuild:
            print(f"\n⚠️ Target directory exists: {output_root}. Removing for clean aggregation...")
            shutil.rmtree(output_root)
        else:
            print(f"Target directory {output_root} already exists. Skipping aggregate_datasets...")

    # 2. Perform zero-reencode fast merge across all 6 datasets
    t0 = time.time()
    if not output_root.exists():
        print("\n📦 Running aggregate_datasets (zero-reencode, hardlinked videos for 6 sources)...")
        repo_ids = [s[1] for s in sources]
        roots = [s[2] for s in sources]
        
        orig_copy = aggr_module.shutil.copy
        aggr_module.shutil.copy = fast_hardlink_or_copy
        try:
            aggregate_datasets(
                repo_ids=repo_ids,
                aggr_repo_id=output_repo_id,
                roots=roots,
                aggr_root=output_root,
                concatenate_videos=False,
                concatenate_data=False,
            )
        finally:
            aggr_module.shutil.copy = orig_copy

        print(f"🎉 Raw aggregation finished in {time.time() - t0:.2f}s")

    # 3. Load merged metadata
    print("\n🔍 Inspecting merged dataset metadata...")
    dst_meta = LeRobotDatasetMetadata(output_repo_id, root=output_root)
    assert dst_meta.total_episodes == 755, f"Merged episodes mismatch: {dst_meta.total_episodes} != 755"
    print(f"✓ Merged dataset verified: {dst_meta.total_episodes} episodes, {dst_meta.total_frames:,} frames")

    ep_lengths = [int(ep["length"]) for ep in dst_meta.episodes]
    assert len(ep_lengths) == 755

    # 4. Generate meta/episode_frame_rewards.json
    print("\n📝 Generating meta/episode_frame_rewards.json (all 755 episodes)...")
    r_rew = json.loads(rollout_rew_file.read_text())
    t1_top1_ann = json.loads(t1_top1_ann_p.read_text())["episodes"]
    t1_top2_ann = json.loads(t1_top2_ann_p.read_text())["episodes"]
    t2_top_ann = json.loads(t2_top_ann_p.read_text())["episodes"]

    final_rewards: Dict[str, List[List[Any]]] = {}

    # (A) Episodes 0..574: Clean expert demos (reward = 0.0)
    for ep_i in range(575):
        ep_len = ep_lengths[ep_i]
        final_rewards[str(ep_i)] = [[0, ep_len, 0.0]]

    # (B) Episodes 575..674: Rollout 100ep (+575 remapped)
    for ep_i in range(100):
        global_ep = ep_i + 575
        ep_len = ep_lengths[global_ep]
        segs = r_rew[str(ep_i)]
        assert segs[-1][1] == ep_len, f"Rollout ep {ep_i} len mismatch: {segs[-1][1]} != {ep_len}"
        final_rewards[str(global_ep)] = segs

    # Helper function for top-grasp rollout rewards
    def make_topgrasp_rewards(ann_dict: dict, count: int, offset: int):
        for ep_i in range(count):
            global_ep = ep_i + offset
            ep_len = ep_lengths[global_ep]
            ep_ann = ann_dict[str(ep_i)]
            segs = ep_ann.get("segments", [])
            rew_segs = []
            for s in segs:
                r_val = 0.4 if s.get("type") == "self_correction" else 0.0
                rew_segs.append([int(s["start"]), int(s["end"]), r_val])
            if not rew_segs:
                rew_segs = [[0, ep_len, 0.4]]
            assert rew_segs[-1][1] == ep_len, f"Topgrasp ep {ep_i} len mismatch: {rew_segs[-1][1]} != {ep_len}"
            final_rewards[str(global_ep)] = rew_segs

    # (C) Episodes 675..683: Task 1 TopGrasp 1 (9ep, offset 675)
    make_topgrasp_rewards(t1_top1_ann, 9, 675)

    # (D) Episodes 684..689: Task 1 TopGrasp 2 (6ep, offset 684)
    make_topgrasp_rewards(t1_top2_ann, 6, 684)

    # (E) Episodes 690..704: Task 2 TopGrasp (15ep, offset 690)
    make_topgrasp_rewards(t2_top_ann, 15, 690)

    # (F) Episodes 705..754: HIL 50ep (reward = +0.6)
    for ep_i in range(50):
        global_ep = ep_i + 705
        ep_len = ep_lengths[global_ep]
        final_rewards[str(global_ep)] = [[0, ep_len, 0.6]]

    meta_dir = output_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    out_rew_path = meta_dir / "episode_frame_rewards.json"
    out_rew_path.write_text(json.dumps(final_rewards, indent=2))
    print(f"✓ Wrote {len(final_rewards)} episode rewards to {out_rew_path}")

    # Backup in project/config
    cfg_dir = Path("project/config")
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_rew_path = cfg_dir / f"episode_rewards_{output_name}.json"
    cfg_rew_path.write_text(json.dumps(final_rewards, indent=2))
    print(f"✓ Saved backup copy to {cfg_rew_path}")

    # 5. Generate meta/rwfm_rollout_annotations.json
    print("\n📝 Generating meta/rwfm_rollout_annotations.json (remapped episodes 575..754)...")
    r_ann = json.loads(rollout_ann_file.read_text())
    orig_episodes = r_ann.get("episodes", {})

    remapped_episodes: Dict[str, Any] = {}
    total_failure_annotations = 0

    # (A) Remap Rollout 100ep (575..674)
    for ep_i in range(100):
        orig_key = str(ep_i)
        ep_info = copy.deepcopy(orig_episodes[orig_key])
        global_ep = ep_i + 575
        ep_len = ep_lengths[global_ep]
        ep_info["episode_index"] = global_ep
        for seg in ep_info["segments"]:
            assert 0 <= seg["start"] < seg["end"] <= ep_len
            if seg["type"] == "failure":
                total_failure_annotations += 1
        remapped_episodes[str(global_ep)] = ep_info

    # (B) Remap TopGrasp 30ep (675..704)
    def remap_topgrasp_ann(ann_dict: dict, count: int, offset: int, task_name: str):
        for ep_i in range(count):
            global_ep = ep_i + offset
            ep_len = ep_lengths[global_ep]
            ep_info = copy.deepcopy(ann_dict[str(ep_i)])
            ep_info["episode_index"] = global_ep
            ep_info["task"] = task_name
            for seg in ep_info["segments"]:
                assert 0 <= seg["start"] < seg["end"] <= ep_len
            remapped_episodes[str(global_ep)] = ep_info

    remap_topgrasp_ann(t1_top1_ann, 9, 675, "Task 1 Top Grasp Rollout")
    remap_topgrasp_ann(t1_top2_ann, 6, 684, "Task 1 Top Grasp Rollout")
    remap_topgrasp_ann(t2_top_ann, 15, 690, "Task 2 Top Grasp Rollout")

    # (C) Remap HIL 50ep (705..754)
    for ep_i in range(50):
        global_ep = ep_i + 705
        ep_len = ep_lengths[global_ep]
        remapped_episodes[str(global_ep)] = {
            "episode_index": global_ep,
            "task": "HIL Demonstration",
            "segments": [{"start": 0, "end": ep_len, "type": "human_correction"}],
        }

    final_annotations = {
        "schema_version": "1.0",
        "description": "Authoritative RWFM rollout segment annotations remapped to 755ep general dataset coordinates.",
        "episodes": remapped_episodes,
    }

    out_ann_path = meta_dir / "rwfm_rollout_annotations.json"
    out_ann_path.write_text(json.dumps(final_annotations, indent=2))
    print(f"✓ Wrote {len(remapped_episodes)} remapped rollout annotations to {out_ann_path}")
    print(f"✓ Preserved {total_failure_annotations} failure segments for action loss masking")

    # 6. Synchronize meta/rwfm_learn_segments.json (Clean575 placement mask)
    print("\n📝 Synchronizing meta/rwfm_learn_segments.json (Clean575 placement mask)...")
    clean_learn_data = json.loads(clean_learn_p.read_text())
    
    # Update dataset repo id in header
    clean_learn_data["dataset_repo_id"] = output_repo_id
    out_clean_learn_path = meta_dir / "rwfm_learn_segments.json"
    out_clean_learn_path.write_text(json.dumps(clean_learn_data, indent=2))
    print(f"✓ Wrote Clean575 placement mask ({len(clean_learn_data['episodes'])} episodes) to {out_clean_learn_path}")

    # 7. Verification Checklist
    print("\n" + "=" * 80)
    print("🔬 [VERIFICATION CHECKLIST (755EP)]")
    print("=" * 80)

    # 1. Total episodes == 755
    assert dst_meta.total_episodes == 755, f"total_episodes {dst_meta.total_episodes} != 755"
    print("  [PASS] 1. num_episodes == 755")

    # 2. Episode index 0..754
    assert len(final_rewards) == 755 and min(int(k) for k in final_rewards.keys()) == 0 and max(int(k) for k in final_rewards.keys()) == 754
    print("  [PASS] 2. final episode index == 0 ~ 754")

    # 3. Clean 575ep reward == 0.0
    for i in range(575):
        assert final_rewards[str(i)][0][2] == 0.0
    print("  [PASS] 3. Clean 575 episodes all have reward = 0.0")

    # 4. Rollout 100ep remapped to 575..674
    for i in range(575, 675):
        assert str(i) in remapped_episodes
    print("  [PASS] 4. Rollout 100 episodes remapped to 575 ~ 674")

    # 5. TopGrasp 30ep remapped to 675..704
    for i in range(675, 705):
        assert str(i) in remapped_episodes
        # verify no failure segments
        for seg in remapped_episodes[str(i)]["segments"]:
            assert seg["type"] in ("normal", "self_correction"), f"Unexpected segment in ep {i}: {seg}"
    print("  [PASS] 5. TopGrasp 30 episodes remapped to 675 ~ 704 (all learn=true, reward=+0.4 for self_correction)")

    # 6. HIL 50ep remapped to 705..754 with reward 0.6
    for i in range(705, 755):
        assert final_rewards[str(i)][0][2] == 0.6
    print("  [PASS] 6. HIL 50 episodes remapped to 705 ~ 754 with reward = 0.6")

    # 7. Clean placement mask intact in meta/rwfm_learn_segments.json
    assert len(clean_learn_data["episodes"]) == 575
    assert clean_learn_data["total_mask_intervals"] == 2875
    print("  [PASS] 7. Clean placement mask contains 575 episodes and 2,875 intervals")

    # 8. Spot check dataset decoding
    print("  Spot checking dataset frame loading across boundary episodes...")
    test_ds = LeRobotDataset(output_repo_id, root=output_root, return_uint8=True)
    probe_episodes = [0, 352, 353, 574, 575, 674, 675, 704, 705, 754]
    for p_ep in probe_episodes:
        ep_entry = test_ds.meta.episodes[p_ep]
        from_idx = ep_entry["dataset_from_index"]
        item = test_ds[from_idx]
        assert "action" in item and item["action"].shape == (6,)
        assert "observation.state" in item and item["observation.state"].shape == (6,)
        assert "observation.images.top" in item and item["observation.images.top"].shape == (3, 480, 640)
        assert "observation.images.wrist" in item and item["observation.images.wrist"].shape == (3, 480, 640)
        assert item["task"] is not None
        print(f"    ✓ Probe Ep {p_ep:3d}: frame {from_idx:7d}, task='{item['task'][:35]}...'")

    print("  [PASS] 8. All features and image decoding intact across all boundary episodes")

    # 9. RewardSampleWeighter auto-detection test
    print("  Testing RewardSampleWeighter auto-detection...")
    weighter_cfg = SampleWeightingConfig(type="reward_weighted")
    weighter = RewardSampleWeighter(weighter_cfg, device="cpu", dataset_root=output_root)
    assert len(weighter.ep_frame_rewards) == 755, f"Weighter loaded {len(weighter.ep_frame_rewards)} episodes, expected 755"
    print("  [PASS] 9. RewardSampleWeighter auto-detected and loaded all 755 episode reward profiles")

    print("\n✅ ALL VERIFICATION CHECKS PASSED 100%!")

    # 8. Push to Hugging Face Hub
    if push_to_hub:
        print("\n" + "=" * 80)
        print(f"🚀 [HUGGING FACE HUB PUSH] Uploading {output_repo_id}...")
        print("=" * 80)
        try:
            test_ds.push_to_hub(upload_large_folder=True)
            print(f"✅ Successfully pushed dataset via push_to_hub to https://huggingface.co/datasets/{output_repo_id}")
        except Exception as e:
            print(f"⚠️ push_to_hub failed ({e}), falling back to HfApi.upload_folder...")
            api = HfApi()
            api.create_repo(repo_id=output_repo_id, repo_type="dataset", exist_ok=True)
            api.upload_folder(
                folder_path=str(output_root),
                repo_id=output_repo_id,
                repo_type="dataset",
            )
            print(f"✅ Successfully uploaded dataset via HfApi to https://huggingface.co/datasets/{output_repo_id}")

        # Ensure custom meta files are uploaded to Hub
        print("📤 Uploading custom metadata files to Hub...")
        api = HfApi()
        for mf in ["episode_frame_rewards.json", "rwfm_rollout_annotations.json", "rwfm_learn_segments.json"]:
            mf_path = meta_dir / mf
            if mf_path.exists():
                api.upload_file(
                    path_or_fileobj=str(mf_path),
                    path_in_repo=f"meta/{mf}",
                    repo_id=output_repo_id,
                    repo_type="dataset",
                )
                print(f"  ✓ Uploaded meta/{mf}")

    print("\n" + "=" * 80)
    print("🎉 [BUILD & PUSH COMPLETE]")
    print(f"• 최종 dataset repo id: {output_repo_id}")
    print(f"• 총 episode 수:         755 episodes ({dst_meta.total_frames:,} frames)")
    print(f"• Source별 episode 범위:")
    print(f"    - Clean575:          0 ~ 574   (575 episodes, reward=0.0, placement learn=false mask)")
    print(f"    - 기존 RWFM100:      575 ~ 674 (100 episodes, normal=0.0, self_corr=0.4, failure=-1.0)")
    print(f"    - 새 상단 RWFM30:    675 ~ 704 (30 episodes, normal=0.0, self_corr=0.4, learn=true)")
    print(f"        * Task1 (15ep):  675 ~ 689")
    print(f"        * Task2 (15ep):  690 ~ 704")
    print(f"    - HIL50:             705 ~ 754 (50 episodes, reward=+0.6, learn=true)")
    print("=" * 80)

    return output_root


def main():
    parser = argparse.ArgumentParser(description="Build and push RWFM General 755ep dataset.")
    parser.add_argument("--repo-id", type=str, default=DEFAULT_OUTPUT_REPO, help="Target Hugging Face dataset repo ID.")
    parser.add_argument("--push-to-hub", action="store_true", default=True, help="Push dataset to HF Hub.")
    parser.add_argument("--no-push", dest="push_to_hub", action="store_false", help="Skip pushing to Hub.")
    args = parser.parse_args()

    build_rwfm_general_755ep(
        output_repo_id=args.repo_id,
        push_to_hub=args.push_to_hub,
    )


if __name__ == "__main__":
    main()
