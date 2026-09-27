#!/usr/bin/env python3
"""
Builds and verifies the final unified RWFM General dataset (725 episodes) for SO-101.

Composition:
  1. 0 ~ 574   (575 episodes): Clean multitask expert demonstrations (reward = 0.0, learn=true)
  2. 575 ~ 674 (100 episodes): Autonomous RWFM rollouts (normal=0.0, self_correction=0.4, failure=-1.0)
  3. 675 ~ 724 (50 episodes) : Correction-only HIL demonstrations (reward = +0.6, learn=true)

Total: 725 episodes (0..724)

Metadata:
  - meta/episode_frame_rewards.json: Frame-level reward intervals for all 725 episodes
  - meta/rwfm_rollout_annotations.json: Preserves failure annotations for action loss masking (episodes 575..674)

Zero-reencode Fast Merge:
  - Uses `lerobot.datasets.aggregate.aggregate_datasets` with concatenate_videos=False and concatenate_data=False.
  - Video files are hard-linked (os.link) to consume 0 additional disk space and merge within seconds.
  - Source datasets are strictly preserved without modification.

Usage:
  python project/scripts/tools/build_rwfm_general_725ep.py
  OUTPUT_REPO_ID="eslab1234/smolvla_rwfm_general_725ep_v1" python project/scripts/tools/build_rwfm_general_725ep.py --push-to-hub
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

HF_USER = os.environ.get("HF_USER", "eslab1234")
CACHE_BASE = Path.home() / ".cache/huggingface/lerobot" / HF_USER

DEFAULT_CLEAN_REPO = f"{HF_USER}/multitask_5blocks_v3_575ep_merged"
DEFAULT_ROLLOUT_REPO = f"{HF_USER}/smolvla_multitask_rwfm_rollout_822ep_50k_100ep_trimmed_merged"
DEFAULT_HIL_REPO = f"{HF_USER}/smolvla_hil_575_285k_v3_50ep_trimmed_merged"

DEFAULT_OUTPUT_REPO = os.environ.get("OUTPUT_REPO_ID", f"{HF_USER}/smolvla_rwfm_general_725ep_v1")


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


def build_rwfm_general_dataset(
    output_repo_id: str,
    clean_repo_id: str,
    rollout_repo_id: str,
    hil_repo_id: str,
    push_to_hub: bool = True,
    force_rebuild: bool = True,
) -> Path:
    clean_root = CACHE_BASE / Path(clean_repo_id).name
    rollout_root = CACHE_BASE / Path(rollout_repo_id).name
    hil_root = CACHE_BASE / Path(hil_repo_id).name

    output_name = Path(output_repo_id).name
    output_root = CACHE_BASE / output_name

    print("\n" + "=" * 80)
    print("🚀 [BUILD RWFM GENERAL 725EP DATASET]")
    print(f"Target Repo ID:    {output_repo_id}")
    print(f"Target Root Path:   {output_root}")
    print(f"1. Clean 575ep:     {clean_repo_id} ({clean_root})")
    print(f"2. Rollout 100ep:   {rollout_repo_id} ({rollout_root})")
    print(f"3. HIL 50ep:        {hil_repo_id} ({hil_root})")
    print(f"Push to Hub:        {push_to_hub}")
    print("=" * 80)

    # 1. Preflight validation
    for name, r_id, p in [
        ("Clean", clean_repo_id, clean_root),
        ("Rollout", rollout_repo_id, rollout_root),
        ("HIL", hil_repo_id, hil_root),
    ]:
        if not p.exists():
            raise FileNotFoundError(f"Source dataset directory missing for {name}: {p}")
        if not (p / "meta/info.json").exists():
            raise FileNotFoundError(f"Missing meta/info.json in {p}")

    rollout_rew_file = rollout_root / "meta/episode_frame_rewards.json"
    rollout_ann_file = rollout_root / "meta/rwfm_rollout_annotations.json"
    hil_rew_file = hil_root / "meta/episode_frame_rewards.json"

    if not rollout_rew_file.exists():
        raise FileNotFoundError(f"Rollout rewards missing: {rollout_rew_file}")
    if not rollout_ann_file.exists():
        raise FileNotFoundError(f"Rollout annotations missing: {rollout_ann_file}")
    if not hil_rew_file.exists():
        raise FileNotFoundError(f"HIL rewards missing: {hil_rew_file}")

    m_clean = LeRobotDatasetMetadata(clean_repo_id, root=clean_root)
    m_rollout = LeRobotDatasetMetadata(rollout_repo_id, root=rollout_root)
    m_hil = LeRobotDatasetMetadata(hil_repo_id, root=hil_root)

    assert m_clean.total_episodes == 575, f"Expected 575 clean episodes, got {m_clean.total_episodes}"
    assert m_rollout.total_episodes == 100, f"Expected 100 rollout episodes, got {m_rollout.total_episodes}"
    assert m_hil.total_episodes == 50, f"Expected 50 HIL episodes, got {m_hil.total_episodes}"

    print(f"✓ Source 1 (Clean 575ep):   {m_clean.total_episodes} episodes, {m_clean.total_frames:,} frames")
    print(f"✓ Source 2 (Rollout 100ep): {m_rollout.total_episodes} episodes, {m_rollout.total_frames:,} frames")
    print(f"✓ Source 3 (HIL 50ep):      {m_hil.total_episodes} episodes, {m_hil.total_frames:,} frames")
    print(f"✓ Total Target:             725 episodes, {m_clean.total_frames + m_rollout.total_frames + m_hil.total_frames:,} frames")

    if output_root.exists():
        if force_rebuild:
            print(f"\n⚠️ Target directory exists: {output_root}. Removing for clean aggregation...")
            shutil.rmtree(output_root)
        else:
            print(f"Target directory {output_root} already exists. Skipping aggregate_datasets...")

    # 2. Perform zero-reencode fast merge
    t0 = time.time()
    if not output_root.exists():
        print("\n📦 Running aggregate_datasets (zero-reencode, hardlinked videos)...")
        # Monkeypatch shutil.copy inside aggregate module to hardlink videos
        orig_copy = aggr_module.shutil.copy
        aggr_module.shutil.copy = fast_hardlink_or_copy
        try:
            aggregate_datasets(
                repo_ids=[clean_repo_id, rollout_repo_id, hil_repo_id],
                aggr_repo_id=output_repo_id,
                roots=[clean_root, rollout_root, hil_root],
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
    assert dst_meta.total_episodes == 725, f"Merged episodes mismatch: {dst_meta.total_episodes} != 725"
    print(f"✓ Merged dataset verified: {dst_meta.total_episodes} episodes, {dst_meta.total_frames:,} frames")

    # Extract episode lengths
    ep_lengths = [int(ep["length"]) for ep in dst_meta.episodes]
    assert len(ep_lengths) == 725

    # 4. Generate meta/episode_frame_rewards.json
    print("\n📝 Generating meta/episode_frame_rewards.json (all 725 episodes)...")
    r_rew = json.loads(rollout_rew_file.read_text())
    h_rew = json.loads(hil_rew_file.read_text())

    final_rewards: Dict[str, List[List[Any]]] = {}

    # (A) Episodes 0..574: Clean expert demos (all frame reward = 0.0)
    for ep_i in range(575):
        ep_len = ep_lengths[ep_i]
        final_rewards[str(ep_i)] = [[0, ep_len, 0.0]]

    # (B) Episodes 575..674: Rollout episodes (+575 remapped)
    rollout_failure_segs = 0
    rollout_self_corr_segs = 0
    for ep_i in range(100):
        global_ep = ep_i + 575
        ep_len = ep_lengths[global_ep]
        segs = r_rew[str(ep_i)]
        assert segs[-1][1] == ep_len, (
            f"Rollout ep {ep_i} length mismatch: segments end at {segs[-1][1]}, episode len is {ep_len}"
        )
        final_rewards[str(global_ep)] = segs
        for s in segs:
            if s[2] < 0:
                rollout_failure_segs += 1
            elif s[2] > 0.05:
                rollout_self_corr_segs += 1

    # (C) Episodes 675..724: Correction-only HIL episodes (+675 remapped, all frame reward = 0.6)
    for ep_i in range(50):
        global_ep = ep_i + 675
        ep_len = ep_lengths[global_ep]
        final_rewards[str(global_ep)] = [[0, ep_len, 0.6]]

    meta_dir = output_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    out_rew_path = meta_dir / "episode_frame_rewards.json"
    out_rew_path.write_text(json.dumps(final_rewards, indent=2))
    print(f"✓ Wrote {len(final_rewards)} episode rewards to {out_rew_path}")

    # Also save backup in project/config/
    cfg_dir = Path("project/config")
    cfg_dir.mkdir(parents=True, exist_ok=True)
    cfg_rew_path = cfg_dir / f"episode_rewards_{output_name}.json"
    cfg_rew_path.write_text(json.dumps(final_rewards, indent=2))
    print(f"✓ Saved backup configuration to {cfg_rew_path}")

    # 5. Generate meta/rwfm_rollout_annotations.json (preserving failure semantic annotations)
    print("\n📝 Generating meta/rwfm_rollout_annotations.json (remapped episodes 575..674)...")
    r_ann = json.loads(rollout_ann_file.read_text())
    orig_episodes = r_ann.get("episodes", {})

    remapped_episodes: Dict[str, Any] = {}
    total_failure_annotations = 0

    for ep_i in range(100):
        orig_key = str(ep_i)
        if orig_key not in orig_episodes:
            raise KeyError(f"Original rollout annotation missing for episode {ep_i}")
        ep_info = copy.deepcopy(orig_episodes[orig_key])
        global_ep = ep_i + 575
        ep_len = ep_lengths[global_ep]

        ep_info["episode_index"] = global_ep
        for seg in ep_info["segments"]:
            assert 0 <= seg["start"] < seg["end"] <= ep_len, (
                f"Invalid segment bounds: {seg} in global ep {global_ep} (len={ep_len})"
            )
            if seg["type"] == "failure":
                total_failure_annotations += 1

        remapped_episodes[str(global_ep)] = ep_info

    final_annotations = {
        "schema_version": r_ann.get("schema_version", "1.0"),
        "description": "Authoritative RWFM rollout segment annotations remapped to 725ep general dataset coordinates.",
        "episodes": remapped_episodes,
    }

    out_ann_path = meta_dir / "rwfm_rollout_annotations.json"
    out_ann_path.write_text(json.dumps(final_annotations, indent=2))
    print(f"✓ Wrote {len(remapped_episodes)} remapped rollout annotations to {out_ann_path}")
    print(f"✓ Preserved {total_failure_annotations} failure segments for action loss masking")

    # 6. Execute Full Verification Checklist
    print("\n" + "=" * 80)
    print("🔬 [VERIFICATION CHECKLIST]")
    print("=" * 80)

    # 1. num_episodes == 725
    assert dst_meta.total_episodes == 725, f"FAIL: total_episodes is {dst_meta.total_episodes}, expected 725"
    print("  [PASS] 1. num_episodes == 725")

    # 2. final episode index == 0~724
    ep_keys_int = [int(k) for k in final_rewards.keys()]
    assert min(ep_keys_int) == 0 and max(ep_keys_int) == 724 and len(ep_keys_int) == 725, (
        f"FAIL: Episode indices do not span 0..724 exactly: min={min(ep_keys_int)}, max={max(ep_keys_int)}, len={len(ep_keys_int)}"
    )
    print("  [PASS] 2. final episode index == 0 ~ 724")

    # 3. rollout annotation 100개가 575~674로 remap됨
    ann_keys_int = [int(k) for k in final_annotations["episodes"].keys()]
    assert len(ann_keys_int) == 100 and min(ann_keys_int) == 575 and max(ann_keys_int) == 674, (
        f"FAIL: Rollout annotations do not span 575..674: min={min(ann_keys_int)}, max={max(ann_keys_int)}, len={len(ann_keys_int)}"
    )
    print("  [PASS] 3. rollout annotation 100 episodes correctly remapped to 575 ~ 674")

    # 4. episode_frame_rewards.json에 0~724 모두 존재
    assert set(final_rewards.keys()) == {str(i) for i in range(725)}, (
        "FAIL: episode_frame_rewards.json missing required episode keys"
    )
    print("  [PASS] 4. episode_frame_rewards.json contains all keys '0' through '724'")

    # 5. clean575 reward = 0.0
    for i in range(575):
        r_list = final_rewards[str(i)]
        assert len(r_list) == 1 and r_list[0][0] == 0 and r_list[0][1] == ep_lengths[i] and r_list[0][2] == 0.0, (
            f"FAIL: Clean episode {i} reward mismatch: {r_list}"
        )
    print("  [PASS] 5. clean 575 episodes all have reward = 0.0")

    # 6. HIL50 reward = +0.6
    for i in range(50):
        global_ep = i + 675
        r_list = final_rewards[str(global_ep)]
        assert len(r_list) == 1 and r_list[0][0] == 0 and r_list[0][1] == ep_lengths[global_ep] and r_list[0][2] == 0.6, (
            f"FAIL: HIL episode {global_ep} reward mismatch: {r_list}"
        )
    print("  [PASS] 6. HIL 50 episodes all have reward = +0.6")

    # 7. rollout reward 값은 기존 annotation과 일치
    for i in range(100):
        global_ep = i + 575
        assert final_rewards[str(global_ep)] == r_rew[str(i)], (
            f"FAIL: Rollout reward mismatch between orig {i} and global {global_ep}"
        )
    print("  [PASS] 7. rollout reward values match original rollout annotations exactly")

    # 8. rollout failure semantic annotation이 유지됨
    assert total_failure_annotations > 0, "FAIL: No failure semantic annotations found"
    print(f"  [PASS] 8. rollout failure semantic annotations preserved ({total_failure_annotations} failure segments)")

    # 9. task prompt / camera / state / action feature가 merge 과정에서 깨지지 않음
    print("  Testing dataset loading and frame decoding with LeRobotDataset facade...")
    test_ds = LeRobotDataset(output_repo_id, root=output_root, return_uint8=True)
    assert test_ds.num_episodes == 725
    assert test_ds.num_frames == dst_meta.total_frames
    assert "observation.images.top" in test_ds.features
    assert "observation.images.wrist" in test_ds.features
    assert "observation.state" in test_ds.features
    assert "action" in test_ds.features

    # Spot-check boundary episodes across all 3 components
    probe_episodes = [0, 574, 575, 674, 675, 724]
    for p_ep in probe_episodes:
        ep_entry = test_ds.meta.episodes[p_ep]
        from_idx = ep_entry["dataset_from_index"]
        item = test_ds[from_idx]
        assert "action" in item and item["action"].shape == (6,), f"Invalid action shape in ep {p_ep}"
        assert "observation.state" in item and item["observation.state"].shape == (6,), f"Invalid state shape in ep {p_ep}"
        assert "observation.images.top" in item and item["observation.images.top"].shape == (3, 480, 640), f"Invalid top shape in ep {p_ep}"
        assert "observation.images.wrist" in item and item["observation.images.wrist"].shape == (3, 480, 640), f"Invalid wrist shape in ep {p_ep}"
        assert item["task"] is not None and len(item["task"]) > 10, f"Task empty in ep {p_ep}"
        print(f"    ✓ Boundary check Ep {p_ep:3d}: frame {from_idx:7d}, task='{item['task'][:35]}...'")

    print("  [PASS] 9. All features, image decoding, and task strings intact across all components")

    # 10. Test RewardSampleWeighter auto-detection
    print("  Testing RewardSampleWeighter auto-detection...")
    weighter_cfg = SampleWeightingConfig(type="reward_weighted")
    weighter = RewardSampleWeighter(weighter_cfg, device="cpu", dataset_root=output_root)
    assert len(weighter.ep_frame_rewards) == 725, f"Weighter loaded {len(weighter.ep_frame_rewards)} episodes, expected 725"
    print("  [PASS] 10. RewardSampleWeighter auto-detected and loaded 725 episode reward profiles")

    print("\n✅ ALL 10 VERIFICATION CHECKS PASSED 100%!")

    # 7. Push to Hugging Face Hub
    push_success = False
    if push_to_hub:
        print("\n" + "=" * 80)
        print(f"🚀 [HUGGING FACE HUB PUSH] Uploading {output_repo_id}...")
        print("=" * 80)
        try:
            test_ds.push_to_hub(upload_large_folder=True)
            print(f"✅ Successfully pushed dataset to https://huggingface.co/datasets/{output_repo_id}")
            push_success = True
        except Exception as e:
            print(f"⚠️ Push via upload_large_folder failed ({e}), attempting standard push_to_hub()...")
            try:
                test_ds.push_to_hub(upload_large_folder=False)
                print(f"✅ Successfully pushed dataset to https://huggingface.co/datasets/{output_repo_id}")
                push_success = True
            except Exception as e2:
                print(f"❌ Failed to push dataset to Hugging Face Hub: {e2}")
                push_success = False
    else:
        print("\n⏩ Push to Hub skipped (--no-push).")

    # 8. Print Final Required Report
    print("\n" + "=" * 80)
    print("📋 [FINAL RWFM GENERAL DATASET REPORT]")
    print("=" * 80)
    print(f"• 최종 dataset repo id:               {output_repo_id}")
    print(f"• 725 episode 확인:                  {dst_meta.total_episodes} episodes ({dst_meta.total_frames:,} frames)")
    print(f"• clean / rollout / HIL episode 범위:")
    print(f"    - Clean 575ep:                    0 ~ 574   (575 episodes, reward = 0.0)")
    print(f"    - RWFM Rollout 100ep:             575 ~ 674 (100 episodes, normal=0.0, self_corr=+0.4, fail=-1.0)")
    print(f"    - Correction-only HIL 50ep:       675 ~ 724 (50 episodes, reward = +0.6)")
    print(f"• episode_frame_rewards.json 경로:    {out_rew_path}")
    print(f"• rwfm_rollout_annotations.json 경로: {out_ann_path}")
    print(f"• Hugging Face push 성공 여부:        {'성공 (SUCCESS)' if push_success else ('스킵됨 (SKIPPED)' if not push_to_hub else '실패 (FAILED)')}")
    print("=" * 80)

    return output_root


def main():
    parser = argparse.ArgumentParser(description="Build and verify final RWFM General dataset (725 episodes).")
    parser.add_argument(
        "--repo-id",
        type=str,
        default=DEFAULT_OUTPUT_REPO,
        help=f"Target HF dataset repository ID (default: {DEFAULT_OUTPUT_REPO})",
    )
    parser.add_argument(
        "--clean-repo",
        type=str,
        default=DEFAULT_CLEAN_REPO,
        help=f"Clean 575ep repository ID (default: {DEFAULT_CLEAN_REPO})",
    )
    parser.add_argument(
        "--rollout-repo",
        type=str,
        default=DEFAULT_ROLLOUT_REPO,
        help=f"RWFM Rollout 100ep repository ID (default: {DEFAULT_ROLLOUT_REPO})",
    )
    parser.add_argument(
        "--hil-repo",
        type=str,
        default=DEFAULT_HIL_REPO,
        help=f"HIL 50ep repository ID (default: {DEFAULT_HIL_REPO})",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        default=True,
        help="Push dataset to Hugging Face Hub after verification passes (default: True)",
    )
    parser.add_argument(
        "--no-push",
        action="store_false",
        dest="push_to_hub",
        help="Do not push to Hugging Face Hub",
    )
    parser.add_argument(
        "--no-force",
        action="store_false",
        dest="force_rebuild",
        help="Do not overwrite existing destination directory if present",
    )

    args = parser.parse_args()

    build_rwfm_general_dataset(
        output_repo_id=args.repo_id,
        clean_repo_id=args.clean_repo,
        rollout_repo_id=args.rollout_repo,
        hil_repo_id=args.hil_repo,
        push_to_hub=args.push_to_hub,
        force_rebuild=args.force_rebuild,
    )


if __name__ == "__main__":
    main()
