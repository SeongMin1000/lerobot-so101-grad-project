#!/usr/bin/env python3
"""
Generate Frame-Level Learn=False Mask Metadata for 222ep Multitask Dataset.

Dataset: eslab1234/smolvla_multitask_t1_111_t2_111_222ep_merged
Rule:
- Observe: observation.state arm 5-joints within ±15 deg of [-0.97, -87.43, 16.66, 95.47, -7.03] for >= 5 consecutive frames.
- Gripper Close: action[:, 5] <= 12.0 deg for >= 3 consecutive frames.
- Mask: [close_frame + 20, next_observe_frame) is learn=false.
"""

import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

HF_USER = "eslab1234"
REPO_ID = f"{HF_USER}/smolvla_multitask_t1_111_t2_111_222ep_merged"
ROOT = Path.home() / ".cache/huggingface/lerobot" / REPO_ID

def main():
    meta = LeRobotDatasetMetadata(REPO_ID, root=ROOT)
    
    runtime_p = Path("project/config/runtime.json")
    rt = json.loads(runtime_p.read_text())
    obs_target = np.array([
        rt["poses"]["observe"]["shoulder_pan.pos"],
        rt["poses"]["observe"]["shoulder_lift.pos"],
        rt["poses"]["observe"]["elbow_flex.pos"],
        rt["poses"]["observe"]["wrist_flex.pos"],
        rt["poses"]["observe"]["wrist_roll.pos"],
    ])

    OBS_TOL = 15.0
    OBS_MIN_F = 5
    GRIP_ACT_TH = 12.0
    GRIP_MIN_F = 3
    GRASP_KEEP_FRAMES = 20

    ep_files = sorted(list({meta.get_data_file_path(ep) for ep in range(meta.total_episodes)}))
    
    total_episodes = meta.total_episodes
    assert total_episodes == 222, f"Expected 222 episodes, got {total_episodes}"
    
    episodes_metadata = {}
    
    total_mask_intervals = 0
    total_learn_false_frames = 0
    total_learn_true_frames = 0
    
    task1_mask_frames = 0
    task2_mask_frames = 0
    
    processed_ep_count = 0

    for f in ep_files:
        df = pd.read_parquet(ROOT / f)
        for ep_i, ep_df in df.groupby("episode_index"):
            processed_ep_count += 1
            ep_idx = int(ep_i)
            states = np.array(ep_df["observation.state"].tolist())
            actions = np.array(ep_df["action"].tolist())
            T = len(states)
            arm_state = states[:, :5]
            act_grip = actions[:, 5]
            
            is_task1 = (ep_idx < 111)
            task_prompt = ep_df["tasks"].iloc[0] if "tasks" in ep_df else ("Task 1" if is_task1 else "Task 2")
            if isinstance(task_prompt, list) or isinstance(task_prompt, np.ndarray):
                task_prompt = task_prompt[0]
            
            # 1. Detect Observe arrivals (5 consecutive frames within tolerance)
            arm_err = np.max(np.abs(arm_state - obs_target), axis=1)
            is_obs = arm_err < OBS_TOL
            
            obs_arrivals = []
            run_len = 0
            for t in range(T):
                if is_obs[t]:
                    run_len += 1
                    if run_len == OBS_MIN_F:
                        obs_arrivals.append(t - OBS_MIN_F + 1)
                else:
                    run_len = 0
            
            assert len(obs_arrivals) >= 6, f"Episode {ep_idx} has fewer than 6 observe arrivals: {obs_arrivals}"
            
            # Exactly 5 block cycles between the 6 observe arrivals
            cycle_bounds = [(obs_arrivals[k], obs_arrivals[k+1]) for k in range(5)]
            
            cycles_data = []
            segments = []
            masked_intervals = []
            
            ep_learn_false = 0
            ep_learn_true = 0
            
            # Pre-cycle 1 Observe rest
            if obs_arrivals[0] > 0:
                segments.append({
                    "start": 0,
                    "end": obs_arrivals[0],
                    "type": "initial_observe_rest",
                    "learn": True
                })
                ep_learn_true += obs_arrivals[0]
            
            for c_i, (o_start, o_next) in enumerate(cycle_bounds, 1):
                # Search for first gripper action <= 12.0 deg for 3 frames
                close_f = None
                for t in range(o_start, o_next):
                    if t + GRIP_MIN_F <= T and np.all(act_grip[t:t+GRIP_MIN_F] <= GRIP_ACT_TH):
                        close_f = t
                        break
                
                assert close_f is not None, f"Failed to detect gripper close in Episode {ep_idx} Cycle {c_i}"
                
                mask_start = close_f + GRASP_KEEP_FRAMES
                mask_end = o_next
                assert mask_start < mask_end, (
                    f"Invalid mask bounds in Ep {ep_idx} Cycle {c_i}: mask_start={mask_start} >= mask_end={mask_end}"
                )
                
                # Segments
                # 1. Approach + Grasp + Initial 20f Lift (learn=true)
                seg_true = {
                    "start": int(o_start),
                    "end": int(mask_start),
                    "type": f"cycle_{c_i}_approach_and_grasp",
                    "learn": True
                }
                # 2. Task-specific movement / place / stack / retract (learn=false)
                seg_false = {
                    "start": int(mask_start),
                    "end": int(mask_end),
                    "type": "failure",  # Marked as failure to seamlessly match existing trainer masking
                    "semantic_type": f"cycle_{c_i}_task_movement",
                    "learn": False
                }
                
                segments.append(seg_true)
                segments.append(seg_false)
                masked_intervals.append([int(mask_start), int(mask_end)])
                
                n_true = mask_start - o_start
                n_false = mask_end - mask_start
                
                ep_learn_true += n_true
                ep_learn_false += n_false
                
                total_mask_intervals += 1
                total_learn_false_frames += n_false
                
                if is_task1:
                    task1_mask_frames += n_false
                else:
                    task2_mask_frames += n_false
                    
                cycles_data.append({
                    "cycle_index": c_i,
                    "observe_start_frame": int(o_start),
                    "close_frame": int(close_f),
                    "mask_start_frame": int(mask_start),
                    "mask_end_frame": int(mask_end),
                    "learn_true_frames": int(n_true),
                    "learn_false_frames": int(n_false)
                })

            # Post-cycle 5 terminal Observe rest
            terminal_obs = obs_arrivals[5]
            if terminal_obs < T:
                segments.append({
                    "start": int(terminal_obs),
                    "end": int(T),
                    "type": "terminal_observe_rest",
                    "learn": True
                })
                ep_learn_true += (T - terminal_obs)

            total_learn_true_frames += ep_learn_true
            
            assert ep_learn_true + ep_learn_false == T, (
                f"Frame sum mismatch in Ep {ep_idx}: {ep_learn_true} + {ep_learn_false} != {T}"
            )

            episodes_metadata[str(ep_idx)] = {
                "episode_index": ep_idx,
                "task": task_prompt,
                "total_frames": int(T),
                "observe_arrivals": [int(x) for x in obs_arrivals],
                "cycles": cycles_data,
                "segments": segments,
                "masked_intervals": masked_intervals,
                "total_masked_frames": int(ep_learn_false),
                "total_unmasked_frames": int(ep_learn_true),
                "masked_frame_ratio": float(ep_learn_false / T)
            }

    assert processed_ep_count == 222, f"Processed {processed_ep_count} episodes, expected 222"
    assert total_mask_intervals == 222 * 5, f"Expected 1110 mask intervals, got {total_mask_intervals}"

    final_payload = {
        "schema_version": "1.0",
        "description": "Task-specific movement and placement/stacking intervals masked as learn=false for 222ep multitask dataset.",
        "dataset_repo_id": REPO_ID,
        "total_episodes": total_episodes,
        "total_frames": int(total_learn_true_frames + total_learn_false_frames),
        "total_mask_intervals": total_mask_intervals,
        "total_learn_false_frames": total_learn_false_frames,
        "total_learn_true_frames": total_learn_true_frames,
        "overall_masked_ratio": float(total_learn_false_frames / (total_learn_true_frames + total_learn_false_frames)),
        "task1_masked_frames": task1_mask_frames,
        "task2_masked_frames": task2_mask_frames,
        "parameters": {
            "observe_pose": list(obs_target),
            "observe_tolerance_deg": OBS_TOL,
            "observe_min_consecutive_frames": OBS_MIN_F,
            "gripper_close_action_deg": GRIP_ACT_TH,
            "gripper_close_min_consecutive_frames": GRIP_MIN_F,
            "grasp_keep_frames": GRASP_KEEP_FRAMES
        },
        "episodes": episodes_metadata
    }

    # Save to dataset meta/
    meta_dir = ROOT / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    
    # Primary: meta/rwfm_learn_segments.json
    out_learn_p = meta_dir / "rwfm_learn_segments.json"
    out_learn_p.write_text(json.dumps(final_payload, indent=2))
    print(f"✓ Saved primary metadata to: {out_learn_p}")
    
    # Authoritative trainer compatibility: meta/rwfm_rollout_annotations.json
    # (Matches train_smolvla_rwfm_pcgrad_combined.py format directly)
    out_ann_p = meta_dir / "rwfm_rollout_annotations.json"
    out_ann_p.write_text(json.dumps(final_payload, indent=2))
    print(f"✓ Saved trainer-compatible annotations to: {out_ann_p}")
    
    # Backup in project/config
    cfg_p = Path("project/config/rwfm_learn_segments_222ep.json")
    cfg_p.write_text(json.dumps(final_payload, indent=2))
    print(f"✓ Saved backup copy to: {cfg_p}")
    
    print("\n" + "=" * 80)
    print("🎉 [GENERATION & VERIFICATION SUCCESSFUL]")
    print("=" * 80)
    print(f"1. Dataset Root:            {ROOT}")
    print(f"2. Processed Episodes:      {processed_ep_count} / 222 (100%)")
    print(f"3. Total Mask Intervals:    {total_mask_intervals} (222ep × 5 cycles)")
    print(f"4. Total learn=false Frames: {total_learn_false_frames:,} ({total_learn_false_frames/final_payload['total_frames']*100:.2f}%)")
    print(f"5. Total learn=true Frames:  {total_learn_true_frames:,} ({total_learn_true_frames/final_payload['total_frames']*100:.2f}%)")
    print(f"6. Task 1 Mask Frames:       {task1_mask_frames:,} (Task 1 total: 157,292f, ratio: {task1_mask_frames/157292*100:.2f}%)")
    print(f"   Task 2 Mask Frames:       {task2_mask_frames:,} (Task 2 total: 219,379f, ratio: {task2_mask_frames/219379*100:.2f}%)")
    print(f"7. Metadata filenames:       meta/rwfm_learn_segments.json & meta/rwfm_rollout_annotations.json")

if __name__ == "__main__":
    main()
