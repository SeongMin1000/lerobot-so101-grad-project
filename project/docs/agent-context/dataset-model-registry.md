# Dataset and Model Registry

Last synchronized: 2026-08-11

Status meanings:

- `active`: current inference candidate, not automatically the final model.
- `candidate`: intended for an upcoming comparison or training run.
- `baseline`: retained for comparison.
- `historical`: superseded but useful for reproducing a result.
- `failed`: known to perform inadequately for the stated objective.
- `external-reference`: another team's artifact; never treat as team-owned.
- `verify`: name or completion state must be checked on disk/Hub before use.

## Team datasets

| Dataset ID | Approximate content | Status | Important note |
| --- | --- | --- | --- |
| `eslab1234/pick_and_place_20260704_merged_v1` | ACT-era multi-block, about 100 episodes | `baseline`, `historical` | Random five-block rollout failed badly; source for later task-text conversion |
| `eslab1234/red_handoff_smoke_20260728_merged` | red-only pilot, 30 episodes | `historical` | Too little spatial diversity |
| `eslab1234/red_handoff_smoke_8zones_64ep_20260730_merged` | red-only, A-H zones, 64 episodes | `historical` | Worked at seen points but generalized poorly |
| `eslab1234/red_full_126ep` | red-only expanded spatial dataset, 126 episodes | `historical` | Basis of several LoRA comparisons |
| `eslab1234/red_full_138ep_recovery_v1` | 126 episodes plus recovery demonstrations, 138 total | `active` | Current red-policy dataset snapshot |
| `eslab1234/pick_and_place_20260704_merged_v1_smolvla_task_v1` | ACT-era data relabeled/converted for SmolVLA task text | `candidate`, `verify` | Inspect actual episode tasks and camera keys before training |
| `eslab1234/task1_hybrid_5blocks_v3_100ep_merged` | Task 1 5-block one-take recording (v3), 100 episodes, 138k frames | `active` | 5 sessions merged, RBF hover + distance S-curve + complete pose (gripper closed) |

Never merge merely because IDs look compatible. Verify FPS, robot type,
feature keys, action/state shapes, camera orientation, task semantics, and
normalization stats.

## Team models

| Model/checkpoint ID | Training summary | Status | Observed result or role |
| --- | --- | --- | --- |
| `eslab1234/grasp_redblock_act_c100_a100_50k_v2` | ACT red-block baseline | `baseline`, `historical` | About 20/25 successes at a known C1 setup; rotation/generalization failures |
| `eslab1234/act_pick_and_place_20260704_merged_v1_chunk20_150k` | ACT, multi-block dataset, 150k | `failed`, `baseline` | Random five-block inference did not approach targets reliably |
| `eslab1234/smolvla_red_handoff_smoke_20260729_v1` | SmolVLA LoRA on 30 episodes | `failed`, `historical` | Turned right and grasped empty space |
| `eslab1234/smolvla_red_handoff_smoke_8zones_64ep_20260730_fullft_v1` | SmolVLA full fine-tune, 20k | `failed`, `historical` | Failed at novel positions despite some seen-position success |
| `eslab1234/smolvla_red_full_126ep_lora_r64_20k_v1` | 126 episodes, LoRA r64, 20k | `historical` | Former general-launcher fallback; retained for historical comparison |
| `eslab1234/smolvla_red_full_126ep_lora_r64_lr1e3_40k_v2` | 126 episodes, LoRA r64, lr1e-3, 40k | `candidate`, `historical` | Fixed safety wrapper reproduces this model |
| `eslab1234/smolvla_red_H_full_30ep_lora_r64_10k_v1` | H-zone/30-episode comparison | `historical` | Used only by previous-model comparison wrapper |
| `eslab1234/smolvla_red_full_138ep_recovery_lora_r64_lr1e3_20k_v1` | recovery dataset, LoRA r64, lr1e-3, 20k | `historical` | Good color choice and much better approach after preprocessing fix; grasp angle/release and residual right bias remain |
| `eslab1234/smolvla_red_138ep_expert_only_nolora_b64_lr1e4_30k_v1` | recovery dataset, no PEFT, frozen vision encoder, expert/state projection training, batch 64, lr1e-4, 30k | `active` | User-selected inference target; normal direct camera order `top/wrist/belly -> camera1/camera2/camera3`; rollout result pending |
| `eslab1234/smolvla_all5_actdata_100ep_lora_r64_40k_v1` | planned all-five SmolVLA LoRA run | `candidate`, `verify` | Confirm whether training completed and Hub artifacts exist before inference |

The active model uses `camera1/camera2/camera3` in the current inference
profile. Inspect its actual config/processor files rather than relying on this
note alone.

## External comparison artifacts

| Artifact | Type | Status | Use |
| --- | --- | --- | --- |
| `Chaenn/so101_cube_multitask_hil_0724_merged` | dataset | `external-reference` | Compare multi-task/HIL episode structure and coverage |
| `Chaenn/smolvla_policy_so101_cube_multitask_edge_0802` | model | `external-reference` | Compare SmolVLA multi-block strategy |
| `Chaenn/act_policy_so101_cube_multitask_egde_0802` | model | `external-reference` | Compare ACT baseline strategy |

Do not upload, modify, or claim ownership of external artifacts.

## W&B references

| Run | Associated work | Status |
| --- | --- | --- |
| `fan9xe62` | 126-episode LoRA experiment | historical reference |
| `c69pjfic` | 138-episode recovery LoRA experiment | active-model training reference |

Project commonly uses W&B project `lerobot`. The entity/workspace may vary by
login; read the actual run URL or configured account before querying it.

## Historical hyperparameter families

These are references, not universal defaults:

| Parameter | Values used |
| --- | --- |
| Base | `lerobot/smolvla_base` |
| LoRA rank / alpha | `64 / 64` in current red runs; earlier rank comparisons were discussed |
| Batch size | 4 in early pilot; 8 in later runs |
| Steps | 10k, 20k, 40k depending on run |
| Learning rate | `1e-4` and `1e-3` in different experiments |
| Warmup | 1000 in later runs |
| Optimizer family | AdamW, beta `(0.9, 0.95)`, weight decay `0.01` in later runs |
| Schedule | cosine decay in later runs |
| PEFT mode | `use_peft=true`, `freeze_vision_encoder=true`, `train_expert_only=true`, `train_state_proj=true` for later LoRA runs |
| AMP | commonly `false` in recorded commands |

Before starting training, create a new registry row with exact dataset,
checkpoint/base, Git commit, command, intended comparison, output path, and
status `candidate`. Update status only after artifacts and rollout evidence are
verified.
