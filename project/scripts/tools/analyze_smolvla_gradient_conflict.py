#!/usr/bin/env python3
"""
analyze_smolvla_gradient_conflict.py

Comparative multi-task gradient conflict diagnostic tool comparing:
  - PRIMARY TARGET  : 822ep Expert-only fine-tuned model
                      (eslab1234/smolvla_multitask_5blocks_v3_822ep_from865_expertonly_b32_lr5e6_50k)
  - BASELINE TARGET : 865ep model
                      (eslab1234/smolvla_multitask_5blocks_v3_865ep_trimmed_from575_285k_fullft_lr1e5_150k)

Evaluated under strictly controlled, identical conditions:
  - Branch source datasets:
      Task 1: smolvla_task1_hil_575_285k_v4_120ep_trimmed_merged
      Task 2: smolvla_task2_hil_575_285k_v4_127ep_trimmed_merged
  - Exact episode color mapping:
      Task 1: Red (0~39), Yellow (40~59), Wood (60~79), Green (80~99), Blue (100~119)
      Task 2: Yellow (0~19), Blue (20~39), Red (40~79), Green (80~101), Wood (102~126)
  - Trajectory phase partitioning:
      EARLY : first 1/3 of active transit (lift & initial routing)
      MID   : middle 1/3 of active transit (maximum task divergence)
      LATE  : last 1/3 of active transit (approach & hover alignment)
      WHOLE : full transit trajectory
  - Preprocessing fidelity:
      Each model's exact training-time normalizer safetensors loaded directly from checkpoint
      Stochastic image transforms disabled for deterministic gradient geometry measurement
  - Sampling:
      Stratified episode-balanced frame sampling (exactly B samples, 0 frame overlap for within-task)
      Identical seeds, samples, Flow Matching noise & time across both models per repeat
"""

import argparse
import csv
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.utils.collate import lerobot_collate_fn


COLOR_NAMES = ["Red", "Yellow", "Wood", "Green", "Blue"]
PHASE_NAMES = ["EARLY", "MID", "LATE", "WHOLE"]

TASK1_EPISODE_RANGES = {
    "Red": (0, 39),
    "Yellow": (40, 59),
    "Wood": (60, 79),
    "Green": (80, 99),
    "Blue": (100, 119),
}

TASK2_EPISODE_RANGES = {
    "Yellow": (0, 19),
    "Blue": (20, 39),
    "Red": (40, 79),
    "Green": (80, 101),
    "Wood": (102, 126),
}


def setup_logger() -> logging.Logger:
    logger = logging.getLogger("gradient_conflict")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Comparative multi-task gradient conflict diagnostic (865ep vs 822ep)."
    )
    parser.add_argument(
        "--primary-policy",
        type=str,
        default="eslab1234/smolvla_multitask_5blocks_v3_822ep_from865_expertonly_b32_lr5e6_50k",
        help="Primary target: 822ep Expert-only fine-tuned model checkpoint.",
    )
    parser.add_argument(
        "--baseline-policy",
        type=str,
        default="eslab1234/smolvla_multitask_5blocks_v3_865ep_trimmed_from575_285k_fullft_lr1e5_150k",
        help="Baseline target: 865ep model checkpoint (set to empty to evaluate only primary).",
    )
    parser.add_argument(
        "--task1-dataset",
        type=str,
        default="eslab1234/smolvla_task1_hil_575_285k_v4_120ep_trimmed_merged",
        help="Task 1 branch dataset repo id or directory.",
    )
    parser.add_argument(
        "--task2-dataset",
        type=str,
        default="eslab1234/smolvla_task2_hil_575_285k_v4_127ep_trimmed_merged",
        help="Task 2 branch dataset repo id or directory.",
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        default=["EARLY", "MID", "LATE", "WHOLE"],
        help="Trajectory phases to analyze (EARLY, MID, LATE, WHOLE).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size (number of training frames per condition).",
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=4,
        help="Micro-batch size for gradient accumulation to prevent VRAM OOM.",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=15,
        help="Number of independent sample repeats for statistics (mean, std, min, max).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run computation on ('cuda' or 'cpu').",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/gradient_conflict_analysis",
        help="Directory to save CSV and JSON results.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for reproducibility.",
    )
    return parser.parse_args()


def resolve_local_path(dataset_identifier: str) -> Path:
    """Finds dataset directory in local cache or returns path."""
    cached = Path.home() / ".cache/huggingface/lerobot" / dataset_identifier
    if cached.exists():
        return cached
    pure_name = dataset_identifier.split("/")[-1]
    cached_pure = Path.home() / ".cache/huggingface/lerobot" / pure_name
    if cached_pure.exists():
        return cached_pure
    cached_eslab = Path.home() / ".cache/huggingface/lerobot/eslab1234" / pure_name
    if cached_eslab.exists():
        return cached_eslab
    return Path(dataset_identifier)


def load_model_exact_stats(policy_path_or_repo: str, logger: logging.Logger) -> Dict[str, Dict[str, torch.Tensor]]:
    """Loads exact training normalization stats from checkpoint normalizer safetensors,
    falling back to train_config.json's dataset meta/stats.json if not present.
    """
    logger.info(f"Resolving normalization statistics for {policy_path_or_repo}...")
    p = Path(policy_path_or_repo)
    safetensors_file = None

    if p.exists() and (p / "policy_preprocessor_step_5_normalizer_processor.safetensors").exists():
        safetensors_file = p / "policy_preprocessor_step_5_normalizer_processor.safetensors"
    else:
        # Check local HF hub cache
        hub_name = f"models--{policy_path_or_repo.replace('/', '--')}"
        cached_snapshots = Path.home() / ".cache/huggingface/hub" / hub_name / "snapshots"
        if cached_snapshots.exists():
            for snap in cached_snapshots.glob("*"):
                cand = snap / "policy_preprocessor_step_5_normalizer_processor.safetensors"
                if cand.exists():
                    safetensors_file = cand
                    break

    if safetensors_file is None:
        try:
            downloaded = hf_hub_download(policy_path_or_repo, "policy_preprocessor_step_5_normalizer_processor.safetensors")
            safetensors_file = Path(downloaded)
        except Exception:
            safetensors_file = None

    if safetensors_file and safetensors_file.exists():
        logger.info(f"  [Direct Checkpoint Fidelity] Loading exact stats from: {safetensors_file.name}")
        stats: Dict[str, Dict[str, torch.Tensor]] = {}
        with safe_open(safetensors_file, framework="pt") as f:
            for k in f.keys():
                parts = k.split(".")
                stat_name = parts[-1]
                feat_name = ".".join(parts[:-1])
                if feat_name not in stats:
                    stats[feat_name] = {}
                stats[feat_name][stat_name] = f.get_tensor(k)
        return stats

    # Fallback to train_config.json dataset
    tc_file = p / "train_config.json"
    if tc_file.exists():
        with open(tc_file) as f:
            tc = json.load(f)
        ds_repo = tc.get("dataset", {}).get("repo_id")
        if ds_repo:
            ds_dir = resolve_local_path(ds_repo)
            s_file = ds_dir / "meta/stats.json"
            if s_file.exists():
                logger.info(f"  [Dataset Fallback] Loading stats from: {s_file}")
                with open(s_file) as f:
                    raw_s = json.load(f)
                return {k: {sk: torch.tensor(sv) for sk, sv in v.items()} for k, v in raw_s.items()}

    raise FileNotFoundError(f"Could not resolve normalization statistics for {policy_path_or_repo}")


def build_phase_episode_frame_dict(
    episodes_parquet_path: Path,
    ranges_dict: Dict[str, Tuple[int, int]],
    phase: str,
    trim_start: int = 5,
    trim_end: int = 15,
) -> Dict[str, Dict[int, List[int]]]:
    """Partitions each episode's active transit frames into:
      EARLY : [s, s + 1/3 * D)
      MID   : [s + 1/3 * D, s + 2/3 * D)
      LATE  : [s + 2/3 * D, e)
      WHOLE : [s, e)
    """
    ep_df = pd.read_parquet(episodes_parquet_path)
    result = {c: {} for c in ranges_dict}

    for color, (start_ep, end_ep) in ranges_dict.items():
        for ep in range(start_ep, end_ep + 1):
            if ep >= len(ep_df):
                continue
            row = ep_df.iloc[ep]
            f_start = int(row["dataset_from_index"])
            f_end = int(row["dataset_to_index"])
            s = f_start + trim_start
            e = max(s + 3, f_end - trim_end)
            duration = e - s

            if phase == "EARLY":
                sub_s = s
                sub_e = s + duration // 3
            elif phase == "MID":
                sub_s = s + duration // 3
                sub_e = s + 2 * (duration // 3)
            elif phase == "LATE":
                sub_s = s + 2 * (duration // 3)
                sub_e = e
            elif phase == "WHOLE":
                sub_s = s
                sub_e = e
            else:
                raise ValueError(f"Unknown phase: {phase}")

            frames = list(range(sub_s, max(sub_s + 1, sub_e)))
            result[color][ep] = frames

    return result


def sample_balanced_batch_indices(
    episodes_dict: Dict[int, List[int]],
    batch_size: int,
    rng: np.random.Generator,
    exclude_indices: Optional[Set[int]] = None,
) -> List[int]:
    """Draws exactly `batch_size` training frames uniformly across available episodes,
    preventing longer episodes from dominating the batch.
    """
    if exclude_indices is None:
        exclude_indices = set()

    available_per_ep = {
        ep: [idx for idx in frames if idx not in exclude_indices]
        for ep, frames in episodes_dict.items()
    }
    valid_eps = [ep for ep, frames in available_per_ep.items() if len(frames) > 0]
    if not valid_eps:
        raise ValueError("No available frames left in episode pool after exclusion.")

    chosen_indices: List[int] = []
    ep_cycle = rng.permutation(valid_eps).tolist()
    ep_ptr = 0

    while len(chosen_indices) < batch_size and ep_cycle:
        ep = ep_cycle[ep_ptr % len(ep_cycle)]
        ep_ptr += 1
        avail = available_per_ep[ep]
        if avail:
            chosen_idx = rng.choice(avail)
            chosen_indices.append(int(chosen_idx))
            avail.remove(chosen_idx)
        else:
            ep_cycle.remove(ep)

    if len(chosen_indices) < batch_size:
        raise ValueError(
            f"Could not sample requested batch_size={batch_size}; only found {len(chosen_indices)} unique frames."
        )

    return chosen_indices


def extract_parameter_groups(policy: SmolVLAPolicy) -> Dict[str, List[torch.nn.Parameter]]:
    """Extracts Action Expert parameter tensors."""
    groups: Dict[str, List[torch.nn.Parameter]] = {
        "expert_all": [],
        "lm_expert": [],
        "action_proj": [],
        "action_time_mlp": [],
    }

    for name, param in policy.model.named_parameters():
        if not param.requires_grad:
            continue
        if "lm_expert" in name:
            groups["lm_expert"].append(param)
            groups["expert_all"].append(param)
        elif "action_in_proj" in name or "action_out_proj" in name:
            groups["action_proj"].append(param)
            groups["expert_all"].append(param)
        elif "action_time_mlp" in name:
            groups["action_time_mlp"].append(param)
            groups["expert_all"].append(param)

    return groups


def prepare_torch_batch(
    raw_samples: List[Dict[str, Any]],
    preprocessor: Any,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Collates and preprocesses a batch of samples into model input tensors.
    Ensures uint8 is scaled to float32 [0.0, 1.0].
    Stochastic image transforms are deliberately disabled for deterministic gradient geometry.
    """
    batch = lerobot_collate_fn(raw_samples)
    for cam_key in ["observation.images.top", "observation.images.wrist"]:
        if cam_key in batch and batch[cam_key].dtype == torch.uint8:
            batch[cam_key] = batch[cam_key].to(dtype=torch.float32) / 255.0

    batch = preprocessor(batch)
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device=device)
    return batch


def compute_gradient_cosine_and_norms(
    g1: Tuple[torch.Tensor, ...],
    g2: Tuple[torch.Tensor, ...],
) -> Tuple[float, float, float, float]:
    """Computes inner product, norms, cosine similarity, and norm ratio layer-by-layer
    without allocating giant concatenated gradient tensors.
    """
    dot = 0.0
    norm1_sq = 0.0
    norm2_sq = 0.0

    for p1, p2 in zip(g1, g2, strict=False):
        if p1 is not None and p2 is not None:
            p1_f = p1.detach().float().view(-1)
            p2_f = p2.detach().float().view(-1)
            dot += torch.dot(p1_f, p2_f).item()
            norm1_sq += torch.dot(p1_f, p1_f).item()
            norm2_sq += torch.dot(p2_f, p2_f).item()

    norm1 = math.sqrt(max(norm1_sq, 0.0))
    norm2 = math.sqrt(max(norm2_sq, 0.0))

    if norm1 > 1e-12 and norm2 > 1e-12:
        cosine = dot / (norm1 * norm2)
    else:
        cosine = 0.0

    norm_ratio = norm1 / norm2 if norm2 > 1e-12 else float("nan")
    return cosine, norm1, norm2, norm_ratio


def compute_accumulated_gradient(
    policy: SmolVLAPolicy,
    target_params: List[torch.nn.Parameter],
    raw_samples: List[Dict[str, Any]],
    preprocessor: Any,
    noise_tensor: torch.Tensor,
    time_tensor: torch.Tensor,
    micro_batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, ...]:
    """Computes exact mean Flow Matching gradient on target parameters using micro-batching
    gradient accumulation to fit strictly within GPU VRAM limits.
    """
    total_samples = len(raw_samples)
    accum_grads: Optional[List[torch.Tensor]] = None

    for start_idx in range(0, total_samples, micro_batch_size):
        end_idx = min(start_idx + micro_batch_size, total_samples)
        mb_raw = raw_samples[start_idx:end_idx]
        mb_noise = noise_tensor[start_idx:end_idx]
        mb_time = time_tensor[start_idx:end_idx]
        mb_weight = (end_idx - start_idx) / float(total_samples)

        batch = prepare_torch_batch(mb_raw, preprocessor, device)
        policy.zero_grad(set_to_none=True)
        loss, _ = policy.forward(batch, noise=mb_noise, time=mb_time)
        scaled_loss = loss * mb_weight

        grads = torch.autograd.grad(scaled_loss, target_params, retain_graph=False, create_graph=False)

        if accum_grads is None:
            accum_grads = [g.detach().clone() if g is not None else None for g in grads]
        else:
            for i, g in enumerate(grads):
                if g is not None:
                    if accum_grads[i] is None:
                        accum_grads[i] = g.detach().clone()
                    else:
                        accum_grads[i].add_(g.detach())

        del batch, loss, scaled_loss, grads

    policy.zero_grad(set_to_none=True)
    return tuple(accum_grads) if accum_grads is not None else tuple()


def main():
    args = parse_args()
    logger = setup_logger()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    logger.info("=" * 80)
    logger.info("SmolVLA Multi-Task Gradient Conflict Diagnostic: 865ep vs 822ep")
    logger.info("=" * 80)
    logger.info(f"Target Device        : {device}")
    logger.info(f"Primary Policy (822) : {args.primary_policy}")
    logger.info(f"Baseline Policy (865): {args.baseline_policy}")
    logger.info(f"Task 1 Dataset       : {args.task1_dataset}")
    logger.info(f"Task 2 Dataset       : {args.task2_dataset}")
    logger.info(f"Phases               : {args.phases}")
    logger.info(f"Training Batch Size  : {args.batch_size} samples per condition")
    logger.info(f"Repeats (N)          : {args.n_repeats}")
    logger.info(f"Seed                 : {args.seed}")

    # 1. Resolve Dataset Paths
    p1 = resolve_local_path(args.task1_dataset)
    p2 = resolve_local_path(args.task2_dataset)
    logger.info(f"Resolved Task 1 Path : {p1}")
    logger.info(f"Resolved Task 2 Path : {p2}")

    # 2. Instantiate LeRobot Datasets
    chunk_size = 50
    fps = 30.0
    delta_ts = {"action": [i / fps for i in range(chunk_size)]}

    logger.info("Instantiating LeRobotDataset instances (50-step action chunking, uint8 enabled)...")
    ds_t1 = LeRobotDataset(args.task1_dataset, root=p1, delta_timestamps=delta_ts)
    ds_t2 = LeRobotDataset(args.task2_dataset, root=p2, delta_timestamps=delta_ts)

    # 3. Target Model Definitions
    models_to_evaluate = [("822", args.primary_policy)]
    if args.baseline_policy:
        models_to_evaluate.append(("865", args.baseline_policy))

    # 4. Phase-Partitioned Episode Frame Pools
    ep_p1 = p1 / "meta/episodes/chunk-000/file-000.parquet"
    ep_p2 = p2 / "meta/episodes/chunk-000/file-000.parquet"

    phase_pools: Dict[str, Dict[str, Dict[str, Dict[int, List[int]]]]] = {}
    for phase in args.phases:
        t1_dict = build_phase_episode_frame_dict(ep_p1, TASK1_EPISODE_RANGES, phase=phase)
        t2_dict = build_phase_episode_frame_dict(ep_p2, TASK2_EPISODE_RANGES, phase=phase)
        phase_pools[phase] = {"Task1": t1_dict, "Task2": t2_dict}

    # 5. Diagnostic Loop across Models, Phases, Colors, and Repeats
    results_records: List[Dict[str, Any]] = []
    beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)

    # Comparison definitions:
    # A) Cross-Task: T1 vs T2 for all 5 colors
    # B) Within-Task: T1 A vs B and T2 A vs B for all 5 colors
    comparisons = []
    for phase in args.phases:
        for c in COLOR_NAMES:
            comparisons.append({
                "type": "cross_task",
                "phase": phase,
                "color": c,
                "label": f"T1 {c} vs T2 {c}",
                "task_a": "Task1",
                "task_b": "Task2",
            })
        for c in COLOR_NAMES:
            comparisons.append({
                "type": "within_t1",
                "phase": phase,
                "color": c,
                "label": f"T1 {c} (A vs B)",
                "task_a": "Task1",
                "task_b": "Task1",
            })
            comparisons.append({
                "type": "within_t2",
                "phase": phase,
                "color": c,
                "label": f"T2 {c} (A vs B)",
                "task_a": "Task2",
                "task_b": "Task2",
            })

    total_evals = len(comparisons) * args.n_repeats * len(models_to_evaluate)
    logger.info(f"\nStarting evaluation ({len(comparisons)} conditions x {args.n_repeats} repeats x {len(models_to_evaluate)} models = {total_evals} evals)...")
    logger.info(f"Using micro_batch_size={args.micro_batch_size} for gradient accumulation (total batch_size={args.batch_size})")

    for tag, policy_path in models_to_evaluate:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Setting up and evaluating {tag} Model Pipeline ({policy_path})")
        logger.info(f"{'=' * 80}")

        policy = SmolVLAPolicy.from_pretrained(policy_path)
        policy.to(device)
        policy.eval()

        param_groups = extract_parameter_groups(policy)
        expert_params = param_groups["expert_all"]
        num_expert_params = sum(p.numel() for p in expert_params)
        logger.info(f"  {tag} Action Expert Parameters: {len(expert_params)} tensors ({num_expert_params:,} parameters)")

        stats = load_model_exact_stats(policy_path, logger)
        logger.info(f"  {tag} Action Mean: {stats['action']['mean'].tolist()[:3]} ...")
        logger.info(f"  {tag} State Mean : {stats['observation.state']['mean'].tolist()[:3]} ...")

        preprocessor, _ = make_pre_post_processors(policy_cfg=policy.config, dataset_stats=stats)

        for c_idx, comp in enumerate(comparisons):
            comp_type = comp["type"]
            phase = comp["phase"]
            color = comp["color"]
            label = comp["label"]
            task_a = comp["task_a"]
            task_b = comp["task_b"]

            logger.info(f"[{tag}] [{c_idx + 1:2d}/{len(comparisons)}] Phase: {phase:<5} | Condition: {label} ({args.n_repeats} repeats)")

            ep_dict_a = phase_pools[phase][task_a][color]
            ep_dict_b = phase_pools[phase][task_b][color]
            ds_a = ds_t1 if task_a == "Task1" else ds_t2
            ds_b = ds_t1 if task_b == "Task1" else ds_t2

            for rep in range(args.n_repeats):
                seed_rep = args.seed + rep * 1007
                rng = np.random.default_rng(seed_rep)

                if comp_type == "cross_task":
                    idx_a = sample_balanced_batch_indices(ep_dict_a, args.batch_size, rng)
                    idx_b = sample_balanced_batch_indices(ep_dict_b, args.batch_size, rng)
                else:
                    idx_a = sample_balanced_batch_indices(ep_dict_a, args.batch_size, rng)
                    idx_b = sample_balanced_batch_indices(ep_dict_b, args.batch_size, rng, exclude_indices=set(idx_a))

                # Fetch identical raw samples
                raw_a = [ds_a[i] for i in idx_a]
                raw_b = [ds_b[i] for i in idx_b]

                # Generate identical flow matching noise & time
                torch.manual_seed(seed_rep)
                time_tensor = beta_dist.sample((args.batch_size,)).to(device=device, dtype=torch.float32) * 0.999 + 0.001
                noise_tensor = torch.randn((args.batch_size, chunk_size, 32), device=device, dtype=torch.float32)

                g_a = compute_accumulated_gradient(
                    policy=policy,
                    target_params=expert_params,
                    raw_samples=raw_a,
                    preprocessor=preprocessor,
                    noise_tensor=noise_tensor,
                    time_tensor=time_tensor,
                    micro_batch_size=args.micro_batch_size,
                    device=device,
                )
                g_b = compute_accumulated_gradient(
                    policy=policy,
                    target_params=expert_params,
                    raw_samples=raw_b,
                    preprocessor=preprocessor,
                    noise_tensor=noise_tensor,
                    time_tensor=time_tensor,
                    micro_batch_size=args.micro_batch_size,
                    device=device,
                )

                cos, norm_a, norm_b, ratio = compute_gradient_cosine_and_norms(g_a, g_b)

                results_records.append({
                    "model_tag": tag,
                    "model_path": policy_path,
                    "phase": phase,
                    "comparison": comp_type,
                    "color": color,
                    "condition": label,
                    "repeat": rep,
                    "cosine": cos,
                    "grad_norm_a": norm_a,
                    "grad_norm_b": norm_b,
                    "norm_ratio": ratio,
                    "batch_size": args.batch_size,
                    "seed": seed_rep,
                })

            if (c_idx + 1) % 5 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        del policy, expert_params, preprocessor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 6. Save Raw Results to CSV and JSON
    csv_file = out_dir / "gradient_cosine_comparison_865_vs_822.csv"
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "model_tag", "model_path", "phase", "comparison", "color", "condition",
            "repeat", "cosine", "grad_norm_a", "grad_norm_b", "norm_ratio",
            "batch_size", "seed"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_records)
    logger.info(f"Saved {len(results_records)} raw records to {csv_file}")

    json_file = out_dir / "gradient_cosine_comparison_865_vs_822.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(results_records, f, indent=2)
    logger.info(f"Saved JSON records to {json_file}")

    # 7. Print Pretty Summary Comparison Tables
    df_res = pd.DataFrame(results_records)
    cross_df = df_res[df_res["comparison"] == "cross_task"]

    evaluated_tags = set(df_res["model_tag"].unique()) if len(df_res) > 0 else set()
    has_both = "865" in evaluated_tags and "822" in evaluated_tags

    print("\n" + "=" * 115)
    print("Cross-Task Gradient Conflict Comparison: 865ep (Baseline) vs 822ep (Primary)")
    print("Note: 865 evaluated on Task2 127ep reflects counterfactual correction response to new demonstrations.")
    print("=" * 115)
    if has_both:
        header = (
            f"{'Color':<8} | {'Phase':<6} | {'865 Cos':>8} | {'822 Cos':>8} | {'Δ Cos':>7} | "
            f"{'865 ||gT1||':>11} | {'865 ||gT2||':>11} | {'822 ||gT1||':>11} | {'822 ||gT2||':>11} | {'822 T1/T2':>9}"
        )
        print(header)
        print("-" * 115)
        for c in COLOR_NAMES:
            for ph in args.phases:
                sub865 = cross_df[(cross_df["model_tag"] == "865") & (cross_df["color"] == c) & (cross_df["phase"] == ph)]
                sub822 = cross_df[(cross_df["model_tag"] == "822") & (cross_df["color"] == c) & (cross_df["phase"] == ph)]
                if len(sub865) == 0 or len(sub822) == 0:
                    continue
                cos865 = sub865["cosine"].mean()
                cos822 = sub822["cosine"].mean()
                d_cos = cos822 - cos865
                na865 = sub865["grad_norm_a"].mean()
                nb865 = sub865["grad_norm_b"].mean()
                na822 = sub822["grad_norm_a"].mean()
                nb822 = sub822["grad_norm_b"].mean()
                ratio822 = sub822["norm_ratio"].mean()
                row = (
                    f"{c:<8} | {ph:<6} | {cos865:>+8.4f} | {cos822:>+8.4f} | {d_cos:>+7.4f} | "
                    f"{na865:>11.3f} | {nb865:>11.3f} | {na822:>11.3f} | {nb822:>11.3f} | {ratio822:>9.2f}"
                )
                print(row)
            print("-" * 115)
    else:
        # Single model output
        tag = list(evaluated_tags)[0] if evaluated_tags else "Unknown"
        print(f"{'Color':<8} | {'Phase':<6} | {'Cosine':>8} | {'||g_T1||':>9} | {'||g_T2||':>9} | {'T1/T2 ratio':>11}")
        print("-" * 80)
        for c in COLOR_NAMES:
            for ph in args.phases:
                sub = cross_df[(cross_df["color"] == c) & (cross_df["phase"] == ph)]
                if len(sub) == 0:
                    continue
                c_mean = sub["cosine"].mean()
                na = sub["grad_norm_a"].mean()
                nb = sub["grad_norm_b"].mean()
                ratio = sub["norm_ratio"].mean()
                print(f"{c:<8} | {ph:<6} | {c_mean:>+8.4f} | {na:>9.3f} | {nb:>9.3f} | {ratio:>11.2f}")

    # Within-Task Comparison Table
    print("\n" + "=" * 95)
    print("Within-Task Baseline Cosine (Batch A vs Batch B, Zero Frame Overlap)")
    print("=" * 95)
    print(f"{'Condition':<25} | {'Phase':<6} | {'865 Baseline':>12} | {'822 Baseline':>12} | {'Δ Baseline':>10}")
    print("-" * 95)
    within_df = df_res[df_res["comparison"].isin(["within_t1", "within_t2"])]
    for comp_k in ["within_t1", "within_t2"]:
        for c in COLOR_NAMES:
            for ph in args.phases:
                cond_label = f"{'T1' if 't1' in comp_k else 'T2'} {c} (A vs B)"
                sub865 = within_df[(within_df["model_tag"] == "865") & (within_df["comparison"] == comp_k) & (within_df["color"] == c) & (within_df["phase"] == ph)]
                sub822 = within_df[(within_df["model_tag"] == "822") & (within_df["comparison"] == comp_k) & (within_df["color"] == c) & (within_df["phase"] == ph)]
                cos865 = sub865["cosine"].mean() if len(sub865) > 0 else float("nan")
                cos822 = sub822["cosine"].mean() if len(sub822) > 0 else float("nan")
                d_b = cos822 - cos865 if not (math.isnan(cos865) or math.isnan(cos822)) else float("nan")
                print(f"{cond_label:<25} | {ph:<6} | {cos865:>+12.4f} | {cos822:>+12.4f} | {d_b:>+10.4f}")

    print("\n" + "=" * 95)
    print("Interpretation Helper & Methodological Notes")
    print("=" * 95)
    print("1. [Task Gradient Geometry]: 본 분석은 task condition만의 순수한 차이를 분리하는 것이 아니며,")
    print("   T1/T2의 observation, state, trajectory distribution 차이를 포함하여")
    print("   '동일한 stochastic augmentation/noise 조건 아래 각 task-conditioned demonstration distribution이")
    print("   Action Expert에 요구하는 gradient geometry'를 측정합니다.")
    print("2. [End-to-End Policy State]: 865와 822는 각각 자신의 training-time normalization을 사용하므로,")
    print("   865 -> 822 비교는 단순 weight 변화만의 효과가 아니라 각 checkpoint의 실제 training-time")
    print("   preprocessing을 포함한 end-to-end policy state의 비교입니다.")
    print("3. [Diagnostic Signal]: gradient cosine은 task interference의 진단 지표이지 실제 failure의")
    print("   단독 원인을 증명하는 것은 아닙니다. cosine < 0이라 하여 무조건 유해한 충돌로 단정하지 않으며,")
    print("   잘 수행되는 Red/Blue vs 편향이 있는 Wood/Green 사이의 differential 패턴과 norm ratio를 종합 평가합니다.")
    print("=" * 95 + "\n")


if __name__ == "__main__":
    main()
