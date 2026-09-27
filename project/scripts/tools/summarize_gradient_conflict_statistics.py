#!/usr/bin/env python3
"""
summarize_gradient_conflict_statistics.py

Reads existing raw 15-repeat gradient conflict results from:
  outputs/gradient_conflict_comparison_865_vs_822/gradient_cosine_comparison_865_vs_822.csv
Computes comprehensive distribution statistics (Student's t 95% CI, quantiles, negative fractions,
norm distributions) across all Color x Phase x Model conditions without performing any new autograd.
Outputs:
  - gradient_conflict_statistical_summary.csv
  - gradient_conflict_statistical_summary.json
  - gradient_conflict_statistical_report.md
"""

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# Degrees of freedom = 14 (N=15). Critical t value for two-tailed alpha=0.05
# t_{0.975, 14} = 2.1447866879169273.
# We explicitly document this fallback value since scipy is not installed in the environment.
T_CRIT_14 = 2.1447866879169273

def compute_group_stats(vals: np.ndarray, t_crit: float = T_CRIT_14) -> Dict[str, Any]:
    n = len(vals)
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
    se = std / math.sqrt(n) if n > 0 else 0.0
    ci_low = mean - t_crit * se
    ci_high = mean + t_crit * se
    median = float(np.median(vals))
    q1 = float(np.percentile(vals, 25))
    q3 = float(np.percentile(vals, 75))
    v_min = float(np.min(vals))
    v_max = float(np.max(vals))
    neg_cnt = int(np.sum(vals < 0))
    pos_cnt = int(np.sum(vals > 0))
    neg_frac = neg_cnt / float(n) if n > 0 else 0.0
    pos_frac = pos_cnt / float(n) if n > 0 else 0.0

    return {
        "n": n,
        "mean": mean,
        "std": std,
        "se": se,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "min": v_min,
        "max": v_max,
        "median": median,
        "q1": q1,
        "q3": q3,
        "neg_count": neg_cnt,
        "neg_fraction": neg_frac,
        "pos_count": pos_cnt,
        "pos_fraction": pos_frac,
    }


def main():
    root = Path(__file__).resolve().parent.parent.parent.parent
    data_dir = root / "outputs/gradient_conflict_comparison_865_vs_822"
    raw_csv = data_dir / "gradient_cosine_comparison_865_vs_822.csv"

    if not raw_csv.exists():
        raise FileNotFoundError(f"Raw CSV file not found: {raw_csv}")

    df = pd.read_csv(raw_csv)
    print(f"Loaded raw records: {len(df)} rows")

    # Verify repeat counts per condition
    grouped = df.groupby(["model_tag", "comparison", "color", "phase"])
    print(f"Total condition groups: {len(grouped)}")

    counts = grouped.size()
    non_15 = counts[counts != 15]
    if len(non_15) > 0:
        print(f"WARNING: Found {len(non_15)} groups with count != 15:\n{non_15}")
    else:
        print("Verification SUCCESS: All 120 condition groups have exactly N=15 repeats.")

    summary_records: List[Dict[str, Any]] = []

    # Iterate in a clean, stable order
    models = ["822", "865"]
    comp_types = ["cross_task", "within_t1", "within_t2"]
    colors = ["Red", "Yellow", "Wood", "Green", "Blue"]
    phases = ["EARLY", "MID", "LATE", "WHOLE"]

    for model in models:
        for comp in comp_types:
            for color in colors:
                for phase in phases:
                    sub = df[
                        (df["model_tag"].astype(str) == model)
                        & (df["comparison"] == comp)
                        & (df["color"] == color)
                        & (df["phase"] == phase)
                    ]
                    if len(sub) == 0:
                        continue

                    cos_vals = sub["cosine"].values
                    g1_vals = sub["grad_norm_a"].values
                    g2_vals = sub["grad_norm_b"].values
                    ratio_vals = sub["norm_ratio"].values
                    cond_name = sub["condition"].iloc[0]

                    cos_stat = compute_group_stats(cos_vals)
                    g1_stat = compute_group_stats(g1_vals)
                    g2_stat = compute_group_stats(g2_vals)
                    ratio_stat = compute_group_stats(ratio_vals)

                    rec = {
                        "model": model,
                        "comparison_type": comp,
                        "color": color,
                        "phase": phase,
                        "condition": cond_name,
                        "N": cos_stat["n"],
                        "cos_mean": cos_stat["mean"],
                        "cos_std": cos_stat["std"],
                        "cos_se": cos_stat["se"],
                        "cos_ci95_low": cos_stat["ci95_low"],
                        "cos_ci95_high": cos_stat["ci95_high"],
                        "cos_min": cos_stat["min"],
                        "cos_max": cos_stat["max"],
                        "cos_median": cos_stat["median"],
                        "cos_q1": cos_stat["q1"],
                        "cos_q3": cos_stat["q3"],
                        "negative_count": cos_stat["neg_count"],
                        "negative_fraction": cos_stat["neg_fraction"],
                        "positive_count": cos_stat["pos_count"],
                        "positive_fraction": cos_stat["pos_fraction"],
                        "g_t1_mean": g1_stat["mean"],
                        "g_t1_std": g1_stat["std"],
                        "g_t1_ci95_low": g1_stat["ci95_low"],
                        "g_t1_ci95_high": g1_stat["ci95_high"],
                        "g_t1_median": g1_stat["median"],
                        "g_t1_min": g1_stat["min"],
                        "g_t1_max": g1_stat["max"],
                        "g_t2_mean": g2_stat["mean"],
                        "g_t2_std": g2_stat["std"],
                        "g_t2_ci95_low": g2_stat["ci95_low"],
                        "g_t2_ci95_high": g2_stat["ci95_high"],
                        "g_t2_median": g2_stat["median"],
                        "g_t2_min": g2_stat["min"],
                        "g_t2_max": g2_stat["max"],
                        "norm_ratio_mean": ratio_stat["mean"],
                        "norm_ratio_std": ratio_stat["std"],
                        "norm_ratio_ci95_low": ratio_stat["ci95_low"],
                        "norm_ratio_ci95_high": ratio_stat["ci95_high"],
                        "norm_ratio_median": ratio_stat["median"],
                        "norm_ratio_min": ratio_stat["min"],
                        "norm_ratio_max": ratio_stat["max"],
                        "raw_cosines": [float(x) for x in cos_vals],
                        "raw_g_t1": [float(x) for x in g1_vals],
                        "raw_g_t2": [float(x) for x in g2_vals],
                        "raw_norm_ratio": [float(x) for x in ratio_vals],
                    }
                    summary_records.append(rec)

    # 1. Save CSV
    out_csv = data_dir / "gradient_conflict_statistical_summary.csv"
    csv_fields = [
        "model",
        "comparison_type",
        "color",
        "phase",
        "condition",
        "N",
        "cos_mean",
        "cos_std",
        "cos_se",
        "cos_ci95_low",
        "cos_ci95_high",
        "cos_min",
        "cos_max",
        "cos_median",
        "cos_q1",
        "cos_q3",
        "negative_count",
        "negative_fraction",
        "positive_count",
        "positive_fraction",
        "g_t1_mean",
        "g_t1_std",
        "g_t1_ci95_low",
        "g_t1_ci95_high",
        "g_t1_median",
        "g_t1_min",
        "g_t1_max",
        "g_t2_mean",
        "g_t2_std",
        "g_t2_ci95_low",
        "g_t2_ci95_high",
        "g_t2_median",
        "g_t2_min",
        "g_t2_max",
        "norm_ratio_mean",
        "norm_ratio_std",
        "norm_ratio_ci95_low",
        "norm_ratio_ci95_high",
        "norm_ratio_median",
        "norm_ratio_min",
        "norm_ratio_max",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary_records)
    print(f"Saved statistical summary CSV: {out_csv} ({len(summary_records)} rows)")

    # 2. Save JSON
    out_json = data_dir / "gradient_conflict_statistical_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary_records, f, indent=2)
    print(f"Saved statistical summary JSON: {out_json}")

    # Build quick lookup dictionary for report generation
    lookup = {
        (r["model"], r["comparison_type"], r["color"], r["phase"]): r
        for r in summary_records
    }

    # Print out detailed results for the 4 critical conditions
    print("\n" + "=" * 90)
    print("DETAILED BREAKOUT FOR 4 CRITICAL CONDITIONS (822 Primary)")
    print("=" * 90)
    for color, phase in [("Green", "MID"), ("Green", "LATE"), ("Wood", "MID"), ("Wood", "LATE")]:
        key = ("822", "cross_task", color, phase)
        r = lookup[key]
        print(f"\n822 {color} {phase}")
        print(f"- N: {r['N']}")
        print(f"- raw: {np.array2string(np.array(r['raw_cosines']), precision=4, separator=', ')}")
        print(f"- mean: {r['cos_mean']:+.4f}")
        print(f"- std: {r['cos_std']:.4f}")
        print(f"- standard error: {r['cos_se']:.4f}")
        print(f"- 95% CI: [{r['cos_ci95_low']:+.4f}, {r['cos_ci95_high']:+.4f}] (Student's t df=14, t_crit=2.1448)")
        print(f"- min / max: {r['cos_min']:+.4f} / {r['cos_max']:+.4f}")
        print(f"- median: {r['cos_median']:+.4f} (Q1: {r['cos_q1']:+.4f}, Q3: {r['cos_q3']:+.4f})")
        print(f"- negative count: {r['negative_count']} / {r['N']}")
        print(f"- fraction(cos < 0): {r['negative_fraction']*100:.1f}%")
        print(f"- positive count: {r['positive_count']} / {r['N']}")
        print(f"- fraction(cos > 0): {r['positive_fraction']*100:.1f}%")
        print(f"- ||g_T1||: {r['g_t1_mean']:.3f} +/- {r['g_t1_std']:.3f} [95% CI: {r['g_t1_ci95_low']:.3f}, {r['g_t1_ci95_high']:.3f}]")
        print(f"- ||g_T2||: {r['g_t2_mean']:.3f} +/- {r['g_t2_std']:.3f} [95% CI: {r['g_t2_ci95_low']:.3f}, {r['g_t2_ci95_high']:.3f}]")
        print(f"- norm_ratio (T1/T2): {r['norm_ratio_mean']:.2f} +/- {r['norm_ratio_std']:.2f} [95% CI: {r['norm_ratio_ci95_low']:.2f}, {r['norm_ratio_ci95_high']:.2f}]")

    # Generate Markdown Report
    out_md = data_dir / "gradient_conflict_statistical_report.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# SmolVLA Multi-Task Gradient Interference Statistical Report (865ep vs 822ep)\n\n")
        f.write("Evaluation based on exact N=15 repeat sampling under strictly controlled noise, time, and batch pairs.\n\n")
        f.write("## 1. 822ep Primary Critical Conditions\n\n")
        for color, phase in [("Green", "MID"), ("Green", "LATE"), ("Wood", "MID"), ("Wood", "LATE")]:
            r = lookup[("822", "cross_task", color, phase)]
            f.write(f"### 822 {color} {phase}\n")
            f.write(f"- **N**: {r['N']}\n")
            f.write(f"- **raw cosines**: `{np.array2string(np.array(r['raw_cosines']), precision=4, separator=', ')}`\n")
            f.write(f"- **mean**: `{r['cos_mean']:+.4f}`\n")
            f.write(f"- **std**: `{r['cos_std']:.4f}` (SE: `{r['cos_se']:.4f}`)\n")
            f.write(f"- **95% CI**: `[{r['cos_ci95_low']:+.4f}, {r['cos_ci95_high']:+.4f}]` (Student's t, df=14, t=2.1448)\n")
            f.write(f"- **min / max**: `{r['cos_min']:+.4f}` / `{r['cos_max']:+.4f}`\n")
            f.write(f"- **median (Q1, Q3)**: `{r['cos_median']:+.4f}` (`{r['cos_q1']:+.4f}`, `{r['cos_q3']:+.4f}`)\n")
            f.write(f"- **negative count**: {r['negative_count']} / {r['N']} ({r['negative_fraction']*100:.1f}%)\n")
            f.write(f"- **norm ratio (||g_T1|| / ||g_T2||)**: `{r['norm_ratio_mean']:.2f}` (95% CI `[{r['norm_ratio_ci95_low']:.2f}, {r['norm_ratio_ci95_high']:.2f}]`)\n\n")

    print(f"\nSaved Markdown report: {out_md}")

if __name__ == "__main__":
    main()
