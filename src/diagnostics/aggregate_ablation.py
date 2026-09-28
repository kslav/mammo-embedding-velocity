"""Aggregate the 8-cell harmonization ablation table for a given
(cohort, model) combination. Replaces the per-combination
per-cohort-and-model variants that previously existed.

Reads summary.json from each ablation cell and prints the manuscript-
format table (primary mal-vs-ctrl, secondary ben-vs-ctrl, exploratory
mal-vs-benign, LME interaction p-values) for both granularities.

Usage:
    python src/diagnostics/aggregate_ablation.py \\
        --baseline-dir /path/to/drift_analysis_<cohort>/ \\
        --ablation-prefix /path/to/drift_ablation_<model>_<cohort>_

The 7 ablation cell directories are derived as
    {ablation-prefix}{cell}  for cell in [l__, _b_, __z, lb_, l_z, _bz, lbz]
plus the baseline (___) at --baseline-dir.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# 3-letter cell tag → display label
CELLS = ["___", "l__", "_b_", "__z", "lb_", "l_z", "_bz", "lbz"]


def fmt_p(p):
    if p is None:
        return "  n/a "
    if p < 1e-3:
        return f"{p:.0e}"
    return f"{p:.3f}"


def fmt_med(m):
    if m is None:
        return "  n/a"
    return f"{m:+.3f}"


def gather(baseline_dir: Path, ablation_prefix: str) -> dict:
    out = {}
    for cell in CELLS:
        path = baseline_dir if cell == "___" else Path(f"{ablation_prefix}{cell}")
        sj = path / "summary.json"
        if not sj.is_file():
            print(f"MISSING: {sj}")
            out[cell] = None
            continue
        out[cell] = json.loads(sj.read_text())
    return out


def print_table(title, summaries, comp_key, test_key, with_median=True):
    print(f"\n{title}")
    print("  cond | breast p   | breast med | study p    | study med")
    print("  ---- | ---------- | ---------- | ---------- | ----------")
    for cell in CELLS:
        s = summaries.get(cell)
        if s is None:
            print(f"  {cell}  | (missing)")
            continue
        b = s.get(f"breast__{comp_key}", {})
        st = s.get(f"study__{comp_key}", {})

        def _val(block):
            if not block:
                return None, None
            if test_key == "patient_wilcoxon":
                blk = block.get(test_key, {})
                return blk.get("pvalue"), blk.get("median_diff")
            if test_key == "mann_whitney_u":
                blk = block.get(test_key) or {}
                return blk.get("pvalue"), None
            if test_key == "lme":
                blk = block.get(test_key, {})
                return blk.get("interaction_pvalue"), blk.get("interaction_beta")
            return None, None

        bp, bm = _val(b)
        sp, sm = _val(st)
        bm_s = fmt_med(bm) if with_median else "    -"
        sm_s = fmt_med(sm) if with_median else "    -"
        print(f"  {cell}  | {fmt_p(bp):>9}  | {bm_s:>10} | {fmt_p(sp):>9}  | {sm_s:>10}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-dir", type=Path, required=True,
                    help="Dir of the no-harm (___) baseline run.")
    ap.add_argument("--ablation-prefix", type=str, required=True,
                    help="Prefix of the 7 ablation dirs; the cell tag is appended.")
    ap.add_argument("--label", type=str, default="(unspecified)",
                    help="Displayed in the title.")
    args = ap.parse_args()

    summaries = gather(args.baseline_dir, args.ablation_prefix)
    print("=" * 70)
    print(f"Harmonization ablation — {args.label}")
    print("=" * 70)

    print_table(
        "PRIMARY — malignant vs control, patient Wilcoxon (median diff in yr⁻¹)",
        summaries, "malignant_vs_control", "patient_wilcoxon",
    )
    print_table(
        "SECONDARY — biopsy_negative vs control, patient Wilcoxon",
        summaries, "biopsy_negative_vs_control", "patient_wilcoxon",
    )
    print_table(
        "EXPLORATORY — malignant vs biopsy_negative, Mann-Whitney",
        summaries, "malignant_vs_biopsy_negative", "mann_whitney_u",
        with_median=False,
    )
    print_table(
        "LME — malignant vs control, time × group interaction (β yr⁻², p)",
        summaries, "malignant_vs_control", "lme",
    )
    print_table(
        "LME — biopsy_negative vs control, time × group interaction",
        summaries, "biopsy_negative_vs_control", "lme",
    )


if __name__ == "__main__":
    main()
