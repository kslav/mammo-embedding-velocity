"""Deeper subgroup: how many pairs match on modality at exam_orders 0..K?

For each K in {0,1,2,3,4}, count pairs where case and control share
MG_modality at every exam_order ≤ K that both patients have. Used to
decide which depth is powered enough for a stratified drift analysis.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.cohort_embeddings import build_group_map  # noqa: E402


def _study_modality_map(df: pd.DataFrame) -> pd.DataFrame:
    """(PATIENTID, exam_order) → MG_modality, deduped."""
    m = df[["PATIENTID", "exam_order", "MG_modality"]].drop_duplicates(
        ["PATIENTID", "exam_order"]
    )
    return m


def _pair_table(df_cases: pd.DataFrame, df_controls: pd.DataFrame) -> pd.DataFrame:
    cases_pairs = (
        df_cases[["PATIENTID", "MATCH_ID"]].dropna().drop_duplicates()
        .rename(columns={"PATIENTID": "case_pid", "MATCH_ID": "control_pid"})
    )
    ctrl_links = (
        df_controls[["PATIENTID", "MATCH_ID"]].dropna().drop_duplicates()
        .rename(columns={"PATIENTID": "control_pid", "MATCH_ID": "case_pid"})
    )
    pairs = cases_pairs.merge(ctrl_links, on=["case_pid", "control_pid"], how="inner")
    return pairs.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases-csv", type=Path, required=True)
    ap.add_argument("--controls-csv", type=Path, required=True)
    args = ap.parse_args()

    df_cases = pd.read_csv(args.cases_csv, low_memory=False)
    df_ctrls = pd.read_csv(args.controls_csv, low_memory=False)
    group_map = build_group_map(df_cases, df_ctrls)

    mods = _study_modality_map(pd.concat([df_cases, df_ctrls], ignore_index=True))
    pairs = _pair_table(df_cases, df_ctrls)
    pairs["case_group"] = pairs["case_pid"].map(group_map)

    # For each exam_order 0..4, look up modality per patient in the pair and
    # test equality. NaN (no exam at that order) is treated as no-match.
    max_k = 4
    for k in range(max_k + 1):
        k_mods = mods[mods["exam_order"] == k].set_index("PATIENTID")["MG_modality"]
        pairs[f"case_mod_{k}"] = pairs["case_pid"].map(k_mods)
        pairs[f"ctrl_mod_{k}"] = pairs["control_pid"].map(k_mods)
        pairs[f"match_{k}"] = (
            pairs[f"case_mod_{k}"] == pairs[f"ctrl_mod_{k}"]
        ) & pairs[f"case_mod_{k}"].notna() & pairs[f"ctrl_mod_{k}"].notna()

    print(f"Total pairs: {len(pairs)}")
    print()
    print(f"{'depth':>25s}  {'n (mal)':>10s}  {'n (BN)':>10s}  {'n (all)':>10s}")
    # Cumulative match up to depth k.
    cum = pd.Series([True] * len(pairs), index=pairs.index)
    for k in range(max_k + 1):
        cum = cum & pairs[f"match_{k}"]
        sub = pairs[cum]
        n_mal = int((sub["case_group"] == "malignant").sum())
        n_bn = int((sub["case_group"] == "biopsy_negative").sum())
        print(f"  {'match through prior_'+str(k):>25s}  "
              f"{n_mal:>10d}  {n_bn:>10d}  {len(sub):>10d}")

    print()
    print("Breakdown of match_through_prior_1 subgroup by shared index modality:")
    sub = pairs[pairs["match_0"] & pairs["match_1"]]
    print(f"  {'modality':>15s}  {'n (mal)':>10s}  {'n (BN)':>10s}  {'n (all)':>10s}")
    for mod in sorted(sub["case_mod_0"].dropna().unique()):
        s = sub[sub["case_mod_0"] == mod]
        n_mal = int((s["case_group"] == "malignant").sum())
        n_bn = int((s["case_group"] == "biopsy_negative").sum())
        print(f"  {mod:>15s}  {n_mal:>10d}  {n_bn:>10d}  {len(s):>10d}")


if __name__ == "__main__":
    main()
