"""Subgroup diagnostic: for each matched pair, do case and control share the
same MG_modality at exam_order 0 (index) and 1 (first prior)?

Produces a per-pair table with columns:
    case_pid, control_pid, case_group, case_mod_idx, ctrl_mod_idx,
    case_mod_p1,  ctrl_mod_p1,  match_idx (bool), match_p1 (bool),
    match_both (bool)

Then reports subgroup sizes:
  - n pairs where match_idx == True
  - n pairs where match_both == True (index AND prior-1)
  - n pairs where match_both == True AND both are FFDM
  - same for Synth2D, S2D+FFDM (few pairs, likely)

Uses build_group_map for the malignant/biopsy_negative classification so it
matches the manuscript's headline convention.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.cohort_embeddings import build_group_map  # noqa: E402


def _study_modality_at(df: pd.DataFrame, exam_order: int) -> pd.Series:
    """Return PATIENTID -> MG_modality for rows at the given exam_order."""
    sub = df[df["exam_order"] == exam_order][["PATIENTID", "MG_modality"]]
    return sub.drop_duplicates("PATIENTID").set_index("PATIENTID")["MG_modality"]


def _pair_table(df_cases: pd.DataFrame, df_controls: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct pair_id -> (case_pid, control_pid), same logic as 04b.build_pairs."""
    cases_pairs = (
        df_cases[["PATIENTID", "MATCH_ID"]]
        .dropna()
        .drop_duplicates()
        .rename(columns={"PATIENTID": "case_pid", "MATCH_ID": "control_pid"})
    )
    ctrl_links = (
        df_controls[["PATIENTID", "MATCH_ID"]]
        .dropna()
        .drop_duplicates()
        .rename(columns={"PATIENTID": "control_pid", "MATCH_ID": "case_pid"})
    )
    pairs = cases_pairs.merge(ctrl_links, on=["case_pid", "control_pid"], how="inner")
    pairs = pairs.reset_index(drop=True)
    pairs["pair_id"] = [f"P{i:06d}" for i in range(len(pairs))]
    return pairs[["pair_id", "case_pid", "control_pid"]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases-csv", type=Path, required=True)
    ap.add_argument("--controls-csv", type=Path, required=True)
    ap.add_argument("--out-csv", type=Path, default=None,
                    help="Optional: write the per-pair table here.")
    args = ap.parse_args()

    df_cases = pd.read_csv(args.cases_csv, low_memory=False)
    df_ctrls = pd.read_csv(args.controls_csv, low_memory=False)
    group_map = build_group_map(df_cases, df_ctrls)

    # exam_order 0 (index) and 1 (first prior) modality per patient.
    mod_all = pd.concat([df_cases, df_ctrls], ignore_index=True)
    idx_mod = _study_modality_at(mod_all, 0)
    p1_mod = _study_modality_at(mod_all, 1)

    pairs = _pair_table(df_cases, df_ctrls)
    pairs["case_group"] = pairs["case_pid"].map(group_map)
    pairs["case_mod_idx"] = pairs["case_pid"].map(idx_mod)
    pairs["ctrl_mod_idx"] = pairs["control_pid"].map(idx_mod)
    pairs["case_mod_p1"] = pairs["case_pid"].map(p1_mod)
    pairs["ctrl_mod_p1"] = pairs["control_pid"].map(p1_mod)
    pairs["match_idx"] = pairs["case_mod_idx"] == pairs["ctrl_mod_idx"]
    pairs["match_p1"] = pairs["case_mod_p1"] == pairs["ctrl_mod_p1"]
    pairs["match_both"] = pairs["match_idx"] & pairs["match_p1"]

    total = len(pairs)
    print(f"Total pairs: {total}")
    print()
    print("== By group ==")
    for grp in ["malignant", "biopsy_negative"]:
        sub = pairs[pairs["case_group"] == grp]
        n = len(sub)
        n_i = int(sub["match_idx"].sum())
        n_p = int(sub["match_p1"].sum())
        n_b = int(sub["match_both"].sum())
        print(f"  {grp}: n={n}  match_idx={n_i} ({n_i/n:.1%})  "
              f"match_p1={n_p} ({n_p/n:.1%})  match_both={n_b} ({n_b/n:.1%})")

    print()
    print("== match_both subgroup by shared modality (index level) ==")
    print(f"  {'modality':>15s}  {'n (mal)':>10s}  {'n (BN)':>10s}  {'n (all)':>10s}")
    for mod in sorted(pairs.loc[pairs["match_both"], "case_mod_idx"].dropna().unique()):
        sub = pairs[pairs["match_both"] & (pairs["case_mod_idx"] == mod)]
        n_mal = int((sub["case_group"] == "malignant").sum())
        n_bn = int((sub["case_group"] == "biopsy_negative").sum())
        print(f"  {mod:>15s}  {n_mal:>10d}  {n_bn:>10d}  {len(sub):>10d}")

    # Also stratify by (case_mod_idx, ctrl_mod_idx) for match_idx AND for
    # the diagonal-only rows so the reader can see the density.
    print()
    print("== index-modality crosstab, case × control (all pairs) ==")
    xt = pd.crosstab(pairs["case_mod_idx"], pairs["ctrl_mod_idx"], dropna=False, margins=True)
    print(xt.to_string())

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        pairs.to_csv(args.out_csv, index=False)
        print(f"\nWrote per-pair table to {args.out_csv}")


if __name__ == "__main__":
    main()
