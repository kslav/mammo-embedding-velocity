"""Build filtered case/control CSVs for the CONTRALATERAL (04c) modality
subgroup — split by whether a patient's MG_modality is consistent ACROSS
THEIR OWN longitudinal exams.

MG_modality is per-study (both breasts share it — verified), so the
contralateral within-patient design is automatically modality-controlled
BETWEEN sides. The meaningful within-patient modality question is
therefore temporal: does a patient keep the same modality across their
index + priors, or switch (e.g. FFDM history -> tomosynthesis at index)?

Per-patient predicate (applied to cases AND controls independently, since
04c is within-patient):
  - present through prior_D: patient has a study at every exam_order 0..D.
  - MATCH  (--depth D): present through D AND all those studies share one
    MG_modality (no switch within 0..D).
  - MISMATCH (--complement, use --depth 2): present through 2 AND the
    modality is NOT constant over 0..2 (a switch somewhere) — the clean
    "modality-switcher" subgroup, analogous to the between-patient
    mismatch (depth 2 avoids attrition contamination).

Downstream: feed the filtered CSVs to 04c_analyze_contralateral_drift.py
(granularity breast_both_sides) with the same --exam-parquet-dir as the
full contralateral run; 04c keys on PATIENTID so the filter restricts the
within-patient comparison.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.cohort_embeddings import build_group_map  # noqa: E402


def _per_patient_modality_by_order(df: pd.DataFrame, max_k: int) -> pd.DataFrame:
    """PATIENTID -> {exam_order: MG_modality} for orders 0..max_k."""
    sub = df[df["exam_order"] <= max_k][["PATIENTID", "exam_order", "MG_modality"]]
    sub = sub.drop_duplicates(["PATIENTID", "exam_order"])
    return sub.pivot(index="PATIENTID", columns="exam_order", values="MG_modality")


def _classify(df: pd.DataFrame, depth: int) -> tuple[set, set]:
    """Return (match_pids, mismatch_pids) at the given depth.

    match: present at every order 0..depth AND constant modality.
    mismatch: present at every order 0..depth AND NOT constant.
    Patients not present through depth fall into neither set.
    """
    piv = _per_patient_modality_by_order(df, depth)
    cols = [k for k in range(depth + 1) if k in piv.columns]
    present = piv[cols].notna().all(axis=1)
    n_distinct = piv[cols].nunique(axis=1, dropna=False)
    match = set(piv.index[present & (n_distinct == 1)])
    mismatch = set(piv.index[present & (n_distinct > 1)])
    return match, mismatch


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases-csv", type=Path, required=True)
    ap.add_argument("--controls-csv", type=Path, required=True)
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--complement", action="store_true",
                    help="Emit modality-SWITCHER (mismatch) subgroup instead of "
                         "modality-stable (match). Use with --depth 2.")
    ap.add_argument("--out-cases-csv", type=Path, required=True)
    ap.add_argument("--out-controls-csv", type=Path, required=True)
    args = ap.parse_args()

    df_cases = pd.read_csv(args.cases_csv, low_memory=False)
    df_ctrls = pd.read_csv(args.controls_csv, low_memory=False)
    group_map = build_group_map(df_cases, df_ctrls)

    def _select(df: pd.DataFrame) -> set:
        match, mismatch = _classify(df, args.depth)
        return mismatch if args.complement else match

    case_pids = _select(df_cases)
    ctrl_pids = _select(df_ctrls)

    label = ("modality-SWITCHER (mismatch)" if args.complement
             else "modality-stable (match)")
    def _grp_counts(pids):
        g = pd.Series({p: group_map.get(p) for p in pids})
        return int((g == "malignant").sum()), int((g == "biopsy_negative").sum())
    mal, bn = _grp_counts(case_pids)
    print(f"{label} through prior_{args.depth}: "
          f"cases n={len(case_pids)} (mal={mal}, BN={bn}); controls n={len(ctrl_pids)}")

    out_cases = df_cases[df_cases["PATIENTID"].isin(case_pids)].copy()
    out_ctrls = df_ctrls[df_ctrls["PATIENTID"].isin(ctrl_pids)].copy()
    print(f"Filtered cases CSV: {len(out_cases)} rows; controls CSV: {len(out_ctrls)} rows")

    args.out_cases_csv.parent.mkdir(parents=True, exist_ok=True)
    out_cases.to_csv(args.out_cases_csv, index=False)
    out_ctrls.to_csv(args.out_controls_csv, index=False)
    print(f"Wrote {args.out_cases_csv} and {args.out_controls_csv}")


if __name__ == "__main__":
    main()
