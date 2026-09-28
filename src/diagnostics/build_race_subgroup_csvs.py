"""Build filtered matched case/control CSVs split by whether case and
control share RACE (5-bin scheme), for the race subgroup analysis.

Mirrors build_mgmod_matched_subgroup_csvs.py but the pair-level predicate
is binned-race equality between the case and its matched control instead
of MG_modality concordance across exams. Race is a per-patient attribute
(taken at the index exam via confounder_check.per_patient_race, binned by
confounder_check.bin_race: White / Black-AA / Asian / Unknown / Other-Multi).

Emits EITHER the race-MATCH stratum (case_bin == ctrl_bin) or, with
--complement, the race-MISMATCH stratum (case_bin != ctrl_bin).

Downstream: point 04b_analyze_drift.py at these filtered CSVs plus a
harmonized exam_embeddings.parquet; attach_pair_ids restricts the paired
comparisons to the surviving patients.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.cohort_embeddings import build_group_map  # noqa: E402
from diagnostics.confounder_check import bin_race, per_patient_race  # noqa: E402


def _pair_table(df_cases: pd.DataFrame, df_controls: pd.DataFrame) -> pd.DataFrame:
    cases_pairs = (
        df_cases[["PATIENTID", "MATCH_ID"]].dropna().drop_duplicates()
        .rename(columns={"PATIENTID": "case_pid", "MATCH_ID": "control_pid"})
    )
    ctrl_links = (
        df_controls[["PATIENTID", "MATCH_ID"]].dropna().drop_duplicates()
        .rename(columns={"PATIENTID": "control_pid", "MATCH_ID": "case_pid"})
    )
    return cases_pairs.merge(ctrl_links, on=["case_pid", "control_pid"], how="inner").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases-csv", type=Path, required=True)
    ap.add_argument("--controls-csv", type=Path, required=True)
    ap.add_argument("--complement", action="store_true",
                    help="Emit the race-MISMATCH stratum (case_bin != ctrl_bin) "
                         "instead of the race-MATCH stratum.")
    ap.add_argument("--out-cases-csv", type=Path, required=True)
    ap.add_argument("--out-controls-csv", type=Path, required=True)
    args = ap.parse_args()

    df_cases = pd.read_csv(args.cases_csv, low_memory=False)
    df_ctrls = pd.read_csv(args.controls_csv, low_memory=False)
    group_map = build_group_map(df_cases, df_ctrls)

    # Per-patient binned race, from index exam (fallback to any exam if NaN).
    all_df = pd.concat([df_cases, df_ctrls], ignore_index=True)
    race_bin = per_patient_race(all_df).map(bin_race)

    pairs = _pair_table(df_cases, df_ctrls)
    pairs["case_group"] = pairs["case_pid"].map(group_map)
    pairs["case_bin"] = pairs["case_pid"].map(race_bin)
    pairs["ctrl_bin"] = pairs["control_pid"].map(race_bin)
    # Both sides must have a resolvable bin (they always do — Unknown is a bin).
    match = pairs["case_bin"] == pairs["ctrl_bin"]

    sel = ~match if args.complement else match
    kept = pairs[sel]
    stratum = "race-MISMATCH" if args.complement else "race-MATCH"
    n_mal = int((kept["case_group"] == "malignant").sum())
    n_bn = int((kept["case_group"] == "biopsy_negative").sum())
    print(f"{stratum} subgroup: {len(kept)} pairs (mal={n_mal}, BN={n_bn})")
    # Composition by shared/differing race bin.
    if args.complement:
        combo = (kept["case_bin"] + " vs " + kept["ctrl_bin"]).value_counts().head(8)
        print("  top case-vs-ctrl race-bin combos:")
        for k, v in combo.items():
            print(f"    {k}: {v}")
    else:
        print("  by shared race bin:")
        for b, v in kept["case_bin"].value_counts().items():
            m = int((kept[kept["case_bin"] == b]["case_group"] == "malignant").sum())
            bn = int((kept[kept["case_bin"] == b]["case_group"] == "biopsy_negative").sum())
            print(f"    {b}: {v} (mal={m}, BN={bn})")

    keep_case_pids = set(kept["case_pid"])
    keep_ctrl_pids = set(kept["control_pid"])
    out_cases = df_cases[df_cases["PATIENTID"].isin(keep_case_pids)].copy()
    out_ctrls = df_ctrls[df_ctrls["PATIENTID"].isin(keep_ctrl_pids)].copy()
    print(f"Filtered cases CSV: {len(out_cases)} rows, {out_cases['PATIENTID'].nunique()} patients")
    print(f"Filtered controls CSV: {len(out_ctrls)} rows, {out_ctrls['PATIENTID'].nunique()} patients")

    args.out_cases_csv.parent.mkdir(parents=True, exist_ok=True)
    out_cases.to_csv(args.out_cases_csv, index=False)
    out_ctrls.to_csv(args.out_controls_csv, index=False)
    print(f"Wrote {args.out_cases_csv} and {args.out_controls_csv}")


if __name__ == "__main__":
    main()
