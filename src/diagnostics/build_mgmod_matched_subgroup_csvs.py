"""Build filtered matched case/control CSVs for the MG-modality-matched
subgroup analysis.

A patient pair is retained iff case and control share MG_modality at
exam_order 0 (index) AND 1 (first prior) AND 2 (second prior). This
mirrors the depth-2 subgroup identified by
`mg_modality_pair_match_depth.py` (n = 1,455 pairs).

Downstream: point `04b_analyze_drift.py` at these filtered CSVs plus any
harmonized `exam_embeddings.parquet` from the main tree, and 04b's
`build_pairs`/`attach_pair_ids` will naturally restrict the analysis to
these patients.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.cohort_embeddings import build_group_map  # noqa: E402


def _study_modality_at(df: pd.DataFrame, k: int) -> pd.Series:
    m = df[df["exam_order"] == k][["PATIENTID", "MG_modality"]]
    return m.drop_duplicates("PATIENTID").set_index("PATIENTID")["MG_modality"]


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
    ap.add_argument("--depth", type=int, default=2,
                    help="Match modality at exam_orders 0..depth (default: 2).")
    ap.add_argument("--complement", action="store_true",
                    help="Emit the MISMATCH subgroup: all pairs NOT in the "
                         "depth-D modality-matched set (i.e. discordant on "
                         "modality somewhere in 0..depth, or one patient lacks "
                         "an exam at that depth). Composition is reported.\n"
                         "NOTE: use --complement only with --depth 2. At depth "
                         "2 every retained pair has an index+prior1+prior2 (a "
                         "consequence of the n_priors>=2 inclusion criterion), "
                         "so the complement is 100%% truly modality-DISCORDANT "
                         "(318 pairs: 220 mal / 98 BN, malignant-heavy exactly "
                         "because malignant patients are the ones who switch to "
                         "tomosynthesis at the index). At depth>=3 the "
                         "complement is dominated by chain-attrition pairs "
                         "(no prior_3) that are actually FFDM-concordant, which "
                         "muddies the 'discordant' interpretation — so depth 2 "
                         "is the only clean mismatch subgroup.")
    ap.add_argument("--out-cases-csv", type=Path, required=True)
    ap.add_argument("--out-controls-csv", type=Path, required=True)
    args = ap.parse_args()

    df_cases = pd.read_csv(args.cases_csv, low_memory=False)
    df_ctrls = pd.read_csv(args.controls_csv, low_memory=False)
    group_map = build_group_map(df_cases, df_ctrls)

    mods = pd.concat([df_cases, df_ctrls], ignore_index=True)
    pairs = _pair_table(df_cases, df_ctrls)
    pairs["case_group"] = pairs["case_pid"].map(group_map)

    keep = pd.Series([True] * len(pairs), index=pairs.index)
    # Track per-order presence + discordance for complement composition.
    present_all = pd.Series([True] * len(pairs), index=pairs.index)
    discordant_any = pd.Series([False] * len(pairs), index=pairs.index)
    for k in range(args.depth + 1):
        k_mods = _study_modality_at(mods, k)
        case_mod = pairs["case_pid"].map(k_mods)
        ctrl_mod = pairs["control_pid"].map(k_mods)
        both_present = case_mod.notna() & ctrl_mod.notna()
        keep &= (case_mod == ctrl_mod) & both_present
        present_all &= both_present
        discordant_any |= (both_present & (case_mod != ctrl_mod))

    if args.complement:
        sel = ~keep
        # Composition: of the mismatch pairs, how many are truly discordant
        # (both present through depth, differ somewhere) vs excluded only for
        # a missing exam at some order in 0..depth.
        n_discordant = int((sel & discordant_any).sum())
        n_missing_only = int((sel & ~discordant_any).sum())
        kept = pairs[sel]
        n_mal = int((kept["case_group"] == "malignant").sum())
        n_bn = int((kept["case_group"] == "biopsy_negative").sum())
        print(f"MISMATCH subgroup (complement of match-through-prior_{args.depth}): "
              f"{len(kept)} pairs (mal={n_mal}, BN={n_bn})")
        print(f"  composition: {n_discordant} truly modality-discordant "
              f"(both present, differ somewhere in 0..{args.depth}); "
              f"{n_missing_only} excluded only for a missing exam at some order")
    else:
        kept = pairs[keep]
        n_mal = int((kept["case_group"] == "malignant").sum())
        n_bn = int((kept["case_group"] == "biopsy_negative").sum())
        print(f"Subgroup (match through prior_{args.depth}): "
              f"{len(kept)} pairs (mal={n_mal}, BN={n_bn})")

    keep_case_pids = set(kept["case_pid"])
    keep_ctrl_pids = set(kept["control_pid"])
    out_cases = df_cases[df_cases["PATIENTID"].isin(keep_case_pids)].copy()
    out_ctrls = df_ctrls[df_ctrls["PATIENTID"].isin(keep_ctrl_pids)].copy()
    print(f"Filtered cases CSV: {len(out_cases)} rows, "
          f"{out_cases['PATIENTID'].nunique()} patients")
    print(f"Filtered controls CSV: {len(out_ctrls)} rows, "
          f"{out_ctrls['PATIENTID'].nunique()} patients")

    args.out_cases_csv.parent.mkdir(parents=True, exist_ok=True)
    out_cases.to_csv(args.out_cases_csv, index=False)
    out_ctrls.to_csv(args.out_controls_csv, index=False)
    print(f"Wrote {args.out_cases_csv} and {args.out_controls_csv}")


if __name__ == "__main__":
    main()
