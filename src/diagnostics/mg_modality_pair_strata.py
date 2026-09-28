"""For each depth D and each shared-modality *pattern* (all-FFDM,
all-Synth2D, mixed), count pairs where case and control share MG_modality
at every exam_order k in [0, D].

A "pattern" for depth D is the tuple of shared modalities at exam_orders
0..D (identical for case and control by definition of match). "All-FFDM"
= every element of the tuple is 'FFDM'. "All-Synth2D" = every element is
'Synth2d'. "Mixed" = anything else (includes S2d_FFDM anywhere or a
switch across depth).

Reports sizes to help decide which strata are powered enough for
stratified drift analysis.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.cohort_embeddings import build_group_map  # noqa: E402


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


def _study_mod_at(df: pd.DataFrame, k: int) -> pd.Series:
    m = df[df["exam_order"] == k][["PATIENTID", "MG_modality"]]
    return m.drop_duplicates("PATIENTID").set_index("PATIENTID")["MG_modality"]


def _classify_pattern(tup: tuple) -> str:
    if all(x == "FFDM" for x in tup):
        return "all_FFDM"
    if all(x == "Synth2d" for x in tup):
        return "all_Synth2d"
    if all(x == "S2d_FFDM" for x in tup):
        return "all_S2d_FFDM"
    return "mixed"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases-csv", type=Path, required=True)
    ap.add_argument("--controls-csv", type=Path, required=True)
    ap.add_argument("--max-depth", type=int, default=6)
    args = ap.parse_args()

    df_cases = pd.read_csv(args.cases_csv, low_memory=False)
    df_ctrls = pd.read_csv(args.controls_csv, low_memory=False)
    group_map = build_group_map(df_cases, df_ctrls)
    mods = pd.concat([df_cases, df_ctrls], ignore_index=True)

    pairs = _pair_table(df_cases, df_ctrls)
    pairs["case_group"] = pairs["case_pid"].map(group_map)
    # Pre-compute per-depth match + shared modality string.
    per_k_case = {}
    per_k_ctrl = {}
    for k in range(args.max_depth + 1):
        kmap = _study_mod_at(mods, k)
        per_k_case[k] = pairs["case_pid"].map(kmap)
        per_k_ctrl[k] = pairs["control_pid"].map(kmap)
        pairs[f"match_{k}"] = (
            (per_k_case[k] == per_k_ctrl[k]) &
            per_k_case[k].notna() &
            per_k_ctrl[k].notna()
        )
        pairs[f"shared_mod_{k}"] = per_k_case[k].where(pairs[f"match_{k}"])

    total = len(pairs)
    print(f"Total pairs: {total}  (malignant: {int((pairs['case_group']=='malignant').sum())}, "
          f"biopsy_negative: {int((pairs['case_group']=='biopsy_negative').sum())})")
    print()

    # For each depth D, compute cumulative-match set and stratify.
    for D in range(args.max_depth + 1):
        # Only consider pairs where BOTH case and control have an exam at each k in [0, D].
        # This is stricter than "match through prior_D" — we also require presence.
        keep = pd.Series([True] * len(pairs), index=pairs.index)
        for k in range(D + 1):
            keep &= pairs[f"match_{k}"]
        sub = pairs[keep].copy()
        if len(sub) == 0:
            print(f"--- Match through prior_{D}: 0 pairs (empty) ---")
            continue
        # Pattern tuple = shared_mod_0, shared_mod_1, ..., shared_mod_D
        patt = sub[[f"shared_mod_{k}" for k in range(D + 1)]].apply(tuple, axis=1)
        sub["pattern"] = patt
        sub["pattern_class"] = patt.apply(_classify_pattern)
        n_mal = int((sub["case_group"] == "malignant").sum())
        n_bn = int((sub["case_group"] == "biopsy_negative").sum())
        print(f"=== Match through prior_{D}: {len(sub)} pairs "
              f"(mal={n_mal}, BN={n_bn}) ===")
        print(f"  {'pattern_class':>15s}  {'n (mal)':>10s}  {'n (BN)':>10s}  {'n (all)':>10s}")
        classes = sub["pattern_class"].value_counts().index.tolist()
        for cls in classes:
            s = sub[sub["pattern_class"] == cls]
            m = int((s["case_group"] == "malignant").sum())
            b = int((s["case_group"] == "biopsy_negative").sum())
            print(f"  {cls:>15s}  {m:>10d}  {b:>10d}  {len(s):>10d}")
        # For "mixed" pattern, print the top distinct patterns.
        mixed = sub[sub["pattern_class"] == "mixed"]
        if len(mixed) > 0:
            patt_counts = mixed["pattern"].value_counts().head(6)
            print(f"  top 'mixed' patterns:")
            for pat, n in patt_counts.items():
                s = mixed[mixed["pattern"] == pat]
                m = int((s["case_group"] == "malignant").sum())
                b = int((s["case_group"] == "biopsy_negative").sum())
                print(f"    {'->'.join(pat)}: mal={m}, BN={b}, all={n}")
        print()


if __name__ == "__main__":
    main()
