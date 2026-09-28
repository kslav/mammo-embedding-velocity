"""MG-modality breakdown restricted to the index exams only.

Reads the augmented matched case/control CSVs and, for each patient's
index row (is_index == True), reports the distribution of MG_modality
per group (malignant / biopsy_negative / control). Used for a Table 2
supplementary row in MANUSCRIPT.md.

Classification of case patients into malignant vs biopsy_negative uses
``build_group_map`` from ``src/utils/cohort_embeddings.py`` so the
grouping is guaranteed identical to the manuscript's headline results.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.cohort_embeddings import build_group_map  # noqa: E402


def _pct(n: int, total: int) -> str:
    return f"{n} ({n / total:.1%})" if total else "0 (—)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-csv", type=Path, required=True)
    parser.add_argument("--controls-csv", type=Path, required=True)
    args = parser.parse_args()

    cases_raw = pd.read_csv(args.cases_csv)
    controls_raw = pd.read_csv(args.controls_csv)
    group_map = build_group_map(cases_raw, controls_raw)

    def _index_only(df: pd.DataFrame) -> pd.DataFrame:
        idx = df[df["is_index"] == True]  # noqa: E712
        return idx.drop_duplicates(subset=["PATIENTID"])[["PATIENTID", "MG_modality"]]

    cases_idx = _index_only(cases_raw)
    controls_idx = _index_only(controls_raw)
    cases_idx = cases_idx.assign(group=cases_idx["PATIENTID"].map(group_map))
    controls_idx = controls_idx.assign(group=controls_idx["PATIENTID"].map(group_map))

    mal = cases_idx[cases_idx["group"] == "malignant"]
    bn = cases_idx[cases_idx["group"] == "biopsy_negative"]
    ctrl = controls_idx[controls_idx["group"] == "control"]

    print(f"Malignant       n_patients = {len(mal)}")
    print(f"Biopsy-negative n_patients = {len(bn)}")
    print(f"Control         n_patients = {len(ctrl)}")

    modalities = sorted(set(cases_raw["MG_modality"].dropna().unique()) |
                        set(controls_raw["MG_modality"].dropna().unique()))
    print()
    print(f"{'Modality':12s}  {'Malignant':>18s}  {'Biopsy-negative':>18s}  {'Control':>18s}")
    for m in modalities:
        n_mal = int((mal["MG_modality"] == m).sum())
        n_bn = int((bn["MG_modality"] == m).sum())
        n_ctrl = int((ctrl["MG_modality"] == m).sum())
        print(f"{m:12s}  {_pct(n_mal, len(mal)):>18s}  {_pct(n_bn, len(bn)):>18s}  {_pct(n_ctrl, len(ctrl)):>18s}")

    for name, df in [("Malignant", mal), ("Biopsy-negative", bn), ("Control", ctrl)]:
        n_missing = int(df["MG_modality"].isna().sum())
        print(f"{name:16s} rows with missing MG_modality: {n_missing}")


if __name__ == "__main__":
    main()
