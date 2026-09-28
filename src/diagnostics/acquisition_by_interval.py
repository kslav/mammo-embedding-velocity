"""Combined (Manufacturer × MG_modality) batch cell distribution by exam_order.

For each group (malignant / biopsy_negative / control), tabulates the
per-patient count of studies at each exam_order broken down by the
combined batch cell ``{MANUFACTURER_CATEGORY}_{MG_modality}``.
Deduplicates on (PATIENTID, STUDYID) so counts are per-study, not
per-image.

exam_order = 0 is the index (or synthesized pseudo-index on the control
arm); exam_order = 1 is the most recent prior; higher exam_orders are
progressively older priors and have decreasing n.

Classification of case patients into malignant vs biopsy_negative uses
``build_group_map`` from ``src/utils/cohort_embeddings.py``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.cohort_embeddings import build_group_map  # noqa: E402


COLS = ["PATIENTID", "STUDYID", "exam_order", "MG_modality", "MANUFACTURER_CATEGORY"]


def _pct(n: int, total: int) -> str:
    return f"{n} ({n / total:.1%})" if total else "—"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-csv", type=Path, required=True)
    parser.add_argument("--controls-csv", type=Path, required=True)
    parser.add_argument("--max-exam-order", type=int, default=6,
                        help="Show rows up to and including this exam_order.")
    args = parser.parse_args()

    cases_raw = pd.read_csv(args.cases_csv)
    controls_raw = pd.read_csv(args.controls_csv)
    group_map = build_group_map(cases_raw, controls_raw)

    def _prep(df: pd.DataFrame) -> pd.DataFrame:
        # Per-study rows (MG_modality and MANUFACTURER_CATEGORY are set per-study)
        return df[COLS].drop_duplicates(subset=["PATIENTID", "STUDYID"])

    cases = _prep(cases_raw).assign(group=lambda d: d["PATIENTID"].map(group_map))
    controls = _prep(controls_raw).assign(group=lambda d: d["PATIENTID"].map(group_map))
    both = pd.concat([cases, controls], ignore_index=True)

    # Combined batch cell = "{MANUFACTURER_CATEGORY}_{MG_modality}" — the same
    # per-study batch label the rejected §3.6.1 sensitivity used. NaN in
    # either component becomes an "Unknown_*" cell.
    both["batch_cell"] = (
        both["MANUFACTURER_CATEGORY"].fillna("Unknown").astype(str)
        + "_"
        + both["MG_modality"].fillna("Unknown").astype(str)
    )
    # Rank cells by overall population size so the wide columns stay stable
    # across groups and readable ones come first. Rare cells (< min_n across
    # the whole cohort) are lumped into "Other" to keep the table narrow.
    cell_counts = both["batch_cell"].value_counts()
    keep_cells = list(cell_counts[cell_counts >= 50].index)
    other_cells = [c for c in cell_counts.index if c not in keep_cells]
    if other_cells:
        both.loc[both["batch_cell"].isin(other_cells), "batch_cell"] = "Other"
    columns = keep_cells + (["Other"] if other_cells else [])

    for group in ["malignant", "biopsy_negative", "control"]:
        g = both[both["group"] == group]
        print(f"\n=== {group}  (n_patients = {g['PATIENTID'].nunique()}) ===")
        header = f"{'exam_order':>10s}  {'n_studies':>9s}  " + \
                 "  ".join(f"{c:>17s}" for c in columns)
        print(header)
        for eo in range(args.max_exam_order + 1):
            sub = g[g["exam_order"] == eo]
            total = len(sub)
            if total == 0:
                continue
            cells = [_pct(int((sub["batch_cell"] == c).sum()), total)
                     for c in columns]
            row = f"{eo:>10d}  {total:>9d}  " + \
                  "  ".join(f"{c:>17s}" for c in cells)
            print(row)


if __name__ == "__main__":
    main()
