"""Add a pseudonymized REGION column to results/cohort_membership.csv.

Real region names are replaced with stable codes R1, R2, ... ordered by
descending study frequency (ties broken by name), so the demographic frame can
be shared without revealing site location. The code->region key is written only
to the authors' private scratch (NOT the repo). Age and race are left as-is;
facility remains excluded.

Run via slurm_jobs/diagnostics/slurm_diagnostic.sh (VENV=mammoclip).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR as D, SCRATCH_DIR as SCRATCH, RESULTS_DIR  # noqa: E402
MESH = "Density_nPriors_d335_d425_mesh"
COH = RESULTS_DIR / "cohort_membership.csv"
KEY = SCRATCH / "region_code_key.csv"  # private, not committed


def region_by_study() -> pd.DataFrame:
    frames = []
    for f in (f"matched_cases_{MESH}.csv", f"matched_controls_{MESH}.csv"):
        frames.append(pd.read_csv(D / f, usecols=["PATIENTID", "STUDYID", "REGION"],
                                  low_memory=False))
    return pd.concat(frames, ignore_index=True).drop_duplicates(["PATIENTID", "STUDYID"])


def main() -> None:
    reg = region_by_study()
    counts = reg["REGION"].value_counts(dropna=True)
    ordered = sorted(counts.index, key=lambda k: (-int(counts[k]), str(k)))
    code = {name: f"R{i + 1}" for i, name in enumerate(ordered)}
    reg["REGION"] = reg["REGION"].map(code)  # NaN region -> NaN

    coh = pd.read_csv(COH, low_memory=False)
    n0 = len(coh)
    merged = coh.merge(reg, on=["PATIENTID", "STUDYID"], how="left")
    assert len(merged) == n0, f"row count changed {n0} -> {len(merged)}"

    # place REGION (coded) right after RACE
    cols = [c for c in merged.columns if c != "REGION"]
    ins = cols.index("RACE") + 1
    merged = merged[cols[:ins] + ["REGION"] + cols[ins:]]
    merged.to_csv(COH, index=False)

    # private key -> scratch only
    KEY.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"code": list(code.values()), "region": list(code.keys()),
                  "n_studies": [int(counts[k]) for k in code]}).to_csv(KEY, index=False)

    print(f"wrote {COH}  ({len(merged)} rows, unchanged from {n0})")
    print(f"regions coded: {len(code)}   missing-region rows: {int(merged['REGION'].isna().sum())}")
    print(f"private key written to {KEY} (scratch, not committed)")
    print("distribution by code:")
    print(merged["REGION"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
