"""Build a de-identified study-level cohort table with subgroup-membership flags,
committed to results/ so the effect-size / drift tables reproduce from the repo
without the raw Data-dir CSVs.

Grain: one row per (PATIENTID, STUDYID). PATIENTID is the existing de-identified
UUID (unchanged, so it joins to the committed interval CSVs). Image-level columns
(SOPINSTANCEUID, IMAGELATERALITY, has_valid_views_*), absolute DICOM-store paths
(fullPath*), and FACILITY/REGION are dropped. STUDYDATE and demographics are kept.

Membership flags (per PATIENTID) reproduce every subsetted table:
  mgmod_p2/p3/p4  depth-adaptive modality-matched pairs (Table 2/S5/S7/S16 Mm)
  modstable       modality-stable cohort, cases OR controls (Table 5/S15/S17)
  racematch/racemismatch  race strata (Table 6)
  ffdm_all        entire chain FFDM (Table S14 FFDM column); from the exam parquet

Run via slurm_jobs/diagnostics/slurm_diagnostic.sh (VENV=mammoclip; needs pyarrow).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO = Path("/path/to/repo")
D = Path("/path/to/data/FM_Mammo/hoppr_data/breast_cancer/drift_analysis")
SCRATCH = Path("/path/to/scratch/drift_cohort")
OUT = REPO / "results" / "cohort_membership.csv"

MESH = "Density_nPriors_d335_d425_mesh"
KEEP = ["PATIENTID", "STUDYID", "STUDYDATE", "cohort", "is_index", "exam_order",
        "date_diff", "Cancer", "Cancer_Description", "subtype", "BI-RADS Category",
        "Breast Density", "PATIENTAGE", "SEX", "RACE", "MG_modality",
        "MANUFACTURER_CATEGORY", "Laterality_Biopsy", "MATCH_ID"]


def pidset(fname: str) -> set:
    return set(pd.read_csv(D / fname, usecols=["PATIENTID"])["PATIENTID"])


def main() -> None:
    cases = pd.read_csv(D / f"matched_cases_{MESH}.csv", low_memory=False)
    ctrls = pd.read_csv(D / f"matched_controls_{MESH}.csv", low_memory=False)
    cases["cohort"], ctrls["cohort"] = "case", "control"
    df = pd.concat([cases, ctrls], ignore_index=True)
    for c in KEEP:
        if c not in df.columns:
            df[c] = pd.NA
    sd = df[KEEP].drop_duplicates(["PATIENTID", "STUDYID"]).reset_index(drop=True)

    # per-PATIENTID membership sets
    p2, p3, p4 = (pidset(f"matched_cases_{MESH}_mgmod_p{k}.csv") for k in (2, 3, 4))
    rm = pidset(f"matched_cases_{MESH}_racematch.csv")
    rmm = pidset(f"matched_cases_{MESH}_racemismatch.csv")
    mod = pidset(f"matched_cases_{MESH}_modstable.csv") | pidset(f"matched_controls_{MESH}_modstable.csv")
    par = pd.read_parquet(
        SCRATCH / "drift_analysis_augmented_335_425_l_combat_long" / "breast" / "exam_embeddings.parquet",
        columns=["PATIENTID", "MG_modality"])
    ffdm = set(par.groupby("PATIENTID")["MG_modality"]
               .agg(lambda s: bool((s == "FFDM").all())).loc[lambda x: x].index)

    for col, s in [("mgmod_p2", p2), ("mgmod_p3", p3), ("mgmod_p4", p4),
                   ("modstable", mod), ("racematch", rm), ("racemismatch", rmm),
                   ("ffdm_all", ffdm)]:
        sd[col] = sd["PATIENTID"].isin(s)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    sd.to_csv(OUT, index=False)

    # aggregate sanity (no PHI printed) — per-patient dedup for flag counts
    pp = sd.drop_duplicates("PATIENTID")
    print(f"wrote {OUT}")
    print(f"  study rows: {len(sd)}   patients: {pp['PATIENTID'].nunique()} "
          f"(case {int((pp.cohort=='case').sum())}, control {int((pp.cohort=='control').sum())})")
    mal = pp[(pp.cohort == 'case') & (pp.Cancer == 1)]
    print(f"  malignant patients: {len(mal)}   with subtype: {int(mal['subtype'].notna().sum())}")
    for col in ["mgmod_p2", "mgmod_p3", "mgmod_p4", "modstable", "racematch", "racemismatch", "ffdm_all"]:
        cflag = pp[(pp.cohort == 'case') & pp[col]]
        print(f"  {col:13s} case-patients={len(cflag):5d}  "
              f"(malignant {int((cflag.Cancer==1).sum())}, benign {int((cflag.Cancer==0).sum())})")


if __name__ == "__main__":
    main()
