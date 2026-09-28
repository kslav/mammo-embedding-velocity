"""Select the 4 patients for the example-views supplement figure.

Picks, from the MG-modality-matched pairs, a malignant case + its matched
control and a benign case + its matched control such that:
  * the case index projection onto u_hat sits far above its matched control
    (large within-pair separation), and
  * the malignant and benign examples are density-matched and age-similar to
    each other, so the four rows are visually comparable.

Ranks by MedImageInsight (the strongest model). Emits the four
(role, PATIENTID, STUDYID, affected_side, group, density, age, proj_gap) rows
as CSV; a downstream figure builder joins the DICOM catalogue on SOPINSTANCEUID
to render LCC/LMLO/RCC/RMLO per patient.

Selection is over index exams only; u_hat and projection z-scoring match the
projection-trajectory figure.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import SCRATCH_DIR as SCRATCH, DATA_DIR  # noqa: E402
MII_PARQUET = (SCRATCH / "drift_analysis_augmented_335_425_mii_l_combat_long"
               / "breast" / "exam_embeddings.parquet")
DRIFT_CSV_DIR = DATA_DIR
MGMOD_CASES = DRIFT_CSV_DIR / "matched_cases_Density_nPriors_d335_d425_mesh_mgmod_p2.csv"
MGMOD_CONTROLS = DRIFT_CSV_DIR / "matched_controls_Density_nPriors_d335_d425_mesh_mgmod_p2.csv"

MALIGNANT, BIOPSY_NEG, CONTROL = "malignant", "biopsy_negative", "control"
AGE_TOL = 5  # years: malignant vs benign example age similarity


def _standard_4view_pids() -> set:
    """PATIENTIDs whose INDEX study has exactly the standard 2L+2R image set.

    A clean LCC/LMLO/RCC/RMLO grid needs exactly two left and two right images
    at the index; studies with extra/spot views (e.g. 3L+1R) are excluded. Uses
    the per-image matched CSVs (which carry IMAGELATERALITY per SOP).
    """
    frames = []
    for f in (MGMOD_CASES, MGMOD_CONTROLS):
        d = pd.read_csv(f, low_memory=False,
                        usecols=["PATIENTID", "is_index", "IMAGELATERALITY"])
        frames.append(d[d["is_index"]])
    idx = pd.concat(frames, ignore_index=True)
    counts = idx.groupby(["PATIENTID", "IMAGELATERALITY"]).size().unstack(fill_value=0)
    ok = counts.get("L", 0).eq(2) & counts.get("R", 0).eq(2)
    return set(counts.index[ok])


def _pairs() -> pd.DataFrame:
    """Modality-matched case<->control pairs (pair_id, case_pid, control_pid)."""
    c = pd.read_csv(MGMOD_CASES, low_memory=False)[["PATIENTID", "MATCH_ID"]].dropna()
    ct = pd.read_csv(MGMOD_CONTROLS, low_memory=False)[["PATIENTID", "MATCH_ID"]].dropna()
    c = c.rename(columns={"PATIENTID": "case_pid", "MATCH_ID": "control_pid"})
    ct = ct.rename(columns={"PATIENTID": "control_pid", "MATCH_ID": "case_pid"})
    return c.merge(ct, on=["case_pid", "control_pid"], how="inner").drop_duplicates()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-csv", type=Path, required=True)
    args = ap.parse_args()

    df = pd.read_parquet(MII_PARQUET)
    idx = df[df["is_index"]].copy()
    m = np.stack(idx.loc[idx["group"] == MALIGNANT, "embedding"].values).mean(0)
    c = np.stack(idx.loc[idx["group"] == CONTROL, "embedding"].values).mean(0)
    u = (m - c) / np.linalg.norm(m - c)
    proj = np.stack(idx["embedding"].values) @ u
    idx["proj_z"] = (proj - proj.mean()) / proj.std()

    info = idx.set_index("PATIENTID")[
        ["STUDYID", "proj_z", "group", "Breast Density", "PATIENTAGE",
         "Laterality_Biopsy"]]
    # Both members of a shown pair must have the standard 2L+2R index view set.
    std = _standard_4view_pids()
    pairs = _pairs()
    pairs = pairs[pairs["case_pid"].isin(info.index)
                  & pairs["control_pid"].isin(info.index)
                  & pairs["case_pid"].isin(std)
                  & pairs["control_pid"].isin(std)].copy()
    pairs["case_group"] = pairs["case_pid"].map(info["group"])
    pairs["gap"] = (pairs["case_pid"].map(info["proj_z"])
                    - pairs["control_pid"].map(info["proj_z"]))
    pairs["density"] = pairs["case_pid"].map(info["Breast Density"])
    pairs["age"] = pairs["case_pid"].map(info["PATIENTAGE"])

    mal = pairs[pairs["case_group"] == MALIGNANT].sort_values("gap", ascending=False)
    ben = pairs[pairs["case_group"] == BIOPSY_NEG].sort_values("gap", ascending=False)

    # Best malignant pair; then best benign pair matching its density and age +-tol.
    best = None
    for _, mr in mal.iterrows():
        cand = ben[(ben["density"] == mr["density"])
                   & (ben["age"].sub(mr["age"]).abs() <= AGE_TOL)]
        if len(cand):
            best = (mr, cand.iloc[0])
            break
    if best is None:  # relax: same density only
        for _, mr in mal.iterrows():
            cand = ben[ben["density"] == mr["density"]]
            if len(cand):
                best = (mr, cand.iloc[0])
                break
    mr, br = best

    rows = []
    for role, pid in [("malignant_case", mr["case_pid"]),
                      ("malignant_control", mr["control_pid"]),
                      ("benign_case", br["case_pid"]),
                      ("benign_control", br["control_pid"])]:
        r = info.loc[pid]
        rows.append({"role": role, "PATIENTID": pid, "STUDYID": r["STUDYID"],
                     "affected_side": r["Laterality_Biopsy"], "group": r["group"],
                     "density": r["Breast Density"], "age": r["PATIENTAGE"],
                     "proj_z": round(float(r["proj_z"]), 3)})
    out = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(out.to_string(index=False))
    print(f"\nmalignant pair gap={mr['gap']:.2f}  benign pair gap={br['gap']:.2f}")
    print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
