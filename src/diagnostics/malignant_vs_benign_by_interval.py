"""Per-interval malignant-vs-benign comparison (unpaired) on drift velocity.

The main analysis compares each case arm against matched *controls*. The two
case arms (malignant, biopsy-negative) are not matched to each other, so
malignant-vs-benign is inherently unpaired. The main text reports it only as a
single patient-mean Mann-Whitney U (Table 9); this diagnostic breaks it out
*per interval*, and adds a within-patient contrast, on two cohorts:

  * full cohort
  * FFDM-only: patients whose entire longitudinal chain is FFDM (the most
    numerous single-modality set), which removes the acquisition-modality
    difference that most distinguishes the malignant arm.

Two families, all four models, headline pipeline (L2 + long-ComBat, breast):

  (A) Between-patient: one-sided Mann-Whitney U on v_signed per interval,
      malignant > benign. Uses the between-patient interval CSVs (affected side
      for cases).
  (B) Within-patient: for each patient compute the affected-minus-contralateral
      v_signed per interval, then Mann-Whitney U comparing that within-patient
      side-asymmetry between the malignant and benign arms. Tests whether the
      affected-side acceleration is *steeper* in malignant than benign patients,
      net of each patient's own contralateral baseline.

Emits a tidy CSV. Reads scratch interval CSVs (which retain PATIENTID) plus the
exam parquet (for MG_modality, to build the FFDM subset).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import SCRATCH_DIR as SCRATCH, BIOMEDCLIP_SCRATCH_DIR as BIOMEDCLIP_ROOT  # noqa: E402

# (between-patient dir, contralateral dir, exam-parquet dir) per model.
MODELS = {
    "Mammo-CLIP": (
        "drift_analysis_augmented_335_425_l_combat_long",
        "drift_analysis_augmented_335_425_mammoclip_l_contralat_combat_long",
        "drift_analysis_augmented_335_425_l_combat_long"),
    "HOPPR": (
        "drift_analysis_augmented_335_425_hoppr_l_combat_long",
        "drift_analysis_augmented_335_425_hoppr_l_contralat_combat_long",
        "drift_analysis_augmented_335_425_hoppr_l_combat_long"),
    "MedImageInsight": (
        "drift_analysis_augmented_335_425_mii_l_combat_long",
        "drift_analysis_augmented_335_425_mii_l_contralat_combat_long",
        "drift_analysis_augmented_335_425_mii_l_combat_long"),
    "BiomedCLIP": (
        None,  # between-patient CSV lives in the USER tree; handled below
        "drift_analysis_augmented_335_425_biomedclip_l_contralat_combat_long",
        None),
}

MALIGNANT, BIOPSY_NEG = "malignant", "biopsy_negative"
MAX_INTERVAL = 4


def _between_dir(model: str, sub: str | None) -> Path:
    if model == "BiomedCLIP":
        return BIOMEDCLIP_ROOT / "drift_analysis_biomedclip_l_combat_long"
    return SCRATCH / sub


def _exam_parquet(model: str, sub: str | None) -> Path:
    base = (BIOMEDCLIP_ROOT / "drift_analysis_biomedclip_l_combat_long"
            if model == "BiomedCLIP" else SCRATCH / sub)
    return base / "breast" / "exam_embeddings.parquet"


def _ffdm_pids(parquet: Path) -> set:
    """Patients whose every exam is FFDM (MG_modality all == FFDM)."""
    df = pd.read_parquet(parquet, columns=["PATIENTID", "MG_modality"])
    per = df.groupby("PATIENTID")["MG_modality"].agg(
        lambda s: bool((s == "FFDM").all()))
    return set(per[per].index)


def _mwu(a: np.ndarray, b: np.ndarray) -> tuple[float, float, int, int]:
    """One-sided MWU (a > b); returns (p, median_diff_of_medians, n_a, n_b)."""
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 10 or len(b) < 10:
        return np.nan, float(np.median(a) - np.median(b)) if len(a) and len(b) else np.nan, len(a), len(b)
    p = mannwhitneyu(a, b, alternative="greater").pvalue
    return p, float(np.median(a) - np.median(b)), len(a), len(b)


def _holm_col(rows: list[dict], key: str = "p_raw") -> None:
    valid = [(i, r[key]) for i, r in enumerate(rows) if not np.isnan(r[key])]
    if valid:
        corr = multipletests([p for _, p in valid], method="holm")[1]
        for (i, _), pc in zip(valid, corr):
            rows[i]["p_holm"] = pc
    for r in rows:
        r.setdefault("p_holm", np.nan)


def _between_family(model: str, between_dir: Path, ffdm: set) -> list[dict]:
    """(A) Unpaired MWU malignant>benign on v_signed per interval."""
    csv = between_dir / "breast" / "breast_malignant_vs_biopsy_negative_intervals.csv"
    df = pd.read_csv(csv)
    rows_full, rows_ffdm = [], []
    for cohort, keep, bucket in [("full", None, rows_full), ("ffdm", ffdm, rows_ffdm)]:
        d = df if keep is None else df[df["PATIENTID"].isin(keep)]
        for iv in range(1, MAX_INTERVAL + 1):
            s = d[d["interval_idx"] == iv]
            a = s.loc[s["group"] == MALIGNANT, "v_signed"].to_numpy()
            b = s.loc[s["group"] == BIOPSY_NEG, "v_signed"].to_numpy()
            p, md, na, nb = _mwu(a, b)
            bucket.append({"model": model, "family": "between_mwu", "cohort": cohort,
                           "interval": iv, "n_malignant": na, "n_benign": nb,
                           "median_diff": md, "p_raw": p})
        _holm_col(bucket)
    return rows_full + rows_ffdm


def _within_family(model: str, contra_dir: Path, ffdm: set) -> list[dict]:
    """(B) MWU comparing within-patient (affected-contralateral) v_signed
    between malignant and benign arms, per interval."""
    frames = []
    for arm in (MALIGNANT, BIOPSY_NEG):
        f = contra_dir / "breast_both_sides" / f"contralateral_{arm}_intervals.csv"
        frames.append(pd.read_csv(f))
    df = pd.concat(frames, ignore_index=True)
    # per (PATIENTID, interval): affected - contralateral v_signed
    piv = (df.pivot_table(index=["PATIENTID", "group", "interval_idx"],
                          columns="side_label", values="v_signed", aggfunc="mean")
             .reset_index())
    piv["asym"] = piv["affected"] - piv["contralateral"]

    rows_full, rows_ffdm = [], []
    for cohort, keep, bucket in [("full", None, rows_full), ("ffdm", ffdm, rows_ffdm)]:
        d = piv if keep is None else piv[piv["PATIENTID"].isin(keep)]
        for iv in range(1, MAX_INTERVAL + 1):
            s = d[d["interval_idx"] == iv]
            a = s.loc[s["group"] == MALIGNANT, "asym"].to_numpy()
            b = s.loc[s["group"] == BIOPSY_NEG, "asym"].to_numpy()
            p, md, na, nb = _mwu(a, b)
            bucket.append({"model": model, "family": "within_asym_mwu", "cohort": cohort,
                           "interval": iv, "n_malignant": na, "n_benign": nb,
                           "median_diff": md, "p_raw": p})
        _holm_col(bucket)
    return rows_full + rows_ffdm


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-csv", type=Path, required=True)
    args = ap.parse_args()

    all_rows: list[dict] = []
    for model, (btw, con, exam) in MODELS.items():
        print(f"[{model}]")
        between_dir = _between_dir(model, btw)
        # FFDM set from the between-patient exam parquet (same patients).
        ffdm = _ffdm_pids(_exam_parquet(model, exam if exam else btw))
        print(f"  FFDM-all-exams patients: {len(ffdm)}")
        all_rows.extend(_between_family(model, between_dir, ffdm))
        all_rows.extend(_within_family(model, SCRATCH / con, ffdm))

    out = pd.DataFrame(all_rows)[
        ["model", "family", "cohort", "interval", "n_malignant", "n_benign",
         "median_diff", "p_raw", "p_holm"]]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"wrote {args.out_csv}  ({len(out)} rows)")


if __name__ == "__main__":
    main()
