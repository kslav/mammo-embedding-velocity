"""Per-exam-order paired tests on u_hat *position* (not velocity).

Companion to the drift-velocity analysis (04b) and the contralateral analysis
(04c). Where those test the *rate* a group moves along the cancer direction
u_hat, this diagnostic tests the *location* of each group on u_hat at each
exam_order (0 = index, 1 = first prior, ...). It answers "how far back in time
are the groups already separated on u_hat?" — a positional, table-form analog
of the projection-trajectory figures (Fig. 3, Fig. 5).

Two comparison families, both headline pipeline (per-image L2-normalized +
longitudinal ComBat, breast level), all four models:

  (1) Case-control (between-patient), paired on pair_id from the step-2 CSVs:
      malignant-vs-control and benign-vs-control projection difference at each
      exam_order.
  (2) Affected-vs-contralateral (within-patient), paired within patient on the
      breast_both_sides parquet: affected-side minus contralateral-side
      projection at each exam_order, per group arm.

For each (model, family, exam_order) we report the one-sided paired Wilcoxon
p (H1: case/affected sits higher on u_hat than control/contralateral), the
median paired projection difference, and n_pairs, Holm-corrected across
exam_orders within each (model, comparison). Emits a tidy CSV.

u_hat is the LOO-free group-mean cancer direction on index embeddings
(mean(malignant index) - mean(control index) for the between-patient family;
the affected-side analog for the contralateral family), L2-normalized — the same
direction the trajectory figures project onto. Projections are z-scored per
model across all exams so positions are comparable across models.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from importlib import import_module

build_pairs = import_module("04b_analyze_drift").build_pairs  # reuse canonical pairing

from config import SCRATCH_DIR as SCRATCH, DATA_DIR, BIOMEDCLIP_SCRATCH_DIR  # noqa: E402
DRIFT_CSV_DIR = DATA_DIR
CASES_CSV = DRIFT_CSV_DIR / "matched_cases_Density_nPriors_d335_d425_mesh.csv"
CONTROLS_CSV = DRIFT_CSV_DIR / "matched_controls_Density_nPriors_d335_d425_mesh.csv"

# Between-patient breast parquets (headline L2 + long-ComBat) per model.
BETWEEN = {
    "Mammo-CLIP": SCRATCH / "drift_analysis_augmented_335_425_l_combat_long",
    "HOPPR": SCRATCH / "drift_analysis_augmented_335_425_hoppr_l_combat_long",
    "MedImageInsight": SCRATCH / "drift_analysis_augmented_335_425_mii_l_combat_long",
    "BiomedCLIP": BIOMEDCLIP_SCRATCH_DIR / "drift_analysis_biomedclip_l_combat_long",
}
# Within-patient contralateral parquets (breast_both_sides).
CONTRA = {
    "Mammo-CLIP": SCRATCH / "drift_analysis_augmented_335_425_mammoclip_l_contralat_combat_long",
    "HOPPR": SCRATCH / "drift_analysis_augmented_335_425_hoppr_l_contralat_combat_long",
    "MedImageInsight": SCRATCH / "drift_analysis_augmented_335_425_mii_l_contralat_combat_long",
    "BiomedCLIP": SCRATCH / "drift_analysis_augmented_335_425_biomedclip_l_contralat_combat_long",
}

MALIGNANT, BIOPSY_NEG, CONTROL = "malignant", "biopsy_negative", "control"
MAX_ORDER = 4  # index + 4 priors; deeper is underpowered


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _project_zscored(df: pd.DataFrame, u: np.ndarray) -> pd.Series:
    proj = np.stack(df["embedding"].values) @ u
    return pd.Series((proj - proj.mean()) / proj.std(), index=df.index)


def _holm(pvals: list[float]) -> list[float]:
    if not pvals:
        return []
    return list(multipletests(pvals, method="holm")[1])


def _paired_rows(diffs: np.ndarray, comparison: str, model: str,
                 order: int) -> dict:
    """One-sided paired Wilcoxon that the case/affected side is higher."""
    diffs = diffs[~np.isnan(diffs)]
    n = len(diffs)
    if n < 10 or np.allclose(diffs, 0):
        return {"model": model, "comparison": comparison, "exam_order": order,
                "n_pairs": n, "median_diff": float(np.median(diffs)) if n else np.nan,
                "p_raw": np.nan}
    try:
        p = wilcoxon(diffs, alternative="greater", zero_method="wilcox").pvalue
    except ValueError:
        p = np.nan
    return {"model": model, "comparison": comparison, "exam_order": order,
            "n_pairs": n, "median_diff": float(np.median(diffs)), "p_raw": p}


def _between_family(model: str, parquet_dir: Path, pairs: pd.DataFrame) -> list[dict]:
    df = pd.read_parquet(parquet_dir / "breast" / "exam_embeddings.parquet")
    idx = df[df["is_index"]]
    u = _unit(np.stack(idx.loc[idx["group"] == MALIGNANT, "embedding"].values).mean(0)
              - np.stack(idx.loc[idx["group"] == CONTROL, "embedding"].values).mean(0))
    df = df.copy()
    df["proj_z"] = _project_zscored(df, u)
    # (PATIENTID, exam_order) -> projection
    proj = df.set_index(["PATIENTID", "exam_order"])["proj_z"]

    rows: list[dict] = []
    for case_grp, label in [(MALIGNANT, "malignant_vs_control"),
                            (BIOPSY_NEG, "benign_vs_control")]:
        sub_pairs = pairs[pairs["case_pid"].isin(
            df.loc[df["group"] == case_grp, "PATIENTID"].unique())]
        per_order: list[dict] = []
        for order in range(MAX_ORDER + 1):
            diffs = []
            for _, r in sub_pairs.iterrows():
                cp = proj.get((r["case_pid"], order), np.nan)
                kp = proj.get((r["control_pid"], order), np.nan)
                diffs.append(cp - kp)
            per_order.append(_paired_rows(np.array(diffs, dtype=float),
                                          label, model, order))
        _apply_holm(per_order)
        rows.extend(per_order)
    return rows


def _contra_family(model: str, parquet_dir: Path) -> list[dict]:
    df = pd.read_parquet(parquet_dir / "breast_both_sides" / "exam_embeddings.parquet")
    idx = df[(df["is_index"]) & (df["side_label"] == "affected")]
    u = _unit(np.stack(idx.loc[idx["group"] == MALIGNANT, "embedding"].values).mean(0)
              - np.stack(idx.loc[idx["group"] == CONTROL, "embedding"].values).mean(0))
    df = df.copy()
    df["proj_z"] = _project_zscored(df, u)
    # (PATIENTID, exam_order, side) -> projection
    proj = df.set_index(["PATIENTID", "exam_order", "side_label"])["proj_z"]

    rows: list[dict] = []
    for arm in (MALIGNANT, BIOPSY_NEG, CONTROL):
        pids = df.loc[df["group"] == arm, "PATIENTID"].unique()
        label = f"affected_vs_contralateral_{arm}"
        per_order: list[dict] = []
        for order in range(MAX_ORDER + 1):
            diffs = []
            for pid in pids:
                a = proj.get((pid, order, "affected"), np.nan)
                c = proj.get((pid, order, "contralateral"), np.nan)
                diffs.append(a - c)
            per_order.append(_paired_rows(np.array(diffs, dtype=float),
                                          label, model, order))
        _apply_holm(per_order)
        rows.extend(per_order)
    return rows


def _apply_holm(per_order: list[dict]) -> None:
    """Holm-correct p_raw across exam_orders in place, writing p_holm."""
    valid = [(i, r["p_raw"]) for i, r in enumerate(per_order) if not np.isnan(r["p_raw"])]
    corrected = _holm([p for _, p in valid])
    for (i, _), pc in zip(valid, corrected):
        per_order[i]["p_holm"] = pc
    for r in per_order:
        r.setdefault("p_holm", np.nan)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-csv", type=Path, required=True)
    args = ap.parse_args()

    df_cases = pd.read_csv(CASES_CSV, low_memory=False)
    df_controls = pd.read_csv(CONTROLS_CSV, low_memory=False)
    pairs = build_pairs(df_cases, df_controls)

    all_rows: list[dict] = []
    for model, pdir in BETWEEN.items():
        print(f"[between] {model}")
        all_rows.extend(_between_family(model, pdir, pairs))
    for model, pdir in CONTRA.items():
        print(f"[contralateral] {model}")
        all_rows.extend(_contra_family(model, pdir))

    out = pd.DataFrame(all_rows)[
        ["model", "comparison", "exam_order", "n_pairs", "median_diff",
         "p_raw", "p_holm"]]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"wrote {args.out_csv}  ({len(out)} rows)")


if __name__ == "__main__":
    main()
