"""Manufacturer-η² ablation across all 8 cells of {l2-norm, baseline-sub, zscore}.

For each cell c ∈ {___, l__, _b_, __z, lb_, l_z, _bz, lbz}:
  1. Apply the corresponding harmonization steps to a stratified subsample.
  2. Compute η² (between-MANUFACTURER variance / total variance).
  3. Permutation null on label shuffle (1,000 perms).
  4. Pairwise manufacturer-centroid cosine similarity.

Sampling design:
  - Subgroups: malignant cases / benign cases / controls (independent — no
    case-control pairing, since pairing is irrelevant for variance
    decomposition on manufacturer).
  - Within each subgroup, stratified-sample up to ``--n-per-cell`` patients
    per (subgroup, MANUFACTURER_CATEGORY) cell. Lorad caps tighter than
    GE/Hologic in the new cohort.
  - One study at random per patient (any exam_order).
  - Within that study, pick one CC + one MLO per laterality (canonical
    bilateral set, 4 images max).
  - Mean-pool per study to get one 2048-d vector per patient.

For step-2 cells (_b_, lb_, _bz, lbz), baseline subtraction needs a second
exam per patient. We additionally pull each sampled patient's *earliest
prior* (highest exam_order in the new cohort), aggregate it the same way,
include it in the harmonization pipeline so the per-patient baseline
exists, then drop it before computing η² (we only score the originally
sampled exam).

Outputs:
  - {output_dir}/harm_ablation_eta.csv  (one row per cell)
  - {output_dir}/sampled_meta.csv       (one row per sampled study)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.cohort_embeddings import (  # noqa: E402
    apply_baseline_subtraction,
    apply_zscore,
    aggregate_exam_embeddings,
)

logger = logging.getLogger(__name__)


CC_VIEWS = {"CC", "CCID"}
MLO_VIEWS = {"MLO", "MLOID"}
DEFAULT_NEW_DIR = Path(
    "/path/to/scratch/drift_cohort/test_rerun_2026_06_16"
)
DEFAULT_EMB_ROOT = Path("/path/to/scratch/drift_cohort/embeddings")
DEFAULT_OUT = Path(
    "/path/to/scratch/drift_cohort/diagnostics/harm_ablation_eta"
)

CELLS = ["___", "l__", "_b_", "__z", "lb_", "l_z", "_bz", "lbz"]


def _cell_flags(cell: str) -> dict:
    return {
        "l2_norm": cell[0] == "l",
        "subtract_baseline": cell[1] == "b",
        "zscore_dim": cell[2] == "z",
    }


# ----------------------------- sampling ----------------------------- #

def stratified_sample_one_study(
    df: pd.DataFrame,
    subgroup_label: str,
    n_per_cell: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """One study per patient, stratified by MANUFACTURER_CATEGORY.

    Returns a frame with the **original SOP-level rows** of the sampled
    studies (so the caller can mean-pool / pick canonical views).
    """
    studies = df.drop_duplicates(["PATIENTID", "STUDYID"])
    counts = studies["MANUFACTURER_CATEGORY"].value_counts()
    keep_cats = sorted(
        counts[(counts >= 30) & (counts.index != "Unknown")].index.tolist()
    )
    logger.info("[%s] viable manuf categories (≥30 patients): %s",
                subgroup_label, {k: int(counts[k]) for k in keep_cats})

    chosen_keys: list[tuple] = []
    for cat in keep_cats:
        cat_studies = studies[studies["MANUFACTURER_CATEGORY"] == cat]
        cat_patients = cat_studies["PATIENTID"].unique()
        n_take = min(n_per_cell, len(cat_patients))
        chosen_pids = rng.choice(cat_patients, size=n_take, replace=False)
        for pid in chosen_pids:
            pat_rows = cat_studies[cat_studies["PATIENTID"] == pid]
            row = pat_rows.sample(n=1, random_state=int(rng.integers(1 << 31))).iloc[0]
            chosen_keys.append((pid, row["STUDYID"]))
        logger.info("  [%s/%s] sampled %d (target %d)", subgroup_label, cat,
                    n_take, n_per_cell)

    key_set = set(chosen_keys)
    return df[df.apply(lambda r: (r["PATIENTID"], r["STUDYID"]) in key_set, axis=1)].copy()


def pull_earliest_prior_rows(df_full: pd.DataFrame, sampled_pids: set[str]) -> pd.DataFrame:
    """For each sampled patient, also pull all SOPs from their earliest prior
    (highest exam_order). Used for step-2 (baseline-subtract) cells which
    need a second per-patient exam to subtract."""
    sub = df_full[df_full["PATIENTID"].isin(sampled_pids)]
    studies = sub.drop_duplicates(["PATIENTID", "STUDYID"])
    earliest = (studies.sort_values(["PATIENTID", "exam_order"], ascending=[True, False])
                       .drop_duplicates("PATIENTID", keep="first"))
    keys = set(zip(earliest["PATIENTID"], earliest["STUDYID"]))
    return sub[sub.apply(lambda r: (r["PATIENTID"], r["STUDYID"]) in keys, axis=1)].copy()


def pick_canonical_views(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Within each (PATIENTID, STUDYID), keep ≤1 CC + ≤1 MLO per laterality.

    Operates on image-level rows; returns image-level rows."""
    keep = []
    for (_, _), study in df.groupby(["PATIENTID", "STUDYID"]):
        for lat in ["L", "R"]:
            side = study[study["IMAGELATERALITY"] == lat]
            cc = side[side["VIEWPOSITION"].isin(CC_VIEWS)] if "VIEWPOSITION" in side.columns else side.iloc[0:0]
            mlo = side[side["VIEWPOSITION"].isin(MLO_VIEWS)] if "VIEWPOSITION" in side.columns else side.iloc[0:0]
            if len(cc):
                keep.append(cc.sample(n=1, random_state=int(rng.integers(1 << 31))).iloc[0])
            if len(mlo):
                keep.append(mlo.sample(n=1, random_state=int(rng.integers(1 << 31))).iloc[0])
    if not keep:
        return df.iloc[0:0]
    return pd.DataFrame(keep)


def attach_view_position(df: pd.DataFrame, all_screening_path: str) -> pd.DataFrame:
    """The matched CSVs no longer carry VIEWPOSITION post-melt; we need it
    to pick canonical views. Join from the source all_screening parquet."""
    src = pd.read_parquet(
        all_screening_path,
        columns=["STUDYID", "SOPINSTANCEUID", "VIEWPOSITION"],
    ).drop_duplicates(["STUDYID", "SOPINSTANCEUID"])
    return df.merge(src, on=["STUDYID", "SOPINSTANCEUID"], how="left")


# ----------------------------- statistics ----------------------------- #

def variance_decomposition(X: np.ndarray, labels: np.ndarray) -> float:
    X = X.astype(np.float64)
    grand = X.mean(axis=0, keepdims=True)
    ss_total = float(((X - grand) ** 2).sum())
    if ss_total <= 0:
        return float("nan")
    ss_between = 0.0
    for lab in np.unique(labels):
        m = labels == lab
        n = int(m.sum())
        mu = X[m].mean(axis=0, keepdims=True)
        ss_between += n * float(((mu - grand) ** 2).sum())
    return ss_between / ss_total


def permute_eta2(X: np.ndarray, labels: np.ndarray, n_perm: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    obs = variance_decomposition(X, labels)
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = variance_decomposition(X, rng.permutation(labels))
    p = (float((null >= obs).sum()) + 1) / (n_perm + 1)
    return {
        "eta2": obs,
        "null_mean": float(null.mean()),
        "null_p95": float(np.quantile(null, 0.95)),
        "pvalue": p,
    }


def centroid_table(X: np.ndarray, labels: np.ndarray) -> dict:
    out: dict = {}
    label_order = sorted(set(labels.tolist()))
    cents = {lab: X[labels == lab].mean(axis=0) for lab in label_order}
    for i, a in enumerate(label_order):
        for b in label_order[i + 1:]:
            ca, cb = cents[a], cents[b]
            cos = float((ca @ cb) / (np.linalg.norm(ca) * np.linalg.norm(cb) + 1e-12))
            out[f"cos_{a}_{b}"] = cos
    return out


# ----------------------------- per-cell harmonization ----------------------------- #

def run_cell(
    cell: str,
    sampled_image_rows: pd.DataFrame,
    sampled_keys: set[tuple],
    extra_baseline_rows: pd.DataFrame,
    emb_root: Path,
    n_perm: int,
    seed: int,
) -> dict:
    """Apply the harm cell's transforms and compute η² on MANUFACTURER_CATEGORY."""
    flags = _cell_flags(cell)

    # Step 1 (l2-norm) is per-image, applied during aggregation.
    # If step 2 is in this cell, we aggregate sampled+earliest-prior together
    # so baseline subtraction has both rows per patient. Otherwise just sampled.
    if flags["subtract_baseline"]:
        agg_input = pd.concat([sampled_image_rows, extra_baseline_rows], ignore_index=True)
    else:
        agg_input = sampled_image_rows

    exam = aggregate_exam_embeddings(
        agg_input, emb_root=emb_root, granularity="study",
        emb_dim=2048, l2_norm=flags["l2_norm"],
    )
    if exam.empty:
        return {"cell": cell, "n": 0, "eta2": float("nan")}

    if flags["subtract_baseline"]:
        exam = apply_baseline_subtraction(exam)
    if flags["zscore_dim"]:
        # apply_zscore needs an is_index column; in our sample the originally-
        # sampled row may or may not be is_index. The function fits μ/σ on
        # is_index==True rows; if none exist, it errors. To make the test
        # well-defined, treat the sampled rows as the "fit population" by
        # temporarily marking them is_index=True.
        exam_z = exam.copy()
        exam_z["is_index"] = exam_z.apply(
            lambda r: (r["PATIENTID"], r["STUDYID"]) in sampled_keys, axis=1,
        )
        exam = apply_zscore(exam_z)

    # Restrict to the originally sampled rows for the η² test.
    exam = exam[exam.apply(
        lambda r: (r["PATIENTID"], r["STUDYID"]) in sampled_keys, axis=1
    )].copy()

    if exam.empty or "MANUFACTURER_CATEGORY" not in exam.columns:
        return {"cell": cell, "n": 0, "eta2": float("nan")}

    X = np.stack(exam["embedding"].to_numpy()).astype(np.float64)
    labs = exam["MANUFACTURER_CATEGORY"].to_numpy()
    perm = permute_eta2(X, labs, n_perm=n_perm, seed=seed)
    cents = centroid_table(X, labs)
    rec = {"cell": cell, "n": int(len(exam)), **perm, **cents}
    rec["manuf_breakdown"] = exam["MANUFACTURER_CATEGORY"].value_counts().to_dict()
    return rec


# ----------------------------- main ----------------------------- #

def build_subgroup_dfs(cases: pd.DataFrame, ctrls: pd.DataFrame) -> dict:
    """Split into malignant cases, benign cases, and controls."""
    mal = cases[cases["Cancer"] == 1].copy()
    benign = cases[cases["Cancer"] == 0].copy()
    return {"malignant": mal, "biopsy_negative": benign, "control": ctrls}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cases-csv", type=Path,
                   default=DEFAULT_NEW_DIR / "matched_cases_Density_nPriors_d335_d425_mesh.csv")
    p.add_argument("--controls-csv", type=Path,
                   default=DEFAULT_NEW_DIR / "matched_controls_Density_nPriors_d335_d425_mesh.csv")
    p.add_argument("--embeddings-dir", type=Path, default=DEFAULT_EMB_ROOT)
    p.add_argument("--all-screening-parquet", type=str,
                   default="/path/to/datasets/mammo/mammo_hoppr/source_tables/"
                           "all_screening_studies_filtered_llm_meta_08262025.parquet")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--n-per-cell", type=int, default=66,
                   help="Target patients per (subgroup, manufacturer) cell. "
                        "Lorad caps tighter (~35–47) in the new cohort.")
    p.add_argument("--n-perm", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    cases = pd.read_csv(args.cases_csv)
    ctrls = pd.read_csv(args.controls_csv)
    logger.info("loaded cases (%d rows, %d patients), controls (%d rows, %d patients)",
                len(cases), cases["PATIENTID"].nunique(),
                len(ctrls), ctrls["PATIENTID"].nunique())

    # The matched CSV has SOPINSTANCEUID + IMAGELATERALITY but NOT VIEWPOSITION.
    # Join VIEWPOSITION from the screening source so pick_canonical_views works.
    cases = attach_view_position(cases, args.all_screening_parquet)
    ctrls = attach_view_position(ctrls, args.all_screening_parquet)

    rng = np.random.default_rng(args.seed)
    subgroups = build_subgroup_dfs(cases, ctrls)

    sampled_per_subgroup: list[pd.DataFrame] = []
    for sg_name, sg_df in subgroups.items():
        s = stratified_sample_one_study(sg_df, sg_name, args.n_per_cell,
                                        rng=np.random.default_rng(args.seed + hash(sg_name) % 1000))
        s = pick_canonical_views(s, rng)
        s["_subgroup"] = sg_name
        sampled_per_subgroup.append(s)
        logger.info("[%s] image-level rows after canonical-view pick: %d "
                    "(across %d studies)", sg_name, len(s),
                    s.drop_duplicates(["PATIENTID", "STUDYID"]).shape[0])

    sampled = pd.concat(sampled_per_subgroup, ignore_index=True)
    sampled_keys = set(zip(sampled["PATIENTID"], sampled["STUDYID"]))
    logger.info("Total sampled studies across subgroups: %d", len(sampled_keys))

    # For step-2 cells, we need each patient's earliest prior. Pull from the
    # full cohort frames (cases or ctrls depending on subgroup).
    extra_rows: list[pd.DataFrame] = []
    for sg_name, sg_df in subgroups.items():
        sg_pids = set(sampled[sampled["_subgroup"] == sg_name]["PATIENTID"])
        if not sg_pids:
            continue
        e = pull_earliest_prior_rows(sg_df, sg_pids)
        # pick canonical views for these too
        e = pick_canonical_views(e, rng)
        # Skip rows that ARE the sampled study (when the sampled study IS
        # the earliest prior — degenerate case for step 2; baseline-sub will
        # zero them out).
        e = e[~e.apply(
            lambda r: (r["PATIENTID"], r["STUDYID"]) in sampled_keys, axis=1
        )].copy()
        e["_subgroup"] = sg_name
        extra_rows.append(e)
    extra_baseline_rows = pd.concat(extra_rows, ignore_index=True) if extra_rows else pd.DataFrame()
    logger.info("Extra (baseline) image rows pulled for step-2 cells: %d",
                len(extra_baseline_rows))

    # Save the sampled metadata for traceability.
    sampled_meta = (sampled[["PATIENTID", "STUDYID", "_subgroup",
                             "MANUFACTURER_CATEGORY"]]
                    .drop_duplicates(["PATIENTID", "STUDYID"]))
    sampled_meta.to_csv(args.output_dir / "sampled_meta.csv", index=False)

    rows: list[dict] = []
    for cell in CELLS:
        logger.info("=== cell %s ===", cell)
        rec = run_cell(
            cell,
            sampled_image_rows=sampled,
            sampled_keys=sampled_keys,
            extra_baseline_rows=extra_baseline_rows,
            emb_root=args.embeddings_dir,
            n_perm=args.n_perm,
            seed=args.seed + 1,
        )
        cents = {k: v for k, v in rec.items() if k.startswith("cos_")}
        logger.info("  cell=%s  n=%d  η²=%.4f  null_mean=%.4f  p=%.4g  cents=%s",
                    rec["cell"], rec["n"], rec.get("eta2", float("nan")),
                    rec.get("null_mean", float("nan")),
                    rec.get("pvalue", float("nan")), cents)
        rows.append(rec)

    # Tabulate
    df = pd.DataFrame(rows)
    out_csv = args.output_dir / "harm_ablation_eta.csv"
    df.to_csv(out_csv, index=False)
    logger.info("Wrote %s", out_csv)
    logger.info("Summary:\n%s", df[["cell", "n", "eta2", "null_mean",
                                    "null_p95", "pvalue"]].to_string(index=False))


if __name__ == "__main__":
    main()
