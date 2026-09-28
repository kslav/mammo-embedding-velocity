"""Quantify how much of the embedding-space variation is explained by
MANUFACTURER_CATEGORY.

Sampling design (per arm):
  - 200 patients, stratified by MANUFACTURER_CATEGORY so each viable bucket
    is represented (Hologic / GE / Lorad in the production cohort).
  - One study per patient (any exam_order, sampled randomly within stratum).
  - Within that study, pick one CC and one MLO from each laterality
    (4 images max per study — the canonical bilateral set).
  - Mean-pool those ≤4 image embeddings into one 2048-d vector per study.

Reports per arm:
  1. Per-manufacturer counts.
  2. η² (multivariate analog: trace(SS_between)/trace(SS_total)).
  3. Pairwise centroid distances (L2 + cosine sim).
  4. PCA scatter PNG colored by MANUFACTURER_CATEGORY.
  5. Optional permutation test on η² (--n-perm).

MANUFACTURER is recovered by joining the all_screening source table on
STUDYID; VIEWPOSITION is joined on (STUDYID, SOPINSTANCEUID). Buckets
are produced via the same map_manufacturer rule as
src/01_curate_longitudinal_cohort.py.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)


PROD_DIR = Path(
    "/path/to/data/FM_Mammo/hoppr_data/breast_cancer/drift_analysis"
)
DEFAULT_EMB_ROOT = Path("/path/to/scratch/drift_cohort/embeddings")
DEFAULT_ALL_SCREENING = (
    "/path/to/datasets/mammo/mammo_hoppr/source_tables/"
    "all_screening_studies_filtered_llm_meta_08262025.parquet"
)
DEFAULT_OUT = Path(
    "/path/to/scratch/drift_cohort/diagnostics/manufacturer_variation"
)

CC_VIEWS = {"CC", "CCID"}
MLO_VIEWS = {"MLO", "MLOID"}


def map_manufacturer(m):
    if pd.isna(m):
        return "Unknown"
    s = m.lower()
    if "hologic" in s: return "Hologic"
    if "ge " in s:    return "GE"
    if "siemens" in s: return "Siemens"
    if "philips" in s: return "Philips"
    if "lorad" in s:  return "Lorad"
    if "fujifilm" in s: return "Fujifilm"
    return "Other"


def attach_manufacturer_and_view(df: pd.DataFrame, all_screening_path: str) -> pd.DataFrame:
    """Join MANUFACTURER (per study) + VIEWPOSITION (per SOP)."""
    src_m = pd.read_parquet(all_screening_path, columns=["STUDYID", "MANUFACTURER"])
    src_m = src_m.drop_duplicates(subset=["STUDYID"])
    src_v = pd.read_parquet(
        all_screening_path,
        columns=["STUDYID", "SOPINSTANCEUID", "VIEWPOSITION"],
    ).drop_duplicates(subset=["STUDYID", "SOPINSTANCEUID"])
    out = df.merge(src_m, on="STUDYID", how="left")
    out = out.merge(src_v, on=["STUDYID", "SOPINSTANCEUID"], how="left")
    out["MANUFACTURER_CATEGORY"] = out["MANUFACTURER"].apply(map_manufacturer)
    return out


def stratified_sample_studies(
    studies_df: pd.DataFrame,
    n_total: int,
    categories: list[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Pick one study per patient, stratified to fill ≈ n_total/len(categories)
    patients per bucket (or all available if a bucket is smaller)."""
    per_bucket = n_total // len(categories)
    selected = []
    for cat in categories:
        # Patients who have ≥1 study in this category
        cat_studies = studies_df[studies_df["MANUFACTURER_CATEGORY"] == cat]
        cat_patients = cat_studies["PATIENTID"].unique()
        n_take = min(per_bucket, len(cat_patients))
        chosen_pids = rng.choice(cat_patients, size=n_take, replace=False)
        # For each chosen patient, sample exactly one of their studies in this category
        for pid in chosen_pids:
            pat_rows = cat_studies[cat_studies["PATIENTID"] == pid]
            row = pat_rows.sample(n=1, random_state=int(rng.integers(1 << 31))).iloc[0]
            selected.append(row)
        logger.info("  sampled %d patients from %s (target %d, pool %d)",
                    n_take, cat, per_bucket, len(cat_patients))
    return pd.DataFrame(selected).reset_index(drop=True)


def pick_canonical_views(
    images: pd.DataFrame, rng: np.random.Generator
) -> pd.DataFrame:
    """Within a study, keep at most 1 CC + 1 MLO per laterality."""
    keep_rows = []
    for lat in ["L", "R"]:
        side = images[images["IMAGELATERALITY"] == lat]
        cc = side[side["VIEWPOSITION"].isin(CC_VIEWS)]
        mlo = side[side["VIEWPOSITION"].isin(MLO_VIEWS)]
        if len(cc):
            keep_rows.append(cc.sample(n=1, random_state=int(rng.integers(1 << 31))).iloc[0])
        if len(mlo):
            keep_rows.append(mlo.sample(n=1, random_state=int(rng.integers(1 << 31))).iloc[0])
    if not keep_rows:
        return images.iloc[0:0]
    return pd.DataFrame(keep_rows)


def load_embedding(path: Path, emb_dim: int = 2048) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        logger.warning("failed to load %s: %s", path, e)
        return None
    if isinstance(payload, torch.Tensor):
        emb = payload
    elif isinstance(payload, dict) and "embedding" in payload:
        emb = payload["embedding"]
    else:
        return None
    if isinstance(emb, torch.Tensor):
        emb = emb.detach().cpu().numpy()
    emb = np.asarray(emb, dtype=np.float32).reshape(-1)
    return emb if emb.shape[0] == emb_dim else None


def variance_decomposition(X: np.ndarray, labels: np.ndarray) -> dict:
    X = X.astype(np.float64)
    grand = X.mean(axis=0, keepdims=True)
    ss_total = float(((X - grand) ** 2).sum())
    ss_between = 0.0
    for lab in np.unique(labels):
        mask = labels == lab
        n = int(mask.sum())
        mu = X[mask].mean(axis=0, keepdims=True)
        ss_between += n * float(((mu - grand) ** 2).sum())
    eta2 = ss_between / ss_total if ss_total > 0 else float("nan")
    return {"eta2": eta2, "ss_total": ss_total, "ss_between": ss_between}


def centroid_table(X: np.ndarray, labels: np.ndarray, label_order: list[str]) -> pd.DataFrame:
    centroids = {lab: X[labels == lab].mean(axis=0) for lab in label_order}
    rows = []
    for i, a in enumerate(label_order):
        for b in label_order[i + 1:]:
            ca, cb = centroids[a], centroids[b]
            l2 = float(np.linalg.norm(ca - cb))
            cos = float((ca @ cb) / (np.linalg.norm(ca) * np.linalg.norm(cb) + 1e-12))
            rows.append({"a": a, "b": b, "l2": l2, "cosine_sim": cos,
                         "n_a": int((labels == a).sum()),
                         "n_b": int((labels == b).sum())})
    return pd.DataFrame(rows)


def permute_eta2(X: np.ndarray, labels: np.ndarray, n_perm: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    obs = variance_decomposition(X, labels)["eta2"]
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = variance_decomposition(X, rng.permutation(labels))["eta2"]
    p = (float((null >= obs).sum()) + 1) / (n_perm + 1)
    return {"eta2_observed": obs, "eta2_null_mean": float(null.mean()),
            "eta2_null_p95": float(np.quantile(null, 0.95)), "pvalue": p}


def plot_pca(X: np.ndarray, labels: np.ndarray, label_order: list[str],
             title: str, out_png: Path) -> None:
    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(X)
    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = plt.get_cmap("tab10")
    for i, lab in enumerate(label_order):
        m = labels == lab
        ax.scatter(Z[m, 0], Z[m, 1], s=18, alpha=0.6,
                   color=cmap(i), label=f"{lab} (n={int(m.sum())})")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    logger.info("Wrote %s", out_png)


def run_arm(
    label: str, df: pd.DataFrame, n_target: int,
    emb_root: Path, out_dir: Path, n_perm: int, seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    studies = df.drop_duplicates(["PATIENTID", "STUDYID"])
    counts = studies["MANUFACTURER_CATEGORY"].value_counts()
    # Drop tiny buckets (<30 patients) and "Unknown".
    keep = counts[(counts >= 30) & (counts.index != "Unknown")].index.tolist()
    keep = sorted(keep)
    logger.info("[%s] viable categories (n>=30): %s", label,
                {k: int(counts[k]) for k in keep})

    sampled = stratified_sample_studies(studies, n_target, keep, rng)
    logger.info("[%s] sampled %d (PATIENTID, STUDYID) rows", label, len(sampled))

    # For each sampled study, gather all of its image rows and pick canonical 1 CC + 1 MLO per laterality.
    rows = []
    sample_keys = set(zip(sampled["PATIENTID"], sampled["STUDYID"]))
    for (pid, sid), g in df.groupby(["PATIENTID", "STUDYID"]):
        if (pid, sid) not in sample_keys:
            continue
        canon = pick_canonical_views(g, rng)
        if len(canon) == 0:
            continue
        embs = []
        for _, r in canon.iterrows():
            e = load_embedding(emb_root / pid / sid / f"{r['SOPINSTANCEUID']}.pt")
            if e is not None:
                embs.append(e)
        if not embs:
            continue
        rows.append({
            "PATIENTID": pid, "STUDYID": sid,
            "MANUFACTURER_CATEGORY": canon["MANUFACTURER_CATEGORY"].iloc[0],
            "n_images_used": len(embs),
            "embedding": np.mean(np.stack(embs), axis=0),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        logger.warning("[%s] no usable studies after embedding load.", label)
        return pd.DataFrame(), pd.DataFrame()

    logger.info("[%s] usable studies: %d (n_images_used distribution: %s)",
                label, len(out), out["n_images_used"].value_counts().to_dict())
    logger.info("[%s] manufacturer breakdown of usable studies:\n%s",
                label, out["MANUFACTURER_CATEGORY"].value_counts().to_string())

    X = np.stack(out["embedding"].to_numpy())
    labs = out["MANUFACTURER_CATEGORY"].to_numpy()
    label_order = sorted(out["MANUFACTURER_CATEGORY"].unique().tolist())

    var = variance_decomposition(X, labs)
    logger.info("[%s] η² (between/total trace) = %.4f", label, var["eta2"])

    cent = centroid_table(X, labs, label_order)
    cent.insert(0, "arm", label)
    logger.info("[%s] pairwise centroid distances:\n%s", label,
                cent.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    if n_perm > 0:
        perm = permute_eta2(X, labs, n_perm=n_perm, seed=seed + 1)
        logger.info("[%s] permutation: obs=%.4f, null mean=%.4f, "
                    "null 95th=%.4f, p=%.4g",
                    label, perm["eta2_observed"], perm["eta2_null_mean"],
                    perm["eta2_null_p95"], perm["pvalue"])

    plot_pca(X, labs, label_order,
             f"PCA of per-exam embeddings — {label} (η²={var['eta2']:.3f})",
             out_dir / f"pca_{label}.png")

    out_df = out[["PATIENTID", "STUDYID", "MANUFACTURER_CATEGORY", "n_images_used"]].copy()
    out_df.to_csv(out_dir / f"sampled_{label}.csv", index=False)
    return out_df, cent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cases-csv", type=Path, default=PROD_DIR / "matched_cases_Density_nPriors_d335_d425_mesh.csv")
    p.add_argument("--controls-csv", type=Path, default=PROD_DIR / "matched_controls_Density_nPriors_d335_d425_mesh.csv")
    p.add_argument("--embeddings-dir", type=Path, default=DEFAULT_EMB_ROOT)
    p.add_argument("--all-screening-parquet", type=str, default=DEFAULT_ALL_SCREENING)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--n-per-arm", type=int, default=200)
    p.add_argument("--n-perm", type=int, default=0,
                   help="Set >0 for a permutation test on η² (e.g. 1000).")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    cases = pd.read_csv(args.cases_csv)
    ctrls = pd.read_csv(args.controls_csv)

    cases = attach_manufacturer_and_view(cases, args.all_screening_parquet)
    ctrls = attach_manufacturer_and_view(ctrls, args.all_screening_parquet)

    _, cent_c = run_arm("cases", cases, args.n_per_arm,
                        args.embeddings_dir, args.output_dir, args.n_perm, args.seed)
    _, cent_k = run_arm("controls", ctrls, args.n_per_arm,
                        args.embeddings_dir, args.output_dir, args.n_perm, args.seed + 100)
    pd.concat([cent_c, cent_k], ignore_index=True).to_csv(
        args.output_dir / "centroid_distances.csv", index=False
    )
    logger.info("Done. Outputs in %s", args.output_dir)


if __name__ == "__main__":
    main()
