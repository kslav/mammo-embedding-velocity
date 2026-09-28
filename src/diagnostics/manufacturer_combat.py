"""Apply NeuroComBat to harmonize MANUFACTURER_CATEGORY as a batch effect on
the same 200 cases / 200 controls sample used by manufacturer_variation.py,
and recompute η² + permutation p-value.

NeuroComBat fits per-feature location/scale shifts conditional on batch label
and an empirical Bayes shrinkage prior across features. We pass `Cancer` as
a *biological* covariate to preserve so ComBat doesn't subtract part of the
case/control signal along with the manufacturer batch effect.

Lorad and Hologic are 0.88 cos-similar in the original space (close to
indistinguishable), but we keep them as separate batch labels for ComBat
because (a) collapsing them constrains us to a 2-batch fit even though we
have 3 detector families, and (b) NeuroComBat handles small batches via EB
shrinkage. If Lorad ends up degenerate we'll see it in the post-η².
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

# Import the canonical ComBat helper from utils. Both this diagnostic and
# the production apply_combat_manuf route through combat_harmonize_array.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logger = logging.getLogger(__name__)


from config import DATA_DIR, SCRATCH_DIR, SOURCE_TABLES_DIR  # noqa: E402
PROD_DIR = DATA_DIR
DEFAULT_EMB_ROOT = SCRATCH_DIR / "embeddings"
DEFAULT_ALL_SCREENING = str(
    SOURCE_TABLES_DIR / "all_screening_studies_filtered_llm_meta_08262025.parquet")
DEFAULT_OUT = SCRATCH_DIR / "diagnostics" / "manufacturer_combat"

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
    studies_df: pd.DataFrame, n_total: int,
    categories: list[str], rng: np.random.Generator,
) -> pd.DataFrame:
    per_bucket = n_total // len(categories)
    selected = []
    for cat in categories:
        cat_studies = studies_df[studies_df["MANUFACTURER_CATEGORY"] == cat]
        cat_patients = cat_studies["PATIENTID"].unique()
        n_take = min(per_bucket, len(cat_patients))
        chosen = rng.choice(cat_patients, size=n_take, replace=False)
        for pid in chosen:
            pat_rows = cat_studies[cat_studies["PATIENTID"] == pid]
            row = pat_rows.sample(n=1, random_state=int(rng.integers(1 << 31))).iloc[0]
            selected.append(row)
        logger.info("  sampled %d patients from %s (target %d, pool %d)",
                    n_take, cat, per_bucket, len(cat_patients))
    return pd.DataFrame(selected).reset_index(drop=True)


def pick_canonical_views(images: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    keep = []
    for lat in ["L", "R"]:
        side = images[images["IMAGELATERALITY"] == lat]
        cc = side[side["VIEWPOSITION"].isin(CC_VIEWS)]
        mlo = side[side["VIEWPOSITION"].isin(MLO_VIEWS)]
        if len(cc):
            keep.append(cc.sample(n=1, random_state=int(rng.integers(1 << 31))).iloc[0])
        if len(mlo):
            keep.append(mlo.sample(n=1, random_state=int(rng.integers(1 << 31))).iloc[0])
    if not keep:
        return images.iloc[0:0]
    return pd.DataFrame(keep)


def load_embedding(path: Path, emb_dim: int = 2048) -> np.ndarray | None:
    if not path.is_file():
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
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


def apply_combat(
    X: np.ndarray,
    batch: np.ndarray,
    categorical_cov: dict[str, np.ndarray] | None = None,
    continuous_cov: dict[str, np.ndarray] | None = None,
) -> np.ndarray:
    """Thin wrapper around the canonical ``combat_harmonize_array`` helper.

    Kept here purely for the ``categorical_cov`` / ``continuous_cov`` kwarg
    names that match this diagnostic's call sites. All actual ComBat work
    (covariate frame construction, near-constant feature passthrough, the
    patched-neuroCombat context) happens inside ``combat_harmonize_array``.
    """
    from utils.cohort_embeddings import combat_harmonize_array
    return combat_harmonize_array(
        X,
        batch=batch,
        categorical=categorical_cov,
        continuous=continuous_cov,
    )


def compute_time_to_index_years(df: pd.DataFrame) -> pd.Series:
    """Per-row time from the index exam, in years.
    Convention: date_diff[k] = days between exam k and the next-newer exam k-1
    (the index has NaN). Cumulative days from the index = sum of date_diff for
    exam_orders 1..k. We sort by exam_order ascending per patient so cumsum
    walks back in time."""
    s = df.sort_values(["PATIENTID", "exam_order"]).copy()
    s["date_diff_filled"] = s["date_diff"].fillna(0).astype(float)
    s["cum_days_from_index"] = (
        s.groupby("PATIENTID")["date_diff_filled"].cumsum()
    )
    s["time_to_index_years"] = s["cum_days_from_index"] / 365.25
    return s.set_index(["PATIENTID", "STUDYID"])["time_to_index_years"]


def sample_arm(
    label: str, df: pd.DataFrame, n_target: int,
    emb_root: Path, rng: np.random.Generator,
) -> pd.DataFrame:
    """Stratify-sample n_target studies (one per patient) from this arm,
    pick canonical 1 CC + 1 MLO per laterality, mean-pool embeddings.
    Returns a DataFrame with one row per sampled study, including
    embedding + preserved covariates."""
    studies = df.drop_duplicates(["PATIENTID", "STUDYID"])
    counts = studies["MANUFACTURER_CATEGORY"].value_counts()
    keep_cats = sorted(
        counts[(counts >= 30) & (counts.index != "Unknown")].index.tolist()
    )
    logger.info("[%s] viable categories: %s", label,
                {k: int(counts[k]) for k in keep_cats})

    sampled = stratified_sample_studies(studies, n_target, keep_cats, rng)
    sample_keys = set(zip(sampled["PATIENTID"], sampled["STUDYID"]))

    # Precompute time_to_index_years for the whole arm (needs full chain to cumsum).
    tti = compute_time_to_index_years(df.drop_duplicates(["PATIENTID", "STUDYID"]))

    rows = []
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
        cancer_val = canon["Cancer"].iloc[0] if "Cancer" in canon.columns else 0
        rows.append({
            "PATIENTID": pid, "STUDYID": sid,
            "arm": label,
            "MANUFACTURER_CATEGORY": canon["MANUFACTURER_CATEGORY"].iloc[0],
            "Cancer": int(cancer_val) if pd.notna(cancer_val) else 0,
            "is_index": int(bool(canon["is_index"].iloc[0])),
            "time_to_index_years": float(tti.get((pid, sid), 0.0)),
            "n_images_used": len(embs),
            "embedding": np.mean(np.stack(embs), axis=0),
        })
    out = pd.DataFrame(rows)
    logger.info(
        "[%s] usable studies: %d (manuf: %s, Cancer mean: %.3f, "
        "is_index mean: %.3f, time_to_index_years: μ=%.2f σ=%.2f)",
        label, len(out),
        out["MANUFACTURER_CATEGORY"].value_counts().to_dict(),
        out["Cancer"].mean(), out["is_index"].mean(),
        out["time_to_index_years"].mean(), out["time_to_index_years"].std(),
    )
    return out


def report_eta(
    X: np.ndarray, batch: np.ndarray, label: str, stage: str,
    n_perm: int, seed: int, label_order: list[str],
) -> tuple[float, dict | None, pd.DataFrame]:
    var = variance_decomposition(X, batch)
    perm = permute_eta2(X, batch, n_perm=n_perm, seed=seed) if n_perm > 0 else None
    cent = centroid_table(X, batch, label_order)
    cent.insert(0, "stage", stage)
    cent.insert(0, "arm", label)

    logger.info("[%s, %s] η² = %.4f", label, stage, var["eta2"])
    if perm is not None:
        logger.info(
            "[%s, %s] perm: obs=%.4f, null mean=%.4f, null 95th=%.4f, p=%.4g",
            label, stage, perm["eta2_observed"], perm["eta2_null_mean"],
            perm["eta2_null_p95"], perm["pvalue"],
        )
    return var["eta2"], perm, cent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cases-csv", type=Path, default=PROD_DIR / "matched_cases_Density_nPriors_d335_d425_mesh.csv")
    p.add_argument("--controls-csv", type=Path, default=PROD_DIR / "matched_controls_Density_nPriors_d335_d425_mesh.csv")
    p.add_argument("--embeddings-dir", type=Path, default=DEFAULT_EMB_ROOT)
    p.add_argument("--all-screening-parquet", type=str, default=DEFAULT_ALL_SCREENING)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--n-per-arm", type=int, default=200)
    p.add_argument("--n-perm", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    cases = pd.read_csv(args.cases_csv)
    ctrls = pd.read_csv(args.controls_csv)
    cases = attach_manufacturer_and_view(cases, args.all_screening_parquet)
    ctrls = attach_manufacturer_and_view(ctrls, args.all_screening_parquet)

    rng = np.random.default_rng(args.seed)
    cases_sample = sample_arm("cases", cases, args.n_per_arm,
                              args.embeddings_dir, rng)
    ctrls_sample = sample_arm("controls", ctrls, args.n_per_arm,
                              args.embeddings_dir,
                              np.random.default_rng(args.seed + 100))

    # ---- Joint cases+controls fit ----
    joint = pd.concat([cases_sample, ctrls_sample], ignore_index=True)
    X_joint = np.stack(joint["embedding"].to_numpy())
    batch_joint = joint["MANUFACTURER_CATEGORY"].to_numpy()
    label_order = sorted(np.unique(batch_joint).tolist())

    logger.info("Joint sample: %d rows total (%d cases + %d controls); "
                "manuf: %s; Cancer mean: %.3f; is_index mean: %.3f",
                len(joint), len(cases_sample), len(ctrls_sample),
                joint["MANUFACTURER_CATEGORY"].value_counts().to_dict(),
                joint["Cancer"].mean(), joint["is_index"].mean())

    # PRE
    cents = []
    for arm_label, mask in [("cases", joint["arm"] == "cases"),
                            ("controls", joint["arm"] == "controls"),
                            ("joint", np.ones(len(joint), dtype=bool))]:
        eta, perm, cent = report_eta(
            X_joint[mask], batch_joint[mask], arm_label, "pre",
            args.n_perm, args.seed + 1, label_order,
        )
        cents.append(cent)
    plot_pca(X_joint, batch_joint, label_order,
             f"PRE-ComBat — joint (η²={variance_decomposition(X_joint, batch_joint)['eta2']:.3f})",
             args.output_dir / "pca_joint_pre.png")

    # Joint ComBat fit, preserving Cancer (categorical), is_index (categorical),
    # and time_to_index_years (continuous). All three are clinical signals we
    # do NOT want absorbed into the manufacturer batch correction.
    logger.info("Fitting joint ComBat with preserved covariates: "
                "Cancer (cat), is_index (cat), time_to_index_years (cont)")
    X_harm = apply_combat(
        X_joint, batch_joint,
        categorical_cov={
            "Cancer": joint["Cancer"].to_numpy(),
            "is_index": joint["is_index"].to_numpy(),
        },
        continuous_cov={
            "time_to_index_years": joint["time_to_index_years"].to_numpy(),
        },
    )

    # POST
    for arm_label, mask in [("cases", joint["arm"] == "cases"),
                            ("controls", joint["arm"] == "controls"),
                            ("joint", np.ones(len(joint), dtype=bool))]:
        eta, perm, cent = report_eta(
            X_harm[mask], batch_joint[mask], arm_label, "post",
            args.n_perm, args.seed + 2, label_order,
        )
        cents.append(cent)
    plot_pca(X_harm, batch_joint, label_order,
             f"POST-ComBat — joint (η²={variance_decomposition(X_harm, batch_joint)['eta2']:.4f})",
             args.output_dir / "pca_joint_post.png")

    pd.concat(cents, ignore_index=True).to_csv(
        args.output_dir / "centroid_distances_pre_post.csv", index=False
    )

    # Save the harmonized + raw embeddings for inspection.
    joint_meta = joint[["PATIENTID", "STUDYID", "arm", "MANUFACTURER_CATEGORY",
                        "Cancer", "is_index", "time_to_index_years",
                        "n_images_used"]].copy()
    np.savez_compressed(
        args.output_dir / "joint_embeddings.npz",
        X_pre=X_joint.astype(np.float32),
        X_post=X_harm.astype(np.float32),
    )
    joint_meta.to_csv(args.output_dir / "joint_meta.csv", index=False)
    logger.info("Done. Outputs in %s", args.output_dir)


if __name__ == "__main__":
    main()
