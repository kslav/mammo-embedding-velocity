#!/usr/bin/env python3
"""
Step 04a — Embedding-distance sanity checks.

Before running the drift analysis (04b), this step verifies that the index
embeddings for our three groups (malignant, biopsy_negative, control) are
actually separated in Mammo-CLIP embedding space. If the groups are
indistinguishable here, no drift analysis along a cancer direction can work.

For each granularity (breast, study):
  1. Aggregate image-level rows → per-exam embeddings (shared helper in
     utils.cohort_embeddings). Writes exam_embeddings.parquet which 04b will
     re-use.
  2. Compute group-mean index embeddings and the L2 and cosine distances
     between them. Run a label-permutation null to quantify whether the
     observed separation is larger than chance given group sizes and
     embedding variance.
  3. Project all index embeddings onto u_hat = mean(malignant) − mean(control)
     and plot overlaid histograms by group — the 1-D view of the direction
     04b will use.
  4. t-SNE over all index embeddings, colored by group.

Outputs → <output_dir>/{granularity}/ :
  - exam_embeddings.parquet  (shared with 04b)
  - distances.json           (observed distances + permutation null)
  - u_hat_projection_hist.png
  - tsne.png
  - tsne_coords.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402

from utils.cohort_embeddings import (  # noqa: E402
    BIOPSY_NEG,
    CONTROL,
    MALIGNANT,
    UHAT_PROJ_LABEL,
    build_exam_table,
    group_label,
)


def group_mean_distances(
    index_df: pd.DataFrame, rng: np.random.Generator, n_perm: int = 1000
) -> dict:
    """
    Observed distances between group-mean index embeddings, plus a
    label-permutation null for each pairwise comparison.
    """
    X_by_group = {
        grp: np.stack(index_df.loc[index_df["group"] == grp, "embedding"].values)
        for grp in index_df["group"].unique()
    }
    means = {grp: X.mean(axis=0) for grp, X in X_by_group.items()}

    def euclid(a, b):
        return float(np.linalg.norm(a - b))

    def cosine_dist(a, b):
        return float(1 - (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    pairs = [(MALIGNANT, CONTROL), (BIOPSY_NEG, CONTROL), (MALIGNANT, BIOPSY_NEG)]
    out = {"n_by_group": {g: int(len(X)) for g, X in X_by_group.items()},
           "pairs": {}}

    for g1, g2 in pairs:
        if g1 not in X_by_group or g2 not in X_by_group:
            continue
        X1, X2 = X_by_group[g1], X_by_group[g2]
        obs_euclid = euclid(means[g1], means[g2])
        obs_cos = cosine_dist(means[g1], means[g2])

        # Permutation null: pool X1 and X2, randomly split into sizes (n1, n2),
        # recompute mean-to-mean distance, repeat n_perm times.
        X_pool = np.vstack([X1, X2])
        n1 = len(X1)
        n_total = len(X_pool)
        null_euclid = np.empty(n_perm)
        null_cos = np.empty(n_perm)
        for k in range(n_perm):
            idx = rng.permutation(n_total)
            a = X_pool[idx[:n1]].mean(axis=0)
            b = X_pool[idx[n1:]].mean(axis=0)
            null_euclid[k] = euclid(a, b)
            null_cos[k] = cosine_dist(a, b)

        p_euclid = float((null_euclid >= obs_euclid).sum() + 1) / (n_perm + 1)
        p_cos = float((null_cos >= obs_cos).sum() + 1) / (n_perm + 1)

        out["pairs"][f"{g1}_vs_{g2}"] = {
            "n1": int(n1),
            "n2": int(n_total - n1),
            "euclid": {
                "observed": obs_euclid,
                "null_mean": float(null_euclid.mean()),
                "null_std": float(null_euclid.std()),
                "null_q95": float(np.quantile(null_euclid, 0.95)),
                "pvalue_perm": p_euclid,
            },
            "cosine_dist": {
                "observed": obs_cos,
                "null_mean": float(null_cos.mean()),
                "null_std": float(null_cos.std()),
                "null_q95": float(np.quantile(null_cos, 0.95)),
                "pvalue_perm": p_cos,
            },
        }
    return out


def projection_hist(index_df: pd.DataFrame, out_dir: Path, tag: str) -> None:
    """
    Project each index embedding onto u_hat = mean(malignant) − mean(control)
    (L2-normalized) and plot overlaid histograms by group. The 1-D view of the
    separation the permutation test is detecting.
    """
    groups_present = set(index_df["group"].unique())
    if MALIGNANT not in groups_present or CONTROL not in groups_present:
        print(f"  skipping projection_hist: need {MALIGNANT}+{CONTROL}, have {groups_present}")
        return

    emb_malig = np.stack(index_df.loc[index_df["group"] == MALIGNANT, "embedding"].values)
    emb_ctrl = np.stack(index_df.loc[index_df["group"] == CONTROL, "embedding"].values)
    u = emb_malig.mean(axis=0) - emb_ctrl.mean(axis=0)
    u = u / np.linalg.norm(u)

    proj = {grp: np.stack(index_df.loc[index_df["group"] == grp, "embedding"].values) @ u
            for grp in index_df["group"].unique()}

    palette = {CONTROL: "#7f7f7f", BIOPSY_NEG: "#1f77b4", MALIGNANT: "#d62728"}
    fig, ax = plt.subplots(figsize=(7, 4))
    for grp in (CONTROL, BIOPSY_NEG, MALIGNANT):
        if grp not in proj:
            continue
        ax.hist(proj[grp], bins=40, alpha=0.5,
                label=f"{group_label(grp)} (n = {len(proj[grp])}, mean = {proj[grp].mean():.3f})",
                color=palette.get(grp, "k"), density=True)
    ax.set_xlabel(UHAT_PROJ_LABEL)
    ax.set_ylabel("Density")
    ax.set_title(f"Index-embedding projection onto cancer direction — {tag}")
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(out_dir / "u_hat_projection_hist.png", dpi=300)
    plt.close(fig)


def tsne_plot(
    index_df: pd.DataFrame, out_dir: Path, tag: str, seed: int = 42
) -> pd.DataFrame:
    """2D t-SNE of index embeddings, colored by group."""
    X = np.stack(index_df["embedding"].values)
    print(f"  running t-SNE on {X.shape[0]} points × {X.shape[1]} dims...")
    ts = TSNE(n_components=2, perplexity=30, init="pca",
              learning_rate="auto", random_state=seed,
              n_iter=1000, metric="euclidean", verbose=0)
    coords = ts.fit_transform(X)

    df = pd.DataFrame({
        "PATIENTID": index_df["PATIENTID"].values,
        "group": index_df["group"].values,
        "tsne_x": coords[:, 0],
        "tsne_y": coords[:, 1],
    })

    palette = {CONTROL: "#7f7f7f", BIOPSY_NEG: "#1f77b4", MALIGNANT: "#d62728"}
    fig, ax = plt.subplots(figsize=(7, 6))
    # plot controls first (background), then cases (foreground)
    for grp in (CONTROL, BIOPSY_NEG, MALIGNANT):
        sub = df[df["group"] == grp]
        if len(sub) == 0:
            continue
        ax.scatter(sub["tsne_x"], sub["tsne_y"],
                   c=palette.get(grp, "k"),
                   s=8, alpha=0.4 if grp == CONTROL else 0.7,
                   label=f"{group_label(grp)} (n = {len(sub)})",
                   edgecolors="none")
    ax.set_title(f"t-SNE of index embeddings — {tag}")
    ax.set_xlabel("t-SNE dimension 1")
    ax.set_ylabel("t-SNE dimension 2")
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    fig.savefig(out_dir / "tsne.png", dpi=300)
    plt.close(fig)

    df.to_csv(out_dir / "tsne_coords.csv", index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Embedding-distance sanity checks (04a)")
    parser.add_argument("--cases-csv", type=Path, required=True,
                        help="matched_cases CSV from step 2 (image-level rows)")
    parser.add_argument("--controls-csv", type=Path, required=True,
                        help="matched_controls CSV from step 2 (image-level rows)")
    parser.add_argument("--embeddings-dir", type=Path, default=None,
                        help="Root of embedding tree from step 3. Required unless "
                             "--exam-parquet-dir is provided.")
    parser.add_argument("--exam-parquet-dir", type=Path, default=None,
                        help="If provided, skip aggregation and load "
                             "{gran}/exam_embeddings.parquet from this directory "
                             "(produced by a prior 04a/04b run or by "
                             "src/apply_harmonization.py). Mutually exclusive with "
                             "--embeddings-dir; harmonization flags --harm-l2-norm "
                             "and --harm-subtract-baseline / --harm-zscore-dim are "
                             "ignored on this path.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Where to write distances.json, plots, exam_embeddings.parquet")
    parser.add_argument("--granularities", nargs="+",
                        default=["breast", "study"], choices=["breast", "study"])
    parser.add_argument("--n-perm", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--emb-dim", type=int, default=2048,
                        help="Dimension of each saved .pt embedding (default: 2048, "
                             "Mammo-CLIP EfficientNet-B5). Override for other models.")
    parser.add_argument("--harm-l2-norm", action="store_true",
                        help="Harmonization step 1: L2-normalize each per-image "
                             "embedding before mean-pooling within an exam.")
    parser.add_argument("--harm-subtract-baseline", action="store_true",
                        help="Harmonization step 2: subtract each patient's "
                             "earliest-prior per-exam embedding from all of "
                             "their exams (anchor at zero in patient frame).")
    parser.add_argument("--harm-zscore-dim", action="store_true",
                        help="Harmonization step 3: per-dimension z-score across "
                             "the population, fit on index rows.")
    parser.add_argument("--harm-combat-manuf", action="store_true",
                        help="Harmonization step 4: NeuroComBat batch-correction "
                             "on MANUFACTURER_CATEGORY, preserving Cancer + "
                             "is_index + time_to_index_years as biological "
                             "covariates.")
    args = parser.parse_args()

    if args.embeddings_dir is None and args.exam_parquet_dir is None:
        parser.error("must provide either --embeddings-dir or --exam-parquet-dir")
    if args.embeddings_dir is not None and args.exam_parquet_dir is not None:
        parser.error("--embeddings-dir and --exam-parquet-dir are mutually exclusive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"Loading cases from {args.cases_csv}")
    df_cases = pd.read_csv(args.cases_csv)
    print(f"Loading controls from {args.controls_csv}")
    df_controls = pd.read_csv(args.controls_csv)
    print(f"  cases: {len(df_cases)} rows, {df_cases['PATIENTID'].nunique()} patients")
    print(f"  controls: {len(df_controls)} rows, {df_controls['PATIENTID'].nunique()} patients")

    overall: dict[str, dict] = {}
    for gran in args.granularities:
        print(f"\n=== Granularity: {gran} ===")
        gran_out = args.output_dir / gran
        gran_out.mkdir(parents=True, exist_ok=True)

        if args.exam_parquet_dir is not None:
            in_path = args.exam_parquet_dir / gran / "exam_embeddings.parquet"
            print(f"  loading precomputed {in_path}")
            exam = pd.read_parquet(in_path)
        else:
            exam = build_exam_table(
                df_cases, df_controls, args.embeddings_dir, gran,
                emb_dim=args.emb_dim,
                harm_l2_norm=args.harm_l2_norm,
                harm_subtract_baseline=args.harm_subtract_baseline,
                harm_zscore_dim=args.harm_zscore_dim,
                harm_combat_manuf=args.harm_combat_manuf,
            )
        exam.to_parquet(gran_out / "exam_embeddings.parquet")

        index_df = exam[exam["is_index"]].copy()
        print(f"  index rows by group: "
              f"{index_df.groupby('group').size().to_dict()}")

        dists = group_mean_distances(index_df, rng, n_perm=args.n_perm)
        with open(gran_out / "distances.json", "w") as f:
            json.dump(dists, f, indent=2)
        overall[gran] = dists
        for pair, res in dists["pairs"].items():
            e = res["euclid"]
            c = res["cosine_dist"]
            print(f"  {pair}: ‖Δmean‖={e['observed']:.4f} "
                  f"(null {e['null_mean']:.4f}±{e['null_std']:.4f}, "
                  f"p={e['pvalue_perm']:.4g}) | "
                  f"cos_dist={c['observed']:.5f} (p={c['pvalue_perm']:.4g})")

        projection_hist(index_df, gran_out, tag=gran)
        tsne_plot(index_df, gran_out, tag=gran, seed=args.seed)

    # Write to distances_summary.json (NOT summary.json) so that 04a and 04b
    # can safely share --output-dir without racing on the same filename. 04b's
    # drift-stats summary.json is the "canonical" one consumed by
    # aggregate_ablation.py and the manuscript pipeline; 04a's distances info
    # is also mirrored per-granularity in {breast,study}/distances.json.
    with open(args.output_dir / "distances_summary.json", "w") as f:
        json.dump(overall, f, indent=2)
    print(f"\n04a outputs written to {args.output_dir}")


if __name__ == "__main__":
    main()
