"""Where does the cancer direction u-hat live in PCA space? (probe, not an analysis layer)

PCA on the per-exam index embeddings is an orthonormal rotation, so projecting
u-hat / drift into full PC space changes NOTHING (all dot products, distances,
p-values are identical). We therefore do NOT re-run the drift analysis in PC
space. Instead this is a *diagnostic*: it characterizes the primary u-hat by
asking what fraction of its length sits in the top-k principal components of
the embedding cloud, and how that compares to the fraction of total embedding
VARIANCE the same PCs carry.

Interpretation:
  - If u-hat concentrates in the SAME PCs that carry most embedding variance,
    the cancer direction is a "high-variance" axis (PCA truncation would keep
    it; a few PCs summarize the signal).
  - If u-hat is spread across many PCs / lands in LOW-variance PCs, the signal
    is a subtle direction that PCA truncation would damage — evidence AGAINST
    dimensionality reduction as a preprocessing step, and a useful answer to
    the reviewer question "is this just the dominant mode of variation?"

Computed per model on the headline (L2 + long-ComBat) index embeddings, breast
granularity, for the malignant-vs-control LOO-global u-hat (the direction the
primary analysis uses; LOO per-patient directions are ~identical in aggregate).

Emits results/uhat_pca_spectrum.csv (model × k with cumulative u-hat energy and
cumulative variance) and prints a compact summary (k for 50/80/90% of u-hat).

Run (login-shell-safe: reads the committed exam parquet is NOT available; uses
the scratch harmonized parquet, in-memory PCA on ~3.5k × D — seconds):
    .venv_mammoclip/bin/python src/diagnostics/uhat_pca_spectrum.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.cohort_embeddings import MALIGNANT, CONTROL  # noqa: E402

from config import SCRATCH_DIR as SCRATCH, BIOMEDCLIP_SCRATCH_DIR  # noqa: E402
REPO = Path(__file__).resolve().parents[2]
# Variant -> {model -> breast exam-parquet dir suffix}. Both are long-ComBat;
# l_longcombat additionally L2-normalizes per-image before mean-pooling.
VARIANT_SUFFIX = {
    "l_longcombat": {"mammoclip": "l_combat_long", "hoppr": "hoppr_l_combat_long",
                     "mii": "mii_l_combat_long", "biomedclip": "biomedclip_l_combat_long"},
    "base_longcombat": {"mammoclip": "combat_long", "hoppr": "hoppr_combat_long",
                        "mii": "mii_combat_long", "biomedclip": "biomedclip_combat_long"},
}
VARIANT_LABEL = {"l_longcombat": "L2 + long-ComBat",
                 "base_longcombat": "long-ComBat (raw features)"}
MODEL_DISP = {"mammoclip": "Mammo-CLIP", "hoppr": "HOPPR",
              "mii": "MedImageInsight", "biomedclip": "BiomedCLIP"}
MODELS = ["mammoclip", "mii", "hoppr", "biomedclip"]
# BiomedCLIP's headline parquets live in a separate extraction tree.
_BMC_TREE = BIOMEDCLIP_SCRATCH_DIR


def _parquet_path(variant: str, model: str) -> Path:
    """Resolve the breast exam-parquet for a (variant, model)."""
    suf = VARIANT_SUFFIX[variant][model]
    if model == "biomedclip":
        return _BMC_TREE / f"drift_analysis_{suf}" / "breast" / "exam_embeddings.parquet"
    return (SCRATCH / f"drift_analysis_augmented_335_425_{suf}"
            / "breast" / "exam_embeddings.parquet")
KS = [1, 2, 3, 5, 10, 20, 50, 100]


def _eta2(scores: np.ndarray, labels: np.ndarray) -> float:
    """One-way η² of a 1-D PC-score vector against a categorical label."""
    df = pd.DataFrame({"s": scores, "g": labels})
    gm = df["s"].mean()
    ss_tot = ((df["s"] - gm) ** 2).sum()
    if ss_tot <= 0:
        return 0.0
    ss_bet = df.groupby("g")["s"].apply(lambda x: len(x) * (x.mean() - gm) ** 2).sum()
    return float(ss_bet / ss_tot)


def _uhat_and_pca(exam: pd.DataFrame):
    """Return (u_hat, eigen-variances, V, scores, idx_df).

    PCA is fit on the INDEX embeddings (mean-centered), matching the space in
    which u-hat is defined. u_hat = mean(malignant index) - mean(control index),
    unit-normalized (global, non-LOO — the aggregate direction). `scores` are
    the index embeddings projected onto the PCs (N × D); `idx_df` carries the
    per-exam labels (MG_modality, MANUFACTURER_CATEGORY, group) for η²."""
    idx = exam[exam["is_index"]].copy()
    X = np.stack(idx["embedding"].values).astype(np.float64)
    g = idx["group"].to_numpy()
    mu = X.mean(axis=0)
    Xc = X - mu
    # PCA via SVD on centered index embeddings.
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = (S ** 2) / (len(Xc) - 1)          # eigen-variance per PC
    V = Vt.T                                 # columns = principal axes
    scores = Xc @ V                          # (N, D) PC scores
    u = X[g == MALIGNANT].mean(0) - X[g == CONTROL].mean(0)
    u = u / (np.linalg.norm(u) or 1.0)
    return u, var, V, scores, idx


def _spectrum(exam: pd.DataFrame):
    """Return (cum_uhat_energy, cum_variance) arrays over PC rank 1..D."""
    u, var, V, _, _ = _uhat_and_pca(exam)
    energy = (V.T @ u) ** 2                       # u-hat energy per PC (sums to 1)
    return np.cumsum(energy), np.cumsum(var) / var.sum()


# Per-PC η² curves: which labels drive each PC's variation.
ETA_LABELS = [("MG_modality", "#e08214", "η²(MG modality)"),
              ("MANUFACTURER_CATEGORY", "#8073ac", "η²(manufacturer)"),
              ("group", "#35978f", "η²(group)")]


def _plot_variant(variant: str, out_dir: Path, max_pc: int = 40) -> list:
    """3-panel figure (one per model): cumulative û-energy + cumulative
    variance (left axis) with per-PC η² for MG_modality / manufacturer /
    group (right axis). Returns tidy rows for the CSV (now incl. per-PC η²)."""
    ncol = len(MODELS)
    nrow = 1 if ncol <= 3 else 2
    per_row = ncol if nrow == 1 else (ncol + 1) // 2
    fig, axes = plt.subplots(nrow, per_row, figsize=(4.85 * per_row, 4.8 * nrow),
                             sharey=True, squeeze=False)
    axes = axes.ravel()
    rows = []
    for ax, m in zip(axes, MODELS):
        parq = _parquet_path(variant, m)
        u, var, V, scores, idx = _uhat_and_pca(pd.read_parquet(parq))
        energy = (V.T @ u) ** 2
        cum_e, cum_v = np.cumsum(energy), np.cumsum(var) / var.sum()
        D = len(cum_e)
        xs = np.arange(1, min(max_pc, D) + 1)
        nk = len(xs)
        # per-PC η² for each label over the shown PCs
        eta = {lab: np.array([_eta2(scores[:, k], idx[lab].to_numpy())
                              for k in range(nk)])
               for lab, _, _ in ETA_LABELS}

        # Left axis: cumulative curves.
        ax.plot(xs, cum_e[:nk], color="#8c1515", lw=2.4,
                label=r"cum. $\hat{u}$ energy")
        ax.plot(xs, cum_v[:nk], color="#3b6fb0", lw=2.0, ls="--",
                label="cum. embedding variance")
        ax.axhline(0.8, color="#bbbbbb", lw=0.8, ls=":")
        ax.set_xlabel("Principal component rank", fontweight="bold", fontsize=9)
        ax.grid(alpha=0.2)
        ke = int(np.searchsorted(cum_e, 0.8) + 1)
        ax.set_title(f"{MODEL_DISP[m]}   " r"($k_{80\%\,\hat{u}}$=" f"{ke})",
                     fontweight="bold", fontsize=11)

        # Right axis: per-PC η² (fraction of each PC's variance from a label).
        ax2 = ax.twinx()
        for lab, col, _ in ETA_LABELS:
            ax2.plot(xs, eta[lab], color=col, lw=1.5, alpha=0.9, marker="o", ms=2.5)
        ax2.set_ylim(0, max(0.4, min(1.0, max(eta["MG_modality"].max(),
                                              eta["group"].max()) * 1.25)))
        if ax is axes[-1]:
            ax2.set_ylabel("per-PC η²", fontsize=9, fontweight="bold")
        else:
            ax2.set_yticklabels([])

        for k in KS:
            if k <= D:
                rows.append({"variant": variant, "model": m, "k": k,
                             "cum_uhat_energy": float(cum_e[k - 1]),
                             "cum_variance": float(cum_v[k - 1]),
                             "pc_uhat_energy": float(energy[k - 1]),
                             "eta2_mgmod": _eta2(scores[:, k - 1], idx["MG_modality"].to_numpy()),
                             "eta2_manuf": _eta2(scores[:, k - 1], idx["MANUFACTURER_CATEGORY"].to_numpy()),
                             "eta2_group": _eta2(scores[:, k - 1], idx["group"].to_numpy())})
    axes[0].set_ylabel("Cumulative fraction", fontweight="bold", fontsize=9)
    axes[0].set_ylim(0, 1.02)
    # Combined legend (cumulative curves + η² lines) below the panels.
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], color="#8c1515", lw=2.4, label=r"cum. $\hat{u}$ energy"),
               Line2D([0], [0], color="#3b6fb0", lw=2.0, ls="--", label="cum. embedding variance")]
    handles += [Line2D([0], [0], color=c, lw=1.5, marker="o", ms=3, label=lbl)
                for lab, c, lbl in ETA_LABELS]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=True,
               fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Where $\\hat{{u}}$ lives in PCA space, and which labels drive each PC "
                 f"— {VARIANT_LABEL[variant]}\n"
                 "û energy above the variance curve ⇒ high-variance-concentrated; "
                 "η² curves show if those PCs are disease (group) vs batch (MG modality) axes",
                 fontweight="bold", fontsize=11.5)
    fig.tight_layout(rect=(0, 0.05, 1, 0.90))
    out = out_dir / f"uhat_pca_spectrum_{variant}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out} (+ .pdf)")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-csv", type=Path,
                    default=REPO / "results" / "uhat_pca_spectrum.csv")
    ap.add_argument("--fig-dir", type=Path,
                    default=REPO / "assets" / "pca_probe")
    args = ap.parse_args()

    all_rows = []
    for variant in ("l_longcombat", "base_longcombat"):
        print(f"[{variant}]")
        all_rows += _plot_variant(variant, args.fig_dir)

    df = pd.DataFrame(all_rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"\nWrote {args.out_csv}: {len(df)} rows")
    # Compact console summary: k for 80% u-hat vs k for 80% variance, per cell.
    print(f"\n{'variant':16s} {'model':10s} {'k(80% u-hat)':>13s} {'k(80% var)':>11s}")
    for variant in ("l_longcombat", "base_longcombat"):
        for m in MODELS:
            sub = df[(df.variant == variant) & (df.model == m)]
            # reconstruct from stored KS rows is coarse; recompute exact k:
            parq = _parquet_path(variant, m)
            cum_e, cum_v = _spectrum(pd.read_parquet(parq))
            ke = int(np.searchsorted(cum_e, 0.8) + 1)
            kv = int(np.searchsorted(cum_v, 0.8) + 1)
            print(f"{variant:16s} {m:10s} {ke:>13d} {kv:>11d}")


if __name__ == "__main__":
    main()
