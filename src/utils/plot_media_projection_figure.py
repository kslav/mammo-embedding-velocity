"""Build the MedIA main-text projection-trajectory figure (Fig. 3).

Four model rows (Mammo-CLIP / HOPPR / MedImageInsight / BiomedCLIP) x 2 columns
(A: full matched cohort; B: MG-modality-matched pairs). Within each panel, every
exam's breast-level embedding is projected onto the model's cancer direction
u_hat (defined from index embeddings, L2-normalized), z-scored across all exams,
and plotted as a group-mean trajectory vs. years before the index exam, with 95%
CI bands. The index exam (which defines u_hat) is drawn as open circles
connected by a dashed segment.

Column B is the between-patient analog of Fig. 5's modality-stable panel: it
restricts to the depth-2 modality-matched pairs (case and control concordant on
MG modality at the index and first two priors), the same subset behind the "Mm"
column of Table 2. It shows the case/control projection separation persists when
acquisition modality is held fixed across the pair.

Headline embeddings: per-image L2-normalized + longitudinal ComBat, breast
level. Reads the committed exam parquet per model. Output is a vector PDF (plus
optional PNG) suitable for the CAS width.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Depth-2 MG-modality-matched cohort CSVs (case + control PIDs concordant on
# modality at exam_order 0,1,2); the subset behind Table 2's "Mm" column.
DRIFT_CSV_DIR = Path(
    "/path/to/data/FM_Mammo/hoppr_data/breast_cancer/drift_analysis")
MGMOD_CASES = DRIFT_CSV_DIR / "matched_cases_Density_nPriors_d335_d425_mesh_mgmod_p2.csv"
MGMOD_CONTROLS = DRIFT_CSV_DIR / "matched_controls_Density_nPriors_d335_d425_mesh_mgmod_p2.csv"


def _mgmod_matched_pids() -> set:
    """PATIENTIDs in the depth-2 MG-modality-matched cohort (cases + controls)."""
    c = pd.read_csv(MGMOD_CASES, usecols=["PATIENTID"], low_memory=False)
    ct = pd.read_csv(MGMOD_CONTROLS, usecols=["PATIENTID"], low_memory=False)
    return set(c["PATIENTID"]) | set(ct["PATIENTID"])

# Analysis dirs (breast granularity) for the headline L2 + long-ComBat pipeline.
SCRATCH = Path("/path/to/scratch/drift_cohort")
# subdir may be relative to SCRATCH, or an absolute path (BiomedCLIP lives in a
# separate extraction tree).
MODELS = [
    ("mammoclip", "Mammo-CLIP", "drift_analysis_augmented_335_425_l_combat_long"),
    ("hoppr", "HOPPR", "drift_analysis_augmented_335_425_hoppr_l_combat_long"),
    ("mii", "MedImageInsight", "drift_analysis_augmented_335_425_mii_l_combat_long"),
    ("biomedclip", "BiomedCLIP",
     "/path/to/scratch/drift_cohort_biomedclip/drift_analysis_biomedclip_l_combat_long"),
]

MALIGNANT, BIOPSY_NEG, CONTROL = "malignant", "biopsy_negative", "control"
# Colours matched to the reference design.
COLORS = {CONTROL: "#5aa9e6", BIOPSY_NEG: "#f2c14e", MALIGNANT: "#f2764e"}
LABELS = {CONTROL: "Screen-negative control", BIOPSY_NEG: "Benign biopsy",
          MALIGNANT: "Malignant biopsy"}
MAX_YEARS = 4  # exam_order 0..4 (index + 4 priors); deeper priors are underpowered
# Shared y-range across Fig. 1 and Fig. 3 (contralateral), so trajectories are
# directly comparable between the two figures. Covers MII's ~1.6 peak + margin.
YLIM = (-0.45, 1.7)


def _uhat(index_df: pd.DataFrame) -> np.ndarray:
    """Cancer direction from index embeddings: mean(malignant) - mean(control), unit-normed."""
    m = np.stack(index_df.loc[index_df["group"] == MALIGNANT, "embedding"].values).mean(0)
    c = np.stack(index_df.loc[index_df["group"] == CONTROL, "embedding"].values).mean(0)
    u = m - c
    return u / np.linalg.norm(u)


def _panel(ax, df_full: pd.DataFrame, u: np.ndarray, mean: float, std: float,
           show_xlabel: bool, show_ylabel: bool, keep_pids: set | None = None,
           show_legend: bool = False) -> None:
    """Plot one panel. u_hat / z-score stats are computed on the FULL cohort and
    passed in, so column A and column B share an identical axis; ``keep_pids``
    optionally restricts the plotted rows to a subset (modality-matched)."""
    df = df_full if keep_pids is None else df_full[df_full["PATIENTID"].isin(keep_pids)]
    df = df.copy()
    df["proj_z"] = (np.stack(df["embedding"].values) @ u - mean) / std

    for grp in (CONTROL, BIOPSY_NEG, MALIGNANT):
        sub = df[df["group"] == grp]
        stats = sub.groupby("exam_order")["proj_z"].agg(["mean", "sem", "count"])
        stats = stats.reindex(range(0, MAX_YEARS + 1))
        x = stats.index.values.astype(float)  # years before index (0 = index)
        mean, sem = stats["mean"].values, stats["sem"].values
        ci = 1.96 * sem
        # Priors (x>=1): solid line + CI band. Index (x=0): open marker + dashed link.
        pri = x >= 1
        ax.plot(x[pri], mean[pri], "-", color=COLORS[grp], lw=2.4, zorder=3,
                label=LABELS[grp])
        ax.fill_between(x[pri], (mean - ci)[pri], (mean + ci)[pri],
                        color=COLORS[grp], alpha=0.18, lw=0, zorder=1)
        # dashed connector from prior-1 to index, open circle at index
        ax.plot([1, 0], [mean[x == 1][0], mean[x == 0][0]], "--",
                color=COLORS[grp], lw=1.6, zorder=2)
        ax.plot([0], mean[x == 0], "o", mfc="white", mec=COLORS[grp],
                mew=2.2, ms=9, zorder=4)

    ax.axhline(0, color="0.6", ls=":", lw=1, zorder=0)
    ax.set_ylim(*YLIM)
    ax.set_xlim(MAX_YEARS + 0.3, -0.4)  # reversed: 4 on left, index (0) on right
    ax.set_xticks(range(0, MAX_YEARS + 1))
    ax.set_xticklabels([f"{i}" if i > 0 else "0\n(index)" for i in range(0, MAX_YEARS + 1)])
    if show_ylabel:
        ax.set_ylabel(r"projection onto $\hat{u}$" + "\n(z-scored)", fontsize=9)
    ax.tick_params(labelsize=8)
    if show_legend:
        # Sit the legend below the model label (which is at y~0.95) to avoid overlap.
        ax.legend(loc="upper left", bbox_to_anchor=(0.015, 0.82), fontsize=7,
                  frameon=False, handlelength=1.4, labelspacing=0.3)
    if show_xlabel:
        ax.set_xlabel("years before the index (biopsy-trigger) exam", fontsize=9,
                      fontweight="bold")


def build(out_path: Path, also_png: bool = False) -> None:
    n = len(MODELS)
    keep = _mgmod_matched_pids()
    # ~1.9in/row x 2 cols; keeps a 4-row figure within a page at text width.
    fig, axes = plt.subplots(n, 2, figsize=(9.0, 1.95 * n), sharex=True, sharey=True)
    for i, (key, label, subdir) in enumerate(MODELS):
        sd = Path(subdir)
        base = sd if sd.is_absolute() else SCRATCH / sd
        df_full = pd.read_parquet(base / "breast" / "exam_embeddings.parquet")
        df_full = df_full[df_full["exam_order"] <= MAX_YEARS].copy()
        # u_hat + z-score stats fit ONCE on the full cohort, shared by both cols.
        u = _uhat(df_full[df_full["is_index"]])
        proj = np.stack(df_full["embedding"].values) @ u
        m, s = proj.mean(), proj.std()
        for c, kp in enumerate((None, keep)):  # col A: full, col B: modality-matched
            ax = axes[i, c]
            _panel(ax, df_full, u, m, s, show_xlabel=(i == n - 1),
                   show_ylabel=(c == 0), keep_pids=kp, show_legend=(i == 0 and c == 0))
            ax.text(0.015, 0.95, label, transform=ax.transAxes, fontweight="bold",
                    fontsize=10, va="top")
            if i == 0:
                title = "A. Full cohort" if c == 0 else "B. Modality-matched pairs"
                ax.set_title(title, fontweight="bold", fontsize=10.5, loc="left", pad=8)
    fig.tight_layout(h_pad=1.3)
    fig.savefig(out_path, bbox_inches="tight")
    if also_png:
        fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--also-png", action="store_true")
    args = ap.parse_args()
    build(args.out, args.also_png)


if __name__ == "__main__":
    main()
