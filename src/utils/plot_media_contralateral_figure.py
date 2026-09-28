"""Build the MedIA contralateral projection-trajectory figure (Fig. 3).

Companion to plot_media_projection_figure.py, for the within-patient
contralateral design. Grid is 3 rows (models: Mammo-CLIP / HOPPR /
MedImageInsight) x 2 columns (full cohort / modality-stable subset). Each panel
overlays all three groups; within a group the AFFECTED side is a dark solid
line and the CONTRALATERAL side a light dashed line:

    malignant   -> red / light red
    benign      -> blue / light blue    (benign = biopsy-negative)
    control     -> green / light green

y is the projection onto the affected-side cancer direction u_hat (z-scored),
by years before the index exam, with the index exam (which defines u_hat) drawn
as open circles on a dashed segment. The vertical gap between a group's two
lines is the within-patient side asymmetry the paired Wilcoxon tests. The y-axis
range is shared with Fig. 1 (see plot_media_projection_figure.YLIM) so the two
figures are directly comparable.

u_hat = mean(malignant affected index) - mean(control affected index),
L2-normalized (the 04c cancer direction). Headline embeddings: per-image
L2-normalized + longitudinal ComBat, breast_both_sides granularity.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diagnostics.contralat_modality_subgroup_stats import _load_pid_sets  # modstable/switch PIDs
from plot_media_projection_figure import MAX_YEARS, YLIM  # shared x-depth + y-range

SCRATCH = Path("/path/to/scratch/drift_cohort")
CONTRALAT_DIR = {
    "mammoclip": "drift_analysis_augmented_335_425_mammoclip_l_contralat_combat_long",
    "hoppr": "drift_analysis_augmented_335_425_hoppr_l_contralat_combat_long",
    "mii": "drift_analysis_augmented_335_425_mii_l_contralat_combat_long",
    "biomedclip": "drift_analysis_augmented_335_425_biomedclip_l_contralat_combat_long",
}
MODELS = [("mammoclip", "Mammo-CLIP"), ("hoppr", "HOPPR"),
          ("mii", "MedImageInsight"), ("biomedclip", "BiomedCLIP")]

MALIGNANT, BIOPSY_NEG, CONTROL = "malignant", "biopsy_negative", "control"
# (affected dark, contralateral light) per group.
GROUP_STYLE = {
    MALIGNANT:  ("Malignant",     "#d62728", "#f4a9a9"),
    BIOPSY_NEG: ("Benign",        "#1f77b4", "#a9cbe8"),
    CONTROL:    ("Control",       "#2ca02c", "#a9d8a9"),
}


def _uhat_affected(df: pd.DataFrame) -> np.ndarray:
    """Affected-side cancer direction: mean(malignant aff. index) - mean(control aff. index)."""
    idx = df[(df["is_index"]) & (df["side_label"] == "affected")]
    m = np.stack(idx.loc[idx["group"] == MALIGNANT, "embedding"].values).mean(0)
    c = np.stack(idx.loc[idx["group"] == CONTROL, "embedding"].values).mean(0)
    u = m - c
    return u / np.linalg.norm(u)


def _side_trajectory(ax, sub: pd.DataFrame, color: str, dashed: bool) -> None:
    stats = sub.groupby("exam_order")["proj_z"].agg(["mean", "sem"]).reindex(
        range(0, MAX_YEARS + 1))
    x = stats.index.values.astype(float)
    mean, ci = stats["mean"].values, 1.96 * stats["sem"].values
    pri = x >= 1
    ls = "--" if dashed else "-"
    ax.plot(x[pri], mean[pri], ls, color=color, lw=2.0, zorder=3)
    ax.fill_between(x[pri], (mean - ci)[pri], (mean + ci)[pri], color=color,
                    alpha=0.15, lw=0, zorder=1)
    ax.plot([1, 0], [mean[x == 1][0], mean[x == 0][0]], ls, color=color, lw=1.4,
            zorder=2, alpha=0.8)
    ax.plot([0], mean[x == 0], "o", mfc="white", mec=color, mew=1.8, ms=7, zorder=4)


def _panel(ax, df: pd.DataFrame, show_xlabel: bool, show_ylabel: bool) -> None:
    sub = df[df["exam_order"] <= MAX_YEARS]
    for grp, (_, dark, light) in GROUP_STYLE.items():
        g = sub[sub["group"] == grp]
        _side_trajectory(ax, g[g["side_label"] == "affected"], dark, dashed=False)
        _side_trajectory(ax, g[g["side_label"] == "contralateral"], light, dashed=True)
    ax.axhline(0, color="0.6", ls=":", lw=1, zorder=0)
    ax.set_ylim(*YLIM)
    ax.set_xlim(MAX_YEARS + 0.3, -0.4)
    ax.set_xticks(range(0, MAX_YEARS + 1))
    ax.set_xticklabels([f"{i}" if i > 0 else "0\n(index)" for i in range(0, MAX_YEARS + 1)])
    if show_ylabel:
        ax.set_ylabel(r"projection onto $\hat{u}$" + "\n(z-scored)", fontsize=9)
    ax.tick_params(labelsize=8)
    if show_xlabel:
        ax.set_xlabel("years before the index exam", fontsize=8.5, fontweight="bold")


def _legend(fig) -> None:
    """Shared legend: group colors (affected solid / contralateral dashed)."""
    from matplotlib.lines import Line2D
    handles, labels = [], []
    for grp, (name, dark, light) in GROUP_STYLE.items():
        # Cases have a real biopsied side; controls have no biopsy, so their
        # solid line is the case-matched pseudo-side ("matched side").
        solid_side = "matched side" if grp == CONTROL else "biopsied"
        handles.append(Line2D([0], [0], color=dark, lw=2.2, ls="-"))
        labels.append(f"{name} {solid_side}")
        handles.append(Line2D([0], [0], color=light, lw=2.0, ls="--"))
        labels.append(f"{name} contralateral")
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, 1.0))


def build(out_path: Path, also_png: bool = False) -> None:
    stable = _load_pid_sets()["modstable"]
    nrow = len(MODELS)
    # ~2.2in/row keeps the 4-row figure within a page at \linewidth.
    fig, axes = plt.subplots(nrow, 2, figsize=(9.0, 2.2 * nrow), sharex=True, sharey=True)
    for r, (mk, mlbl) in enumerate(MODELS):
        df = pd.read_parquet(
            SCRATCH / CONTRALAT_DIR[mk] / "breast_both_sides" / "exam_embeddings.parquet")
        u = _uhat_affected(df)
        proj = np.stack(df["embedding"].values) @ u
        df = df.copy()
        df["proj_z"] = (proj - proj.mean()) / proj.std()
        cols = [("Full cohort", df),
                ("Modality-stable", df[df["PATIENTID"].isin(stable)])]
        for c, (col_lbl, col_df) in enumerate(cols):
            ax = axes[r, c]
            _panel(ax, col_df, show_xlabel=(r == 2), show_ylabel=(c == 0))
            ax.text(0.015, 0.95, mlbl, transform=ax.transAxes, fontweight="bold",
                    fontsize=10, va="top")
            if r == 0:
                ax.set_title(f"{'A' if c == 0 else 'B'}. {col_lbl}",
                             fontweight="bold", fontsize=10.5, loc="left", pad=10)
    # Reserve a strip at the top for the shared legend, above the column titles.
    fig.tight_layout(h_pad=1.1, rect=[0, 0, 1, 0.95])
    _legend(fig)
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
