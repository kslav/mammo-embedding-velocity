#!/usr/bin/env python3
"""RSNA 2026 late-breaking abstract figure — single composite PDF.

Three panels:
  A  Embedding-space schematic (3-axis): control and malignant group-mean
     centroids, the unit cancer direction u-hat between them, and a sample
     patient trajectory drifting along u-hat over successive exams. Hand-
     drawn (no data) — illustrates the method.
  B  Data-flow schematic: cohort → 1:1 case-control matching (density,
     n_priors, age) → per-image foundation-model embeddings → per-interval
     drift velocity along u-hat. Hand-drawn.
  C/D  Results: 3 rows (Mammo-CLIP / MedImageInsight / HOPPR) × 2 columns,
       L2 + longitudinal-ComBat headline, per-interval Holm-corrected
       paired-Wilcoxon stars.
         C (left)  between-patient v_malignant - v_control, restricted to
                   MG-modality-concordant pairs (subgroup_mgmod_p2).
         D (right) within-patient v_affected - v_contralateral, restricted to
                   MG-modality-stable patients (modstable) — same partition and
                   stats path as src/utils/plot_subgroup_figures.py.

Output: a single vector PDF (embedded fonts) suitable for the RSNA figure
upload. Login-shell-safe (reads committed results CSVs + summary.json).

Run:
    .venv_mammoclip/bin/python src/utils/plot_rsna_abstract_figure.py \
        --out /path/to/scratch/rsna2026_figure.pdf
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from mpl_toolkits.mplot3d import proj3d  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3D

from plot_assets import _p_to_stars, _paired_diffs_for_interval
# Contralateral (within-patient) helpers + the modality-stable PID partition,
# reused so Panel C's right column matches src/utils/plot_subgroup_figures.py.
import plot_subgroup_figures as _psf
from plot_subgroup_figures import (
    _analysis_dir,
    _diffs_affected_contra_subset,
    _holm_per_interval,
)

# Embed TrueType (Type 42), never Type 3 — RSNA/production requirement.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

REPO = Path(__file__).resolve().parents[2]
BOX_COLOR = "#8c1515"
CTRL_COLOR = "#7f7f7f"
MAL_COLOR = "#d62728"
MODELS = [("mammoclip", "Mammo-CLIP"), ("mii", "MedImageInsight"), ("hoppr", "HOPPR")]


class _Arrow3D(FancyArrowPatch):
    """A 3D arrow (matplotlib has no native one)."""
    def __init__(self, xs, ys, zs, *args, **kw):
        super().__init__((0, 0), (0, 0), *args, **kw)
        self._xyz = (xs, ys, zs)

    def do_3d_projection(self, renderer=None):
        (x0, x1), (y0, y1), (z0, z1) = self._xyz
        xs, ys, _ = proj3d.proj_transform((x0, x1), (y0, y1), (z0, z1), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return float(np.min([z0, z1]))


def _gradient_arrow(ax, p0, p1, c0, c1, n=60, lw=3.0, zorder=6):
    """Arrow from p0→p1 whose shaft fades color c0→c1, with an arrowhead at p1."""
    from matplotlib.collections import LineCollection
    from matplotlib.colors import to_rgb
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    ts = np.linspace(0, 0.92, n)  # stop short so the head caps the tip
    pts = p0 + np.outer(ts, p1 - p0)
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    cols = [tuple((1 - t) * np.array(to_rgb(c0)) + t * np.array(to_rgb(c1)))
            for t in ts[:-1]]
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=lw, zorder=zorder))
    ax.add_patch(FancyArrowPatch(pts[-2], p1, mutation_scale=16, lw=0,
                                 arrowstyle="-|>", color=c1, zorder=zorder + 1))


def _callout(ax, xy, text, color="#222222"):
    ax.annotate(text, xy, ha="center", va="center", fontsize=8.2,
                fontweight="bold", color=color,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#888888", lw=1.0))


def panel_a_embedding(ax) -> None:
    """2D schematic (drawn to look 3D): three feature axes from a shared
    origin, red 'cancer location' and green 'normal location' position
    vectors, the green→red cancer-direction u-hat between the tips, and a
    definition of the drift-velocity v below the axes."""
    GREEN, RED, AXcol = "#2ca02c", "#d62728", "#3b6fb0"
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    # NB: no equal-aspect — that letterboxed the panel and wasted the width.
    # Coordinates are tuned to fill the (wider-than-tall) axes box.
    o = np.array([0.20, 0.42])  # origin

    # Three feature axes: f1 straight up, f3 straight right (exactly 90° to
    # each other), f2 the oblique depth axis (down-left).
    axes_def = {"f_1": o + [0.0, 0.52], "f_2": o + [-0.16, -0.34], "f_3": o + [0.74, 0.0]}
    for lbl, tip in axes_def.items():
        ax.add_patch(FancyArrowPatch(o, tip, mutation_scale=11, lw=1.2,
                                     arrowstyle="-|>", color=AXcol, zorder=2))
        off = (tip - o); off = off / np.linalg.norm(off) * 0.045
        extra = np.array([-0.045, 0.0]) if lbl == "f_1" else np.array([0.0, 0.0])
        ax.text(*(tip + off + extra), f"${lbl}$", color=AXcol, fontsize=10,
                ha="center", va="center")

    # Position vectors from origin, well separated in angle.
    cancer_tip = o + np.array([0.30, 0.44])   # red, upper
    normal_tip = o + np.array([0.62, 0.10])   # green, lower-right
    ax.add_patch(FancyArrowPatch(o, cancer_tip, mutation_scale=16, lw=3.0,
                                 arrowstyle="-|>", color=RED, zorder=5))
    ax.add_patch(FancyArrowPatch(o, normal_tip, mutation_scale=16, lw=3.0,
                                 arrowstyle="-|>", color=GREEN, zorder=5))
    # Cancer direction u-hat = green tip → red tip.
    _gradient_arrow(ax, normal_tip, cancer_tip, GREEN, RED)
    mid = (normal_tip + cancer_tip) / 2
    ax.text(*(mid + [0.055, 0.0]), r"$\hat{u}$", fontsize=15, fontweight="bold",
            color="#7a3b3b", ha="left", va="center", zorder=8)

    # Callouts placed clear of the vectors and of u-hat (which sits mid-right).
    _callout(ax, (0.16, 0.80), '"Cancer location"', color=RED)
    _callout(ax, (0.74, 0.93), "Direction from normal\nto cancer  ($\\hat{u}$)")
    _callout(ax, (0.66, 0.30), '"Normal\nlocation"', color=GREEN)

    # Definition of drift velocity v, below the axes.
    ax.text(0.5, -0.03,
            r"Drift velocity  $v = \langle \Delta\mathrm{embedding},\ \hat{u}\rangle\,/\,\Delta t$",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.5,
            fontweight="bold")
    ax.text(0.5, -0.13,
            "(signed movement toward the cancer direction per year)",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.2,
            style="italic", color="#555555")

    ax.set_title("A. Simplified embedding space",
                 fontweight="bold", fontsize=10.5, loc="left", y=1.02)


def _box(ax, xy, w, h, text, fc="#f0f0f0", ec="#333333"):
    ax.add_patch(FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.008,rounding_size=0.02",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
            fontsize=7.6, fontweight="bold")


def _flow_arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), mutation_scale=12,
                                 arrowstyle="-|>", color="#333333", lw=1.5,
                                 clip_on=False))


def panel_b_dataflow(ax, n_models: int = 3) -> None:
    """Case-control matching + embedding workflow (top→bottom flow).

    clip_on=False on every patch so no box is ever cut at the axes edge;
    the four rows are spread over the full panel height with clear gaps for
    the connecting arrows."""
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    W, xc = 0.94, 0.03  # near-full-width boxes

    def _tbox(y, h, text, fc, fs=8.0):
        ax.add_patch(FancyBboxPatch((xc, y), W, h,
                     boxstyle="round,pad=0.01,rounding_size=0.02",
                     fc=fc, ec="#333333", lw=1.2, clip_on=False))
        ax.text(xc + W / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, fontweight="bold")

    # Top: two source boxes side by side (kept where they were).
    for x0, fc, txt in [(0.03, "#fbe4e4", "565 malignant +\n890 biopsy-neg"),
                        (0.52, "#eeeeee", "Screen-negative\ncontrol pool")]:
        ax.add_patch(FancyBboxPatch((x0, 0.86), 0.45, 0.14,
                     boxstyle="round,pad=0.01,rounding_size=0.02",
                     fc=fc, ec="#333333", lw=1.2, clip_on=False))
        ax.text(x0 + 0.225, 0.93, txt, ha="center", va="center",
                fontsize=8.0, fontweight="bold")
    # Remaining three rows: 1:1-match and embeddings boxes taller and pushed
    # further apart; drift box extended down into the whitespace (below y=0,
    # safe with clip_on=False). Match is restricted to MG-modality-concordant
    # pairs (index + priors 1-2) -> 1,455 pairs.
    _tbox(0.56, 0.20, "1:1 match on density · n_priors · age,\n"
          "MG-modality-concordant\n(1,455 pairs)", "#e7eef7", fs=7.6)
    _tbox(0.20, 0.22,
          f"Per-image embeddings\n({n_models} foundation models)\n"
          "L2-norm + long-ComBat on manufacturer", "#eef7ee", fs=7.6)
    _tbox(-0.14, 0.17, "Drift velocity along $\\hat{u}$\nper screening interval", "#f7f0e7")
    # Vertical flow arrows down the centre, sitting in the (larger) gaps.
    _flow_arrow(ax, 0.25, 0.86, 0.45, 0.765)
    _flow_arrow(ax, 0.75, 0.86, 0.55, 0.765)
    _flow_arrow(ax, 0.50, 0.56, 0.50, 0.425)
    _flow_arrow(ax, 0.50, 0.20, 0.50, 0.04)
    ax.set_title("B. Data flow", fontweight="bold",
                 fontsize=10.5, loc="left", y=1.02)


def _draw_c_boxes(ax, diffs_by_iv: dict, holm: dict, ylabel: str,
                  max_interval: int = 4) -> None:
    """Shared Panel-C box drawer: per-interval paired-diff boxplot + Holm stars.

    ``diffs_by_iv`` maps interval_idx -> 1-D array of per-unit paired diffs;
    ``holm`` maps interval_idx -> Holm-corrected p (None if untested)."""
    import seaborn as sns
    rows = [{"interval_idx": iv, "diff": float(d)}
            for iv in range(1, max_interval + 1)
            for d in diffs_by_iv.get(iv, [])]
    long = pd.DataFrame(rows)
    sns.boxplot(data=long, x="interval_idx", y="diff", ax=ax, color=BOX_COLOR,
                showfliers=False, linewidth=1.1, width=0.55)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    # Extra top headroom so the stars + n-labels sit clearly ABOVE the boxes.
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, y1 + 0.42 * (y1 - y0))
    trans = ax.get_xaxis_transform()
    for xpos, iv in enumerate(sorted(long["interval_idx"].unique())):
        s = _p_to_stars(holm.get(int(iv)))
        n_iv = len(diffs_by_iv.get(int(iv), []))
        if s:
            ax.text(xpos, 0.93, s, transform=trans, ha="center", va="center",
                    fontsize=11, fontweight="bold")
        ax.text(xpos, 0.80, f"n={n_iv}", transform=trans, ha="center",
                va="center", fontsize=6.8, color="#555555")
    ax.set_ylabel(ylabel, fontweight="bold", fontsize=8.5)
    ax.tick_params(labelsize=8)
    for lbl in ax.get_yticklabels():
        lbl.set_fontweight("bold")


_SUF = {"mammoclip": "l_combat_long", "hoppr": "hoppr_l_combat_long",
        "mii": "mii_l_combat_long"}
_SCRATCH = Path("/path/to/scratch/drift_cohort")
# Per-model full-cohort between-patient root. BiomedCLIP's full-cohort dir lives
# in a separate extraction tree; its modality-matched subgroup was written into
# our scratch (see _biomedclip_mm_root).
_BP_ROOT = {
    "biomedclip": Path("/path/to/scratch/drift_cohort_biomedclip/"
                       "drift_analysis_biomedclip_l_combat_long"),
}
_BIOMEDCLIP_MM = _SCRATCH / "drift_analysis_augmented_335_425_biomedclip_l_combat_long"


def _between_patient_row(ax, model_key: str, label: str, subdir: str | None,
                         max_interval: int = 4, arm: str = "malignant") -> None:
    """Between-patient case-vs-control per-interval boxplot.

    arm="malignant" (default) or "biopsy_negative" selects the case arm.
    subdir=None  -> FULL matched cohort (no modality restriction).
    subdir="subgroup_mgmod_p2" -> modality-concordant pairs (confound-controlled).
    Boxes are the per-interval paired diffs; stars are Holm-corrected p from the
    matching summary.json.
    """
    if model_key == "biomedclip":
        # full-cohort CSV in Eric's tree; mgmod_p2 subgroup in our scratch.
        root = _BP_ROOT["biomedclip"] if subdir is None else _BIOMEDCLIP_MM
    else:
        root = _SCRATCH / f"drift_analysis_augmented_335_425_{_SUF[model_key]}"
    base = root if subdir is None else root / subdir
    ivcsv = base / "breast" / f"breast_{arm}_vs_control_intervals.csv"
    summ = json.loads((base / "summary.json").read_text())
    holm = {int(k): v.get("pvalue_holm")
            for k, v in summ[f"breast__{arm}_vs_control"]["interval_wilcoxon"].items()}
    df = pd.read_csv(ivcsv)
    diffs = {iv: _paired_diffs_for_interval(df, iv)
             for iv in range(1, max_interval + 1)}
    sym = r"mal" if arm == "malignant" else r"BN"
    _draw_c_boxes(ax, diffs, holm,
                  f"{label}\n" r"$v_{\mathrm{" + sym + r"}}-v_{\mathrm{ctrl}}$", max_interval)


def _panel_c_row(ax, model_key: str, label: str, max_interval: int = 4) -> None:
    # Modality-concordant between-patient result (confound-controlled).
    _between_patient_row(ax, model_key, label, "subgroup_mgmod_p2", max_interval)


def _panel_c_contralat_row(ax, model_key: str, label: str,
                           max_interval: int = 4, arm: str = "malignant") -> None:
    # RIGHT column — within-patient contralateral, restricted to MG-modality-STABLE
    # patients (modality does not switch across index+priors). arm selects the
    # malignant or biopsy_negative case arm. Confound-robust two ways: biopsied vs
    # contralateral shares the same exam/scanner, and modstable drops switchers.
    # Reuses the exact partition + stats path from plot_subgroup_figures.py.
    from diagnostics.contralat_modality_subgroup_stats import _load_pid_sets
    stable = _load_pid_sets()["modstable"]
    base = _analysis_dir(model_key, contralat=True) / "breast_both_sides"
    df = pd.read_csv(base / f"contralateral_{arm}_intervals.csv")
    diffs = _diffs_affected_contra_subset(df, stable, max_interval)
    holm = _holm_per_interval(diffs)
    _draw_c_boxes(ax, diffs, holm,
                  f"{label}\n" r"$v_{\mathrm{biop}}-v_{\mathrm{contra}}$", max_interval)


def build(out_path: Path, models=MODELS, panels_only: bool = False,
          arm: str = "malignant") -> None:
    """Build the composite figure.

    panels_only=True drops the top row (embedding schematic + data flow) and
    emits only the three per-interval boxplot grids (A/B between-patient,
    C within-patient) — the manuscript Figure 4 variant. arm="malignant"
    (Fig. 4) or "biopsy_negative" (supplement) selects the case arm.
    """
    _psf._VARIANT = "l_longcombat"  # ensure the contralateral dirs resolve to L2 headline
    n = len(models)
    if panels_only:
        _build_panels_only(out_path, models, arm=arm)
        return
    fig = plt.figure(figsize=(9.4, 9.4))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.35, 1.55, 1.0],
                  width_ratios=[1.15, 1.0], hspace=0.62, wspace=0.18)
    # Row 0: A (embedding schematic) + B (data flow)
    axA = fig.add_subplot(gs[0, 0])
    panel_a_embedding(axA)
    axB = fig.add_subplot(gs[0, 1])
    panel_b_dataflow(axB, n_models=n)
    # Rows 1-2 span both columns: an n×2 grid — one row per model, two
    # columns (LEFT between-patient concordant; RIGHT within-patient
    # contralateral). Tighter vertical gap when fewer (taller) rows.
    hspace_c = 0.42 if n >= 3 else 0.15
    gsC = gs[1:, :].subgridspec(n, 2, hspace=hspace_c, wspace=0.28)
    for i, (mk, lbl) in enumerate(models):
        axL = fig.add_subplot(gsC[i, 0])
        _panel_c_row(axL, mk, lbl)
        axL.set_xlabel("")
        axR = fig.add_subplot(gsC[i, 1])
        _panel_c_contralat_row(axR, mk, lbl)
        axR.set_xlabel("")
        if i == 0:
            axL.set_title("C. Between-patient: malignant vs matched control",
                          fontweight="bold", fontsize=9.5, loc="left", pad=16)
            axR.set_title("D. Within-patient: affected vs contralateral",
                          fontweight="bold", fontsize=9.5, loc="left", pad=16)
        if i == n - 1:
            for ax in (axL, axR):
                ax.set_xlabel("Interval index (1 = closest to index exam)",
                              fontweight="bold", fontsize=9)
            axL.annotate(
                "Asterisks: paired per-interval Wilcoxon, Holm-corrected.   "
                "****p<10⁻⁴  ***p<10⁻³  **p<0.01  *p<0.05;  ns = not significant.",
                xy=(1.15, -0.42), xycoords="axes fraction", ha="center", va="top",
                fontsize=6.8, style="italic")
    # 300 DPI so the PNG raster is print-quality; the PDF is vector (DPI moot).
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote {out_path}")


def _build_panels_only(out_path: Path, models, arm: str = "malignant") -> None:
    """n model-rows × 3 columns of per-interval boxplots —
    (A) full matched cohort between-patient, (B) modality-matched between-patient,
    (C) within-patient contralateral (modality-stable). arm="malignant" (Fig. 4)
    or "biopsy_negative" (supplement). Columns A→B isolate how much modality
    matching costs the between-patient signal; C is the orthogonal within-patient
    check."""
    n = len(models)
    arm_word = "malignant" if arm == "malignant" else "biopsy-negative"
    fig = plt.figure(figsize=(13.2, 3.0 * n + 0.6))
    gsC = fig.add_gridspec(n, 3, hspace=0.15 if n < 3 else 0.42, wspace=0.30)
    titles = ["A. Between-patient, full cohort",
              "B. Between-patient, modality-matched",
              f"C. Within-patient, {arm_word} vs contralateral"]
    for i, (mk, lbl) in enumerate(models):
        axA = fig.add_subplot(gsC[i, 0])
        _between_patient_row(axA, mk, lbl, None, arm=arm)  # full cohort
        axB = fig.add_subplot(gsC[i, 1])
        _between_patient_row(axB, mk, lbl, "subgroup_mgmod_p2", arm=arm)  # modality-matched
        axC = fig.add_subplot(gsC[i, 2])
        _panel_c_contralat_row(axC, mk, lbl, arm=arm)  # within-patient contralateral
        for ax in (axA, axB, axC):
            ax.set_xlabel("")
        if i == 0:
            for ax, t in zip((axA, axB, axC), titles):
                ax.set_title(t, fontweight="bold", fontsize=9.5, loc="left", pad=16)
        if i == n - 1:
            for ax in (axA, axB, axC):
                ax.set_xlabel("Interval index (1 = closest to index exam)",
                              fontweight="bold", fontsize=9)
            axB.annotate(
                "Asterisks: paired per-interval Wilcoxon, Holm-corrected.   "
                "****p<10⁻⁴  ***p<10⁻³  **p<0.01  *p<0.05;  ns = not significant.",
                xy=(0.5, -0.42), xycoords="axes fraction", ha="center", va="top",
                fontsize=6.8, style="italic")
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    if out_path.suffix == ".pdf":
        fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                    default=Path("/path/to/scratch/rsna2026_figure.pdf"))
    ap.add_argument("--also-png", action="store_true")
    ap.add_argument("--no-hoppr", action="store_true",
                    help="Drop the HOPPR row (2-model figure: Mammo-CLIP + MII).")
    ap.add_argument("--panels-only", action="store_true",
                    help="Manuscript Fig. 2: drop the schematic + data-flow row "
                         "and emit the 3-column per-interval boxplot grid "
                         "(full cohort / modality-matched / contralateral). "
                         "Writes PDF + PNG.")
    ap.add_argument("--with-biomedclip", action="store_true",
                    help="Append BiomedCLIP as a 4th row (modality-confound control).")
    ap.add_argument("--arm", choices=["malignant", "biopsy_negative"],
                    default="malignant",
                    help="Case arm for the panels-only figure: malignant (Fig. 4) "
                         "or biopsy_negative (supplement).")
    args = ap.parse_args()
    models = [m for m in MODELS if m[0] != "hoppr"] if args.no_hoppr else list(MODELS)
    if args.with_biomedclip:
        models = models + [("biomedclip", "BiomedCLIP")]
    build(args.out, models=models, panels_only=args.panels_only, arm=args.arm)
    if args.also_png and not args.panels_only:
        build(args.out.with_suffix(".png"), models=models, arm=args.arm)


if __name__ == "__main__":
    main()
