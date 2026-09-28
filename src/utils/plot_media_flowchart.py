"""Build the MedIA methods figures.

--which flow  -> Figure 1: cohort-flow CONSORT diagram (standalone).
    [ biopsy-outcome within 6 mo of BI-RADS 0 index ]  [ screen-negative
       N = x malignant, y benign                          BI-RADS 1/2 index ]
        filter: >= 2 prior negative screening exams (BI-RADS 1/2), 335-425 d
    [ final pool of cases ]  [ final pool of controls ]
        1:1 matching -> 1,773 matched pairs
    [ between-patient design ]  [ within-patient design ]

--which extraction  -> Figure 2: (A) embedding extraction from a real example
    study's four views (L/R x CC/MLO) -> foundation model -> one vector per
    image -> mean-pool per breast; (B) schematic of the cancer direction u_hat
    (control vs malignant group-mean index embeddings) and drift velocity.

Default intermediate counts are the cohort's real numbers (overridable via CLI).
The example-study view PNGs live in figures/example_views/ (generated once from
a de-identified 4-view FFDM index study).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

C_CASE = "#f2d3c0"    # cases (warm tan/orange)
C_CTRL = "#cfe0f0"    # controls (light blue)
C_FILTER = "white"    # dashed filter boxes
EDGE = "#222222"

VIEW_DIR = Path(__file__).resolve().parents[2] / "documents/media_submission/figures/example_views"

# Real cohort funnel counts (see conversation / curation notebook).
DEFAULT_COUNTS = {"x": "12,683", "y": "29,771", "z": "2,495,666",
                  "xx": "785", "yy": "988", "zz": "134,772"}


def _rbox(ax, cx, cy, w, h, text, fc, fs=10.5, dashed=False, ec=EDGE):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.004,rounding_size=0.012",
                                fc=fc, ec=ec, lw=1.4,
                                linestyle=(":" if dashed else "-"), zorder=2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, zorder=3,
            linespacing=1.35)


def _arrow(ax, x0, y0, x1, y1, lw=1.6):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), mutation_scale=14,
                                 arrowstyle="-|>", color=EDGE, lw=lw,
                                 shrinkA=1, shrinkB=1, zorder=1))


def _line(ax, x0, y0, x1, y1, lw=1.6):
    ax.plot([x0, x1], [y0, y1], color=EDGE, lw=lw, zorder=1, solid_capstyle="round")


# --------------------------------------------------------------------------- #
# Panel A: cohort funnel (y runs 0..1; rows spaced with bottom headroom)
# --------------------------------------------------------------------------- #
def panel_a(ax, counts: dict) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    c = counts

    # Row 1 — two source pools.
    _rbox(ax, 0.27, 0.93, 0.48, 0.13,
          r"$\bf{Patients\ with\ biopsy\ outcome\ within}$" "\n"
          r"$\bf{6\ months\ of\ BI}$-$\bf{RADS\ 0\ index}$" "\n"
          f"N = {c['x']} malignant, N = {c['y']} benign", C_CASE)
    _rbox(ax, 0.77, 0.93, 0.42, 0.13,
          r"$\bf{Patients\ with\ screen}$-$\bf{negative}$" "\n"
          r"$\bf{(BI}$-$\bf{RADS\ 1/2)\ index\ exam}$" "\n"
          f"N = {c['z']} controls", C_CTRL)

    # Row 2 — filter (dashed), fed by both sources.
    _line(ax, 0.27, 0.865, 0.27, 0.815); _line(ax, 0.27, 0.815, 0.5, 0.815)
    _line(ax, 0.77, 0.865, 0.77, 0.815); _line(ax, 0.77, 0.815, 0.5, 0.815)
    _arrow(ax, 0.5, 0.815, 0.5, 0.775)
    _rbox(ax, 0.5, 0.71, 0.92, 0.10,
          r"Filter: $\bf{\geq 2\ prior\ negative\ screening\ exams}$ (BI-RADS 1/2)" "\n"
          "before index, 335–425 days apart", C_FILTER, dashed=True)

    # Row 3 — final pools.
    _line(ax, 0.5, 0.66, 0.5, 0.62); _line(ax, 0.27, 0.62, 0.77, 0.62)
    _arrow(ax, 0.27, 0.62, 0.27, 0.58); _arrow(ax, 0.77, 0.62, 0.77, 0.58)
    _rbox(ax, 0.27, 0.515, 0.48, 0.11,
          r"$\bf{Final\ pool\ of\ cases}$" "\n"
          f"N = {c['xx']} malignant, N = {c['yy']} benign", C_CASE)
    _rbox(ax, 0.77, 0.515, 0.42, 0.11,
          r"$\bf{Final\ pool\ of\ controls}$" "\n"
          f"N = {c['zz']} controls", C_CTRL)

    # Row 4 — matching (dashed).
    _line(ax, 0.27, 0.46, 0.27, 0.42); _line(ax, 0.27, 0.42, 0.5, 0.42)
    _line(ax, 0.77, 0.46, 0.77, 0.42); _line(ax, 0.77, 0.42, 0.5, 0.42)
    _arrow(ax, 0.5, 0.42, 0.5, 0.38)
    _rbox(ax, 0.5, 0.305, 0.92, 0.115,
          r"$\bf{1{:}1\ matching}$ on breast density (exact), $n_{\mathrm{priors}}$ (exact),"
          "\nage ($\\pm$3 y)\n"
          r"$\bf{N = 1{,}773\ matched\ case}$-$\bf{control\ pairs}$",
          C_FILTER, dashed=True)

    # Row 5 — two designs (kept clear of the bottom edge).
    _line(ax, 0.5, 0.2475, 0.5, 0.205); _line(ax, 0.27, 0.205, 0.77, 0.205)
    _arrow(ax, 0.27, 0.205, 0.27, 0.165); _arrow(ax, 0.77, 0.205, 0.77, 0.165)
    _rbox(ax, 0.27, 0.09, 0.48, 0.135,
          r"$\bf{Between}$-$\bf{patient\ design}$" "\n\n"
          "Biopsied vs. matched\ncontrol breast\n"
          "1,773 pairs\n(785 malignant, 988 benign)", C_FILTER, fs=10.0)
    _rbox(ax, 0.77, 0.09, 0.42, 0.135,
          r"$\bf{Within}$-$\bf{patient\ design}$" "\n\n"
          "Biopsied vs. contralateral\nbreast of same patient\n"
          "785 malignant / 988 benign\n/ 1,773 control", C_FILTER, fs=10.0)


# --------------------------------------------------------------------------- #
# Panel B: embedding extraction schematic (uses real example views)
# --------------------------------------------------------------------------- #
def _view(ax, cx, cy, side, label):
    """Draw a real example mammogram view centered at (cx, cy) with a frame."""
    img = mpimg.imread(VIEW_DIR / f"{side}.png")
    h = 0.19
    w = h  # square 480x480 crops
    ax.imshow(img, cmap="gray", extent=(cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2),
              zorder=3, aspect="auto")
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, fill=False,
                           ec="#444444", lw=1.2, zorder=4))
    ax.text(cx, cy - h / 2 - 0.02, label, ha="center", va="top", fontsize=9,
            fontweight="bold", zorder=4)


def _vec(ax, cx, cy, n=7, cell=0.022, color="#4a7fb5"):
    total = n * cell
    for i in range(n):
        ax.add_patch(Rectangle((cx - total / 2 + i * cell, cy - cell / 2),
                               cell, cell, fc=color, ec="white", lw=0.7, zorder=3))


def panel_b(ax) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    views = [("LCC", "L-CC", 0.86), ("LMLO", "L-MLO", 0.62),
             ("RCC", "R-CC", 0.38), ("RMLO", "R-MLO", 0.14)]
    icon_x = 0.13
    for side, lbl, y in views:
        _view(ax, icon_x, y, side, lbl)
        _arrow(ax, icon_x + 0.11, y, 0.37, y, lw=1.3)

    # Foundation model black box.
    ax.add_patch(FancyBboxPatch((0.37, 0.05), 0.17, 0.90,
                                boxstyle="round,pad=0.004,rounding_size=0.02",
                                fc="#111111", ec="#000000", lw=1.3, zorder=2))
    ax.text(0.455, 0.5, "Foundation\nmodel", color="white", ha="center",
            va="center", fontsize=11, fontweight="bold", rotation=90, zorder=3)

    # One vector per image out of the model.
    for side, lbl, y in views:
        _arrow(ax, 0.54, y, 0.61, y, lw=1.3)
        _vec(ax, 0.70, y, color="#4a7fb5")
    ax.text(0.70, 1.0, "per-image\nembeddings", ha="center", va="top", fontsize=9)

    # Mean-pool per breast.
    def _pool(y_top, y_bot, y_out, color, lbl):
        xj = 0.815
        _line(ax, 0.78, y_top, xj, y_top); _line(ax, 0.78, y_bot, xj, y_bot)
        _line(ax, xj, y_top, xj, y_bot)
        _arrow(ax, xj, (y_top + y_bot) / 2, 0.88, y_out, lw=1.3)
        _vec(ax, 0.94, y_out, color=color)
        ax.text(0.94, y_out + 0.06, lbl, ha="center", va="bottom", fontsize=9,
                fontweight="bold")

    _pool(0.86, 0.62, 0.74, "#c0504d", "Left breast")
    _pool(0.38, 0.14, 0.26, "#c0504d", "Right breast")
    ax.text(0.845, 0.5, "mean-pool\nper breast", ha="center", va="center",
            fontsize=9, rotation=90)
    ax.text(0.94, 0.03, "one embedding\nper breast", ha="center", va="top", fontsize=9)

    ax.set_title("A", fontweight="bold", fontsize=15, loc="left", x=-0.02, y=0.99)


# --------------------------------------------------------------------------- #
# Figure 2, Panel B: cancer-direction u_hat schematic (2D, drawn to look 3D)
# --------------------------------------------------------------------------- #
def _gradient_arrow(ax, p0, p1, c0, c1, n=60, lw=3.0, zorder=6):
    """Arrow p0->p1 whose shaft fades color c0->c1, with an arrowhead at p1."""
    from matplotlib.collections import LineCollection
    from matplotlib.colors import to_rgb
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    ts = np.linspace(0, 0.92, n)
    pts = p0 + np.outer(ts, p1 - p0)
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    cols = [tuple((1 - t) * np.array(to_rgb(c0)) + t * np.array(to_rgb(c1)))
            for t in ts[:-1]]
    ax.add_collection(LineCollection(segs, colors=cols, linewidths=lw, zorder=zorder))
    ax.add_patch(FancyArrowPatch(pts[-2], p1, mutation_scale=16, lw=0,
                                 arrowstyle="-|>", color=c1, zorder=zorder + 1))


def _callout(ax, xy, text, color="#222222", fs=9.5):
    ax.annotate(text, xy, ha="center", va="center", fontsize=fs, fontweight="bold",
                color=color, bbox=dict(boxstyle="round,pad=0.35", fc="white",
                                       ec="#888888", lw=1.0), zorder=9)


def panel_uhat(ax) -> None:
    """Embedding-space schematic: control ('normal') and malignant ('cancer')
    group-mean index embeddings, the unit cancer direction u_hat between them,
    and the drift-velocity definition."""
    GREEN, RED, AXcol = "#2ca02c", "#d62728", "#3b6fb0"
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    o = np.array([0.20, 0.42])

    axes_def = {"f_1": o + [0.0, 0.50], "f_2": o + [-0.16, -0.32], "f_3": o + [0.72, 0.0]}
    for lbl, tip in axes_def.items():
        ax.add_patch(FancyArrowPatch(o, tip, mutation_scale=11, lw=1.2,
                                     arrowstyle="-|>", color=AXcol, zorder=2))
        off = (tip - o); off = off / np.linalg.norm(off) * 0.05
        extra = np.array([-0.05, 0.0]) if lbl == "f_1" else np.array([0.0, 0.0])
        ax.text(*(tip + off + extra), f"${lbl}$", color=AXcol, fontsize=12,
                ha="center", va="center")

    cancer_tip = o + np.array([0.30, 0.44])
    normal_tip = o + np.array([0.62, 0.10])
    ax.add_patch(FancyArrowPatch(o, cancer_tip, mutation_scale=16, lw=3.0,
                                 arrowstyle="-|>", color=RED, zorder=5))
    ax.add_patch(FancyArrowPatch(o, normal_tip, mutation_scale=16, lw=3.0,
                                 arrowstyle="-|>", color=GREEN, zorder=5))
    _gradient_arrow(ax, normal_tip, cancer_tip, GREEN, RED)
    mid = (normal_tip + cancer_tip) / 2
    ax.text(*(mid + [0.06, 0.02]), r"$\hat{u}$", fontsize=17, fontweight="bold",
            color="#7a3b3b", ha="left", va="center", zorder=8)

    _callout(ax, (0.17, 0.82), "Malignant\ngroup mean", color=RED)
    _callout(ax, (0.75, 0.94), "Cancer direction $\\hat{u}$")
    _callout(ax, (0.68, 0.28), "Control\ngroup mean", color=GREEN)

    ax.text(0.5, -0.02,
            r"Drift velocity  $v = \langle \Delta\mathrm{embedding},\ \hat{u}\rangle\,/\,\Delta t$",
            transform=ax.transAxes, ha="center", va="top", fontsize=10,
            fontweight="bold")
    ax.text(0.5, -0.12, "(signed movement toward the cancer direction per year)",
            transform=ax.transAxes, ha="center", va="top", fontsize=8.5,
            style="italic", color="#555555")

    ax.set_title("B", fontweight="bold", fontsize=15, loc="left", x=-0.02, y=0.99)


def build_flow(out_path: Path, counts: dict, also_png: bool = False) -> None:
    """Figure 1: standalone cohort-flow funnel."""
    fig, ax = plt.subplots(figsize=(8.2, 8.6))
    panel_a(ax, counts)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    if also_png:
        fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote {out_path}")


def build_extraction(out_path: Path, also_png: bool = False) -> None:
    """Figure 2: (A) embedding extraction + (B) u_hat / drift-velocity schematic."""
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.0, 6.2),
                                   gridspec_kw={"width_ratios": [1.15, 1.0]})
    panel_b(axA)       # extraction schematic (labeled A)
    panel_uhat(axB)    # u_hat schematic (labeled B)
    fig.tight_layout(w_pad=3.0)
    fig.savefig(out_path, bbox_inches="tight")
    if also_png:
        fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--which", choices=["flow", "extraction"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--also-png", action="store_true")
    for k, v in DEFAULT_COUNTS.items():
        ap.add_argument(f"--{k}", default=v)
    args = ap.parse_args()
    if args.which == "flow":
        counts = {k: getattr(args, k) for k in DEFAULT_COUNTS}
        build_flow(args.out, counts, args.also_png)
    else:
        build_extraction(args.out, args.also_png)


if __name__ == "__main__":
    main()
