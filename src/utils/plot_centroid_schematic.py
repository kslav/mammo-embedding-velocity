#!/usr/bin/env python3
"""Standalone 'centroid framework' schematic — the alternative to the
axis-vector schematic in the RSNA figure Panel A.

Shows a simplified 3-feature embedding space with:
  - the control group-mean centroid (grey) and malignant group-mean
    centroid (red),
  - the cancer direction u-hat as the unit vector between them,
  - an example patient's exam trajectory drifting from near the control
    centroid toward the malignant centroid over successive screening
    exams (prior exams -> index), i.e. positive drift velocity along u-hat.

Rendered large and roomy (unlike the cramped inline 3D attempt). Vector
PDF with embedded fonts. Login-shell-safe (no data read; pure schematic).

Run:
    .venv_mammoclip/bin/python src/utils/plot_centroid_schematic.py \
        --out /path/to/scratch/centroid_schematic.pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

CTRL_COLOR = "#7f7f7f"
MAL_COLOR = "#d62728"
PATH_COLOR = "#1f77b4"


class _Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kw):
        super().__init__((0, 0), (0, 0), *args, **kw)
        self._xyz = (xs, ys, zs)

    def do_3d_projection(self, renderer=None):
        (x0, x1), (y0, y1), (z0, z1) = self._xyz
        xs, ys, _ = proj3d.proj_transform((x0, x1), (y0, y1), (z0, z1), self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return float(np.min([z0, z1]))


def _text3d(ax, xyz, s, **kw):
    ax.text(xyz[0], xyz[1], xyz[2], s, **kw)


def build(out_path: Path) -> None:
    fig = plt.figure(figsize=(8.5, 7.0))
    ax = fig.add_subplot(111, projection="3d")

    ctrl = np.array([0.25, 0.28, 0.30])
    mal = np.array([0.80, 0.82, 0.78])

    # Centroids.
    ax.scatter(*ctrl, s=340, color=CTRL_COLOR, edgecolor="k", lw=1.5,
               depthshade=False, zorder=6)
    ax.scatter(*mal, s=340, color=MAL_COLOR, edgecolor="k", lw=1.5,
               depthshade=False, zorder=6)
    _text3d(ax, ctrl + [-0.02, -0.16, -0.14], "Control\ncentroid",
            color=CTRL_COLOR, fontsize=11, fontweight="bold", ha="center")
    _text3d(ax, mal + [0.0, 0.08, 0.10], "Malignant\ncentroid",
            color=MAL_COLOR, fontsize=11, fontweight="bold", ha="center")

    # Cancer direction u-hat between centroids.
    ax.add_artist(_Arrow3D([ctrl[0], mal[0]], [ctrl[1], mal[1]], [ctrl[2], mal[2]],
                           mutation_scale=22, lw=2.6, arrowstyle="-|>",
                           color="#222222", zorder=5))
    mid = (ctrl + mal) / 2
    _text3d(ax, mid + [0.02, -0.05, 0.12], r"$\hat{u}$  (cancer direction)",
            fontsize=13, fontweight="bold", ha="left")

    # Example patient trajectory: starts near control, drifts toward malignant.
    # Offset the whole path BELOW/in-front of the u-hat line so it reads as a
    # separate trajectory, not on top of the direction arrow.
    u = (mal - ctrl)
    offset = np.array([0.02, -0.16, -0.02])  # push the path off the centroid axis
    start = ctrl + offset + np.array([-0.02, 0.0, -0.04])
    fracs = [0.0, 0.30, 0.58, 0.86]
    jitter = [[0, 0, 0], [0.02, -0.02, 0.03], [-0.03, 0.02, -0.02], [0.0, 0.0, 0.02]]
    steps = np.array([start + f * u + np.array(j) for f, j in zip(fracs, jitter)])
    ax.plot(steps[:, 0], steps[:, 1], steps[:, 2], color=PATH_COLOR, lw=2.0,
            ls="--", zorder=4)
    ax.scatter(steps[:, 0], steps[:, 1], steps[:, 2], s=55, color=PATH_COLOR,
               edgecolor="white", lw=0.8, depthshade=False, zorder=7)
    # Arrowheads along the path to convey direction of drift.
    for a, b in zip(steps[:-1], steps[1:]):
        ax.add_artist(_Arrow3D([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                               mutation_scale=13, lw=0, arrowstyle="-|>",
                               color=PATH_COLOR, zorder=7))
    # Single trajectory label in open space (below the path); direction is
    # conveyed by the arrowheads, so per-point exam labels are omitted to
    # avoid crowding the control centroid.
    _text3d(ax, steps[2] + [0.10, -0.02, -0.16],
            "example patient trajectory\n(prior exams → index)",
            color=PATH_COLOR, fontsize=10, fontweight="bold", ha="center")

    ax.set_xlabel("feature 1", fontsize=9, labelpad=-6)
    ax.set_ylabel("feature 2", fontsize=9, labelpad=-6)
    ax.set_zlabel("feature 3", fontsize=9, labelpad=-6)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
    ax.view_init(elev=20, azim=-60)
    ax.set_title("Drift toward the cancer direction in embedding space",
                 fontweight="bold", fontsize=13, y=0.97)
    fig.text(0.5, 0.03,
             r"Drift velocity $v = \langle \Delta\mathrm{embedding},\ \hat{u}\rangle / \Delta t$ "
             "— tested faster in cancer cases than matched controls",
             ha="center", fontsize=9.5, style="italic")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path,
                    default=Path("/path/to/scratch/centroid_schematic.pdf"))
    ap.add_argument("--also-png", action="store_true")
    args = ap.parse_args()
    build(args.out)
    if args.also_png:
        build(args.out.with_suffix(".png"))


if __name__ == "__main__":
    main()
