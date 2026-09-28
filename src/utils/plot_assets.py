#!/usr/bin/env python3
"""
Generate the figures embedded in documents/MANUSCRIPT.md and write them to
assets/.

Inputs (must already exist — run 04a and 04b first):
  - scratch/drift_cohort/drift_analysis/{breast,study}/u_hat_projection_hist.png
    (from 04a, under a separate sanity subdir — see PROJECT_NOTES.md)
  - scratch/drift_cohort/drift_analysis/{breast,study}/{tag}_intervals.csv
    (from 04b)
  - scratch/drift_cohort/drift_analysis/pairs.csv

Produces (in assets/):
  - u_hat_projection_hist_{breast,study}.png      (copied from 04a)
  - vsigned_by_interval_{breast,study}.png        (pooled boxplot, malig vs ctrl)
  - vsigned_vs_time_{breast,study}.png            (scatter, copied from 04b)
  - paired_diff_interval1_{breast,study}.png      (NEW: paired differences)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from cohort_embeddings import GROUP_DISPLAY, VSIGNED_LABEL, group_label


PALETTE = {"control": "#7f7f7f", "biopsy_negative": "#1f77b4", "malignant": "#d62728"}
# Display palette keyed by the pretty labels seaborn will see after remapping.
PALETTE_DISPLAY = {group_label(k): v for k, v in PALETTE.items()}


def _paired_diffs_for_interval(
    df: pd.DataFrame, interval_idx: int
) -> np.ndarray:
    """Return the array of (case - control) v_signed differences per pair
    at a given interval_idx. Empty array if no pairs."""
    iv = df[df["interval_idx"] == interval_idx]
    if iv.empty or "pair_id" not in iv.columns:
        return np.array([])
    per = (iv.groupby(["pair_id", "group"], dropna=False)["v_signed"]
           .mean().reset_index())
    cases = per[per["group"] != "control"][["pair_id", "v_signed"]]
    ctrls = per[per["group"] == "control"][["pair_id", "v_signed"]]
    merged = cases.merge(ctrls, on="pair_id", suffixes=("_case", "_ctrl"),
                         how="inner")
    if merged.empty:
        return np.array([])
    return merged["v_signed_case"].to_numpy() - merged["v_signed_ctrl"].to_numpy()


def paired_diff_hist_grid(
    intervals_csv: Path,
    out_path: Path,
    tag: str,
    interval_indices: tuple[int, ...] = (1, 2, 3),
) -> None:
    """
    1×N subplot of paired v_signed differences, one panel per interval_idx.

    Each panel histograms `v_signed_case − v_signed_control` across all pairs
    that are complete at that interval. A vertical dashed line marks zero;
    a solid red line marks the median paired difference.

    Positive shift = case drifts toward u_hat faster than its matched control.
    """
    df = pd.read_csv(intervals_csv)
    n = len(interval_indices)

    # First pass: collect all diffs so we can use a shared x-axis for easy comparison.
    diffs_by_iv: dict[int, np.ndarray] = {
        iv: _paired_diffs_for_interval(df, iv) for iv in interval_indices
    }
    all_vals = np.concatenate([d for d in diffs_by_iv.values() if len(d) > 0])
    if all_vals.size == 0:
        print(f"  skipping {out_path.name}: no paired differences at any interval")
        return
    xlim = (np.quantile(all_vals, 0.005), np.quantile(all_vals, 0.995))
    # Pad a little so rare outliers are visible without dominating
    pad = 0.05 * (xlim[1] - xlim[0])
    xlim = (xlim[0] - pad, xlim[1] + pad)

    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.6), sharex=True, sharey=False)
    if n == 1:
        axes = [axes]

    for ax, iv in zip(axes, interval_indices):
        diff = diffs_by_iv[iv]
        if len(diff) == 0:
            ax.set_title(f"interval {iv} — no pairs")
            ax.set_xlim(xlim)
            continue
        med = float(np.median(diff))
        n_pos = int((diff > 0).sum())
        n_neg = int((diff < 0).sum())
        nz = diff[diff != 0]
        p_str = ""
        if len(nz) >= 5:
            res = stats.wilcoxon(nz, alternative="greater", zero_method="wilcox")
            p_str = f"Wilcoxon p = {res.pvalue:.2g}"

        ax.hist(diff, bins=30, color="#555555", alpha=0.85, edgecolor="white")
        ax.axvline(0, color="k", lw=0.8, ls="--")
        ax.axvline(med, color="#d62728", lw=1.2, ls="-",
                   label=f"Median = {med:+.4f}")
        ax.set_xlim(xlim)
        sub = f"Interval {iv}  ·  $n$ pairs = {len(diff)}  ({n_pos} > 0, {n_neg} < 0)"
        if p_str:
            sub += f"\n{p_str}"
        ax.set_title(sub, fontsize=10)
        ax.set_xlabel("Case − control drift velocity along $\\hat{u}$ (yr$^{-1}$)")
        ax.legend(loc="best", frameon=True, fontsize=8)

    axes[0].set_ylabel("Number of pairs")
    fig.suptitle(f"Paired drift-velocity differences by interval — {tag}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _p_to_stars(p: float) -> str:
    """Significance-star code for a p-value (n.s. shown as 'ns')."""
    if p is None or not np.isfinite(p):
        return ""
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 5e-2:
        return "*"
    return "ns"


def _holm(pvals: dict[int, float]) -> dict[int, float]:
    """Holm-Bonferroni step-down correction over a {key: p} dict."""
    items = [(k, p) for k, p in pvals.items() if p is not None and np.isfinite(p)]
    m = len(items)
    if m == 0:
        return {}
    items.sort(key=lambda kp: kp[1])
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min((m - i) * p, 1.0)
        running = max(running, adj)  # enforce monotonicity
        out[k] = running
    return out


def pooled_vsigned_by_interval(
    intervals_csv: Path,
    out_path: Path,
    tag: str,
    max_interval: int = 4,
) -> None:
    """
    Pooled (not paired) boxplot of v_signed by interval_idx × group, with
    an **unpaired** significance test above each interval.

    The asterisks report a one-sided Mann-Whitney U (malignant > control),
    Holm-corrected across the shown intervals, computed from exactly the
    pooled values plotted — so the annotation is internally consistent with
    the boxes. Note this is deliberately *less* powerful than the
    manuscript's primary paired test (§3.3); the paired view with its own
    stars is `paired_diff_boxplot`. Limits to the first `max_interval`
    intervals since later ones have very few pairs.
    """
    df = pd.read_csv(intervals_csv)
    df = df[df["interval_idx"] <= max_interval].copy()
    if df.empty:
        print(f"  skipping {out_path.name}: no rows ≤ interval {max_interval}")
        return

    # Unpaired Mann-Whitney (malignant > control) per interval, from the
    # pooled per-pair mean v_signed shown in the boxes; then Holm-correct.
    raw_p: dict[int, float] = {}
    for iv in sorted(df["interval_idx"].unique()):
        sub = df[df["interval_idx"] == iv]
        mal = sub.loc[sub["group"] == "malignant", "v_signed"].to_numpy()
        ctl = sub.loc[sub["group"] == "control", "v_signed"].to_numpy()
        if len(mal) >= 5 and len(ctl) >= 5:
            raw_p[int(iv)] = stats.mannwhitneyu(
                mal, ctl, alternative="greater").pvalue
    holm_p = _holm(raw_p)

    # Remap internal group codes -> display labels for legend + palette.
    df["Group"] = df["group"].map(GROUP_DISPLAY).fillna(df["group"])
    hue_order = [GROUP_DISPLAY[g] for g in ("control", "biopsy_negative", "malignant")
                 if GROUP_DISPLAY[g] in set(df["Group"])]

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    sns.boxplot(data=df, x="interval_idx", y="v_signed", hue="Group",
                hue_order=hue_order, ax=ax, palette=PALETTE_DISPLAY,
                showfliers=False, linewidth=1.3)
    ax.axhline(0, color="k", lw=0.7, ls="--")

    # Significance stars in a reserved headroom band (top ~15% of axes) so
    # they never collide with whiskers, title, or legend.
    intervals = sorted(df["interval_idx"].unique())
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, y1 + 0.15 * (y1 - y0))
    trans = ax.get_xaxis_transform()  # x=data, y=axis-fraction
    for xpos, iv in enumerate(intervals):
        stars = _p_to_stars(holm_p.get(int(iv)))
        if not stars:
            continue
        ax.text(xpos, 0.94, stars, transform=trans, ha="center",
                va="center", fontsize=13, fontweight="bold", color="#222222")

    ax.set_xlabel("Interval index (1 = closest to index exam)",
                  fontweight="bold")
    ax.set_ylabel(VSIGNED_LABEL, fontweight="bold")
    ax.set_title(f"Drift velocity by interval\n{tag}", fontweight="bold")
    # Legend below the axes so it never overlaps the star band.
    leg = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16),
                    ncol=len(hue_order), frameon=True, title=None)
    for txt in leg.get_texts():
        txt.set_fontweight("bold")
    ax.tick_params(labelsize=10)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")
    fig.text(0.01, 0.005,
             "Asterisks: unpaired Mann-Whitney U (malignant > control), "
             "Holm-corrected. ****p<10⁻⁴ ***p<10⁻³ **p<0.01 *p<0.05; "
             "ns = not significant.",
             fontsize=6.5, style="italic", ha="left")
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def paired_diff_boxplot(
    intervals_csv: Path,
    out_path: Path,
    tag: str,
    max_interval: int = 4,
    interval_holm_p: dict[int, float] | None = None,
) -> None:
    """
    Paired boxplot: distribution of the per-pair difference
    `v_signed(malignant) − v_signed(control)` by interval, one box each.

    This is the paired analogue of `pooled_vsigned_by_interval`. The y-axis
    IS the quantity the manuscript's primary paired Wilcoxon tests against
    zero (§3.3), so the paired, Holm-corrected per-interval p-values
    (`interval_holm_p`, from summary.json) are the natural annotation and
    match the headline table exactly. A dashed line marks zero (no
    difference); boxes shifted above it mean the malignant side drifts
    faster than its matched control.
    """
    df = pd.read_csv(intervals_csv)
    intervals = list(range(1, max_interval + 1))
    diffs_by_iv = {iv: _paired_diffs_for_interval(df, iv) for iv in intervals}
    rows = [
        {"interval_idx": iv, "diff": float(d)}
        for iv, arr in diffs_by_iv.items() for d in arr
    ]
    if not rows:
        print(f"  skipping {out_path.name}: no paired differences")
        return
    long = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    sns.boxplot(data=long, x="interval_idx", y="diff", ax=ax,
                color="#8c1515", showfliers=False, linewidth=1.3,
                width=0.55)
    ax.axhline(0, color="k", lw=0.9, ls="--")

    present = sorted(long["interval_idx"].unique())
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, y1 + 0.15 * (y1 - y0))
    trans = ax.get_xaxis_transform()
    for xpos, iv in enumerate(present):
        stars = _p_to_stars((interval_holm_p or {}).get(int(iv)))
        if not stars:
            continue
        ax.text(xpos, 0.94, stars, transform=trans, ha="center",
                va="center", fontsize=13, fontweight="bold", color="#222222")

    ax.set_xlabel("Interval index (1 = closest to index exam)",
                  fontweight="bold")
    ax.set_ylabel("Paired difference in drift velocity\n"
                  r"$v_{\mathrm{malignant}}-v_{\mathrm{control}}$ (yr$^{-1}$)",
                  fontweight="bold")
    ax.set_title(f"Paired drift-velocity difference\n{tag}",
                 fontweight="bold")
    ax.tick_params(labelsize=10)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")
    if interval_holm_p:
        fig.text(0.01, 0.005,
                 "Asterisks: paired per-interval Wilcoxon, Holm-corrected "
                 "(malignant vs matched\ncontrol, one-sided). ****p<10⁻⁴ "
                 "***p<10⁻³ **p<0.01 *p<0.05; ns = not significant.",
                 fontsize=6.5, style="italic", ha="left")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate manuscript figures (u_hat hist, vsigned boxplot, paired-diff "
            "grid) from a directory containing {breast,study}/ with intervals CSVs "
            "and u_hat_projection_hist.png. Works against either the raw scratch "
            "output tree (drift_analysis_augmented_335_425/) or the de-identified "
            "results/{model}_{variant}/ tree checked into the repo."
        )
    )
    parser.add_argument("--analysis-dir", type=Path,
                        default=Path(__file__).resolve().parents[2] / "results" / "mammoclip_combat",
                        help=(
                            "Analysis output directory (containing {breast,study}/). "
                            "Default: results/mammoclip_combat/ (the manuscript headline)."
                        ))
    parser.add_argument("--assets-dir", type=Path, required=True,
                        help="Where to write manuscript figures (e.g. assets/hoppr_augmented_combat).")
    args = parser.parse_args()

    args.assets_dir.mkdir(parents=True, exist_ok=True)

    gran_display = {"breast": "Breast level", "study": "Study level"}

    # Load the authoritative per-interval Holm p-values (paired Wilcoxon,
    # malignant vs control) from summary.json so the boxplot asterisks match
    # the manuscript §3.3 tables exactly. Absent file -> no stars drawn.
    summary_path = args.analysis_dir / "summary.json"
    holm_by_gran: dict[str, dict[int, float]] = {}
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        for gran in ("breast", "study"):
            cell = summary.get(f"{gran}__malignant_vs_control", {})
            iv = cell.get("interval_wilcoxon", {})
            holm_by_gran[gran] = {
                int(k): v.get("pvalue_holm") for k, v in iv.items()
            }
    else:
        print(f"  !! no summary.json in {args.analysis_dir}; boxplot stars skipped")

    for gran in ("breast", "study"):
        gran_dir = args.analysis_dir / gran
        uhat_src = gran_dir / "u_hat_projection_hist.png"
        iv_csv = gran_dir / f"{gran}_malignant_vs_control_intervals.csv"
        tag = f"{gran_display.get(gran, gran)}: Malignant vs Control"

        # 1. u_hat projection histogram (copied from 04a output)
        if uhat_src.is_file():
            shutil.copy(uhat_src, args.assets_dir / f"u_hat_projection_hist_{gran}.png")
            print(f"  copied {uhat_src.name} → {args.assets_dir.name}/u_hat_projection_hist_{gran}.png")
        else:
            print(f"  !! missing {uhat_src}")

        # 2. Pooled v_signed boxplot (unpaired Mann-Whitney stars).
        if iv_csv.is_file():
            pooled_vsigned_by_interval(
                iv_csv, args.assets_dir / f"vsigned_by_interval_{gran}.png",
                tag=tag,
            )
            # 3. Paired-difference boxplot (paired Wilcoxon Holm stars, §3.3).
            paired_diff_boxplot(
                iv_csv, args.assets_dir / f"paired_diff_boxplot_{gran}.png",
                tag=tag,
                interval_holm_p=holm_by_gran.get(gran),
            )
            # 4. Paired-difference 1×3 histogram grid for intervals 1, 2, 3.
            # Intervals ≥4 are excluded because n_pairs drops below ~50 and
            # the histograms become uninformative.
            paired_diff_hist_grid(
                iv_csv, args.assets_dir / f"paired_diff_intervals_{gran}.png",
                tag=tag,
                interval_indices=(1, 2, 3),
            )
        else:
            print(f"  !! missing {iv_csv}")

    print(f"\nAssets written to {args.assets_dir}")


if __name__ == "__main__":
    main()
