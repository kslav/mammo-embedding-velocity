#!/usr/bin/env python3
"""
Multi-panel paired-difference boxplots (Figure-5 style) for the subgroup and
contralateral analyses. Asset-only — these are NOT embedded in the manuscript;
they land under assets/subgroup_figures/<variant>/ (see --variant) with a
clear naming convention (documented in that folder's README.md).

Every panel is the same object as manuscript Figure 5: a box-and-whisker of a
per-pair paired difference by interval, with a dashed zero line and
significance asterisks from the *paired* per-interval Wilcoxon (Holm-corrected)
read from the relevant summary.json — so the stars match the analysis's own
stats, not a test recomputed on the pooled boxes.

Figures produced (per --variant embedding pipeline):
  mgmod_subgroup          2-panel: MG-modality match (depth 2) / mismatch
  race_subgroup           2-panel: race match / mismatch
  contralateral           3-panel: malignant / biopsy_negative / control
  contralateral_by_race   5×3 grid: race bin × class
  contralateral_modality_grid  3×2 grid: class × modality stable/switch
  mgmod_match_by_model    3×1 stack: modality-concordant, one row per model
  (contralat_modality     2-panel stable/switch; not in default set)

--variant selects the embedding pipeline (both longitudinal ComBat on
manufacturer): l_longcombat (L2-normalized, headline) or base_longcombat
(raw features). Run (per model; default Mammo-CLIP, l_longcombat):
    .venv_mammoclip/bin/python src/utils/plot_subgroup_figures.py \
        --variant l_longcombat --model mammoclip
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from cohort_embeddings import VSIGNED_LABEL  # noqa: F401  (kept for parity)
from plot_assets import _p_to_stars, _paired_diffs_for_interval, _holm
from scipy import stats

SCRATCH = Path("/path/to/scratch/drift_cohort")
CASES_CSV = Path("/path/to/data/"
                 "drift_analysis/matched_cases_Density_nPriors_d335_d425_mesh.csv")
CONTROLS_CSV = Path(str(CASES_CSV).replace("matched_cases", "matched_controls"))

# Two embedding pipelines are supported via --variant. Both are longitudinal
# ComBat on manufacturer; they differ only in whether per-image embeddings were
# L2-normalized before mean-pooling. Scratch dir suffixes per (variant, model).
# NB the drift dirs drop the "mammoclip" prefix for MC; the contralateral dirs
# carry an explicit model prefix for all three models.
VARIANTS = {
    "l_longcombat": {
        "label": "L2 + long-ComBat",
        "drift": {"mammoclip": "l_combat_long", "hoppr": "hoppr_l_combat_long",
                  "mii": "mii_l_combat_long"},
        "contralat": {"mammoclip": "mammoclip_l_contralat_combat_long",
                      "hoppr": "hoppr_l_contralat_combat_long",
                      "mii": "mii_l_contralat_combat_long",
                      "biomedclip": "biomedclip_l_contralat_combat_long"},
    },
    "base_longcombat": {
        "label": "long-ComBat (raw features)",
        "drift": {"mammoclip": "combat_long", "hoppr": "hoppr_combat_long",
                  "mii": "mii_combat_long"},
        "contralat": {"mammoclip": "mammoclip_contralat_combat_long",
                      "hoppr": "hoppr_contralat_combat_long",
                      "mii": "mii_contralat_combat_long"},
    },
}
# Active variant — set by main(); default is the L2 headline.
_VARIANT = "l_longcombat"
BOX_COLOR = "#8c1515"
STAR_FS = 13
FOOT_FS = 6.5
FOOT_PAIRED = ("Asterisks: paired per-interval Wilcoxon, Holm-corrected. "
               "****p<10⁻⁴ ***p<10⁻³ **p<0.01 *p<0.05; ns = not significant.")


def _plabel() -> str:
    """Human-readable pipeline label for the active variant."""
    return VARIANTS[_VARIANT]["label"]


def _analysis_dir(model: str, contralat: bool = False) -> Path:
    key = "contralat" if contralat else "drift"
    suf = VARIANTS[_VARIANT][key][model]
    return SCRATCH / f"drift_analysis_augmented_335_425_{suf}"


def _holm_from_summary(summary_path: Path, cell_key: str) -> dict[int, float]:
    """Return {interval_idx: pvalue_holm} for a summary.json cell."""
    if not summary_path.is_file():
        return {}
    s = json.loads(summary_path.read_text())
    cell = s.get(cell_key, {})
    iv = cell.get("interval_wilcoxon", {})
    return {int(k): v.get("pvalue_holm") for k, v in iv.items()}


def _draw_panel(ax, diffs_by_iv: dict[int, np.ndarray],
                holm_p: dict[int, float], ylabel: str, panel_title: str,
                max_interval: int = 4) -> None:
    """One Figure-5-style paired-diff box panel on the given axis."""
    rows = [{"interval_idx": iv, "diff": float(d)}
            for iv in range(1, max_interval + 1)
            for d in diffs_by_iv.get(iv, np.array([]))]
    if not rows:
        ax.set_visible(False)
        return
    long = pd.DataFrame(rows)
    sns.boxplot(data=long, x="interval_idx", y="diff", ax=ax, color=BOX_COLOR,
                showfliers=False, linewidth=1.3, width=0.55)
    ax.axhline(0, color="k", lw=0.9, ls="--")

    present = sorted(long["interval_idx"].unique())
    # Reserve a headroom band for the stars + n-labels (top ~22% of axis).
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, y1 + 0.22 * (y1 - y0))
    trans = ax.get_xaxis_transform()
    for xpos, iv in enumerate(present):
        stars = _p_to_stars(holm_p.get(int(iv)))
        n = len(diffs_by_iv.get(iv, []))
        if stars:
            ax.text(xpos, 0.965, stars, transform=trans, ha="center", va="center",
                    fontsize=STAR_FS, fontweight="bold", color="#222222")
        # n-label sits just under the star, inside the reserved band (never
        # over the whiskers, which live below y-fraction ~0.78 now).
        ax.text(xpos, 0.88, f"n={n}", transform=trans, ha="center", va="center",
                fontsize=7.5, color="#555555")

    ax.set_ylabel(ylabel, fontweight="bold", fontsize=9)
    ax.set_title(panel_title, fontweight="bold", fontsize=11, loc="left")
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontweight("bold")


def _diffs_case_control(intervals_csv: Path,
                        max_interval: int = 4) -> dict[int, np.ndarray]:
    """Per-pair (case - control) v_signed diffs by interval (reuses Fig-5 helper)."""
    df = pd.read_csv(intervals_csv)
    return {iv: _paired_diffs_for_interval(df, iv) for iv in range(1, max_interval + 1)}


def _diffs_affected_contralateral(df: pd.DataFrame,
                                  max_interval: int = 4) -> dict[int, np.ndarray]:
    """Per-patient (affected - contralateral) v_signed diffs by interval."""
    out: dict[int, np.ndarray] = {}
    for iv in range(1, max_interval + 1):
        sub = df[df["interval_idx"] == iv]
        if sub.empty:
            out[iv] = np.array([]); continue
        per = (sub.groupby(["PATIENTID", "side_label"])["v_signed"].mean()
               .reset_index())
        aff = per[per["side_label"] == "affected"][["PATIENTID", "v_signed"]]
        con = per[per["side_label"] == "contralateral"][["PATIENTID", "v_signed"]]
        m = aff.merge(con, on="PATIENTID", suffixes=("_aff", "_con"), how="inner")
        out[iv] = (m["v_signed_aff"] - m["v_signed_con"]).to_numpy()
    return out


def _save(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.text(0.01, 0.005, FOOT_PAIRED, fontsize=FOOT_FS, style="italic", ha="left")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---- Figure A: MG_modality match vs mismatch (case-vs-control) --------------
def fig_mgmod_subgroup(model: str, out_dir: Path, depth: int = 2) -> None:
    base = _analysis_dir(model)
    panels = [("subgroup_mgmod_p2", f"MG modality MATCH (index + priors 1–{depth})"),
              ("subgroup_mgmod_p2_mismatch", "MG modality MISMATCH (discordant, depth 2)")]
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 8.0), sharex=True)
    for ax, (sg, title) in zip(axes, panels):
        d = base / sg
        diffs = _diffs_case_control(d / "breast" / "breast_malignant_vs_control_intervals.csv")
        holm = _holm_from_summary(d / "summary.json", "breast__malignant_vs_control")
        _draw_panel(ax, diffs, holm,
                    r"$v_{\mathrm{malig}}-v_{\mathrm{ctrl}}$ (yr$^{-1}$)", title)
    axes[-1].set_xlabel("Interval index (1 = closest to index exam)", fontweight="bold")
    fig.suptitle(f"MG-modality subgroup — malignant vs matched control ({model}, "
                 f"{_plabel()})", fontweight="bold", fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    _save(fig, out_dir / f"mgmod_subgroup_paired_boxplot_{model}_breast.png")


def fig_mgmod_match_by_model(out_dir: Path, models=("mammoclip", "mii", "hoppr"),
                             depth: int = 2) -> None:
    """3×1 stack of the MG-modality-MATCHED paired boxplot, one row per model.

    This is the confound-controlled version of the primary drift result:
    malignant vs matched control, restricted to case/control pairs concordant
    on MG modality at index + priors 1–2, so per-interval significance cannot
    be inflated by the disease/modality confound. Used as RSNA Panel C and
    reusable in the manuscript."""
    disp = {"mammoclip": "Mammo-CLIP", "mii": "MedImageInsight", "hoppr": "HOPPR"}
    fig, axes = plt.subplots(len(models), 1, figsize=(7.2, 8.4), sharex=True)
    for ax, m in zip(axes, models):
        d = _analysis_dir(m) / "subgroup_mgmod_p2"
        diffs = _diffs_case_control(d / "breast" / "breast_malignant_vs_control_intervals.csv")
        holm = _holm_from_summary(d / "summary.json", "breast__malignant_vs_control")
        _draw_panel(ax, diffs, holm,
                    f"{disp.get(m, m)}\n" r"$v_{\mathrm{malig}}-v_{\mathrm{ctrl}}$ (yr$^{-1}$)",
                    disp.get(m, m))
        ax.set_xlabel("")
    axes[-1].set_xlabel("Interval index (1 = closest to index exam)", fontweight="bold")
    fig.suptitle("Malignant vs matched control within MG-modality-concordant pairs\n"
                 f"(index + priors 1–{depth}; {_plabel()}) — confound-controlled",
                 fontweight="bold", fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    _save(fig, out_dir / "mgmod_match_by_model_paired_boxplot_breast.png")


# ---- Figure C: race match vs mismatch (case-vs-control) ---------------------
def fig_race_subgroup(model: str, out_dir: Path) -> None:
    base = _analysis_dir(model)
    panels = [("subgroup_racematch", "Race MATCH (case & control same 5-bin race)"),
              ("subgroup_racemismatch", "Race MISMATCH (discordant race)")]
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 8.0), sharex=True)
    for ax, (sg, title) in zip(axes, panels):
        d = base / sg
        diffs = _diffs_case_control(d / "breast" / "breast_malignant_vs_control_intervals.csv")
        holm = _holm_from_summary(d / "summary.json", "breast__malignant_vs_control")
        _draw_panel(ax, diffs, holm,
                    r"$v_{\mathrm{malig}}-v_{\mathrm{ctrl}}$ (yr$^{-1}$)", title)
    axes[-1].set_xlabel("Interval index (1 = closest to index exam)", fontweight="bold")
    fig.suptitle(f"Race subgroup — malignant vs matched control ({model}, "
                 f"{_plabel()})", fontweight="bold", fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    _save(fig, out_dir / f"race_subgroup_paired_boxplot_{model}_breast.png")


# ---- Figure B: contralateral, 3-panel by group ------------------------------
def fig_contralateral(model: str, out_dir: Path) -> None:
    base = _analysis_dir(model, contralat=True) / "breast_both_sides"
    summ = _analysis_dir(model, contralat=True) / "summary.json"
    groups = [("malignant", "Malignant"), ("biopsy_negative", "Biopsy-negative"),
              ("control", "Control (seed-42 pseudo-affected null)")]
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 11.0), sharex=True)
    for ax, (g, title) in zip(axes, groups):
        df = pd.read_csv(base / f"contralateral_{g}_intervals.csv")
        diffs = _diffs_affected_contralateral(df)
        holm = _holm_from_summary(summ, g)  # contralat summary keyed by group
        _draw_panel(ax, diffs, holm,
                    r"$v_{\mathrm{aff}}-v_{\mathrm{contra}}$ (yr$^{-1}$)", title)
    axes[-1].set_xlabel("Interval index (1 = closest to index exam)", fontweight="bold")
    fig.suptitle(f"Within-patient contralateral — affected − contralateral "
                 f"({model}, {_plabel()})", fontweight="bold", fontsize=12)
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    _save(fig, out_dir / f"contralateral_paired_boxplot_{model}_breast.png")


# ---- Figure E: contralateral × MG-modality temporal subgroup ----------------
def _diffs_affected_contra_subset(df: pd.DataFrame, keep_pids: set,
                                  max_interval: int = 4) -> dict[int, np.ndarray]:
    """affected - contralateral diffs by interval, restricted to keep_pids."""
    return _diffs_affected_contralateral(
        df[df["PATIENTID"].isin(keep_pids)].copy(), max_interval)


def _holm_per_interval(diffs_by_iv: dict[int, np.ndarray]) -> dict[int, float]:
    """Holm-corrected one-sided paired Wilcoxon (affected > contra) per interval."""
    raw = {}
    for iv, arr in diffs_by_iv.items():
        nz = arr[arr != 0]
        if len(nz) >= 5:
            raw[iv] = stats.wilcoxon(nz, alternative="greater",
                                     zero_method="wilcox").pvalue
    return _holm(raw)


def fig_contralateral_by_modality(model: str, out_dir: Path,
                                  group: str = "malignant") -> None:
    """Within-patient contralateral paired diff, split by whether the patient's
    MG_modality is stable vs switches across their index+priors (depth 2).

    The stable/switch PID partition is loaded from the same pre-filtered CSVs
    used to build results/subgroup_contralat_modality_numbers.csv, so this
    figure's grouping matches that table exactly. Per-interval stars are the
    Holm-corrected paired Wilcoxon computed on each subgroup's diffs (the same
    one-sided affected>contralateral test the analysis uses)."""
    from diagnostics.contralat_modality_subgroup_stats import _load_pid_sets
    base = _analysis_dir(model, contralat=True) / "breast_both_sides"
    df = pd.read_csv(base / f"contralateral_{group}_intervals.csv")
    pid_sets = _load_pid_sets()

    panels = [("modstable", "MG modality STABLE across index + priors 1–2"),
              ("modswitch", "MG modality SWITCH (discordant across intervals, depth 2)")]
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 8.0), sharex=False)
    for ax, (sg, title) in zip(axes, panels):
        diffs = _diffs_affected_contra_subset(df, pid_sets[sg])
        _draw_panel(ax, diffs, _holm_per_interval(diffs),
                    r"$v_{\mathrm{aff}}-v_{\mathrm{contra}}$ (yr$^{-1}$)", title)
        ax.set_xlabel("")  # clear default; only the bottom panel is labeled
    axes[-1].set_xlabel("Interval index (1 = closest to index exam)", fontweight="bold")
    fig.suptitle(f"Contralateral × MG-modality temporal subgroup — {group} arm "
                 f"({model}, {_plabel()})", fontweight="bold", fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 0.98))
    _save(fig, out_dir /
          f"contralateral_by_modality_paired_boxplot_{model}_{group}_breast.png")


def fig_contralateral_modality_grid(model: str, out_dir: Path) -> None:
    """3×2 grid: rows = malignant / biopsy-negative / control, columns =
    MG-modality STABLE / SWITCH. One figure per model. Within-patient
    affected − contralateral paired diff by interval, per-interval
    Holm-corrected paired-Wilcoxon stars. Partition matches
    subgroup_contralat_modality_numbers.csv (see fig_contralateral_by_modality)."""
    from diagnostics.contralat_modality_subgroup_stats import _load_pid_sets
    base = _analysis_dir(model, contralat=True) / "breast_both_sides"
    pid_sets = _load_pid_sets()
    rows = [("malignant", "Malignant"),
            ("biopsy_negative", "Biopsy-negative"),
            ("control", "Control (pseudo-affected null)")]
    cols = [("modstable", "MG modality STABLE"),
            ("modswitch", "MG modality SWITCH")]

    fig, axes = plt.subplots(3, 2, figsize=(13.0, 12.5), sharex=True)
    for r, (grp, grp_lbl) in enumerate(rows):
        df = pd.read_csv(base / f"contralateral_{grp}_intervals.csv")
        for c, (sg, sg_lbl) in enumerate(cols):
            ax = axes[r, c]
            diffs = _diffs_affected_contra_subset(df, pid_sets[sg])
            _draw_panel(ax, diffs, _holm_per_interval(diffs),
                        r"$v_{\mathrm{aff}}-v_{\mathrm{contra}}$ (yr$^{-1}$)"
                        if c == 0 else "",
                        f"{grp_lbl} — {sg_lbl}")
            ax.set_xlabel("")
    for c in range(2):
        axes[-1, c].set_xlabel("Interval index (1 = closest to index exam)",
                               fontweight="bold")
    fig.suptitle(f"Contralateral × MG-modality temporal subgroup — {model} "
                 f"({_plabel()})\nrows: malignant / biopsy-negative / control  ·  "
                 "columns: modality stable / switch",
                 fontweight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    _save(fig, out_dir /
          f"contralateral_modality_grid_{model}_breast.png")


# ---- Figure D: contralateral by race, n-panel -------------------------------
def _patient_race_map() -> pd.Series:
    from diagnostics.confounder_check import bin_race, per_patient_race
    cases = pd.read_csv(CASES_CSV, low_memory=False)
    ctrls = pd.read_csv(CONTROLS_CSV, low_memory=False)
    both = pd.concat([cases, ctrls], ignore_index=True)
    return per_patient_race(both).map(bin_race)


def fig_contralateral_by_race(model: str, out_dir: Path) -> None:
    """race × group grid: one row per 5-bin race, one column per class
    (malignant / biopsy-negative / control). Within-patient affected −
    contralateral paired diff by interval, per-interval Holm-corrected
    paired-Wilcoxon stars (same one-sided affected>contralateral test the
    analysis uses, computed per race×group cell)."""
    base = _analysis_dir(model, contralat=True) / "breast_both_sides"
    race = _patient_race_map()
    order = ["White", "Black/AA", "Asian", "Unknown", "Other/Multi"]
    groups = [("malignant", "Malignant"),
              ("biopsy_negative", "Biopsy-negative"),
              ("control", "Control (null)")]

    # Load each group's intervals CSV once, annotate race.
    dfs = {}
    for grp, _ in groups:
        d = pd.read_csv(base / f"contralateral_{grp}_intervals.csv")
        d["race_bin"] = d["PATIENTID"].map(race)
        dfs[grp] = d
    present = [r for r in order
               if any((dfs[g]["race_bin"] == r).any() for g, _ in groups)]

    nrow, ncol = len(present), len(groups)
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.0 * ncol, 2.9 * nrow),
                             sharex=True, squeeze=False)
    for ri, r in enumerate(present):
        for ci, (grp, grp_lbl) in enumerate(groups):
            ax = axes[ri][ci]
            sub = dfs[grp][dfs[grp]["race_bin"] == r]
            diffs = _diffs_affected_contralateral(sub)
            n_pat = sub["PATIENTID"].nunique()
            _draw_panel(ax, diffs, _holm_per_interval(diffs),
                        r"$v_{\mathrm{aff}}-v_{\mathrm{contra}}$ (yr$^{-1}$)"
                        if ci == 0 else "",
                        f"{r} — {grp_lbl}  (n={n_pat})")
            ax.set_xlabel("")
    for ci in range(ncol):
        axes[-1][ci].set_xlabel("Interval index (1 = closest to index exam)",
                                fontweight="bold")
    fig.suptitle(f"Contralateral by race × class — {model} ({_plabel()})\n"
                 "rows: race (5-bin)  ·  columns: malignant / biopsy-negative / control",
                 fontweight="bold", fontsize=13)
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    _save(fig, out_dir / f"contralateral_by_race_paired_boxplot_{model}_breast.png")


def _boot_median_ci(d: np.ndarray, n_boot: int = 5000, seed: int = 42) -> tuple:
    """Percentile bootstrap 95% CI for the median (seeded → reproducible)."""
    if len(d) < 5:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    meds = np.median(d[idx], axis=1)
    return (float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5)))


def fig_contralateral_modstable_by_race(model: str, out_dir: Path,
                                        group: str = "malignant") -> None:
    """EXPLORATORY: within-patient contralateral drift, restricted to
    MG-modality-STABLE patients, split by 5-bin race (one panel per race).

    This intersects two filters (modality-stable ∩ race), so per-race cells
    are small and underpowered for non-White strata. To avoid over-reading
    the stars, each interval-1 cell is annotated with a bootstrap 95% CI on
    the median (affected−contralateral); stars are still shown for reference
    but the CI is the honest summary. Reported as a robustness supplement,
    not a headline (see PROJECT_NOTES 2026-07-22 race-effect-size caveats)."""
    from diagnostics.contralat_modality_subgroup_stats import _load_pid_sets
    base = _analysis_dir(model, contralat=True) / "breast_both_sides"
    stable = _load_pid_sets()["modstable"]
    race = _patient_race_map()
    df = pd.read_csv(base / f"contralateral_{group}_intervals.csv")
    df = df[df["PATIENTID"].isin(stable)]           # modality-stable only
    df["race_bin"] = df["PATIENTID"].map(race)
    order = ["White", "Black/AA", "Asian", "Unknown", "Other/Multi"]
    present = [r for r in order if (df["race_bin"] == r).any()]

    fig, axes = plt.subplots(len(present), 1, figsize=(7.6, 2.8 * len(present)),
                             sharex=True, squeeze=False)
    for ri, r in enumerate(present):
        ax = axes[ri][0]
        sub = df[df["race_bin"] == r]
        diffs = _diffs_affected_contralateral(sub)
        n_pat = sub["PATIENTID"].nunique()
        _draw_panel(ax, diffs, _holm_per_interval(diffs),
                    r"$v_{\mathrm{aff}}-v_{\mathrm{contra}}$ (yr$^{-1}$)",
                    f"{r}  (n={n_pat} modality-stable patients)")
        ax.set_xlabel("")
        # Interval-1 bootstrap median CI, placed in the top headroom band
        # (left of the star column) so it never collides with whiskers.
        iv1 = diffs.get(1, np.array([]))
        if len(iv1) >= 5:
            lo, hi = _boot_median_ci(iv1)
            ax.text(0.015, 0.90, f"iv1 median 95% CI [{lo:+.3f}, {hi:+.3f}]",
                    transform=ax.transAxes, ha="left", va="center",
                    fontsize=7, color="#333333")
    axes[-1][0].set_xlabel("Interval index (1 = closest to index exam)",
                           fontweight="bold")
    fig.suptitle(f"Contralateral, MG-modality-stable, by race — {group} arm "
                 f"({model}, {_plabel()})\nEXPLORATORY: small non-White strata; "
                 "read the iv1 median 95% CI, not only the stars",
                 fontweight="bold", fontsize=12)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    _save(fig, out_dir /
          f"contralateral_modstable_by_race_{model}_{group}_breast.png")


def main() -> None:
    global _VARIANT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="mammoclip",
                    choices=["mammoclip", "hoppr", "mii"])
    ap.add_argument("--variant", default="l_longcombat", choices=list(VARIANTS),
                    help="Embedding pipeline: l_longcombat (L2-norm + long-ComBat, "
                         "default headline) or base_longcombat (raw features + "
                         "long-ComBat). Figures land in a same-named subfolder.")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).resolve().parents[2] / "assets" / "subgroup_figures",
                    help="Base dir; figures are written to <out-dir>/<variant>/.")
    ap.add_argument("--only", nargs="*",
                    choices=["mgmod", "race", "contralat", "contralat_race",
                             "contralat_modality", "contralat_modality_grid",
                             "mgmod_match_by_model", "contralat_modstable_by_race"],
                    help="Subset of figures to build (default: all). Note "
                         "mgmod_match_by_model spans all 3 models in one figure "
                         "(the --model flag is ignored for it). "
                         "contralat_modstable_by_race is exploratory (not in "
                         "the default set) — build it explicitly.")
    ap.add_argument("--contralat-arm", default="malignant",
                    choices=["malignant", "biopsy_negative", "control"],
                    help="Arm for the by-modality contralateral figure "
                         "(the by-race figure now grids all three arms).")
    args = ap.parse_args()
    _VARIANT = args.variant
    out_dir = args.out_dir / args.variant  # variant-specific subfolder
    # Default set excludes contralat_modality (the per-model 2-panel
    # stable/switch fig): the 3×2 contralat_modality_grid supersedes it.
    # Still buildable explicitly via --only contralat_modality.
    todo = set(args.only) if args.only else {"mgmod", "race", "contralat",
                                             "contralat_race",
                                             "contralat_modality_grid",
                                             "mgmod_match_by_model"}

    print(f"Building subgroup figures [{args.variant}] for {args.model} → {out_dir}")
    if "mgmod_match_by_model" in todo:
        fig_mgmod_match_by_model(out_dir)  # spans all 3 models
    if "mgmod" in todo:
        fig_mgmod_subgroup(args.model, out_dir)
    if "race" in todo:
        fig_race_subgroup(args.model, out_dir)
    if "contralat" in todo:
        fig_contralateral(args.model, out_dir)
    if "contralat_race" in todo:
        fig_contralateral_by_race(args.model, out_dir)
    if "contralat_modality" in todo:
        fig_contralateral_by_modality(args.model, out_dir, group=args.contralat_arm)
    if "contralat_modality_grid" in todo:
        fig_contralateral_modality_grid(args.model, out_dir)
    if "contralat_modstable_by_race" in todo:  # exploratory, opt-in only
        fig_contralateral_modstable_by_race(args.model, out_dir,
                                            group=args.contralat_arm)
    print("Done.")


if __name__ == "__main__":
    main()
