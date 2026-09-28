#!/usr/bin/env python3
"""
Drift velocity analysis for Mammo-CLIP embeddings.

Implements the analysis plan in PROJECT_NOTES.md:
  1. Malignant vs matched controls (primary)
  2. Biopsy-negative vs matched controls
  3. Malignant vs biopsy-negative

u_hat is computed leave-one-out: for each patient p, the cancer direction is
rebuilt with p's own index embedding excluded from whichever arm (malignant
or control) they contribute to. This avoids the optimism at interval 1 where
the same malignant index embedding that defines u_hat (via its contribution
to the malignant-mean) would otherwise also sit at exam_order=0 in p's own
intervals.

All velocities are expressed per YEAR (dt_years = date_diff / 365.25); the
LME uses time_to_index_years as its time covariate.

For each comparison we run a patient-level Wilcoxon (primary), interval-level
Wilcoxons with Holm-Bonferroni correction, and a mixed-effects model with
time_to_index_years * group interaction. Subgroups by n_priors (2 vs 3+).

Granularities:
  - breast-level : mean embedding over the affected breast only
  - study-level  : mean embedding over all images in a study

Velocity metrics per interval (displacement = newer - older, dt in YEARS):
  - v_signed : dot(displacement, u_hat_p) / dt_years  [units: 1/year; primary]
  - v_L2     : ||displacement||_2 / dt_years          [units: 1/year; direction-agnostic]
  - cos_sim  : cosine_similarity(displacement, u_hat_p) [unitless; directional]

Usage (typically via SLURM):
    python 04_analyze_drift.py \\
        --cases-csv   <matched_cases.csv> \\
        --controls-csv <matched_controls.csv> \\
        --embeddings-dir /path/to/scratch/drift_cohort/embeddings \\
        --output-dir  /path/to/scratch/drift_cohort/drift_analysis
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from utils.cohort_embeddings import (  # noqa: E402
    BIOPSY_NEG,
    CONTROL,
    DAYS_PER_YEAR,
    MALIGNANT,
    build_exam_table,
)


# ----------------------------- pairing ----------------------------- #


def build_pairs(
    df_cases: pd.DataFrame, df_controls: pd.DataFrame
) -> pd.DataFrame:
    """
    Build a canonical case↔control pairing table from the raw step-2 CSVs.

    In step 2, MATCH_ID on a case row stores the control's PATIENTID and vice
    versa — NOT a shared pair key. This function reconstructs each (case, control)
    pair and assigns a pair_id. A case matched to N controls (1:N) appears in
    N rows of the returned frame; a control appears in exactly one.

    Returns
    -------
    DataFrame with columns: pair_id, case_pid, control_pid.
    """
    cases_pairs = (
        df_cases[["PATIENTID", "MATCH_ID"]]
        .dropna()
        .drop_duplicates()
        .rename(columns={"PATIENTID": "case_pid", "MATCH_ID": "control_pid"})
    )
    # Sanity check: every control_pid should appear in df_controls and its
    # MATCH_ID should point back at the case.
    ctrl_links = (
        df_controls[["PATIENTID", "MATCH_ID"]]
        .dropna()
        .drop_duplicates()
        .rename(columns={"PATIENTID": "control_pid", "MATCH_ID": "case_pid"})
    )
    pairs = cases_pairs.merge(ctrl_links, on=["case_pid", "control_pid"], how="inner")

    n_dropped = len(cases_pairs) - len(pairs)
    if n_dropped > 0:
        print(f"Warning: {n_dropped} case→control links had no reciprocal "
              f"control→case row; dropping for safety.")

    pairs = pairs.reset_index(drop=True)
    pairs["pair_id"] = [f"P{i:06d}" for i in range(len(pairs))]
    return pairs[["pair_id", "case_pid", "control_pid"]]


def attach_pair_ids(intervals: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    """
    Add a pair_id column to an interval-level frame.

    Controls appear in exactly one pair; cases may appear in multiple (1:N).
    For cases, we replicate interval rows once per pair so each case-control
    comparison is explicit. Rows that belong to neither arm of a valid pair
    are dropped.
    """
    case_map = pairs[["case_pid", "pair_id"]].rename(columns={"case_pid": "PATIENTID"})
    ctrl_map = pairs[["control_pid", "pair_id"]].rename(columns={"control_pid": "PATIENTID"})

    case_rows = intervals[intervals["group"] != CONTROL].merge(
        case_map, on="PATIENTID", how="inner"
    )
    ctrl_rows = intervals[intervals["group"] == CONTROL].merge(
        ctrl_map, on="PATIENTID", how="inner"
    )
    return pd.concat([case_rows, ctrl_rows], ignore_index=True)


# ----------------------------- u_hat + velocities ----------------------------- #


def build_loo_u_hat_map(
    index_df: pd.DataFrame, case_group: str, control_group: str = CONTROL,
) -> dict[str, np.ndarray]:
    """
    Leave-one-out u_hat: one unit vector per patient.

    For patient p:
      - if p is in the case arm: u_hat uses mean(case \\ {p}) − mean(control)
      - if p is in the control arm: u_hat uses mean(case) − mean(control \\ {p})
      - else (e.g., biopsy_negative in comparison 1): u_hat uses the global
        mean(case) − mean(control) (p contributed nothing, so LOO is a no-op)

    Efficient: the group sums are computed once, then per-patient means are
    obtained by subtracting that patient's index embedding from the sum.

    Parameters
    ----------
    index_df : DataFrame
        is_index rows only, with columns PATIENTID, group, embedding.
    case_group, control_group : str
        Group labels used to compute the direction.

    Returns
    -------
    dict : PATIENTID → (EMB_DIM,) unit-norm np.ndarray
    """
    # Collect per-patient index embeddings (one row per patient expected).
    index_df = index_df.reset_index(drop=True)
    if not index_df["PATIENTID"].is_unique:
        # If this fires, an upstream change is feeding image-level or
        # multi-index rows into the LOO direction builder; the resulting
        # u_hat would silently double-count those patients. Cheap guard
        # against that footgun.
        n_dup = int(index_df["PATIENTID"].duplicated().sum())
        raise ValueError(
            f"build_loo_u_hat_map expects one index row per patient, "
            f"got {n_dup} duplicate PATIENTID(s)."
        )

    case_mask = (index_df["group"] == case_group).to_numpy()
    ctrl_mask = (index_df["group"] == control_group).to_numpy()

    if not case_mask.any() or not ctrl_mask.any():
        raise ValueError(f"LOO u_hat needs both groups present "
                         f"(n_case={case_mask.sum()}, n_ctrl={ctrl_mask.sum()})")

    X = np.stack(index_df["embedding"].values).astype(np.float64)  # (N, D)
    case_sum = X[case_mask].sum(axis=0)
    ctrl_sum = X[ctrl_mask].sum(axis=0)
    n_case = int(case_mask.sum())
    n_ctrl = int(ctrl_mask.sum())

    # Global (non-LOO) direction — used for patients who didn't contribute to
    # either arm (e.g. biopsy_negative patients in a malignant-vs-control fit).
    global_diff = case_sum / n_case - ctrl_sum / n_ctrl
    global_u = global_diff / (np.linalg.norm(global_diff) or 1.0)

    u_map: dict[str, np.ndarray] = {}
    for i, pid in enumerate(index_df["PATIENTID"].values):
        if case_mask[i]:
            diff = (case_sum - X[i]) / (n_case - 1) - ctrl_sum / n_ctrl
        elif ctrl_mask[i]:
            diff = case_sum / n_case - (ctrl_sum - X[i]) / (n_ctrl - 1)
        else:
            u_map[pid] = global_u
            continue
        norm = np.linalg.norm(diff)
        u_map[pid] = diff / norm if norm > 0 else global_u

    # Also store the global direction under a sentinel so callers can fall back.
    u_map["__GLOBAL__"] = global_u
    return u_map


def compute_intervals(
    exam_df: pd.DataFrame,
    u_hat_map: dict[str, np.ndarray],
) -> pd.DataFrame:
    """
    For each consecutive pair of exams per patient, compute signed velocity,
    L2 speed, and cosine similarity with that patient's LOO u_hat.

    displacement = emb(k-1) - emb(k)  (newer minus older)
    dt_years     = date_diff / 365.25 on the newer row (= gap to its newer
                   neighbor on the older row, i.e. the interval length)

    Returns one row per interval:
      PATIENTID, group, n_priors, interval_idx (1 = closest to index),
      exam_order_newer, exam_order_older, dt_days, dt_years,
      v_signed (1/yr), v_L2 (1/yr), cos_sim, time_to_index_years (≤0).
    """
    global_u = u_hat_map.get("__GLOBAL__")

    rows: list[dict] = []
    for pid, g in exam_df.groupby("PATIENTID", sort=False):
        g = g.sort_values("exam_order").reset_index(drop=True)
        u = u_hat_map.get(pid, global_u)
        if u is None:
            continue
        for i in range(len(g) - 1):
            newer = g.iloc[i]
            older = g.iloc[i + 1]
            dt_days = older["date_diff"]
            if pd.isna(dt_days) or dt_days <= 0:
                continue
            dt_days = float(dt_days)
            dt_years = dt_days / DAYS_PER_YEAR
            disp = newer["embedding"] - older["embedding"]
            disp_norm = np.linalg.norm(disp)
            cos_sim = float(disp @ u / disp_norm) if disp_norm > 0 else np.nan
            rows.append({
                "PATIENTID": pid,
                "group": newer["group"],
                "n_priors": int(newer.get("n_priors", 0)),
                "interval_idx": i + 1,
                "exam_order_newer": int(newer["exam_order"]),
                "exam_order_older": int(older["exam_order"]),
                "dt_days": dt_days,
                "dt_years": dt_years,
                "v_signed": float(disp @ u / dt_years),    # per year
                "v_L2": float(disp_norm / dt_years),       # per year
                "cos_sim": cos_sim,
                # time_to_index_years: signed years-to-index (≤0), midpoint of interval.
                # Midpoint days = −(T_newer + T_older)/2 where T_newer = cumulative
                # date_diff up to and including the newer row (= sum of g.iloc[:i+1])
                # and T_older = T_newer + dt_days. So midpoint = −T_newer − dt_days/2.
                "time_to_index_years": -(
                    g.iloc[:i + 1]["date_diff"].fillna(0).sum() + dt_days / 2.0
                ) / DAYS_PER_YEAR,
            })
    return pd.DataFrame(rows)


# ----------------------------- statistical tests ----------------------------- #

def paired_patient_wilcoxon(
    intervals: pd.DataFrame, case_group: str
) -> dict:
    """
    One-sided paired Wilcoxon on patient-mean v_signed, cases > controls.

    Pairing uses pair_id (must be attached upstream via attach_pair_ids).
    For 1:N matching, a case contributes one paired difference per pair_id.
    """
    required = {"pair_id", "PATIENTID", "group", "v_signed"}
    missing = required - set(intervals.columns)
    if missing:
        raise ValueError(f"paired_patient_wilcoxon requires columns {missing}")

    per = (
        intervals.groupby(["pair_id", "PATIENTID", "group"], dropna=False)
        ["v_signed"].mean().reset_index()
    )

    cases = per[per["group"] == case_group][["pair_id", "v_signed"]]
    ctrls = per[per["group"] == CONTROL][["pair_id", "v_signed"]]
    pairs = cases.merge(ctrls, on="pair_id", suffixes=("_case", "_ctrl"), how="inner")

    if len(pairs) < 5:
        return {"n_pairs": len(pairs), "statistic": None, "pvalue": None,
                "median_diff": None, "note": "insufficient pairs"}

    diff = pairs["v_signed_case"].to_numpy() - pairs["v_signed_ctrl"].to_numpy()
    nz = diff != 0
    if nz.sum() < 5:
        return {"n_pairs": len(pairs), "n_nonzero": int(nz.sum()),
                "statistic": None, "pvalue": None, "note": "too many ties"}
    res = stats.wilcoxon(diff[nz], alternative="greater", zero_method="wilcox")
    return {
        "n_pairs": int(len(pairs)),
        "n_nonzero": int(nz.sum()),
        "statistic": float(res.statistic),
        "pvalue": float(res.pvalue),
        "median_diff": float(np.median(diff)),
    }


def interval_wilcoxon(
    intervals: pd.DataFrame, case_group: str
) -> dict:
    """
    Paired Wilcoxon separately per interval_idx (1 = closest to index, 2 = next),
    with Holm-Bonferroni correction. Pairing uses pair_id.
    """
    required = {"pair_id", "PATIENTID", "group", "v_signed", "interval_idx"}
    missing = required - set(intervals.columns)
    if missing:
        raise ValueError(f"interval_wilcoxon requires columns {missing}")

    per = (
        intervals.groupby(["pair_id", "PATIENTID", "group", "interval_idx"],
                          dropna=False)["v_signed"].mean().reset_index()
    )
    results: dict[int, dict] = {}
    raw_pvals: list[tuple[int, float]] = []

    for iv, sub in per.groupby("interval_idx"):
        cases = sub[sub["group"] == case_group][["pair_id", "v_signed"]]
        ctrls = sub[sub["group"] == CONTROL][["pair_id", "v_signed"]]
        pairs = cases.merge(ctrls, on="pair_id", suffixes=("_case", "_ctrl"),
                            how="inner")
        if len(pairs) < 5:
            results[int(iv)] = {"n_pairs": int(len(pairs)), "pvalue": None}
            continue
        diff = pairs["v_signed_case"].to_numpy() - pairs["v_signed_ctrl"].to_numpy()
        nz = diff != 0
        if nz.sum() < 5:
            results[int(iv)] = {"n_pairs": int(len(pairs)), "pvalue": None}
            continue
        res = stats.wilcoxon(diff[nz], alternative="greater", zero_method="wilcox")
        results[int(iv)] = {
            "n_pairs": int(len(pairs)),
            "statistic": float(res.statistic),
            "pvalue_raw": float(res.pvalue),
            "median_diff": float(np.median(diff)),
        }
        raw_pvals.append((int(iv), float(res.pvalue)))

    # Holm-Bonferroni
    if raw_pvals:
        order = sorted(raw_pvals, key=lambda t: t[1])
        m = len(order)
        adj: dict[int, float] = {}
        running_max = 0.0
        for rank, (iv, p) in enumerate(order):
            a = min(1.0, (m - rank) * p)
            running_max = max(running_max, a)
            adj[iv] = running_max
        for iv, p_adj in adj.items():
            results[iv]["pvalue_holm"] = p_adj

    return results


def fit_lme(intervals: pd.DataFrame, case_group: str) -> dict:
    """
    velocity ~ time_to_index_years * group + (1 | pair_id)

    Random intercept by matched pair: case and control in the same pair share
    a within-pair offset on top of the OLS slope. We previously used a nested
    spec with an additional patient-level variance component, but in 1:1
    matching pair_id and PATIENTID are at adjacent levels with one-to-one
    nesting, so the patient variance was empirically driven to the optimizer's
    numerical floor in 3 of 4 cells (see src/diagnostics/lme_spec_check.py).
    The simpler spec retained here gives virtually identical fixed-effect
    estimates (β within ±5%, p within ≤2×) and converges robustly.
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        return {"error": "statsmodels not installed"}

    required = {"pair_id", "PATIENTID", "group", "v_signed", "time_to_index_years"}
    missing = required - set(intervals.columns)
    if missing:
        return {"error": f"missing columns {missing}"}

    sub = intervals[intervals["group"].isin([case_group, CONTROL])].copy()
    sub = sub.dropna(subset=["v_signed", "time_to_index_years", "pair_id", "PATIENTID"])
    if sub.empty or sub["pair_id"].nunique() < 5:
        return {"error": "insufficient data", "n": int(len(sub))}

    sub["group"] = pd.Categorical(sub["group"], categories=[CONTROL, case_group])
    sub["pair_id"] = sub["pair_id"].astype(str)
    sub["PATIENTID"] = sub["PATIENTID"].astype(str)

    try:
        model = smf.mixedlm(
            "v_signed ~ time_to_index_years * C(group)",
            data=sub,
            groups=sub["pair_id"],
            re_formula="~1",
        )
    except Exception as e:
        return {"error": f"model construction failed: {e}", "n": int(len(sub))}

    # Try optimizers in sequence; accept the first that converges, else return
    # the last fit so downstream code still sees params with converged=False.
    # Statsmodels' `converged` flag is unreliable for Nelder-Mead — a fit that
    # produces sensible parameters can still report converged=False — so we
    # surface a non-fatal warning string rather than dropping the result.
    result = None
    last_err = None
    methods_tried: list[str] = []
    for method in ("lbfgs", "bfgs", "nm"):
        methods_tried.append(method)
        try:
            result = model.fit(method=method, reml=True)
            if getattr(result, "converged", False):
                break
        except Exception as e:
            last_err = f"{method}: {e}"
    if result is None:
        return {"error": f"all optimizers failed ({last_err})", "n": int(len(sub))}

    converged = bool(getattr(result, "converged", False))
    out = {
        "n_obs": int(len(sub)),
        "n_pairs": int(sub["pair_id"].nunique()),
        "n_patients": int(sub["PATIENTID"].nunique()),
        "converged": converged,
        "warning": (
            None if converged else
            f"did not converge under any of {methods_tried}; interaction "
            f"estimates are best-available but unverified — interpret with "
            f"the matched bootstrap or the Wilcoxon as the primary test."
        ),
        "params": {k: float(v) for k, v in result.params.items()},
        "pvalues": {k: float(v) for k, v in result.pvalues.items()},
    }
    interaction_keys = [k for k in result.params.index
                        if "time_to_index_years" in k and "group" in k]
    if interaction_keys:
        k = interaction_keys[0]
        out["interaction_term"] = k
        out["interaction_beta"] = float(result.params[k])
        out["interaction_pvalue"] = float(result.pvalues[k])
    return out


# ----------------------------- plots ----------------------------- #

def save_plots(intervals: pd.DataFrame, out_dir: Path, tag: str) -> None:
    """Quick diagnostic plots: velocity & cosine distributions by group/interval."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn not available — skipping plots")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    for metric in ("v_signed", "v_L2", "cos_sim"):
        fig, ax = plt.subplots(figsize=(7, 4))
        sns.boxplot(data=intervals, x="interval_idx", y=metric, hue="group", ax=ax)
        ax.set_title(f"{tag}: {metric} by interval and group")
        ax.axhline(0, color="k", lw=0.5, ls="--")
        fig.tight_layout()
        fig.savefig(out_dir / f"{tag}_{metric}_by_interval.png", dpi=120)
        plt.close(fig)

    # v_signed vs time_to_index scatter with group regression
    fig, ax = plt.subplots(figsize=(7, 4))
    for grp, sub in intervals.groupby("group"):
        ax.scatter(sub["time_to_index_years"], sub["v_signed"], alpha=0.4, s=14, label=grp)
    ax.axhline(0, color="k", lw=0.5, ls="--")
    ax.set_xlabel("time_to_index (years, <=0)")
    ax.set_ylabel("v_signed (1/year)")
    ax.set_title(f"{tag}: drift over time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{tag}_v_signed_vs_time.png", dpi=120)
    plt.close(fig)


# ----------------------------- pipeline ----------------------------- #

def run_comparison(
    exam_df: pd.DataFrame,
    pairs: pd.DataFrame,
    case_group: str,
    tag: str,
    out_dir: Path,
) -> dict:
    """
    Case (malignant or biopsy_negative) vs. matched CONTROL at a fixed
    granularity. Pairing uses pair_id built from the raw matched CSVs.
    Writes per-interval CSV + plots; returns result summary.
    """
    index_df = exam_df[exam_df["is_index"]].copy()
    # Build a leave-one-out u_hat for every patient who will have intervals.
    # u_hat is ALWAYS the malignant-vs-control cancer direction, regardless of
    # which arm is the "case" in this comparison: every arm's drift is measured
    # as projection onto the single fixed cancer axis. (For the secondary
    # biopsy-negative-vs-control comparison this means benign drift is projected
    # onto the malignant direction, not a benign-defined one.) The LOO exclusion
    # only removes a patient's own index embedding when that patient is in the
    # malignant or control group that defines u_hat; biopsy-negative patients
    # never contribute to u_hat, so they receive the (non-LOO) global direction.
    u_hat_map = build_loo_u_hat_map(index_df, case_group=MALIGNANT,
                                    control_group=CONTROL)
    global_u = u_hat_map["__GLOBAL__"]
    # Summary fields to carry into the output
    n_case_idx = int((index_df["group"] == case_group).sum())
    n_ctrl_idx = int((index_df["group"] == CONTROL).sum())

    arm = exam_df[exam_df["group"].isin([case_group, CONTROL])].copy()
    intervals = compute_intervals(arm, u_hat_map)
    intervals["comparison"] = tag

    # Attach pair_id. Only keep pairs where both arms have at least one interval.
    intervals = attach_pair_ids(intervals, pairs)
    valid_pairs = (
        intervals.groupby("pair_id")["group"].nunique()
        .pipe(lambda s: s[s == 2].index)
    )
    intervals = intervals[intervals["pair_id"].isin(valid_pairs)].copy()

    intervals.to_csv(out_dir / f"{tag}_intervals.csv", index=False)

    patient_res = paired_patient_wilcoxon(intervals, case_group=case_group)
    interval_res = interval_wilcoxon(intervals, case_group=case_group)
    lme_res = fit_lme(intervals, case_group=case_group)

    subgroup_res: dict[str, dict] = {}
    pt_n_priors = intervals.groupby("PATIENTID")["n_priors"].first()
    for label, mask in [("n_priors_2", pt_n_priors == 2),
                        ("n_priors_3plus", pt_n_priors >= 3)]:
        pids = set(pt_n_priors[mask].index)
        sub_iv = intervals[intervals["PATIENTID"].isin(pids)]
        if sub_iv["PATIENTID"].nunique() < 5:
            subgroup_res[label] = {"note": "insufficient patients",
                                   "n_patients": int(sub_iv["PATIENTID"].nunique())}
            continue
        subgroup_res[label] = paired_patient_wilcoxon(sub_iv, case_group=case_group)

    save_plots(intervals, out_dir, tag)

    return {
        "tag": tag,
        "case_group": case_group,
        "control_group": CONTROL,
        "u_hat": {
            "name": f"{MALIGNANT}_vs_{CONTROL}_LOO",
            "n_case_idx": n_case_idx,
            "n_control_idx": n_ctrl_idx,
            "scheme": "leave-one-out per patient; direction always malignant vs control",
            "global_norm": float(np.linalg.norm(global_u)),
        },
        "velocity_units": "1/year",
        "n_pairs": int(len(valid_pairs)),
        "patient_wilcoxon": patient_res,
        "interval_wilcoxon": interval_res,
        "lme": lme_res,
        "subgroup_wilcoxon": subgroup_res,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Drift velocity analysis")
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
                        help="Where to write CSVs, plots, and summary.json")
    parser.add_argument("--granularities", nargs="+",
                        default=["breast", "study"],
                        choices=["breast", "study"])
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

    print(f"Loading cases from {args.cases_csv}")
    df_cases = pd.read_csv(args.cases_csv)
    print(f"Loading controls from {args.controls_csv}")
    df_controls = pd.read_csv(args.controls_csv)
    print(f"  cases: {len(df_cases)} rows, {df_cases['PATIENTID'].nunique()} patients")
    print(f"  controls: {len(df_controls)} rows, {df_controls['PATIENTID'].nunique()} patients")

    pairs = build_pairs(df_cases, df_controls)
    print(f"  matched pairs: {len(pairs)} "
          f"({pairs['case_pid'].nunique()} unique cases × "
          f"{pairs['control_pid'].nunique()} unique controls)")
    pairs.to_csv(args.output_dir / "pairs.csv", index=False)

    summary: dict[str, dict] = {}

    for gran in args.granularities:
        print(f"\n=== Granularity: {gran} ===")
        gran_dir = args.output_dir / gran
        gran_dir.mkdir(parents=True, exist_ok=True)

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
        exam.to_parquet(gran_dir / "exam_embeddings.parquet")

        n_by_group = exam.groupby("group")["PATIENTID"].nunique().to_dict()
        print(f"  patients per group: {n_by_group}")

        # Comparison 1: malignant vs controls
        if MALIGNANT in exam["group"].values and CONTROL in exam["group"].values:
            summary[f"{gran}__malignant_vs_control"] = run_comparison(
                exam, pairs, case_group=MALIGNANT,
                tag=f"{gran}_malignant_vs_control", out_dir=gran_dir,
            )

        # Comparison 2: biopsy-negative vs controls
        if BIOPSY_NEG in exam["group"].values and CONTROL in exam["group"].values:
            summary[f"{gran}__biopsy_negative_vs_control"] = run_comparison(
                exam, pairs, case_group=BIOPSY_NEG,
                tag=f"{gran}_biopsy_negative_vs_control", out_dir=gran_dir,
            )

        # Comparison 3: malignant vs biopsy-negative (UNPAIRED).
        # No 1:1 match exists between malignant and benign cases — each has its
        # own matched control — so we use Mann-Whitney on patient-mean v_signed.
        # u_hat uses "malignant - control" (the direction of interest).
        if MALIGNANT in exam["group"].values and BIOPSY_NEG in exam["group"].values \
                and CONTROL in exam["group"].values:
            # LOO u_hat derived from malignant vs control index embeddings.
            # Malignant patients get their own LOO direction; biopsy-negatives
            # did not contribute to u_hat, so they receive the global direction.
            u_hat_map = build_loo_u_hat_map(exam[exam["is_index"]],
                                            case_group=MALIGNANT,
                                            control_group=CONTROL)
            pair = exam[exam["group"].isin([MALIGNANT, BIOPSY_NEG])].copy()
            iv = compute_intervals(pair, u_hat_map)
            iv["comparison"] = f"{gran}_malignant_vs_biopsy_negative"
            iv.to_csv(gran_dir / f"{gran}_malignant_vs_biopsy_negative_intervals.csv",
                      index=False)

            # For this comparison there's no explicit 1:1 pairing; use unpaired
            # Mann-Whitney on patient-mean v_signed.
            per_pt = iv.groupby(["PATIENTID", "group"])["v_signed"].mean().reset_index()
            m = per_pt.loc[per_pt["group"] == MALIGNANT, "v_signed"]
            b = per_pt.loc[per_pt["group"] == BIOPSY_NEG, "v_signed"]
            mw = (stats.mannwhitneyu(m, b, alternative="greater")
                  if len(m) >= 5 and len(b) >= 5 else None)
            summary[f"{gran}__malignant_vs_biopsy_negative"] = {
                "tag": f"{gran}_malignant_vs_biopsy_negative",
                "u_hat": {
                    "name": f"{MALIGNANT}_vs_{CONTROL}_LOO",
                    "scheme": "leave-one-out per malignant patient; biopsy_neg uses global",
                },
                "velocity_units": "1/year",
                "mann_whitney_u": (None if mw is None else
                                   {"statistic": float(mw.statistic), "pvalue": float(mw.pvalue),
                                    "n_malignant": int(len(m)), "n_biopsy_negative": int(len(b))}),
            }
            save_plots(iv, gran_dir, f"{gran}_malignant_vs_biopsy_negative")

    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary written to {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
