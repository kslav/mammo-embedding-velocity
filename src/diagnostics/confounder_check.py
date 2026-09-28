"""Race + index-year confounder check for the breast malignant-vs-control
result. Parameterized for any (cohort, model) pair.

Runs three tests:
  (1) Distribution balance — chi-square on race, Mann-Whitney on index_year.
  (2) Stratified paired Wilcoxon — by race bin and by index-year quartile.
  (3) LME interaction β before/after adding race + index_year as covariates.

Writes a markdown notes file suitable for the manuscript supplement.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats


def bin_race(s: str) -> str:
    if pd.isna(s):
        return "Unknown"
    s = s.strip()
    if "," in s:
        return "Other/Multi"
    if s == "White":
        return "White"
    if s == "Black or African American":
        return "Black/AA"
    if s == "Asian":
        return "Asian"
    if s in ("Declined to Specify", "No race assigned"):
        return "Unknown"
    return "Other/Multi"


def per_patient_race(df: pd.DataFrame) -> pd.Series:
    idx = df[df["exam_order"] == 0].drop_duplicates("PATIENTID").set_index("PATIENTID")
    out = idx["RACE"].copy()
    nan_mask = out.isna()
    if nan_mask.any():
        fallback = (
            df.loc[df["PATIENTID"].isin(out[nan_mask].index)]
            .dropna(subset=["RACE"])
            .drop_duplicates("PATIENTID")
            .set_index("PATIENTID")["RACE"]
        )
        out.loc[fallback.index] = fallback
    return out


def per_patient_index_year(df: pd.DataFrame, date_col: str = "STUDYDATE") -> pd.Series:
    idx = df[df["exam_order"] == 0].drop_duplicates("PATIENTID").set_index("PATIENTID")
    parsed = pd.to_datetime(idx[date_col], errors="coerce")
    return parsed.dt.year.rename("index_year")


def patient_mean_v_signed(intervals: pd.DataFrame) -> pd.DataFrame:
    return (
        intervals.groupby(["pair_id", "PATIENTID", "group"], dropna=False)
        ["v_signed"].mean().reset_index()
    )


def stratified_wilcoxon(intervals: pd.DataFrame, stratum_col: str,
                        strata_order: list = None) -> dict:
    per = patient_mean_v_signed(intervals)
    case_stratum = (
        intervals[intervals["group"] == "malignant"]
        .drop_duplicates("PATIENTID")
        .set_index("PATIENTID")[stratum_col]
        .to_dict()
    )
    pair_to_case = (
        intervals[intervals["group"] == "malignant"]
        .drop_duplicates("pair_id")
        .set_index("pair_id")["PATIENTID"]
        .to_dict()
    )
    pair_stratum = {pid: case_stratum.get(cpid)
                    for pid, cpid in pair_to_case.items()}
    cases = per[per["group"] == "malignant"][["pair_id", "v_signed"]]
    ctrls = per[per["group"] == "control"][["pair_id", "v_signed"]]
    pairs = cases.merge(ctrls, on="pair_id", suffixes=("_case", "_ctrl"))
    pairs["stratum"] = pairs["pair_id"].map(pair_stratum)
    pairs["diff"] = pairs["v_signed_case"] - pairs["v_signed_ctrl"]
    out = {}
    if strata_order is None:
        strata_order = sorted(pairs["stratum"].dropna().unique().tolist())
    for s in strata_order:
        sub = pairs[pairs["stratum"] == s]
        if len(sub) < 5:
            out[s] = {"n_pairs": len(sub), "pvalue": None, "median_diff": None}
            continue
        diff = sub["diff"].to_numpy()
        nz = diff != 0
        if nz.sum() < 5:
            out[s] = {"n_pairs": len(sub), "n_nz": int(nz.sum()), "pvalue": None}
            continue
        res = stats.wilcoxon(diff[nz], alternative="greater")
        out[s] = {
            "n_pairs": int(len(sub)),
            "n_nz": int(nz.sum()),
            "median_diff": float(np.median(diff)),
            "pvalue": float(res.pvalue),
        }
    return out


def fit_one(df: pd.DataFrame, formula: str, label: str) -> dict:
    sub = df.dropna(subset=["v_signed", "time_to_index_years", "pair_id",
                            "race", "index_year"]).copy()
    sub["group"] = pd.Categorical(sub["group"], categories=["control", "malignant"])
    sub["pair_id"] = sub["pair_id"].astype(str)
    sub["race"] = sub["race"].astype(str)
    try:
        model = smf.mixedlm(formula, data=sub, groups=sub["pair_id"],
                            re_formula="~1")
        res = None
        for method in ("lbfgs", "bfgs", "nm"):
            try:
                res = model.fit(method=method, reml=True)
                if getattr(res, "converged", False):
                    break
            except Exception:
                continue
        if res is None:
            return {"label": label, "error": "no fit"}
        ix_keys = [k for k in res.params.index
                   if "time_to_index_years" in k and "group" in k]
        if not ix_keys:
            return {"label": label, "error": "no interaction term found"}
        k = ix_keys[0]
        return {
            "label": label,
            "n_obs": int(len(sub)),
            "converged": bool(getattr(res, "converged", False)),
            "interaction_beta": float(res.params[k]),
            "interaction_pvalue": float(res.pvalues[k]),
        }
    except Exception as e:
        return {"label": label, "error": str(e)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases-csv", type=Path, required=True)
    ap.add_argument("--controls-csv", type=Path, required=True)
    ap.add_argument("--analysis-dir", type=Path, required=True,
                    help="dir with breast/breast_malignant_vs_control_intervals.csv")
    ap.add_argument("--out-notes", type=Path, required=True)
    ap.add_argument("--label", type=str, default="(unspecified cohort)")
    args = ap.parse_args()

    print(f"Cohort label: {args.label}")
    print(f"Loading {args.cases_csv}, {args.controls_csv}")
    cases_csv = pd.read_csv(args.cases_csv, low_memory=False)
    ctrls_csv = pd.read_csv(args.controls_csv, low_memory=False)
    print(f"  cases: {cases_csv['PATIENTID'].nunique()} patients")
    print(f"  controls: {ctrls_csv['PATIENTID'].nunique()} patients")

    cases_race = per_patient_race(cases_csv).map(bin_race)
    ctrls_race = per_patient_race(ctrls_csv).map(bin_race)
    print(f"\nCase race bins:    {cases_race.value_counts().to_dict()}")
    print(f"Control race bins: {ctrls_race.value_counts().to_dict()}")

    cases_year = per_patient_index_year(cases_csv)
    ctrls_year = per_patient_index_year(ctrls_csv)
    print(f"\nCase index-year range: {cases_year.min():.0f}-{cases_year.max():.0f}, "
          f"median {cases_year.median():.0f}")
    print(f"Control index-year range: {ctrls_year.min():.0f}-{ctrls_year.max():.0f}, "
          f"median {ctrls_year.median():.0f}")

    intervals_path = args.analysis_dir / "breast" / "breast_malignant_vs_control_intervals.csv"
    print(f"\nLoading intervals: {intervals_path}")
    intervals = pd.read_csv(intervals_path)
    print(f"  {len(intervals)} interval rows; pairs: {intervals['pair_id'].nunique()}")
    intervals = intervals[intervals["group"].isin(["malignant", "control"])].copy()
    intervals["race"] = intervals["PATIENTID"].map(
        pd.concat([cases_race, ctrls_race])
    ).fillna("Unknown")
    intervals["index_year"] = intervals["PATIENTID"].map(
        pd.concat([cases_year, ctrls_year])
    )

    # (1) balance
    pp = intervals.drop_duplicates(["PATIENTID", "group"])
    race_xt = pd.crosstab(pp["race"], pp["group"], dropna=False)
    chi2, p_chi, _, _ = stats.chi2_contingency(race_xt)
    print(f"\n(1) RACE balance:\n{race_xt.to_string()}")
    print(f"χ² test: χ²={chi2:.2f}, p={p_chi:.3g}")

    year_xt = pd.crosstab(pp["index_year"], pp["group"], dropna=False)
    print(f"\nIndex-year balance:\n{year_xt.to_string()}")
    year_pp = pp.dropna(subset=["index_year"])
    mw_yr = stats.mannwhitneyu(
        year_pp.loc[year_pp["group"] == "malignant", "index_year"].to_numpy(),
        year_pp.loc[year_pp["group"] == "control", "index_year"].to_numpy(),
        alternative="two-sided",
    )
    print(f"Mann-Whitney on year: U={mw_yr.statistic:.0f}, p={mw_yr.pvalue:.3g}")

    # (2) stratified Wilcoxon
    race_strata = ["White", "Black/AA", "Asian", "Unknown", "Other/Multi"]
    race_strat = stratified_wilcoxon(intervals, "race", race_strata)
    print("\n(2) STRATIFIED — race")
    print(f"  {'stratum':>14} {'n_pairs':>8} {'med diff':>10} {'p':>10}")
    for s in race_strata:
        r = race_strat[s]
        n = r["n_pairs"]
        md = "  n/a" if r.get("median_diff") is None else f"{r['median_diff']:+7.4f}"
        p = "  n/a" if r["pvalue"] is None else f"{r['pvalue']:.3g}"
        print(f"  {s:>14} {n:>8} {md:>10} {p:>10}")

    case_years = intervals[intervals["group"] == "malignant"]["index_year"].dropna()
    qcuts = case_years.quantile([0, 0.25, 0.50, 0.75, 1.0]).to_list()
    qcuts[0] -= 0.5
    qcuts[-1] += 0.5
    intervals["year_bin"] = pd.cut(
        intervals["index_year"], bins=qcuts,
        labels=[f"Q1({qcuts[0]+0.5:.0f}-{qcuts[1]:.0f})",
                f"Q2({qcuts[1]+1:.0f}-{qcuts[2]:.0f})",
                f"Q3({qcuts[2]+1:.0f}-{qcuts[3]:.0f})",
                f"Q4({qcuts[3]+1:.0f}-{qcuts[-1]-0.5:.0f})"],
        include_lowest=True,
    )
    year_strata = sorted(intervals["year_bin"].dropna().astype(str).unique().tolist())
    year_strat = stratified_wilcoxon(intervals, "year_bin", year_strata)
    print("\n(2) STRATIFIED — index-year quartile")
    print(f"  {'stratum':>20} {'n_pairs':>8} {'med diff':>10} {'p':>10}")
    for s in year_strata:
        r = year_strat[s]
        n = r["n_pairs"]
        md = "  n/a" if r.get("median_diff") is None else f"{r['median_diff']:+7.4f}"
        p = "  n/a" if r["pvalue"] is None else f"{r['pvalue']:.3g}"
        print(f"  {s:>20} {n:>8} {md:>10} {p:>10}")

    # (3) LME
    print("\n(3) LME — interaction β before/after covariate adjustment")
    m_base = fit_one(intervals, "v_signed ~ time_to_index_years * C(group)",
                     "baseline (no covariates)")
    m_race = fit_one(intervals, "v_signed ~ time_to_index_years * C(group) + C(race)",
                     "+ race")
    m_year = fit_one(intervals, "v_signed ~ time_to_index_years * C(group) + index_year",
                     "+ index_year")
    m_both = fit_one(intervals, "v_signed ~ time_to_index_years * C(group) + C(race) + index_year",
                     "+ race + index_year")
    print(f"  {'model':>30} {'β':>10} {'p':>12} {'conv':>6}")
    for r in (m_base, m_race, m_year, m_both):
        if "error" in r:
            print(f"  {r['label']:>30} ERROR: {r['error']}")
            continue
        conv = "✓" if r["converged"] else "✗"
        print(f"  {r['label']:>30} {r['interaction_beta']:>10.4f} "
              f"{r['interaction_pvalue']:>12.3g} {conv:>6}")

    # Markdown notes
    args.out_notes.parent.mkdir(parents=True, exist_ok=True)
    notes = [f"# Confounder check — {args.label}", "",
             "**Source:** generated by `src/diagnostics/confounder_check.py` "
             "(submitted via `slurm_jobs/diagnostics/`).",
             "", "## (1) Distribution balance",
             "", "**Race (per-patient, 5 bins):**",
             "", "```", race_xt.to_string(), "```", "",
             f"χ² test of race × group independence: χ²={chi2:.2f}, p={p_chi:.3g}",
             "", "**Index year:**",
             "",
             f"- Cases: {cases_year.min():.0f}–{cases_year.max():.0f} "
             f"(median {cases_year.median():.0f})",
             f"- Controls: {ctrls_year.min():.0f}–{ctrls_year.max():.0f} "
             f"(median {ctrls_year.median():.0f})",
             f"- Mann-Whitney mal-vs-ctrl on year: U={mw_yr.statistic:.0f}, "
             f"p={mw_yr.pvalue:.3g}",
             "", "## (2) Stratified paired Wilcoxon",
             "", "All p-values are one-sided, case > control.",
             "", "**By race:**",
             "", "| stratum | n_pairs | median Δ | p |",
             "|---|---:|---:|---:|"]
    for s in race_strata:
        r = race_strat[s]
        md = "n/a" if r.get("median_diff") is None else f"{r['median_diff']:+.4f}"
        p = "n/a" if r["pvalue"] is None else f"{r['pvalue']:.3g}"
        notes.append(f"| {s} | {r['n_pairs']} | {md} | {p} |")
    notes += ["", "**By index-year quartile:**",
              "", "| stratum | n_pairs | median Δ | p |",
              "|---|---:|---:|---:|"]
    for s in year_strata:
        r = year_strat[s]
        md = "n/a" if r.get("median_diff") is None else f"{r['median_diff']:+.4f}"
        p = "n/a" if r["pvalue"] is None else f"{r['pvalue']:.3g}"
        notes.append(f"| {s} | {r['n_pairs']} | {md} | {p} |")
    notes += ["", "## (3) LME interaction β with covariate adjustment",
              "", "| model | β (interaction) | p | converged |",
              "|---|---:|---:|:---:|"]
    for r in (m_base, m_race, m_year, m_both):
        if "error" in r:
            notes.append(f"| {r['label']} | ERROR | — | — |")
            continue
        notes.append(
            f"| {r['label']} | {r['interaction_beta']:+.4f} "
            f"| {r['interaction_pvalue']:.3g} "
            f"| {'✓' if r['converged'] else '✗'} |"
        )
    if "interaction_beta" in m_base:
        notes += ["", "**β shift relative to baseline:**", ""]
        for r in (m_race, m_year, m_both):
            if "interaction_beta" in r:
                shift = (r["interaction_beta"] - m_base["interaction_beta"]) / abs(m_base["interaction_beta"]) * 100
                notes.append(f"- {r['label']}: Δβ = {shift:+.1f}%")

    args.out_notes.write_text("\n".join(notes) + "\n")
    print(f"\nWrote {args.out_notes}")


if __name__ == "__main__":
    main()
