"""Diagnostics for d365_425 vs d365_800 cohort relaxation."""
from pathlib import Path
import pandas as pd
import numpy as np

DATA = Path("/path/to/data/FM_Mammo/hoppr_data/breast_cancer/drift_analysis")

cases_tight = pd.read_parquet(DATA / "cases_longitudinal_2priorsminimum.parquet")
ctrls_tight = pd.read_parquet(DATA / "controls_longitudinal_2priorsminimum.parquet")
cases_relaxed = pd.read_parquet(DATA / "cases_longitudinal_2priorsminimum_d365_d800.parquet")
ctrls_relaxed = pd.read_parquet(DATA / "controls_longitudinal_2priorsminimum_d365_d800.parquet")


def patient_summary(df, name):
    p = df["PATIENTID"].nunique()
    s = df["STUDYID"].nunique()
    print(f"  {name}: {p:,} patients, {s:,} studies")
    return p


def case_breakdown(df):
    if "Cancer_Description" not in df.columns:
        return None, None
    pid_groups = df.groupby("PATIENTID")["Cancer_Description"].agg(
        lambda s: "malignant" if (s == "path_positive").any() else "benign"
    )
    n_mal = (pid_groups == "malignant").sum()
    n_ben = (pid_groups == "benign").sum()
    return n_mal, n_ben


def n_priors_dist(df, name):
    """Per-patient n_priors (number of non-index exams)."""
    n_priors = df[~df["is_index"]].groupby("PATIENTID").size()
    # Patients with 0 priors aren't represented in the groupby; add them.
    all_pids = set(df["PATIENTID"].unique())
    missing = all_pids - set(n_priors.index)
    if missing:
        n_priors = pd.concat([n_priors, pd.Series(0, index=list(missing))])
    return n_priors.value_counts().sort_index()


def date_diff_by_prior(df, name):
    """date_diff distribution grouped by exam_order (1, 2, 3, ...)."""
    priors = df[~df["is_index"]].copy()
    out = priors.groupby("exam_order")["date_diff"].agg(
        ["count", "median", lambda s: s.quantile(0.25),
         lambda s: s.quantile(0.75), "max"]
    )
    out.columns = ["n", "median", "q25", "q75", "max"]
    return out


print("=" * 70)
print("CASES")
print("=" * 70)
print("\nTight (365–425):")
patient_summary(cases_tight, "cases")
n_mal_t, n_ben_t = case_breakdown(cases_tight)
print(f"    malignant: {n_mal_t}, benign: {n_ben_t}")

print("\nRelaxed (365–800):")
patient_summary(cases_relaxed, "cases")
n_mal_r, n_ben_r = case_breakdown(cases_relaxed)
print(f"    malignant: {n_mal_r}, benign: {n_ben_r}")

print(f"\nGain: malignant +{n_mal_r - n_mal_t} ({100*(n_mal_r-n_mal_t)/max(1,n_mal_t):.0f}%), "
      f"benign +{n_ben_r - n_ben_t} ({100*(n_ben_r-n_ben_t)/max(1,n_ben_t):.0f}%)")

print("\nn_priors distribution (per-patient counts):")
print("  tight:  ", dict(n_priors_dist(cases_tight, "tight")))
print("  relaxed:", dict(n_priors_dist(cases_relaxed, "relaxed")))

print("\ndate_diff by prior (relaxed cases):")
print(date_diff_by_prior(cases_relaxed, "relaxed").to_string())

print("\n" + "=" * 70)
print("CONTROLS")
print("=" * 70)
print("\nTight (365–425):")
patient_summary(ctrls_tight, "controls")
print("\nRelaxed (365–800):")
patient_summary(ctrls_relaxed, "controls")

print("\nn_priors distribution (per-patient counts):")
nptight = n_priors_dist(ctrls_tight, "tight")
nprelaxed = n_priors_dist(ctrls_relaxed, "relaxed")
print("  tight:  ", dict(nptight))
print("  relaxed:", dict(nprelaxed))

print("\ndate_diff by prior (relaxed controls):")
print(date_diff_by_prior(ctrls_relaxed, "relaxed").to_string())

# Pool ratio: at each (n_priors, density, age-bucket) stratum, controls/case.
# Use n_priors exact, density exact, age ±3 y as the matching criteria
# (same as today's match_cases_controls).

def parse_age(age_str):
    if isinstance(age_str, float):
        return age_str
    if not isinstance(age_str, str):
        return np.nan
    try:
        unit = age_str.strip('"')[-1].upper()
        value = int(age_str.strip('"')[:-1])
        if unit == 'Y':
            return float(value)
        elif unit == 'M':
            return value / 12
        elif unit == 'W':
            return value / 52
        elif unit == 'D':
            return value / 365
        else:
            return np.nan
    except Exception:
        return np.nan


def per_patient_index(df, with_n_priors=True):
    """One row per patient at exam_order==0 with parsed age."""
    if with_n_priors:
        npr = df[~df["is_index"]].groupby("PATIENTID").size().rename("n_priors")
    idx = (df[df["exam_order"] == 0]
           .drop_duplicates("PATIENTID")
           .copy())
    if with_n_priors:
        idx = idx.merge(npr, on="PATIENTID", how="left").fillna({"n_priors": 0})
        idx["n_priors"] = idx["n_priors"].astype(int)
    idx["age"] = idx["PATIENTAGE"].apply(parse_age)
    return idx


def pool_ratio(cases, ctrls, age_tol=3, label=""):
    cases_idx = per_patient_index(cases).dropna(subset=["age", "Breast Density"])
    ctrls_idx = per_patient_index(ctrls).dropna(subset=["age", "Breast Density"])
    print(f"\n  {label}: cases with full match data: {len(cases_idx)}, "
          f"controls with full match data: {len(ctrls_idx)}")
    # For each case, count eligible controls (density exact, n_priors exact, age ±3)
    pool_counts = []
    for _, c in cases_idx.iterrows():
        elig = ctrls_idx[
            (ctrls_idx["Breast Density"] == c["Breast Density"]) &
            (ctrls_idx["n_priors"] == c["n_priors"]) &
            ((ctrls_idx["age"] - c["age"]).abs() <= age_tol)
        ]
        pool_counts.append(len(elig))
    pc = pd.Series(pool_counts)
    print(f"    eligible controls per case: median={pc.median():.0f}, "
          f"q25={pc.quantile(0.25):.0f}, q75={pc.quantile(0.75):.0f}, max={pc.max()}")
    print(f"    cases with 0 eligible controls: {(pc == 0).sum()} / {len(pc)}")
    # Break down by n_priors
    cases_idx = cases_idx.copy()
    cases_idx["pool"] = pool_counts
    by_np = cases_idx.groupby("n_priors")["pool"].agg(["count", "median", "min", "max"])
    print("    pool by n_priors:")
    for _, row in by_np.reset_index().iterrows():
        print(f"      n_priors={int(row['n_priors'])}: "
              f"n_cases={int(row['count'])}, median pool={row['median']:.0f}, "
              f"min={int(row['min'])}, max={int(row['max'])}")
    return pc


print("\n" + "=" * 70)
print("POOL RATIO (controls per case, density × n_priors × age±3)")
print("=" * 70)
pool_ratio(cases_tight, ctrls_tight, label="tight (365–425)")
pool_ratio(cases_relaxed, ctrls_relaxed, label="relaxed (365–800)")

# dt-vector compatible pool: simulate per-prior ±180 d filter on top of base matching.
print("\n" + "=" * 70)
print("DT-VECTOR MATCHING SIMULATION (relaxed cohort)")
print("=" * 70)
print("Eligible controls per case under: density exact, n_priors exact, age ±3 y, "
      "AND |dt_case[k] - dt_ctrl[k]| <= 180 d for all priors k=1..n_priors.\n")


def dt_vector_per_patient(df):
    """Return dict: PATIENTID -> sorted dt array of length n_priors."""
    priors = df[~df["is_index"]].copy()
    priors = priors.sort_values(["PATIENTID", "exam_order"])
    out = {}
    for pid, g in priors.groupby("PATIENTID"):
        out[pid] = g["date_diff"].dropna().astype(float).to_numpy()
    return out


cases_idx_r = per_patient_index(cases_relaxed).dropna(subset=["age", "Breast Density"])
ctrls_idx_r = per_patient_index(ctrls_relaxed).dropna(subset=["age", "Breast Density"])
case_dt = dt_vector_per_patient(cases_relaxed)
ctrl_dt = dt_vector_per_patient(ctrls_relaxed)

for tol in (90, 180, 270, 365):
    pool_counts = []
    for _, c in cases_idx_r.iterrows():
        cdt = case_dt.get(c["PATIENTID"], np.array([]))
        if len(cdt) != c["n_priors"]:
            pool_counts.append(0)
            continue
        elig = ctrls_idx_r[
            (ctrls_idx_r["Breast Density"] == c["Breast Density"]) &
            (ctrls_idx_r["n_priors"] == c["n_priors"]) &
            ((ctrls_idx_r["age"] - c["age"]).abs() <= 3)
        ]
        if elig.empty:
            pool_counts.append(0)
            continue
        # Apply per-prior dt tolerance
        n_pass = 0
        for cpid in elig["PATIENTID"]:
            ctd = ctrl_dt.get(cpid, np.array([]))
            if len(ctd) != len(cdt):
                continue
            if np.all(np.abs(cdt - ctd) <= tol):
                n_pass += 1
        pool_counts.append(n_pass)
    pc = pd.Series(pool_counts)
    n_zero = (pc == 0).sum()
    cases_idx_r2 = cases_idx_r.copy()
    cases_idx_r2["pool"] = pool_counts
    by_np = cases_idx_r2.groupby("n_priors")["pool"].agg(["count", "median",
                                                          lambda s: (s == 0).sum()])
    by_np.columns = ["n_cases", "median_pool", "n_with_zero_pool"]
    print(f"  tolerance ±{tol} d: median pool/case={pc.median():.0f}, "
          f"cases with 0 pool={n_zero}/{len(pc)} ({100*n_zero/len(pc):.0f}%)")
    for _, row in by_np.reset_index().iterrows():
        print(f"    n_priors={int(row['n_priors'])}: "
              f"n_cases={int(row['n_cases'])}, median={row['median_pool']:.0f}, "
              f"zero-pool={int(row['n_with_zero_pool'])}")
