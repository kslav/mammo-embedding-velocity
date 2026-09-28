"""Diagnose LME variance-component degeneracy on the relaxed-cohort intervals.

The current spec in 04b is:
    v_signed ~ time_to_index_years * C(group)
    groups = pair_id
    re_formula = "~1"
    vc_formula = {"patient": "0 + C(PATIENTID)"}

In the d800_dt180 results, three of four LMEs converge with both
"Group Var" and "patient Var" at ~1e-10 (numerical floor), meaning the
optimizer is reducing the model to OLS. We want to identify the simplest
spec that yields stable, non-degenerate variance estimates while keeping
the matched-pair structure.

Tries 4 specs on each comparison × granularity combination:
  S1 (current):  groups=pair_id, re=~1, vc=patient        # current
  S2 (pair-only): groups=pair_id, re=~1, no vc            # only pair RE
  S3 (patient-only): groups=PATIENTID, re=~1, no vc       # only patient RE
  S4 (random slope): groups=PATIENTID, re="~time_t..."    # patient slope+intercept

Reports for each: converged?, variance estimates, fixed-effect interaction
β + p, and (where computable) AIC for direct comparison.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

ROOT = Path("/path/to/scratch/drift_cohort/drift_analysis_d365_d800_dt180")

CASES = [
    ("breast", "breast_malignant_vs_control"),
    ("breast", "breast_biopsy_negative_vs_control"),
    ("study",  "study_malignant_vs_control"),
    ("study",  "study_biopsy_negative_vs_control"),
]


def load_intervals(gran: str, tag: str) -> pd.DataFrame:
    p = ROOT / gran / f"{tag}_intervals.csv"
    df = pd.read_csv(p)
    df = df.dropna(subset=["v_signed", "time_to_index_years", "pair_id", "PATIENTID"])
    case_group = tag.replace(f"{gran}_", "").split("_vs_")[0]
    df = df[df["group"].isin([case_group, "control"])].copy()
    df["group"] = pd.Categorical(df["group"], categories=["control", case_group])
    df["pair_id"] = df["pair_id"].astype(str)
    df["PATIENTID"] = df["PATIENTID"].astype(str)
    return df


def fit(model_factory, label: str, df: pd.DataFrame) -> dict:
    try:
        model = model_factory(df)
    except Exception as e:
        return {"label": label, "error": f"build: {e}"}
    last = None
    for method in ("lbfgs", "bfgs", "nm"):
        try:
            res = model.fit(method=method, reml=True)
            last = res
            if getattr(res, "converged", False):
                break
        except Exception as e:
            last = e
    if last is None or isinstance(last, Exception):
        return {"label": label, "error": str(last)}

    out = {
        "label": label,
        "converged": bool(getattr(last, "converged", False)),
        "n_obs": int(last.nobs),
    }
    # Variance estimates: Group var (top-level group) + any vc_formula entries.
    out["group_var"] = float(last.cov_re.iloc[0, 0]) if hasattr(last, "cov_re") and last.cov_re.size else None
    if hasattr(last, "vcomp") and last.vcomp is not None and len(last.vcomp):
        out["vc"] = {f"vc_{i}": float(v) for i, v in enumerate(last.vcomp)}
    # Fixed effects: focus on the time × group interaction
    interaction_keys = [k for k in last.params.index
                        if "time_to_index_years" in k and "group" in k]
    if interaction_keys:
        k = interaction_keys[0]
        out["interaction_beta"] = float(last.params[k])
        out["interaction_pvalue"] = float(last.pvalues[k])
    try:
        out["aic"] = float(last.aic)
    except Exception:
        pass
    return out


def make_S1_pair_groups_patient_vc(df):
    return smf.mixedlm(
        "v_signed ~ time_to_index_years * C(group)",
        data=df, groups=df["pair_id"],
        re_formula="~1",
        vc_formula={"patient": "0 + C(PATIENTID)"},
    )


def make_S2_pair_only(df):
    return smf.mixedlm(
        "v_signed ~ time_to_index_years * C(group)",
        data=df, groups=df["pair_id"],
        re_formula="~1",
    )


def make_S3_patient_only(df):
    return smf.mixedlm(
        "v_signed ~ time_to_index_years * C(group)",
        data=df, groups=df["PATIENTID"],
        re_formula="~1",
    )


def make_S4_patient_random_slope(df):
    return smf.mixedlm(
        "v_signed ~ time_to_index_years * C(group)",
        data=df, groups=df["PATIENTID"],
        re_formula="~time_to_index_years",
    )


SPECS = [
    ("S1 pair_groups + patient_vc (current)", make_S1_pair_groups_patient_vc),
    ("S2 pair_only", make_S2_pair_only),
    ("S3 patient_only", make_S3_patient_only),
    ("S4 patient_random_slope", make_S4_patient_random_slope),
]


def fmt(v, n=4):
    if v is None:
        return "    n/a"
    if isinstance(v, float):
        if abs(v) < 1e-3:
            return f"{v:.2e}"
        return f"{v:.3f}"
    return str(v)


def main():
    for gran, tag in CASES:
        print("=" * 78)
        print(f"{gran} :: {tag}")
        print("=" * 78)
        df = load_intervals(gran, tag)
        print(f"  n_obs={len(df)}, pairs={df['pair_id'].nunique()}, "
              f"patients={df['PATIENTID'].nunique()}")

        for label, factory in SPECS:
            res = fit(factory, label, df)
            if "error" in res:
                print(f"  {label:48s} ERROR: {res['error']}")
                continue
            conv = "✓" if res["converged"] else "✗"
            gv = fmt(res.get("group_var"))
            vc = ""
            if "vc" in res:
                vc = " " + " ".join(f"{k}={fmt(v)}" for k, v in res["vc"].items())
            beta = fmt(res.get("interaction_beta"))
            pv = fmt(res.get("interaction_pvalue"))
            aic = fmt(res.get("aic"), n=1)
            print(f"  {label:48s} conv={conv}  gv={gv}{vc}  "
                  f"β={beta}  p={pv}  AIC={aic}")
        print()


if __name__ == "__main__":
    main()
