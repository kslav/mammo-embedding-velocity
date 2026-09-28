"""LME + race check for the contralateral within-patient analysis.

For every {model} x {baseline, cs-ComBat, long-ComBat} x {raw, L2} contralat
analysis directory, and for every group in {malignant, biopsy_negative,
control}, fits two per-patient mixed-effects models on the already-produced
intervals CSV:

    (a) v_signed ~ time_to_index_years * C(side_label) + (1|PATIENTID)
    (b) v_signed ~ time_to_index_years * C(side_label) + C(race) + (1|PATIENTID)

side_label is Treatment-coded with contralateral as reference so the
interaction term `time_to_index_years:C(side_label)[T.affected]` is
positive when the affected side drifts faster toward u_hat as the index
approaches — the within-patient analog of the case-vs-control
interaction reported in the between-patient LME (04b, MANUSCRIPT §3.3).

Race is binned identically to src/diagnostics/confounder_check.py
(bin_race: White / Black-AA / Asian / Unknown / Other-Multi).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diagnostics.confounder_check import bin_race, per_patient_race  # noqa: E402
from config import SCRATCH_DIR  # noqa: E402


CELLS = [
    # (model, harm_variant_tag, scratch_dir_suffix)
    ("mammoclip", "baseline",       "contralat"),
    ("mammoclip", "cs_combat",      "contralat_combat"),
    ("mammoclip", "long_combat",    "contralat_combat_long"),
    ("mammoclip", "l_baseline",     "l_contralat"),
    ("mammoclip", "l_cs_combat",    "l_contralat_combat"),
    ("mammoclip", "l_long_combat",  "l_contralat_combat_long"),
    ("hoppr",     "baseline",       "contralat"),
    ("hoppr",     "cs_combat",      "contralat_combat"),
    ("hoppr",     "long_combat",    "contralat_combat_long"),
    ("hoppr",     "l_baseline",     "l_contralat"),
    ("hoppr",     "l_cs_combat",    "l_contralat_combat"),
    ("hoppr",     "l_long_combat",  "l_contralat_combat_long"),
    ("mii",       "baseline",       "contralat"),
    ("mii",       "cs_combat",      "contralat_combat"),
    ("mii",       "long_combat",    "contralat_combat_long"),
    ("mii",       "l_baseline",     "l_contralat"),
    ("mii",       "l_cs_combat",    "l_contralat_combat"),
    ("mii",       "l_long_combat",  "l_contralat_combat_long"),
]

GROUPS = ["malignant", "biopsy_negative", "control"]


def fit_lme(df: pd.DataFrame, formula: str) -> dict:
    """Return {beta, pvalue, converged, n_obs} for the interaction term."""
    try:
        model = smf.mixedlm(formula, data=df, groups=df["PATIENTID"], re_formula="~1")
        res = None
        for method in ("lbfgs", "bfgs", "nm"):
            try:
                res = model.fit(method=method, reml=True)
                if getattr(res, "converged", False):
                    break
            except Exception:
                continue
        if res is None:
            return {"error": "no fit"}
        ix_keys = [k for k in res.params.index
                   if "time_to_index_years" in k and "side_label" in k]
        if not ix_keys:
            return {"error": "no interaction term"}
        k = ix_keys[0]
        return {
            "n_obs": int(len(df)),
            "converged": bool(getattr(res, "converged", False)),
            "interaction_beta": float(res.params[k]),
            "interaction_pvalue": float(res.pvalues[k]),
        }
    except Exception as e:
        return {"error": str(e)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases-csv", type=Path, required=True)
    ap.add_argument("--controls-csv", type=Path, required=True)
    ap.add_argument("--scratch-base", type=Path,
                    default=SCRATCH_DIR)
    ap.add_argument("--cohort-prefix", type=str,
                    default="drift_analysis_augmented_335_425_")
    ap.add_argument("--out-md", type=Path, required=True)
    args = ap.parse_args()

    print(f"Loading raw CSVs to build race map...")
    cases_raw = pd.read_csv(args.cases_csv, low_memory=False)
    ctrls_raw = pd.read_csv(args.controls_csv, low_memory=False)
    cases_race = per_patient_race(cases_raw).map(bin_race)
    ctrls_race = per_patient_race(ctrls_raw).map(bin_race)
    race_map = pd.concat([cases_race, ctrls_race])
    print(f"  race bins: {race_map.value_counts().to_dict()}")

    rows: list[dict] = []
    for model, harm_tag, suffix in CELLS:
        cell_dir = args.scratch_base / f"{args.cohort_prefix}{model}_{suffix}"
        gran_dir = cell_dir / "breast_both_sides"
        for group in GROUPS:
            iv_path = gran_dir / f"contralateral_{group}_intervals.csv"
            if not iv_path.is_file():
                print(f"  MISSING: {iv_path}")
                continue
            iv = pd.read_csv(iv_path)
            iv = iv.dropna(subset=["v_signed", "time_to_index_years", "PATIENTID",
                                   "side_label"]).copy()
            iv["race"] = iv["PATIENTID"].map(race_map).fillna("Unknown")
            iv["side_label"] = pd.Categorical(iv["side_label"],
                                              categories=["contralateral", "affected"])
            iv["PATIENTID"] = iv["PATIENTID"].astype(str)

            fit_base = fit_lme(iv,
                "v_signed ~ time_to_index_years * C(side_label) + (1|PATIENTID)"
                if False else  # (1|PATIENTID) is provided via `groups`, not formula
                "v_signed ~ time_to_index_years * C(side_label)")
            fit_race = fit_lme(iv,
                "v_signed ~ time_to_index_years * C(side_label) + C(race)")

            def _shift(a, b):
                if "interaction_beta" not in a or "interaction_beta" not in b:
                    return None
                base_b = a["interaction_beta"]
                if abs(base_b) < 1e-12:
                    return None
                return (b["interaction_beta"] - base_b) / abs(base_b) * 100.0

            shift = _shift(fit_base, fit_race)
            row = {
                "model": model,
                "harm": harm_tag,
                "group": group,
                "n_obs": fit_base.get("n_obs"),
                "base_beta": fit_base.get("interaction_beta"),
                "base_p": fit_base.get("interaction_pvalue"),
                "base_conv": fit_base.get("converged"),
                "race_beta": fit_race.get("interaction_beta"),
                "race_p": fit_race.get("interaction_pvalue"),
                "race_conv": fit_race.get("converged"),
                "shift_pct": shift,
            }
            rows.append(row)
            b_beta = fit_base.get("interaction_beta"); b_p = fit_base.get("interaction_pvalue")
            r_beta = fit_race.get("interaction_beta"); r_p = fit_race.get("interaction_pvalue")
            b_beta_str = "n/a" if b_beta is None else f"{b_beta:+.4f}"
            b_p_str = "n/a" if b_p is None else f"{b_p:.3g}"
            r_beta_str = "n/a" if r_beta is None else f"{r_beta:+.4f}"
            r_p_str = "n/a" if r_p is None else f"{r_p:.3g}"
            shift_str = "n/a" if shift is None else f"{shift:+.1f}%"
            print(f"  [{model:9s} {harm_tag:14s} {group:16s}] "
                  f"β={b_beta_str} p={b_p_str}   "
                  f"+race β={r_beta_str} p={r_p_str}  Δβ={shift_str}")

    # Markdown output
    lines = ["# Contralateral within-patient LME + race check",
             "",
             "Interaction term: `time_to_index_years:C(side_label)[T.affected]`.",
             "Positive β = affected side drifts faster toward `u_hat` as the",
             "index approaches. Fits are per-patient random intercept mixed-",
             "effects. Race binned as in `src/diagnostics/confounder_check.py`.",
             "",
             "| Model | Harm | Group | n_obs | β | p | β+race | p+race | Δβ |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        def _f(v, fmt):
            return "n/a" if v is None else format(v, fmt)
        lines.append(
            f"| {r['model']} | {r['harm']} | {r['group']} | "
            f"{r['n_obs']} | {_f(r['base_beta'], '+.4f')} | {_f(r['base_p'], '.3g')} | "
            f"{_f(r['race_beta'], '+.4f')} | {_f(r['race_p'], '.3g')} | "
            f"{_f(r['shift_pct'], '+.1f')}% |"
        )
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text("\n".join(lines) + "\n")
    print(f"\nWrote {args.out_md} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
