"""Contralateral (04c) MG_modality subgroup — post-hoc stats.

Keeps the FULL-COHORT u_hat: reads the already-written
contralateral_{group}_intervals.csv from each 04c run (whose v_signed was
computed against the full-cohort affected-side LOO u_hat), then restricts
to the modality-STABLE vs modality-SWITCHER patient sets and recomputes
the within-patient paired Wilcoxon (affected > contralateral, one-sided)
per group. This mirrors how the between-patient (A)/(B) subgroups held
u_hat fixed and only subset the comparison.

Stable / switcher patient sets are read from the filtered CSVs produced
by build_contralat_modality_subgroup_csvs.py (depth 2).

Emits results/subgroup_contralat_modality_numbers.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import SCRATCH_DIR as SCRATCH, DATA_DIR as CSV_BASE  # noqa: E402

# Contralateral scratch dirs are named
#   drift_analysis_augmented_335_425_{model}{l2}_contralat{harm}
# where l2 is "" or "_l" and harm is ""/"_combat"/"_combat_long".
MODELS = [("mammoclip", ""), ("mammoclip", "_l"),
          ("hoppr", ""), ("hoppr", "_l"),
          ("mii", ""), ("mii", "_l")]
HARMS = [("baseline", ""),
         ("cs_combat", "_combat"),
         ("long_combat", "_combat_long")]
GROUPS = ["malignant", "biopsy_negative", "control"]


def _paired_wilcoxon(iv: pd.DataFrame, keep_pids: set) -> dict:
    """Affected vs contralateral paired Wilcoxon on the patient subset."""
    sub = iv[iv["PATIENTID"].isin(keep_pids)]
    per = sub.groupby(["PATIENTID", "side_label"])["v_signed"].mean().reset_index()
    aff = per[per["side_label"] == "affected"][["PATIENTID", "v_signed"]]
    con = per[per["side_label"] == "contralateral"][["PATIENTID", "v_signed"]]
    m = aff.merge(con, on="PATIENTID", suffixes=("_aff", "_con"), how="inner")
    if len(m) < 5:
        return {"n_pairs": int(len(m)), "pvalue": None, "median_diff": None}
    diff = m["v_signed_aff"].to_numpy() - m["v_signed_con"].to_numpy()
    nz = diff != 0
    if nz.sum() < 5:
        return {"n_pairs": int(len(m)), "n_nonzero": int(nz.sum()),
                "pvalue": None, "median_diff": float(np.median(diff))}
    res = stats.wilcoxon(diff[nz], alternative="greater", zero_method="wilcox")
    return {"n_pairs": int(len(m)), "n_nonzero": int(nz.sum()),
            "pvalue": float(res.pvalue), "median_diff": float(np.median(diff))}


def _load_pid_sets() -> dict[str, set]:
    """Patient IDs for stable/switcher, cases + controls pooled (04c is
    within-patient, so any patient in either arm participates)."""
    out = {}
    for label, suf in [("modstable", "modstable"), ("modswitch", "modswitch")]:
        c = pd.read_csv(CSV_BASE / f"matched_cases_Density_nPriors_d335_d425_mesh_{suf}.csv",
                        usecols=["PATIENTID"])
        k = pd.read_csv(CSV_BASE / f"matched_controls_Density_nPriors_d335_d425_mesh_{suf}.csv",
                        usecols=["PATIENTID"])
        out[label] = set(c["PATIENTID"]) | set(k["PATIENTID"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-csv", type=Path,
                    default=Path("results/subgroup_contralat_modality_numbers.csv"))
    args = ap.parse_args()

    pid_sets = _load_pid_sets()
    print("stable patients:", len(pid_sets["modstable"]),
          "| switcher patients:", len(pid_sets["modswitch"]))

    rows = []
    for model, l2suf in MODELS:
        l2 = "L2" if l2suf == "_l" else "raw"
        for harm_name, hsuf in HARMS:
            gdir = SCRATCH / f"drift_analysis_augmented_335_425_{model}{l2suf}_contralat{hsuf}" / "breast_both_sides"
            for group in GROUPS:
                ivp = gdir / f"contralateral_{group}_intervals.csv"
                if not ivp.is_file():
                    continue
                iv = pd.read_csv(ivp)
                for sg in ["modstable", "modswitch"]:
                    r = _paired_wilcoxon(iv, pid_sets[sg])
                    rows.append({
                        "subgroup": sg, "model": model, "l2norm": l2,
                        "harm_variant": harm_name, "group": group,
                        "n_pairs": r["n_pairs"], "wilcoxon_p": r["pvalue"],
                        "median_diff": r["median_diff"],
                    })

    df = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv}: {len(df)} rows")
    # Preview: malignant, long-ComBat
    q = df[(df.group == "malignant") & (df.harm_variant == "long_combat")]
    print("\n=== malignant, long-ComBat (affected>contralateral) ===")
    for _, r in q.sort_values(["model", "l2norm", "subgroup"]).iterrows():
        p = f"{r.wilcoxon_p:.2e}" if pd.notna(r.wilcoxon_p) else "n/a"
        md = f"{r.median_diff:+.3f}" if pd.notna(r.median_diff) else "n/a"
        print(f"  {r.model:9s} {r.l2norm:3s} {r.subgroup:9s} n={int(r.n_pairs):4d} p={p} med={md}")


if __name__ == "__main__":
    main()
