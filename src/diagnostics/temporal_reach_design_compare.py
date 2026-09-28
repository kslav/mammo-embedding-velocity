"""Why does the CONTRALATERAL modality-subgroup analysis retain per-interval
significance one interval deeper than the CASE-CONTROL modality-subgroup?

Compares, per interval, two modality-restricted subgroups on the L2 +
longitudinal-ComBat headline:
  - case-control  MG-modality MATCH  (subgroup_mgmod_p2): the between-patient
    paired diff v_malignant - v_control, restricted to pairs concordant on
    MG_modality at index + priors 1-2. Read from that subgroup's own
    summary.json interval_wilcoxon (raw per-interval p).
  - contralateral MG-modality STABLE (modstable pids): the within-patient
    paired diff v_affected - v_contralateral, restricted to patients whose
    MG_modality is constant across index + priors (depth 2). Recomputed here
    from the full-cohort-u_hat contralateral intervals CSV.

For a fair comparison BOTH designs are re-Holm-corrected across intervals
1..max_interval here (the committed case-control subgroup summary is already
Holm'd across all its intervals; we re-derive from raw so the correction
window is identical for the two designs).

Also reports the paired-diff SD per interval, which is the mechanism: the
within-patient design cancels inter-subject anatomy, roughly halving the
per-pair SD, so in these smaller modality subgroups (where case-control
noise is the binding constraint) the tighter noise floor lets a later
interval clear significance.

NB this is a design/subgroup-specific effect: on the FULL cohort the
case-control design is often the MORE significant of the two. Do not read
this as a general power advantage of the within-patient design.

Emits results/temporal_reach_design_compare.csv (one row per
model x interval x design) and prints a per-model iv1/iv2 summary.

Run (login-shell-safe: reads committed CSVs + scratch intervals CSVs only,
no heavy compute):
    .venv_mammoclip/bin/python src/diagnostics/temporal_reach_design_compare.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diagnostics.contralat_modality_subgroup_stats import _load_pid_sets  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SCRATCH = Path("/path/to/scratch/drift_cohort")

# Case-control drift subgroup dirs drop the "mammoclip" prefix for MC;
# contralateral dirs carry an explicit model prefix for all three.
CC_DRIFT_SUFFIX = {"mammoclip": "l_combat_long",
                   "hoppr": "hoppr_l_combat_long",
                   "mii": "mii_l_combat_long"}
CONTRALAT_SUFFIX = {"mammoclip": "mammoclip_l_contralat_combat_long",
                    "hoppr": "hoppr_l_contralat_combat_long",
                    "mii": "mii_l_contralat_combat_long"}
MODELS = ["mammoclip", "hoppr", "mii"]


def _holm(praw: dict[int, float]) -> dict[int, float]:
    """Holm-Bonferroni step-down over {interval: p}."""
    items = sorted([(k, v) for k, v in praw.items() if v is not None and np.isfinite(v)],
                   key=lambda kv: kv[1])
    m = len(items)
    out, run = {}, 0.0
    for i, (k, p) in enumerate(items):
        run = max(run, min((m - i) * p, 1.0))
        out[k] = run
    return out


def _cc_match_per_interval(model: str, max_interval: int) -> dict[int, dict]:
    """Case-control MG-modality MATCH: raw per-interval Wilcoxon p + median +
    SD, read/derived from the subgroup summary + intervals CSV."""
    d = SCRATCH / f"drift_analysis_augmented_335_425_{CC_DRIFT_SUFFIX[model]}" / "subgroup_mgmod_p2"
    summ = json.loads((d / "summary.json").read_text())
    iw = summ["breast__malignant_vs_control"]["interval_wilcoxon"]
    # SD of the paired diff from the intervals CSV (summary only has p/median).
    ivcsv = pd.read_csv(d / "breast" / "breast_malignant_vs_control_intervals.csv")
    out = {}
    for iv in range(1, max_interval + 1):
        cell = iw.get(str(iv))
        if not cell:
            continue
        s = ivcsv[ivcsv["interval_idx"] == iv]
        per = s.groupby(["pair_id", "group"])["v_signed"].mean().reset_index()
        ca = per[per["group"] != "control"][["pair_id", "v_signed"]]
        ct = per[per["group"] == "control"][["pair_id", "v_signed"]]
        m = ca.merge(ct, on="pair_id", suffixes=("_a", "_c"))
        diff = (m["v_signed_a"] - m["v_signed_c"]).to_numpy()
        out[iv] = {"n": int(cell["n_pairs"]), "p_raw": float(cell["pvalue_raw"]),
                   "median": float(cell["median_diff"]),
                   "sd": float(diff.std()) if len(diff) else float("nan")}
    return out


def _contralat_stable_per_interval(model: str, keep_pids: set,
                                   max_interval: int) -> dict[int, dict]:
    """Contralateral MG-modality STABLE: raw per-interval affected>contra
    Wilcoxon p + median + SD, restricted to keep_pids."""
    ivcsv = pd.read_csv(
        SCRATCH / f"drift_analysis_augmented_335_425_{CONTRALAT_SUFFIX[model]}"
        / "breast_both_sides" / "contralateral_malignant_intervals.csv")
    ivcsv = ivcsv[ivcsv["PATIENTID"].isin(keep_pids)]
    out = {}
    for iv in range(1, max_interval + 1):
        s = ivcsv[ivcsv["interval_idx"] == iv]
        per = s.groupby(["PATIENTID", "side_label"])["v_signed"].mean().reset_index()
        aff = per[per["side_label"] == "affected"][["PATIENTID", "v_signed"]]
        con = per[per["side_label"] == "contralateral"][["PATIENTID", "v_signed"]]
        m = aff.merge(con, on="PATIENTID", suffixes=("_a", "_c"))
        if len(m) < 5:
            continue
        diff = (m["v_signed_a"] - m["v_signed_c"]).to_numpy()
        nz = diff[diff != 0]
        p = (stats.wilcoxon(nz, alternative="greater", zero_method="wilcox").pvalue
             if len(nz) >= 5 else float("nan"))
        out[iv] = {"n": int(len(m)), "p_raw": float(p),
                   "median": float(np.median(diff)), "sd": float(diff.std())}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-csv", type=Path,
                    default=REPO / "results" / "temporal_reach_design_compare.csv")
    ap.add_argument("--max-interval", type=int, default=4)
    args = ap.parse_args()

    stable = _load_pid_sets()["modstable"]
    rows = []
    for model in MODELS:
        cc = _cc_match_per_interval(model, args.max_interval)
        cl = _contralat_stable_per_interval(model, stable, args.max_interval)
        cc_holm = _holm({iv: c["p_raw"] for iv, c in cc.items()})
        cl_holm = _holm({iv: c["p_raw"] for iv, c in cl.items()})
        for iv in range(1, args.max_interval + 1):
            if iv in cc:
                rows.append({"model": model, "design": "case_control_MATCH",
                             "interval": iv, **cc[iv],
                             "p_holm": cc_holm.get(iv)})
            if iv in cl:
                rows.append({"model": model, "design": "contralat_STABLE",
                             "interval": iv, **cl[iv],
                             "p_holm": cl_holm.get(iv)})

    df = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv}: {len(df)} rows\n")

    # Per-model iv1/iv2 summary + SD ratio (the mechanism).
    print(f"{'model':10s} {'design':20s} {'iv':>2s} {'n':>5s} "
          f"{'median':>8s} {'SD':>7s} {'p_Holm':>9s}")
    for model in MODELS:
        sub = df[df.model == model]
        for _, r in sub[sub.interval <= 2].sort_values(["interval", "design"]).iterrows():
            ph = f"{r.p_holm:.1e}" if pd.notna(r.p_holm) else "n/a"
            print(f"{model:10s} {r.design:20s} {int(r.interval):>2d} "
                  f"{int(r.n):>5d} {r['median']:>+8.4f} {r.sd:>7.4f} {ph:>9s}")
        # SD ratio at iv1 (mechanism check)
        cc1 = sub[(sub.design == "case_control_MATCH") & (sub.interval == 1)]
        cl1 = sub[(sub.design == "contralat_STABLE") & (sub.interval == 1)]
        if len(cc1) and len(cl1):
            ratio = cl1.iloc[0].sd / cc1.iloc[0].sd
            print(f"           -> iv1 SD ratio contralat/case-control = {ratio:.2f} "
                  f"(<1 means within-patient design is tighter)\n")


if __name__ == "__main__":
    main()
