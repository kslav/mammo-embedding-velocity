"""Race-stratified contralateral drift: effect size vs statistical power.

Answers "does HOPPR (or any model) actually perform worse on non-White
races, or is it just underpowered on the smaller race bins?" by reporting
the EFFECT SIZE (median and median/SD of the within-patient
affected−contralateral paired diff), not only the p-value, per
model × race × group × interval, with a bootstrap 95% CI on the median.

If the median (and median/SD) is comparable across races but p differs,
the difference is power (small n), not a race effect. If the median/SD
itself is attenuated in a race, that is a genuine per-race effect gap.

Finding on the L2 + long-ComBat headline (see PROJECT_NOTES 2026-07-22):
- MC / MII: non-White effect sizes are as large or larger than White →
  the race gaps in the significance grid are power artifacts, supporting
  race as a non-confounder.
- HOPPR: non-White med/SD genuinely below White (Asian lowest) → a real
  per-race attenuation, worth flagging as a model-specific limitation.
  The bootstrap CI here is what guards the small-n non-White bins.

Run (login-shell-safe: reads committed matched CSVs + scratch contralat
intervals CSVs; the bootstrap is a few thousand resamples, <1 min):
    .venv_mammoclip/bin/python src/diagnostics/race_effectsize_by_model.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from diagnostics.confounder_check import bin_race, per_patient_race  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
from config import SCRATCH_DIR as SCRATCH, DATA_DIR as CSV_BASE  # noqa: E402

CONTRALAT_SUFFIX = {"mammoclip": "mammoclip_l_contralat_combat_long",
                    "hoppr": "hoppr_l_contralat_combat_long",
                    "mii": "mii_l_contralat_combat_long"}
MODELS = ["mammoclip", "hoppr", "mii"]
GROUPS = ["malignant", "biopsy_negative", "control"]
RACE_ORDER = ["White", "Black/AA", "Asian", "Unknown", "Other/Multi"]
# Seeded RNG so bootstrap CIs are reproducible (Math.random-style entropy
# would make the committed CSV non-deterministic across runs).
SEED = 42
N_BOOT = 5000


def _race_map() -> pd.Series:
    cases = pd.read_csv(CSV_BASE / "matched_cases_Density_nPriors_d335_d425_mesh.csv",
                        low_memory=False)
    ctrls = pd.read_csv(CSV_BASE / "matched_controls_Density_nPriors_d335_d425_mesh.csv",
                        low_memory=False)
    return per_patient_race(pd.concat([cases, ctrls], ignore_index=True)).map(bin_race)


def _paired_diffs(df: pd.DataFrame, iv: int) -> np.ndarray:
    """Per-patient affected − contralateral v_signed at one interval."""
    s = df[df["interval_idx"] == iv]
    per = s.groupby(["PATIENTID", "side_label"])["v_signed"].mean().reset_index()
    aff = per[per["side_label"] == "affected"][["PATIENTID", "v_signed"]]
    con = per[per["side_label"] == "contralateral"][["PATIENTID", "v_signed"]]
    m = aff.merge(con, on="PATIENTID", suffixes=("_a", "_c"))
    return (m["v_signed_a"] - m["v_signed_c"]).to_numpy()


def _boot_median_ci(d: np.ndarray, rng: np.random.Generator,
                    n_boot: int = N_BOOT) -> tuple[float, float]:
    """Percentile bootstrap 95% CI for the median."""
    if len(d) < 5:
        return (float("nan"), float("nan"))
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    meds = np.median(d[idx], axis=1)
    return (float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-csv", type=Path,
                    default=REPO / "results" / "race_effectsize_by_model.csv")
    ap.add_argument("--max-interval", type=int, default=4)
    args = ap.parse_args()

    race = _race_map()
    rng = np.random.default_rng(SEED)
    rows = []
    for model in MODELS:
        base = SCRATCH / f"drift_analysis_augmented_335_425_{CONTRALAT_SUFFIX[model]}" / "breast_both_sides"
        for group in GROUPS:
            df = pd.read_csv(base / f"contralateral_{group}_intervals.csv")
            df["race_bin"] = df["PATIENTID"].map(race)
            for r in RACE_ORDER:
                sub = df[df["race_bin"] == r]
                for iv in range(1, args.max_interval + 1):
                    d = _paired_diffs(sub, iv)
                    if len(d) == 0:
                        continue
                    med, sd = float(np.median(d)), float(d.std())
                    nz = d[d != 0]
                    p = (stats.wilcoxon(nz, alternative="greater",
                                        zero_method="wilcox").pvalue
                         if len(nz) >= 5 else float("nan"))
                    lo, hi = _boot_median_ci(d, rng)
                    rows.append({
                        "model": model, "race": r, "group": group,
                        "interval": iv, "n": int(len(d)),
                        "median": med, "sd": sd,
                        "median_over_sd": med / sd if sd > 0 else float("nan"),
                        "wilcoxon_p": p,
                        "median_ci_lo": lo, "median_ci_hi": hi,
                    })

    df = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv}: {len(df)} rows "
          f"(seed={SEED}, n_boot={N_BOOT})\n")

    # Headline view: interval-1 malignant, effect size vs power, per model.
    print("=== interval-1 MALIGNANT: median [95% CI], med/SD, p ===")
    q = df[(df.interval == 1) & (df.group == "malignant")]
    for model in MODELS:
        print(f"  {model}:")
        for _, r in q[q.model == model].iterrows():
            p = f"{r.wilcoxon_p:.1e}" if pd.notna(r.wilcoxon_p) else "n/a"
            print(f"    {r.race:12s} n={int(r.n):>4d}  "
                  f"med={r['median']:+.4f} [{r.median_ci_lo:+.4f}, {r.median_ci_hi:+.4f}]  "
                  f"med/SD={r.median_over_sd:+.3f}  p={p}")


if __name__ == "__main__":
    main()
