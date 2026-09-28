#!/usr/bin/env python3
"""
Within-patient contralateral-breast drift analysis.

Companion to 04b that swaps the between-patient matched control for a
within-patient contralateral-breast control. For each patient we aggregate
one exam embedding per (STUDYID, laterality), tag each row with
side_label ∈ {"affected", "contralateral"}, and run a paired one-sided
Wilcoxon on (affected v_signed − contralateral v_signed) *within each
group* (malignant, biopsy_negative, control).

Side labels:
  - Malignant / biopsy_negative: affected = Laterality_Biopsy;
    contralateral = the other side.
  - Control: no biological affected side. We assign one side per patient
    via a seeded PRNG (default seed=42) so the control-arm Wilcoxon is
    a well-defined *null* — a well-behaved pipeline should produce
    p ≈ 0.5 on the control arm.

u_hat is the same cancer direction used by 04b (§2.4 of MANUSCRIPT.md):
mean(malignant_affected_index) − mean(control_affected_index),
L2-normalized, fit leave-one-out per patient.

Usage (typically via SLURM; longitudinal-ComBat inputs consumed via
--exam-parquet-dir produced upstream by apply_harmonization.py):
    python 04c_analyze_contralateral_drift.py \\
        --cases-csv    <matched_cases.csv> \\
        --controls-csv <matched_controls.csv> \\
        --exam-parquet-dir <harm_out> \\
        --output-dir       <contralat_out> \\
        --emb-dim 1024 \\
        --seed 42
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


# ------------------------- u_hat over affected-side index ------------------- #

def build_loo_u_hat_map_affected(
    exam: pd.DataFrame,
    case_group: str = MALIGNANT,
    control_group: str = CONTROL,
) -> dict[str, np.ndarray]:
    """LOO u_hat = mean(case affected index) − mean(control affected index).

    Fit on ``side_label == "affected"`` + ``is_index == True`` rows. Every
    patient contributes exactly one embedding (their affected-side index).
    Leave-one-out excludes p's own affected-index embedding when computing
    u_hat_p.
    """
    idx = exam[(exam["is_index"]) & (exam["side_label"] == "affected")].reset_index(drop=True)
    if not idx["PATIENTID"].is_unique:
        n_dup = int(idx["PATIENTID"].duplicated().sum())
        raise ValueError(
            f"build_loo_u_hat_map_affected: expected one affected-side index row "
            f"per patient, got {n_dup} duplicates."
        )

    case_mask = (idx["group"] == case_group).to_numpy()
    ctrl_mask = (idx["group"] == control_group).to_numpy()
    if not case_mask.any() or not ctrl_mask.any():
        raise ValueError(
            f"LOO u_hat needs both groups present "
            f"(n_case={case_mask.sum()}, n_ctrl={ctrl_mask.sum()})"
        )
    X = np.stack(idx["embedding"].values).astype(np.float64)
    case_sum = X[case_mask].sum(axis=0)
    ctrl_sum = X[ctrl_mask].sum(axis=0)
    n_case = int(case_mask.sum())
    n_ctrl = int(ctrl_mask.sum())

    global_diff = case_sum / n_case - ctrl_sum / n_ctrl
    global_u = global_diff / (np.linalg.norm(global_diff) or 1.0)

    u_map: dict[str, np.ndarray] = {}
    for i, pid in enumerate(idx["PATIENTID"].values):
        if case_mask[i]:
            diff = (case_sum - X[i]) / (n_case - 1) - ctrl_sum / n_ctrl
        elif ctrl_mask[i]:
            diff = case_sum / n_case - (ctrl_sum - X[i]) / (n_ctrl - 1)
        else:
            u_map[pid] = global_u
            continue
        norm = np.linalg.norm(diff)
        u_map[pid] = diff / norm if norm > 0 else global_u
    u_map["__GLOBAL__"] = global_u
    return u_map


# ------------------------- interval velocities per side --------------------- #

def compute_intervals_per_side(
    exam: pd.DataFrame,
    u_hat_map: dict[str, np.ndarray],
) -> pd.DataFrame:
    """One row per (PATIENTID, side_label, interval_idx).

    Consecutive exams within a (PATIENTID, side_label) chain define an
    interval. Displacement = newer − older; v_signed = <disp, u_hat_p> / dt_yr.
    """
    global_u = u_hat_map.get("__GLOBAL__")
    rows: list[dict] = []
    for (pid, side), g in exam.groupby(["PATIENTID", "side_label"], sort=False):
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
                "side_label": side,
                "group": newer["group"],
                "n_priors": int(newer.get("n_priors", 0)),
                "interval_idx": i + 1,
                "exam_order_newer": int(newer["exam_order"]),
                "exam_order_older": int(older["exam_order"]),
                "dt_days": dt_days,
                "dt_years": dt_years,
                "v_signed": float(disp @ u / dt_years),
                "v_L2": float(disp_norm / dt_years),
                "cos_sim": cos_sim,
                "time_to_index_years": -(
                    g.iloc[:i + 1]["date_diff"].fillna(0).sum() + dt_days / 2.0
                ) / DAYS_PER_YEAR,
            })
    return pd.DataFrame(rows)


# ------------------------- paired within-patient tests ---------------------- #

def paired_within_patient_wilcoxon(intervals: pd.DataFrame, group: str) -> dict:
    """One-sided paired Wilcoxon on affected − contralateral v_signed, within group.

    Collapse to one v_signed per (PATIENTID, side_label) — mean across their
    intervals. Pair on PATIENTID; test alternative="greater".
    """
    sub = intervals[intervals["group"] == group]
    per = (
        sub.groupby(["PATIENTID", "side_label"])["v_signed"].mean().reset_index()
    )
    aff = per[per["side_label"] == "affected"][["PATIENTID", "v_signed"]]
    con = per[per["side_label"] == "contralateral"][["PATIENTID", "v_signed"]]
    pairs = aff.merge(con, on="PATIENTID", suffixes=("_aff", "_con"), how="inner")

    if len(pairs) < 5:
        return {"n_pairs": int(len(pairs)),
                "statistic": None, "pvalue": None,
                "median_diff": None, "note": "insufficient pairs"}

    diff = pairs["v_signed_aff"].to_numpy() - pairs["v_signed_con"].to_numpy()
    nz = diff != 0
    if nz.sum() < 5:
        return {"n_pairs": int(len(pairs)), "n_nonzero": int(nz.sum()),
                "statistic": None, "pvalue": None, "note": "too many ties"}
    res = stats.wilcoxon(diff[nz], alternative="greater", zero_method="wilcox")
    return {
        "n_pairs": int(len(pairs)),
        "n_nonzero": int(nz.sum()),
        "statistic": float(res.statistic),
        "pvalue": float(res.pvalue),
        "median_diff": float(np.median(diff)),
    }


def interval_within_patient_wilcoxon(intervals: pd.DataFrame, group: str) -> dict:
    """Per-interval paired Wilcoxon (Holm-corrected) within a single group."""
    sub = intervals[intervals["group"] == group]
    out: dict[str, dict] = {}
    intervals_present = sorted(sub["interval_idx"].dropna().unique().astype(int))
    pvals: list[float] = []
    keys: list[int] = []
    for iv in intervals_present:
        s = sub[sub["interval_idx"] == iv]
        per = (
            s.groupby(["PATIENTID", "side_label"])["v_signed"].mean().reset_index()
        )
        aff = per[per["side_label"] == "affected"][["PATIENTID", "v_signed"]]
        con = per[per["side_label"] == "contralateral"][["PATIENTID", "v_signed"]]
        pairs = aff.merge(con, on="PATIENTID", suffixes=("_aff", "_con"), how="inner")
        if len(pairs) < 5:
            continue
        diff = pairs["v_signed_aff"].to_numpy() - pairs["v_signed_con"].to_numpy()
        nz = diff != 0
        if nz.sum() < 5:
            continue
        res = stats.wilcoxon(diff[nz], alternative="greater", zero_method="wilcox")
        out[str(iv)] = {
            "n_pairs": int(len(pairs)),
            "statistic": float(res.statistic),
            "pvalue_raw": float(res.pvalue),
            "median_diff": float(np.median(diff)),
        }
        pvals.append(float(res.pvalue))
        keys.append(iv)
    # Holm-Bonferroni across intervals
    if pvals:
        order = np.argsort(pvals)
        m = len(pvals)
        holm = np.empty(m)
        running_max = 0.0
        for rank, i in enumerate(order):
            adj = min(1.0, (m - rank) * pvals[i])
            running_max = max(running_max, adj)
            holm[i] = running_max
        for i, iv in enumerate(keys):
            out[str(iv)]["pvalue_holm"] = float(holm[i])
    return out


# ---------------------------- run per group --------------------------------- #

def run_group(
    exam: pd.DataFrame, u_hat_map: dict[str, np.ndarray],
    group: str, out_dir: Path,
) -> dict:
    sub = exam[exam["group"] == group]
    print(f"\n--- Group: {group}  (n_patients = {sub['PATIENTID'].nunique()}) ---")
    if sub.empty:
        return {"group": group, "n_patients": 0, "note": "no rows"}

    # Only keep patients with BOTH sides present in the exam table (else there
    # is no paired within-patient comparison to run).
    both_sides = (
        sub.groupby("PATIENTID")["side_label"].nunique().rename("n_sides")
    )
    keep = set(both_sides[both_sides == 2].index)
    n_dropped_singleton = len(both_sides) - len(keep)
    if n_dropped_singleton > 0:
        print(f"  Dropping {n_dropped_singleton} patient(s) with only one side "
              f"retained; unpaired.")
    sub = sub[sub["PATIENTID"].isin(keep)]

    intervals = compute_intervals_per_side(sub, u_hat_map)
    intervals["comparison"] = f"contralateral_{group}"
    csv_path = out_dir / f"contralateral_{group}_intervals.csv"
    intervals.to_csv(csv_path, index=False)
    print(f"  wrote {csv_path.name}  ({len(intervals)} rows)")

    patient_wilc = paired_within_patient_wilcoxon(intervals, group)
    print(f"  patient Wilcoxon (affected > contralateral): "
          f"n={patient_wilc.get('n_pairs')} p={patient_wilc.get('pvalue')} "
          f"median_diff={patient_wilc.get('median_diff')}")

    per_interval = interval_within_patient_wilcoxon(intervals, group)

    return {
        "group": group,
        "tag": f"contralateral_{group}",
        "n_patients_paired": int(len(keep)),
        "velocity_units": "1/year",
        "patient_wilcoxon": patient_wilc,
        "interval_wilcoxon": per_interval,
    }


# --------------------------------- main ------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases-csv", type=Path, required=True)
    p.add_argument("--controls-csv", type=Path, required=True)
    p.add_argument("--embeddings-dir", type=Path, default=None,
                   help="Root of raw .pt tree from step 3. Required unless "
                        "--exam-parquet-dir is provided.")
    p.add_argument("--exam-parquet-dir", type=Path, default=None,
                   help="If provided, load breast_both_sides/exam_embeddings.parquet "
                        "from this directory (produced by apply_harmonization.py "
                        "with --granularities breast_both_sides). Mutually "
                        "exclusive with --embeddings-dir.")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--emb-dim", type=int, default=2048)
    p.add_argument("--seed", type=int, default=42,
                   help="Seed for control-arm affected-side assignment.")
    p.add_argument("--harm-l2-norm", action="store_true")
    p.add_argument("--harm-combat-manuf", action="store_true")
    args = p.parse_args()

    if args.embeddings_dir is None and args.exam_parquet_dir is None:
        p.error("must provide either --embeddings-dir or --exam-parquet-dir")
    if args.embeddings_dir is not None and args.exam_parquet_dir is not None:
        p.error("--embeddings-dir and --exam-parquet-dir are mutually exclusive")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading cases from {args.cases_csv}")
    df_cases = pd.read_csv(args.cases_csv)
    print(f"Loading controls from {args.controls_csv}")
    df_controls = pd.read_csv(args.controls_csv)
    print(f"  cases: {len(df_cases)} rows, {df_cases['PATIENTID'].nunique()} patients")
    print(f"  controls: {len(df_controls)} rows, {df_controls['PATIENTID'].nunique()} patients")

    gran = "breast_both_sides"
    out_gran = args.output_dir / gran
    out_gran.mkdir(parents=True, exist_ok=True)

    if args.exam_parquet_dir is not None:
        in_path = args.exam_parquet_dir / gran / "exam_embeddings.parquet"
        print(f"  loading precomputed {in_path}")
        exam = pd.read_parquet(in_path)
    else:
        exam = build_exam_table(
            df_cases, df_controls, args.embeddings_dir, gran,
            emb_dim=args.emb_dim,
            harm_l2_norm=args.harm_l2_norm,
            harm_combat_manuf=args.harm_combat_manuf,
            contralateral_seed=args.seed,
        )
    exam.to_parquet(out_gran / "exam_embeddings.parquet")

    if "side_label" not in exam.columns:
        raise ValueError(
            "exam_embeddings.parquet is missing 'side_label' — was it built "
            "with granularity=breast_both_sides?"
        )

    n_by_group_side = (
        exam.groupby(["group", "side_label"])["PATIENTID"].nunique()
        .unstack(fill_value=0)
    )
    print(f"\nPatients per (group, side_label):\n{n_by_group_side}")

    u_hat_map = build_loo_u_hat_map_affected(exam)

    summary: dict[str, dict] = {"seed": int(args.seed)}
    for group in [MALIGNANT, BIOPSY_NEG, CONTROL]:
        if group not in exam["group"].values:
            continue
        summary[group] = run_group(exam, u_hat_map, group, out_gran)

    with open(args.output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary written to {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
