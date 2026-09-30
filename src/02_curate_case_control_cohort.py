"""Step 02 — case-control matching + image-path attachment.

Ported from src/notebooks/cohort_curation_notebook.ipynb. Reads the
longitudinal parquets produced by step 01, performs 1:1 matching on
density + n_priors + age (and optional per-prior dt-vector tolerance),
melts ``valid_images_L`` / ``valid_images_R`` into long form, and joins
the DICOM catalogue to attach file paths.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Allow `from utils.cohort_suffix import build_suffix` whether step 02
# is run from repo root or from src/.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.cohort_suffix import build_suffix  # noqa: E402

logger = logging.getLogger(__name__)


DEFAULT_DATA_DIR = Path(
    "/path/to/data/drift_analysis"
)
# Notebook-pinned catalogue. Older than the 260327 catalogue used elsewhere
# in the repo, but the user prefers this one for cohort curation — keep it.
DEFAULT_DICOM_CATALOGUE_PARQUET = (
    "/path/to/datasets/mammo/mammo_hoppr/manifests/catalogues/"
    "mfs_mammo_allInternalData_251107_catalogue.parquet"
)


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def parse_age(age_str):
    """Parse DICOM-style age strings (e.g. '065Y') to float years.
    Duplicated from step 01 so step 02 stands alone."""
    if isinstance(age_str, float):
        return age_str
    if not isinstance(age_str, str):
        return pd.NaT
    try:
        unit = age_str.strip('"')[-1].upper()
        value = int(age_str.strip('"')[:-1])
        if unit == "Y":
            return float(value)
        elif unit == "M":
            return value / 12
        elif unit == "W":
            return value / 52
        elif unit == "D":
            return value / 365
        else:
            return pd.NaT
    except Exception:
        return pd.NaT


def _prior_dd_seq(df: pd.DataFrame) -> pd.Series:
    """Per-patient ordered tuple of prior date_diff values, ascending exam_order.
    NaN entries are dropped (the index has date_diff=NA and is excluded by
    is_index=False anyway)."""
    return (
        df[~df["is_index"]]
        .drop_duplicates(subset=["PATIENTID", "STUDYID"])
        .sort_values(["PATIENTID", "exam_order"])
        .groupby("PATIENTID")["date_diff"]
        .apply(lambda s: tuple(int(x) for x in s if pd.notna(x)))
    )


# --------------------------------------------------------------------------- #
# Matching                                                                    #
# --------------------------------------------------------------------------- #

def match_cases_controls(
    df_cases: pd.DataFrame,
    df_controls: pd.DataFrame,
    age_col: Optional[str] = None,
    density_col: str = "Breast Density",
    age_tolerance: int = 3,
    manufacturer_col: Optional[str] = None,
    dt_tolerance: Optional[int] = None,
    dt_selection: str = "random",
    n_controls_to_match: int = 1,
    random_state: Optional[int] = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Match each case to ``n_controls_to_match`` controls without replacement.

    Required matching: density exact + n_priors exact (always).
    Optional: age ±``age_tolerance``, manufacturer exact, per-prior dt within
    ``dt_tolerance`` days.

    ``dt_selection`` controls how controls are picked from the eligible pool
    (only meaningful when ``dt_tolerance`` is set):
      - ``"random"``  : sample uniformly at random (notebook default).
      - ``"tightest"``: pick the controls with the smallest L2 distance
        between the case and control per-prior dt vectors.
    """
    if dt_selection not in {"random", "tightest"}:
        raise ValueError(f"dt_selection must be 'random' or 'tightest', got {dt_selection!r}")
    rng = np.random.default_rng(random_state)

    case_dd_seqs = _prior_dd_seq(df_cases)
    ctrl_dd_seqs = _prior_dd_seq(df_controls)

    n_priors = case_dd_seqs.map(len).rename("n_priors")
    n_priors_ctrl = ctrl_dd_seqs.map(len).rename("n_priors")

    case_index = (
        df_cases[df_cases["exam_order"] == 0]
        .drop_duplicates(subset=["PATIENTID", "STUDYID"])
        .join(n_priors, on="PATIENTID")
        .fillna({"n_priors": 0})
        .assign(n_priors=lambda d: d["n_priors"].astype(int))
        .copy()
    )
    control_index = (
        df_controls[df_controls["exam_order"] == 0]
        .drop_duplicates(subset=["PATIENTID", "STUDYID"])
        .join(n_priors_ctrl, on="PATIENTID")
        .fillna({"n_priors": 0})
        .assign(n_priors=lambda d: d["n_priors"].astype(int))
        .copy()
    )

    if age_col is not None:
        case_index[age_col] = case_index[age_col].apply(parse_age)
        control_index[age_col] = control_index[age_col].apply(parse_age)
        case_index = case_index.dropna(subset=[age_col])
        control_index = control_index.dropna(subset=[age_col])

    matched_case_rows: list[dict] = []
    matched_control_rows: list[dict] = []
    available_controls = control_index.copy()
    n_pairs = 0

    for _, case_row in case_index.iterrows():
        eligible_mask = (
            (available_controls[density_col] == case_row[density_col])
            & (available_controls["n_priors"] == case_row["n_priors"])
        )
        if manufacturer_col is not None:
            eligible_mask &= (
                available_controls[manufacturer_col] == case_row[manufacturer_col]
            )
        if age_col is not None:
            eligible_mask &= (
                (available_controls[age_col] - case_row[age_col]).abs() <= age_tolerance
            )

        if dt_tolerance is not None:
            case_seq = case_dd_seqs.get(case_row["PATIENTID"], ())

            def seq_within_tol(ctrl_pid, case_seq=case_seq):
                ctrl_seq = ctrl_dd_seqs.get(ctrl_pid, ())
                if len(ctrl_seq) != len(case_seq):
                    return False
                return all(abs(c - k) <= dt_tolerance for c, k in zip(ctrl_seq, case_seq))

            eligible_mask &= available_controls["PATIENTID"].map(seq_within_tol)

        eligible = available_controls[eligible_mask]

        if eligible.empty:
            logger.warning(
                "No eligible controls for case %s (density=%s, n_priors=%d%s%s) — skipping.",
                case_row["PATIENTID"], case_row[density_col], case_row["n_priors"],
                f", age={case_row[age_col]}" if age_col is not None else "",
                f", dt_seq={case_dd_seqs.get(case_row['PATIENTID'], ())}"
                if dt_tolerance is not None else "",
            )
            continue

        n_sample = min(n_controls_to_match, len(eligible))
        if n_sample < n_controls_to_match:
            logger.warning(
                "Only %d/%d controls available for case %s.",
                n_sample, n_controls_to_match, case_row["PATIENTID"],
            )

        if dt_tolerance is not None and dt_selection == "tightest":
            # Sort eligibles by L2 distance between dt vectors; ties broken by PATIENTID.
            case_seq = case_dd_seqs.get(case_row["PATIENTID"], ())
            l2 = eligible["PATIENTID"].map(
                lambda cpid, case_seq=case_seq: float(
                    np.linalg.norm(
                        np.asarray(case_seq, dtype=float)
                        - np.asarray(ctrl_dd_seqs.get(cpid, ()), dtype=float)
                    )
                )
            )
            order = l2.sort_values(kind="stable").index
            sampled = eligible.loc[order].head(n_sample)
        else:
            sampled = eligible.sample(n=n_sample, random_state=int(rng.integers(1e6)))
        for _, control_row in sampled.iterrows():
            matched_case_rows.append(
                {"PATIENTID": case_row["PATIENTID"], "MATCH_ID": control_row["PATIENTID"]}
            )
            matched_control_rows.append(
                {"PATIENTID": control_row["PATIENTID"], "MATCH_ID": case_row["PATIENTID"]}
            )
        n_pairs += 1
        available_controls = available_controls[
            ~available_controls["PATIENTID"].isin(sampled["PATIENTID"])
        ]

    if not matched_case_rows:
        raise ValueError(
            "No matches found — check matching column names and distributions."
        )

    case_match_ids = pd.DataFrame(matched_case_rows)
    control_match_ids = pd.DataFrame(matched_control_rows)

    matched_cases = (
        df_cases.merge(case_match_ids, on="PATIENTID", how="inner")
        .sort_values(["MATCH_ID", "PATIENTID", "exam_order"])
        .reset_index(drop=True)
    )
    matched_controls = (
        df_controls.merge(control_match_ids, on="PATIENTID", how="inner")
        .sort_values(["MATCH_ID", "PATIENTID", "exam_order"])
        .reset_index(drop=True)
    )

    n_cases = matched_cases["PATIENTID"].nunique()
    n_controls = matched_controls["PATIENTID"].nunique()
    logger.info(
        "Matched %d cases to %d controls (%d case-control pairs, ratio 1:%d)",
        n_cases, n_controls, n_pairs, n_controls_to_match,
    )
    return matched_cases, matched_controls


# --------------------------------------------------------------------------- #
# Quality check                                                               #
# --------------------------------------------------------------------------- #

def quality_check(
    df_matched_cases: pd.DataFrame,
    df_matched_controls: pd.DataFrame,
    min_days: int,
    max_days: int,
    dt_tolerance_days: Optional[int],
) -> None:
    """Validate matched cohort structure: per-arm exam_order contiguity +
    in-window date_diff (warn-only on the latter — there is a known small
    leak from step 01's date_diff recompute), then within-pair dt tolerance
    when --dt-tolerance-days is set.

    Cohort frames here are study-grain (one row per (PATIENTID, STUDYID))."""
    logger.info("=== Quality check ===")

    def _study_level(df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates(subset=["PATIENTID", "STUDYID"]).copy()

    def _per_arm_checks(df: pd.DataFrame, label: str) -> None:
        df = _study_level(df)
        idx_counts = df[df["exam_order"] == 0].groupby("PATIENTID").size()
        bad_idx = idx_counts[idx_counts != 1]
        if len(bad_idx):
            raise ValueError(
                f"{label}: {len(bad_idx)} patient(s) without exactly one "
                f"exam_order=0 row. Examples: {bad_idx.head(5).to_dict()}"
            )
        missing_idx = set(df["PATIENTID"].unique()) - set(idx_counts.index)
        if missing_idx:
            raise ValueError(
                f"{label}: {len(missing_idx)} patient(s) missing exam_order=0. "
                f"Examples: {list(missing_idx)[:5]}"
            )

        bad_chain = []
        for pid, g in df.groupby("PATIENTID", sort=False):
            orders = sorted(g["exam_order"].astype(int).tolist())
            if orders != list(range(0, len(orders))):
                bad_chain.append((pid, orders))
        if bad_chain:
            raise ValueError(
                f"{label}: {len(bad_chain)} patient(s) with non-contiguous "
                f"exam_order chain. Examples: {bad_chain[:3]}"
            )

        non_idx = df[df["exam_order"] != 0]
        if non_idx["date_diff"].isna().any():
            n_nan = non_idx["date_diff"].isna().sum()
            raise ValueError(f"{label}: {n_nan} non-index row(s) with NaN date_diff.")
        oor = non_idx[
            (non_idx["date_diff"] < min_days) | (non_idx["date_diff"] > max_days)
        ]
        if len(oor):
            ex = oor[["PATIENTID", "exam_order", "date_diff"]].head(5).to_dict("records")
            logger.warning(
                "%s: %d non-index row(s) with date_diff outside [%d, %d] — "
                "known step 01 leak, not fatal. Examples: %s",
                label, len(oor), min_days, max_days, ex,
            )

        logger.info(
            "  %s: %d patients, %d studies — index/contiguity checks PASSED.",
            label, df["PATIENTID"].nunique(), df["STUDYID"].nunique(),
        )

    _per_arm_checks(df_matched_cases, "cases")
    _per_arm_checks(df_matched_controls, "controls")

    if dt_tolerance_days is not None:
        cases = _study_level(df_matched_cases)
        ctrls = _study_level(df_matched_controls)
        case_seqs = _prior_dd_seq(cases.assign(is_index=lambda d: d["exam_order"] == 0))
        ctrl_seqs = _prior_dd_seq(ctrls.assign(is_index=lambda d: d["exam_order"] == 0))
        case_to_ctrl = (
            cases[["PATIENTID", "MATCH_ID"]]
            .drop_duplicates()
            .set_index("PATIENTID")["MATCH_ID"]
            .to_dict()
        )
        violations = []
        n_pairs_checked = 0
        for case_pid, ctrl_pid in case_to_ctrl.items():
            cs = case_seqs.get(case_pid)
            ct = ctrl_seqs.get(ctrl_pid)
            if cs is None or ct is None or len(cs) != len(ct):
                violations.append((case_pid, ctrl_pid, "missing/mismatched dt vector"))
                continue
            diff = [abs(c - k) for c, k in zip(cs, ct)]
            if any(d > dt_tolerance_days for d in diff):
                violations.append(
                    (case_pid, ctrl_pid, f"max |Δdt|={max(diff)} > {dt_tolerance_days}")
                )
            n_pairs_checked += 1
        if violations:
            raise ValueError(
                f"dt-tolerance violations: {len(violations)} pair(s) with per-prior "
                f"|Δdt| > {dt_tolerance_days} d. Examples: {violations[:5]}"
            )
        logger.info(
            "  pairs: %d checked — within-pair |Δdt| ≤ %d d on every prior PASSED.",
            n_pairs_checked, dt_tolerance_days,
        )
    else:
        logger.info("  pairs: dt-tolerance check skipped (--dt-tolerance-days not set).")

    logger.info("=== Quality check PASSED ===")


# --------------------------------------------------------------------------- #
# Image-path attachment                                                       #
# --------------------------------------------------------------------------- #

def get_image_paths(df: pd.DataFrame, dicom_catalogue_parquet: str) -> pd.DataFrame:
    """Melt valid_images_{L,R} into long form, recover IMAGELATERALITY from
    the column name, then join the DICOM catalogue on SOPINSTANCEUID."""
    df_catalogue = pd.read_parquet(dicom_catalogue_parquet)
    df_catalogue = df_catalogue.drop_duplicates(subset=["SOPINSTANCEUID"])
    assert df_catalogue["SOPINSTANCEUID"].is_unique, "catalogue has duplicate SOPs"

    id_cols = [c for c in df.columns if c not in ["valid_images_L", "valid_images_R"]]
    long = df.melt(
        id_vars=id_cols,
        value_vars=["valid_images_L", "valid_images_R"],
        var_name="IMAGELATERALITY",
        value_name="SOPINSTANCEUID",
    )
    long["IMAGELATERALITY"] = long["IMAGELATERALITY"].str[-1]  # 'L' or 'R'

    long = long.explode("SOPINSTANCEUID", ignore_index=True)
    long = long.dropna(subset=["SOPINSTANCEUID"])

    df_with_paths = long.merge(
        df_catalogue,
        on="SOPINSTANCEUID",
        how="left",
        validate="m:1",  # explodes if catalogue has dup SOPs — fail loud
    )
    return df_with_paths.drop_duplicates(subset=["STUDYID", "SOPINSTANCEUID"])


# --------------------------------------------------------------------------- #
# main                                                                        #
# --------------------------------------------------------------------------- #

# Suffix logic moved to src/utils/cohort_suffix.py so other scripts can preview
# what suffix a given flag combination produces. _build_suffix kept as alias.
_build_suffix = build_suffix


def main(
    input_dir: Path = DEFAULT_DATA_DIR,
    output_dir: Path = DEFAULT_DATA_DIR,
    dicom_catalogue_parquet: str = DEFAULT_DICOM_CATALOGUE_PARQUET,
    min_days: int = 365,
    max_days: int = 425,
    dt_tolerance_days: Optional[int] = None,
    dt_selection: str = "random",
    include_mesh_path_proven: bool = False,
) -> None:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    in_suffix = _build_suffix(
        min_days, max_days, dt_tolerance_days=None,
        include_mesh_path_proven=include_mesh_path_proven,
    )
    out_suffix = _build_suffix(
        min_days, max_days, dt_tolerance_days,
        include_mesh_path_proven=include_mesh_path_proven,
    )
    logger.info("Input suffix: '%s'  output suffix: '%s'", in_suffix, out_suffix)

    df_longitudinal_2priors = pd.read_parquet(
        input_dir / f"cases_longitudinal_2priorsminimum{in_suffix}.parquet"
    )
    df_controls_longitudinal_2priors = pd.read_parquet(
        input_dir / f"controls_longitudinal_2priorsminimum{in_suffix}.parquet"
    )

    df_matched_cases, df_matched_controls = match_cases_controls(
        df_longitudinal_2priors,
        df_controls_longitudinal_2priors,
        density_col="Breast Density",
        age_col="PATIENTAGE",
        age_tolerance=3,
        n_controls_to_match=1,
        random_state=42,
        dt_tolerance=dt_tolerance_days,
        dt_selection=dt_selection,
    )

    quality_check(
        df_matched_cases,
        df_matched_controls,
        min_days=min_days,
        max_days=max_days,
        dt_tolerance_days=dt_tolerance_days,
    )

    # Sanity prints from the notebook (cell 52, 53).
    logger.info(
        "Cancer_Description in matched cases:\n%s",
        df_matched_cases["Cancer_Description"].value_counts(dropna=False),
    )
    logger.info(
        "has_valid_views_L=True in matched cases: %d  has_valid_views_R=True: %d",
        (df_matched_cases["has_valid_views_L"] == True).sum(),  # noqa: E712
        (df_matched_cases["has_valid_views_R"] == True).sum(),  # noqa: E712
    )

    # Attach image paths via valid_images_{L,R} → SOPINSTANCEUID → catalogue.
    df_matched_cases = get_image_paths(df_matched_cases, dicom_catalogue_parquet)
    df_matched_controls = get_image_paths(df_matched_controls, dicom_catalogue_parquet)
    logger.info(
        "After image-path attach: cases records=%d, controls records=%d",
        len(df_matched_cases), len(df_matched_controls),
    )
    logger.info(
        "Cases: STUDYIDs=%d, PATIENTIDs=%d  | Controls: STUDYIDs=%d, PATIENTIDs=%d",
        df_matched_cases["STUDYID"].nunique(),
        df_matched_cases["PATIENTID"].nunique(),
        df_matched_controls["STUDYID"].nunique(),
        df_matched_controls["PATIENTID"].nunique(),
    )

    cases_out = output_dir / f"matched_cases_Density_nPriors{out_suffix}.csv"
    ctrls_out = output_dir / f"matched_controls_Density_nPriors{out_suffix}.csv"
    df_matched_cases.to_csv(cases_out, index=False)
    df_matched_controls.to_csv(ctrls_out, index=False)
    logger.info("Wrote %s", cases_out)
    logger.info("Wrote %s", ctrls_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Case-control matching + image paths.")
    parser.add_argument(
        "--input-dir", type=Path, default=DEFAULT_DATA_DIR,
        help="Directory containing cases/controls longitudinal parquets from step 01. "
             f"Default: {DEFAULT_DATA_DIR}",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_DATA_DIR,
        help=f"Directory to write matched CSVs. Default: {DEFAULT_DATA_DIR}",
    )
    parser.add_argument(
        "--dicom-catalogue-parquet", type=str,
        default=DEFAULT_DICOM_CATALOGUE_PARQUET,
    )
    parser.add_argument(
        "--min-days", type=int, default=335,
        help="Step 01's --min-days, used to find input parquets. "
             "Default 335 matches the augmented manuscript cohort.",
    )
    parser.add_argument(
        "--max-days", type=int, default=425,
        help="Step 01's --max-days, used to find input parquets.",
    )
    parser.add_argument(
        "--dt-tolerance-days", type=int, default=None,
        help="If set, additionally require |dt_case[k] - dt_ctrl[k]| <= tol "
             "for every prior k. See --dt-selection for how controls are picked.",
    )
    parser.add_argument(
        "--dt-selection", choices=["random", "tightest"], default="random",
        help="When --dt-tolerance-days is set: 'random' samples within tolerance "
             "(notebook behavior); 'tightest' picks the smallest L2 distance match.",
    )
    parser.add_argument(
        "--include-mesh-path-proven", action=argparse.BooleanOptionalAction, default=True,
        help="Step 01's --include-mesh-path-proven, used to find input parquets "
             "and tag output filenames consistently. Default True (manuscript "
             "cohort); pass --no-include-mesh-path-proven to reproduce the "
             "pre-augmentation cohort.",
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        dicom_catalogue_parquet=args.dicom_catalogue_parquet,
        min_days=args.min_days,
        max_days=args.max_days,
        dt_tolerance_days=args.dt_tolerance_days,
        dt_selection=args.dt_selection,
        include_mesh_path_proven=args.include_mesh_path_proven,
    )
