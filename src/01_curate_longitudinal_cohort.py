"""Step 01 — longitudinal cohort curation (cases + controls).

Ported from src/notebooks/cohort_curation_notebook.ipynb. The notebook
flow is preserved: a long linear main() with regression-check counts
emitted as INFO logs, plus a small set of helpers (parse_dates,
parse_age, add_view_validity, map_manufacturer, select_one_study_per_patient,
build_longitudinal_frame, filter_annual_priors).
"""
from __future__ import annotations

import argparse
import importlib
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# Make src/utils importable regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent))

logger = logging.getLogger(__name__)


def _load_birads0_laterality_fn():
    """Lazy-import to dodge the workspace formatter that strips top-level
    imports placed after `sys.path.insert(...)`."""
    mod = importlib.import_module("utils.birads0_laterality")
    return mod.get_birads0_with_laterality_from_annotations


DEFAULT_OUTPUT_DIR = Path(
    "/path/to/data/FM_Mammo/hoppr_data/breast_cancer/drift_analysis"
)

# Source-table paths (frozen here rather than data_paths.yaml because the
# notebook references them inline; centralizing is a separate refactor).
PATH_PATH_PROVEN = (
    "/path/to/data/FM_Mammo/hoppr_data/breast_cancer/"
    "mammo_mesh_risBiopsy_linkage_dicomsInVault_strictScreeningBIRADS0_withLLMlabels.parquet"
)
PATH_ALL_SCREENING = (
    "/path/to/datasets/mammo/mammo_hoppr/source_tables/"
    "all_screening_studies_filtered_llm_meta_08262025.parquet"
)
PATH_ALL_LABELS = (
    "/path/to/datasets/mammo/mammo_hoppr/source_tables/"
    "mammo_all_labels_v0_03122024.parquet"
)
PATH_RIS_DEMOG = (
    "/path/to/datasets/mammo/mammo_hoppr/source_tables/deprecated/"
    "ris_mammo_scr_from2013_onward.parquet"
)
# Augmentation source: a previous mesh-path-proven cohort (cancer
# classification CSV, ~4,521 studies). Studies here have a different
# linkage path to pathology than the RIS-biopsy-linked default source,
# so they expand the biopsied cohort with non-overlapping patients.
PATH_MESH_PATH_PROVEN = (
    "/path/to/datasets/mammo/mammo_hoppr/experiment_spreadsheets/breast_cancer/"
    "cancer_class/mesh/cancer_classification_combined_modality_numStudies_n4521_20250915.csv"
)

# CC/MLO view families used by add_view_validity. XCCM, XCCL, LM etc. are
# excluded — the analysis cohort needs at least one CC and one MLO per
# laterality. See cell 10 of the notebook for the rationale.
CC_VIEWS = {"CC", "CCID"}
MLO_VIEWS = {"MLO", "MLOID"}


# --------------------------------------------------------------------------- #
# Parsing helpers                                                             #
# --------------------------------------------------------------------------- #

def parse_age(age_str):
    """Parse DICOM-style age strings (e.g. '065Y') to float years."""
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


def parse_dates(s: pd.Series, col_name: str = "") -> pd.Series:
    """Robustly parse a date column, trying common formats before falling
    back to pandas inference. Unparseable values become NaT."""
    label = f"'{col_name}'" if col_name else "Series"
    FORMATS = [
        "%Y%m%d",       # 20210610  (DICOM standard)
        "%Y-%m-%d",     # 2021-06-10
        "%m/%d/%Y",     # 06/10/2021
        "%Y.%m.%d",     # 2021.06.10
        "%d-%m-%Y",     # 10-06-2021
        "%d/%m/%Y",     # 10/06/2021
    ]

    result = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    remaining_mask = s.notna()

    for fmt in FORMATS:
        if not remaining_mask.any():
            break
        parsed = pd.to_datetime(s[remaining_mask], format=fmt, errors="coerce")
        matched = parsed.notna()
        result[remaining_mask] = parsed
        remaining_mask[remaining_mask] = ~matched

    if remaining_mask.any():
        parsed = pd.to_datetime(s[remaining_mask], errors="coerce")
        result[remaining_mask] = parsed
        remaining_mask[remaining_mask] = parsed.isna()

    failed_mask = remaining_mask & s.notna()
    if failed_mask.any():
        bad_values = s[failed_mask].unique()[:10]
        logger.warning(
            "Could not parse %d values in %s — dropping those studies. Example values: %s",
            failed_mask.sum(), label, bad_values,
        )

    return result


# --------------------------------------------------------------------------- #
# De-duplication helpers                                                      #
# --------------------------------------------------------------------------- #

def merge_mesh_path_proven(
    df_trigger_recent_prior: pd.DataFrame,
    mesh_csv_path: str = PATH_MESH_PATH_PROVEN,
) -> pd.DataFrame:
    """Augment the biopsied cohort with mesh-path-proven studies.

    The mesh-path-proven CSV is a separate cancer-classification source where
    each study is a screening exam linked to a follow-up diagnostic with
    pathology. Linkage path differs from the default RIS-biopsy-linked
    source, so it adds biopsied patients we'd otherwise miss.

    Steps (matching the cohort_curation_notebook cell-9 logic):
      1. Read the mesh CSV and reduce to study-level (PATIENTID, STUDYID,
         Cancer, Cancer_Description, IMAGELATERALITY, STUDYDATE).
      2. Keep only path_negative / path_positive Cancer_Description rows.
      3. Drop STUDYIDs with bilateral IMAGELATERALITY records (laterality
         ambiguous → can't safely assign a side).
      4. Set assumed columns: BI-RADS Category=0 (these are screenings
         linked to a downstream pathology biopsy, so BIRADS-0 by inference),
         timing_category="trigger", is_screening_llm_meta=True,
         has_dicom_in_vault=True, linked_study_type="screening".
      5. Rename STUDYDATE → linked_study_date and IMAGELATERALITY →
         Laterality_Biopsy to match df_trigger_recent_prior columns.
      6. Drop any patients already in df_trigger_recent_prior to keep
         the two sources disjoint at the patient level.
      7. Concat; return the augmented frame.
    """
    df_mesh = pd.read_csv(mesh_csv_path)
    df_mesh_studyLevel = df_mesh[[
        "PATIENTID", "STUDYID", "Cancer", "Cancer_Description",
        "IMAGELATERALITY", "STUDYDATE",
    ]].drop_duplicates(
        subset=["PATIENTID", "STUDYID", "Cancer", "Cancer_Description", "IMAGELATERALITY"]
    )
    df_mesh_studyLevel = df_mesh_studyLevel[
        df_mesh_studyLevel["Cancer_Description"].isin(["path_negative", "path_positive"])
    ]
    logger.info(
        "mesh source raw: patients=%d, studies=%d, records=%d",
        df_mesh_studyLevel["PATIENTID"].nunique(),
        df_mesh_studyLevel["STUDYID"].nunique(),
        len(df_mesh_studyLevel),
    )

    # Drop STUDYIDs with bilateral records (>1 unique IMAGELATERALITY).
    study_lat_counts = df_mesh_studyLevel.groupby("STUDYID")["IMAGELATERALITY"].nunique()
    bilateral = study_lat_counts[study_lat_counts > 1].index
    if len(bilateral):
        logger.info(
            "Dropping %d mesh STUDYIDs with bilateral IMAGELATERALITY", len(bilateral),
        )
        df_mesh_studyLevel = df_mesh_studyLevel[
            ~df_mesh_studyLevel["STUDYID"].isin(bilateral)
        ]

    # Inferred fields — these screenings link to downstream pathology biopsies
    # within ≤62 days, so timing_category="trigger" and BI-RADS=0 are safe.
    df_mesh_studyLevel["BI-RADS Category"] = 0.0
    df_mesh_studyLevel["timing_category"] = "trigger"
    df_mesh_studyLevel["is_screening_llm_meta"] = True
    df_mesh_studyLevel["has_dicom_in_vault"] = True
    df_mesh_studyLevel["linked_study_type"] = "screening"
    df_mesh_studyLevel = df_mesh_studyLevel.rename(columns={
        "STUDYDATE": "linked_study_date",
        "IMAGELATERALITY": "Laterality_Biopsy",
    })

    # Drop overlap with the existing trigger/recent_prior cohort to keep
    # the two sources disjoint at the patient level (notebook prioritizes
    # the RIS-linked source for any overlapping patient).
    overlap_pids = (
        set(df_trigger_recent_prior["PATIENTID"])
        & set(df_mesh_studyLevel["PATIENTID"])
    )
    overlap_sids = (
        set(df_trigger_recent_prior["STUDYID"])
        & set(df_mesh_studyLevel["STUDYID"])
    )
    logger.info(
        "Overlap with trigger/recent_prior: patients=%d, studies=%d → dropping mesh patients in overlap",
        len(overlap_pids), len(overlap_sids),
    )
    df_mesh_studyLevel = df_mesh_studyLevel[
        ~df_mesh_studyLevel["PATIENTID"].isin(overlap_pids)
    ]

    out = pd.concat([df_trigger_recent_prior, df_mesh_studyLevel], ignore_index=True)
    logger.info(
        "After mesh-path-proven augmentation: patients=%d, studies=%d, records=%d",
        out["PATIENTID"].nunique(), out["STUDYID"].nunique(), len(out),
    )
    _vc(out["Cancer_Description"], "Cancer_Description (post-mesh-merge)")
    return out


def select_one_study_per_patient(df: pd.DataFrame) -> pd.DataFrame:
    """For patients with multiple studies, retain the one with the most
    severe Cancer_Description (positive > negative > unknown), and break
    ties by most recent linked_study_date."""
    df = df.copy()

    def _cancer_description_severity(desc: Optional[str]) -> int:
        if pd.isna(desc):
            return 0
        d = desc.lower()
        if "positive" in d:
            return 3
        elif "negative" in d:
            return 2
        else:
            return 1

    df["cancer_severity"] = df["Cancer_Description"].apply(_cancer_description_severity)
    df = df.sort_values(
        ["PATIENTID", "cancer_severity", "linked_study_date"],
        ascending=[True, False, False],
    )
    df = df.drop_duplicates(subset=["PATIENTID"], keep="first")
    return df.drop(columns=["cancer_severity"])


def add_view_validity(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a study-level dataframe with view validity flags per laterality.
    A laterality is 'valid' iff it has at least one CC-family view (CC/CCID)
    and at least one MLO-family view (MLO/MLOID). Rows with IMAGELATERALITY
    not in {"L", "R"} (e.g. "B") are dropped."""
    sub = df[df["VIEWPOSITION"].isin(CC_VIEWS | MLO_VIEWS)].copy()
    sub = sub[sub["IMAGELATERALITY"].isin(["L", "R"])]
    sub["is_cc"] = sub["VIEWPOSITION"].isin(CC_VIEWS)
    sub["is_mlo"] = sub["VIEWPOSITION"].isin(MLO_VIEWS)

    agg = (
        sub.groupby(["STUDYID", "IMAGELATERALITY"], as_index=False)
        .agg(
            n_cc=("is_cc", "sum"),
            n_mlo=("is_mlo", "sum"),
            sops=("SOPINSTANCEUID", lambda s: sorted(set(s))),
        )
    )
    agg["is_valid"] = (agg["n_cc"] >= 1) & (agg["n_mlo"] >= 1)

    # Seed with ALL studies so studies with zero recognized views still appear,
    # with False/[] filled in by the left merge below.
    result = pd.DataFrame({"STUDYID": df["STUDYID"].unique()})

    for lat in ["L", "R"]:
        lat_df = (
            agg[agg["IMAGELATERALITY"] == lat]
            [["STUDYID", "is_valid", "sops"]]
            .rename(columns={
                "is_valid": f"has_valid_views_{lat}",
                "sops": f"valid_images_{lat}",
            })
        )
        result = result.merge(lat_df, on="STUDYID", how="left")
        result[f"has_valid_views_{lat}"] = (
            result[f"has_valid_views_{lat}"].fillna(False).astype(bool)
        )
        result[f"valid_images_{lat}"] = result[f"valid_images_{lat}"].apply(
            lambda x: x if isinstance(x, list) else []
        )
    return result


def map_manufacturer(manufacturer):
    """Bucket raw MANUFACTURER_DICOM strings into a small categorical."""
    if pd.isna(manufacturer):
        return "Unknown"
    m = manufacturer.lower()
    if "hologic" in m:
        return "Hologic"
    elif "ge " in m:
        return "GE"
    elif "siemens" in m:
        return "Siemens"
    elif "philips" in m:
        return "Philips"
    elif "lorad" in m:  # Lorad was acquired by Hologic but kept as its own bucket
        return "Lorad"
    elif "fujifilm" in m:
        return "Fujifilm"
    else:
        return "Other"


# --------------------------------------------------------------------------- #
# Longitudinal frame construction                                             #
# --------------------------------------------------------------------------- #

def build_longitudinal_frame(
    index_df: pd.DataFrame,
    longitudinal_df: pd.DataFrame,
    index_date_col: str = "STUDYDATE",
    longitudinal_date_col: str = "STUDYDATE",
) -> pd.DataFrame:
    """Build a long-format frame of index + prior studies for each patient.

    Each output row corresponds to one (PATIENTID, STUDYID) pair. The index
    study is tagged ``is_index=True``; priors are tagged ``is_longitudinal=True``.
    ``exam_order`` is 0 for the index and increases moving back in time.
    ``date_diff`` is the number of days between consecutive exams within a
    patient.

    Index-row attributes come from ``index_df``; prior-row attributes come from
    ``longitudinal_df``. Biopsy-event multiplicity in ``index_df`` (multiple
    rows per (PATIENTID, STUDYID)) is preserved on the index rows.
    """
    index_df = index_df.copy()
    longitudinal_df = longitudinal_df.copy()

    cols_to_drop = ["exam_order", "date_diff", "is_index", "is_longitudinal"]
    index_df = index_df.drop(columns=[c for c in cols_to_drop if c in index_df.columns])
    longitudinal_df = longitudinal_df.drop(
        columns=[c for c in cols_to_drop if c in longitudinal_df.columns]
    )

    # 1. Parse dates
    index_df[index_date_col] = parse_dates(index_df[index_date_col], index_date_col)
    longitudinal_df[longitudinal_date_col] = parse_dates(
        longitudinal_df[longitudinal_date_col], longitudinal_date_col
    )
    index_df = index_df.dropna(subset=[index_date_col])
    longitudinal_df = longitudinal_df.dropna(subset=[longitudinal_date_col])

    # 2. Build skeleton: identity cols only, one row per study.
    # Index skeleton — most-recent index date per patient (collapses biopsy-event duplicates).
    index_skel = (
        index_df[["PATIENTID", "STUDYID", index_date_col]]
        .sort_values(index_date_col, ascending=False)
        .drop_duplicates("PATIENTID", keep="first")
        .rename(columns={index_date_col: "STUDYDATE"})
        .assign(is_index=True, is_longitudinal=False)
    )
    # Prior skeleton — longitudinal rows for index patients, excluding the index study.
    index_patients = set(index_skel["PATIENTID"])
    index_study_ids = set(index_skel["STUDYID"])
    prior_skel = (
        longitudinal_df.loc[
            longitudinal_df["PATIENTID"].isin(index_patients)
            & ~longitudinal_df["STUDYID"].isin(index_study_ids),
            ["PATIENTID", "STUDYID", longitudinal_date_col],
        ]
        .drop_duplicates(["PATIENTID", "STUDYID"])
        .rename(columns={longitudinal_date_col: "STUDYDATE"})
        .assign(is_index=False, is_longitudinal=True)
    )

    skel = pd.concat([index_skel, prior_skel], ignore_index=True)

    # 3. Drop priors dated after their index
    idx_dates = index_skel[["PATIENTID", "STUDYDATE"]].rename(
        columns={"STUDYDATE": "_idx_date"}
    )
    skel = skel.merge(idx_dates, on="PATIENTID", how="left")
    future = ~skel["is_index"] & (skel["STUDYDATE"] > skel["_idx_date"])
    if future.any():
        logger.info(
            "Dropping %d longitudinal studies dated after their index exam.",
            int(future.sum()),
        )
    skel = skel.loc[~future].drop(columns=["_idx_date"])

    # 4. exam_order (0 = index, increasing into the past) and date_diff
    skel = skel.sort_values(["PATIENTID", "STUDYDATE"], ascending=[True, False])
    skel["exam_order"] = skel.groupby("PATIENTID").cumcount()
    skel = skel.sort_values(["PATIENTID", "exam_order"]).reset_index(drop=True)
    skel["date_diff"] = (
        skel.groupby("PATIENTID")["STUDYDATE"]
        .diff(periods=1)
        .dt.days.abs()
        .astype("Int64")
    )

    # 5. Sanity check
    bad = skel.loc[skel["exam_order"] == 0].groupby("PATIENTID").size()
    bad = bad[bad != 1]
    if not bad.empty:
        raise ValueError(
            f"{len(bad)} patients have != 1 study at exam_order=0:\n{bad}"
        )

    # 6. Merge attrs back: split by row type so column overlaps don't suffix-collide.
    long_attrs = longitudinal_df.drop(columns=[longitudinal_date_col]).drop_duplicates(
        subset=["PATIENTID", "STUDYID"]
    )
    idx_attrs = index_df.drop(columns=[index_date_col])  # keep biopsy multiplicity

    prior_full = skel.loc[~skel["is_index"]].merge(
        long_attrs, on=["PATIENTID", "STUDYID"], how="left"
    )
    index_full = skel.loc[skel["is_index"]].merge(
        idx_attrs, on=["PATIENTID", "STUDYID"], how="left"
    )

    out = (
        pd.concat([index_full, prior_full], ignore_index=True)
        .sort_values(["PATIENTID", "exam_order"])
        .reset_index(drop=True)
    )
    logger.info(
        "Longitudinal frame built: rows=%d, patients=%d, studies=%d",
        len(out), out["PATIENTID"].nunique(), out["STUDYID"].nunique(),
    )
    return out


def filter_annual_priors(
    df: pd.DataFrame,
    min_days: int = 365,
    max_days: int = 425,
    min_consecutive: int = 1,
) -> pd.DataFrame:
    """Retain patients whose first ``min_consecutive`` priors form an
    unbroken in-window chain anchored at the index. The chain runs from
    exam_order=1 upward; the first out-of-window date_diff breaks the chain
    and all later priors are dropped, even if they fall back into the window.
    exam_order and date_diff are recomputed after filtering.
    """
    df = df.copy()
    df["date_diff"] = df["date_diff"].astype("Int64")

    # Work at study level (collapses biopsy-event multiplicity on index rows).
    study_df = (
        df.drop_duplicates(subset=["PATIENTID", "STUDYID"])
        .sort_values(["PATIENTID", "exam_order"])
        .reset_index(drop=True)
    )

    in_window = (
        (study_df["date_diff"] >= min_days)
        & (study_df["date_diff"] <= max_days)
    )
    # Index rows have date_diff=NA → treat as "passing" so the chain starts alive.
    passing = (study_df["is_index"] | in_window).fillna(False)
    chain_alive = passing.groupby(study_df["PATIENTID"]).cummin().astype(bool)
    kept = study_df[chain_alive].copy()

    # Enforce min_consecutive (excluding the index from the count).
    prior_counts = (
        kept.groupby("PATIENTID")["is_index"].apply(lambda s: (~s).sum())
    )
    qualifying_patients = prior_counts[prior_counts >= min_consecutive].index
    kept = kept[kept["PATIENTID"].isin(qualifying_patients)].copy()

    # Recompute exam_order / date_diff at study level.
    # Sort descending so .diff() lands NaN on the *index* (exam_order=0),
    # matching build_longitudinal_frame's convention. date_diff[k] = days
    # between exam k and the next-newer exam k-1; the index has no newer.
    kept = kept.sort_values(["PATIENTID", "STUDYDATE"], ascending=[True, False])
    kept["date_diff"] = (
        kept.groupby("PATIENTID")["STUDYDATE"].diff().dt.days.abs().astype("Int64")
    )
    kept["exam_order"] = (
        kept.groupby("PATIENTID")["STUDYDATE"]
        .rank(method="first", ascending=False)
        .astype(int)
        - 1
    )

    # Join study-level results back to biopsy-granular frame.
    qualifying_keys = kept.set_index(["PATIENTID", "STUDYID"]).index
    out = df[
        df.set_index(["PATIENTID", "STUDYID"]).index.isin(qualifying_keys)
    ].drop(columns=["exam_order", "date_diff"])
    out = out.merge(
        kept[["PATIENTID", "STUDYID", "exam_order", "date_diff"]],
        on=["PATIENTID", "STUDYID"],
        how="left",
    )

    n_before = df["PATIENTID"].nunique()
    n_after = out["PATIENTID"].nunique()
    logger.info(
        "Patients retained: %d/%d (%d dropped); "
        "consecutive prior requirement: >= %d exams within [%d, %d] days",
        n_after, n_before, n_before - n_after, min_consecutive, min_days, max_days,
    )

    return out.sort_values(["PATIENTID", "exam_order"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# main                                                                        #
# --------------------------------------------------------------------------- #

def _vc(series: pd.Series, name: str) -> None:
    """Log value_counts(dropna=False) — same role as the notebook's print
    statements: regression check at each curation step."""
    logger.info("Value counts for %s:\n%s", name, series.value_counts(dropna=False))


def main(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    min_days: int = 365,
    max_days: int = 425,
    include_mesh_path_proven: bool = False,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use the canonical cohort_suffix helper so step 02 + downstream
    # consumers find the same files. Mesh augmentation contributes a
    # `_mesh` component when enabled.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from utils.cohort_suffix import build_suffix as _build_suffix
    suffix = _build_suffix(
        min_days=min_days, max_days=max_days,
        include_mesh_path_proven=include_mesh_path_proven,
    )
    logger.info(
        "Window: [%d, %d] days; mesh-path-proven=%s; output suffix: '%s'",
        min_days, max_days, include_mesh_path_proven, suffix,
    )

    # ----------------------------------------------------------------- #
    # Cancer cohort: load + collapse to one biopsy study per patient    #
    # ----------------------------------------------------------------- #
    df_path_proven = pd.read_parquet(PATH_PATH_PROVEN)
    logger.info(
        "df_path_proven: patients=%d, studies=%d, records=%d",
        df_path_proven["PATIENTID"].nunique(),
        df_path_proven["STUDYID"].nunique(),
        len(df_path_proven),
    )
    df_path_proven = df_path_proven.rename(columns={"Laterality_Cleaned": "Laterality_Biopsy"})

    _vc(df_path_proven["timing_category"], "timing_category")
    _vc(df_path_proven["BI-RADS Category"], "BI-RADS Category")
    _vc(df_path_proven["Cancer_Description"], "Cancer_Description")
    _vc(df_path_proven["Laterality_Biopsy"], "Laterality_Biopsy")

    # Filter to trigger/recent_prior + drop indeterminate/unknown pathology
    # FIRST, so select_one_study_per_patient operates on biopsy-event rows
    # only (otherwise unrelated non-biopsy rows could win the dedupe).
    df_trigger_recent_prior = df_path_proven[
        df_path_proven["timing_category"].isin(["trigger", "recent_prior"])
    ]
    df_trigger_recent_prior = df_trigger_recent_prior.drop_duplicates(
        subset=["PATIENTID", "STUDYID", "biopsy_study_id"]
    )
    df_trigger_recent_prior = df_trigger_recent_prior[
        ~df_trigger_recent_prior["Cancer_Description"].isin(
            ["path_indeterminate", "path_unknown"]
        )
    ]
    logger.info(
        "trigger/recent_prior (pre-dedupe): patients=%d, studies=%d, records=%d",
        df_trigger_recent_prior["PATIENTID"].nunique(),
        df_trigger_recent_prior["STUDYID"].nunique(),
        len(df_trigger_recent_prior),
    )

    # Now collapse multi-study patients to a single biopsy event:
    # most-severe Cancer_Description, ties broken by most recent linked_study_date.
    patient_study_counts = df_trigger_recent_prior.groupby("PATIENTID")["STUDYID"].nunique()
    n_multi = (patient_study_counts > 1).sum()
    logger.info("Patients with multiple biopsy studies (pre-collapse): %d", n_multi)
    df_trigger_recent_prior = select_one_study_per_patient(df_trigger_recent_prior)
    logger.info(
        "After select_one_study_per_patient: patients=%d, studies=%d, records=%d",
        df_trigger_recent_prior["PATIENTID"].nunique(),
        df_trigger_recent_prior["STUDYID"].nunique(),
        len(df_trigger_recent_prior),
    )

    # Drop STUDYIDs that map to multiple PATIENTIDs (data-quality issue, rare).
    study_patient_counts = df_trigger_recent_prior.groupby("STUDYID")["PATIENTID"].nunique()
    studies_multiple_patients = study_patient_counts[study_patient_counts > 1].index
    if len(studies_multiple_patients):
        logger.info(
            "Dropping %d STUDYIDs assigned to multiple patients in df_trigger_recent_prior.",
            len(studies_multiple_patients),
        )
        df_trigger_recent_prior = df_trigger_recent_prior[
            ~df_trigger_recent_prior["STUDYID"].isin(studies_multiple_patients)
        ]

    # Optional: augment the biopsied cohort with a separate mesh-path-proven
    # source. Adds disjoint patients (overlap dropped); fills inferred
    # columns (timing_category, BI-RADS, etc.) so the merged frame flows
    # through the rest of the pipeline unchanged.
    if include_mesh_path_proven:
        df_trigger_recent_prior = merge_mesh_path_proven(df_trigger_recent_prior)

    # ----------------------------------------------------------------- #
    # Screening reference: image-level metadata + view validity flags   #
    # ----------------------------------------------------------------- #
    df_all_screening = pd.read_parquet(PATH_ALL_SCREENING)
    df_all_screening = df_all_screening[[
        "PATIENTID", "STUDYID", "SOPINSTANCEUID",
        "VIEWPOSITION", "IMAGELATERALITY", "EXAM_TYPE",
        "MANUFACTURER", "SERIESDESCRIPTION",
    ]].drop_duplicates()
    # Rename image-meta MANUFACTURER → MANUFACTURER_DICOM so the column
    # provenance is unambiguous downstream (RIS-derived MANUFACTURER is dropped).
    df_all_screening = df_all_screening.rename(columns={"MANUFACTURER": "MANUFACTURER_DICOM"})
    _vc(df_all_screening["IMAGELATERALITY"], "IMAGELATERALITY (all_screening)")

    df_valid_views = add_view_validity(df_all_screening)

    # MG_modality (Synth2D vs FFDM vs both within a study). Compute on
    # the *valid-view subset only* — restrict to CC/CCID/MLO/MLOID
    # × IMAGELATERALITY ∈ {L,R} before classifying, so a study's
    # MG_modality reflects only the images that survive the bilateral
    # filter and end up in the matched CSV. Otherwise an XCCL spot-
    # compression with a different modality could mis-tag a study as
    # mixed.
    from utils.mg_modality import classify_image_modality, study_level_mg_modality
    df_valid_image_rows = df_all_screening[
        df_all_screening["VIEWPOSITION"].isin(CC_VIEWS | MLO_VIEWS)
        & df_all_screening["IMAGELATERALITY"].isin(["L", "R"])
    ].copy()
    df_image_modality = classify_image_modality(df_valid_image_rows)
    df_mg_modality = study_level_mg_modality(df_image_modality)

    df_all_screening = (
        df_all_screening.drop_duplicates(
            subset=["PATIENTID", "STUDYID", "EXAM_TYPE", "MANUFACTURER_DICOM"]
        )[["PATIENTID", "STUDYID", "EXAM_TYPE", "MANUFACTURER_DICOM"]]
        .merge(df_valid_views, on="STUDYID", how="left")
        .merge(df_mg_modality, on="STUDYID", how="left")
    )
    logger.info(
        "df_all_screening (study-level + view validity): patients=%d, studies=%d, records=%d",
        df_all_screening["PATIENTID"].nunique(),
        df_all_screening["STUDYID"].nunique(),
        len(df_all_screening),
    )
    bilat_frac = (
        df_all_screening["has_valid_views_L"] & df_all_screening["has_valid_views_R"]
    ).sum() / len(df_all_screening)
    logger.info("Proportion of studies with valid bilateral views: %.4f", bilat_frac)

    # Collapse remaining (PATIENTID, STUDYID) duplicates (manufacturer multiplicity).
    df_all_screening = df_all_screening.drop_duplicates(subset=["PATIENTID", "STUDYID"])

    # Drop STUDYIDs assigned to multiple patients.
    studyid_patient_counts = df_all_screening.groupby("STUDYID")["PATIENTID"].nunique()
    studyids_multi = studyid_patient_counts[studyid_patient_counts > 1].index
    if len(studyids_multi):
        logger.info(
            "Dropping %d STUDYIDs assigned to multiple patients in df_all_screening.",
            len(studyids_multi),
        )
        df_all_screening = df_all_screening[~df_all_screening["STUDYID"].isin(studyids_multi)]

    # ----------------------------------------------------------------- #
    # Labels: BI-RADS + Breast Density                                  #
    # ----------------------------------------------------------------- #
    df_all_labels = pd.read_parquet(PATH_ALL_LABELS)
    df_all_labels = df_all_labels.rename(columns={"Study ID": "STUDYID"})
    df_all_labels = df_all_labels.drop_duplicates(
        subset=["STUDYID", "BI-RADS Category", "Breast Density"]
    )
    df_all_labels = df_all_labels[["STUDYID", "BI-RADS Category", "Breast Density"]]

    # Drop STUDYIDs with conflicting BI-RADS or Density values.
    studyid_birads_counts = df_all_labels.groupby("STUDYID")["BI-RADS Category"].nunique()
    studyid_density_counts = df_all_labels.groupby("STUDYID")["Breast Density"].nunique()
    studyids_to_drop = (
        set(studyid_birads_counts[studyid_birads_counts > 1].index)
        | set(studyid_density_counts[studyid_density_counts > 1].index)
    )
    df_all_labels = df_all_labels[~df_all_labels["STUDYID"].isin(studyids_to_drop)]
    df_all_labels = df_all_labels.drop_duplicates(subset=["STUDYID"])

    # Cohort exclusions: BI-RADS NaN/12, Density Unknown.
    df_all_labels = df_all_labels[~df_all_labels["BI-RADS Category"].isin([None, 12])]
    df_all_labels = df_all_labels[~df_all_labels["Breast Density"].isin(["Unknown"])]
    logger.info("df_all_labels (cleaned): records=%d", len(df_all_labels))
    _vc(df_all_labels["BI-RADS Category"], "BI-RADS Category (cleaned)")
    _vc(df_all_labels["Breast Density"], "Breast Density (cleaned)")

    # ----------------------------------------------------------------- #
    # RIS demographics                                                  #
    # ----------------------------------------------------------------- #
    df_ris_source = pd.read_parquet(PATH_RIS_DEMOG)
    # MANUFACTURER intentionally dropped from RIS — image-meta (DICOM) is the
    # authoritative source and lives in df_all_screening as MANUFACTURER_DICOM.
    df_ris_demog = (
        df_ris_source[[
            "HOPPR_PATIENT_ID", "HOPPR_STUDY_ID", "SEX", "AGE", "RACE",
            "COMPLETED_DTTM_SHIFTED", "REGION", "FACILITY",
        ]]
        .copy()
        .drop_duplicates(subset=["HOPPR_PATIENT_ID", "HOPPR_STUDY_ID"])
        .rename(columns={
            "AGE": "PATIENTAGE",
            "HOPPR_STUDY_ID": "STUDYID",
            "HOPPR_PATIENT_ID": "PATIENTID_RIS",
        })
    )
    logger.info(
        "df_ris_demog: patients=%d, studies=%d, valid_age_frac=%.4f",
        df_ris_demog["PATIENTID_RIS"].nunique(),
        df_ris_demog["STUDYID"].nunique(),
        df_ris_demog["PATIENTAGE"].notna().sum() / len(df_ris_demog),
    )

    # Merge labels ← RIS demog ← screening (valid views + EXAM_TYPE + MANUFACTURER_DICOM).
    df_all_labels = df_all_labels.merge(df_ris_demog, on="STUDYID", how="left")
    df_all_labels = df_all_labels.merge(df_all_screening, on="STUDYID", how="left")
    logger.info("df_all_labels after merges: records=%d", len(df_all_labels))

    # Split BI-RADS 0 (potential index exams) vs BI-RADS 1/2 (screening negatives).
    df_all_labels_BIRADS0 = df_all_labels[df_all_labels["BI-RADS Category"] == 0].drop(
        columns="BI-RADS Category"  # cases df brings its own BI-RADS Category
    )
    df_all_labels_BIRADS12 = df_all_labels[df_all_labels["BI-RADS Category"].isin([1, 2])]
    df_screening_labels = df_all_labels_BIRADS12.copy()
    logger.info(
        "df_screening_labels (BIRADS 1/2): records=%d", len(df_screening_labels)
    )

    # Attach BI-RADS-0 labels to the cases cohort (gives Density + valid view info).
    df_trigger_recent_prior_withLabels = pd.merge(
        df_trigger_recent_prior,
        df_all_labels_BIRADS0,
        on=["STUDYID", "PATIENTID"],
        how="inner",
    )
    logger.info(
        "df_trigger_recent_prior_withLabels: records=%d",
        len(df_trigger_recent_prior_withLabels),
    )

    # ----------------------------------------------------------------- #
    # Recover Laterality_Biopsy="U" from BI-RADS-0 annotations table    #
    # ----------------------------------------------------------------- #
    get_birads0 = _load_birads0_laterality_fn()
    df_birads0_laterality = get_birads0()
    df_birads0_laterality = df_birads0_laterality.rename(
        columns={"IMAGELATERALITY": "Laterality_Biopsy"}
    )

    df_lat_unknown = df_trigger_recent_prior_withLabels[
        df_trigger_recent_prior_withLabels["Laterality_Biopsy"] == "U"
    ].drop(columns=["Laterality_Biopsy"])
    df_lat_unknown = df_lat_unknown.merge(
        df_birads0_laterality[["STUDYID", "Laterality_Biopsy"]],
        on="STUDYID", how="inner",
    )
    df_lat_unknown = df_lat_unknown.drop_duplicates(
        subset=["PATIENTID", "STUDYID", "biopsy_study_id"]
    )
    df_lat_known = df_trigger_recent_prior_withLabels[
        df_trigger_recent_prior_withLabels["Laterality_Biopsy"] != "U"
    ]
    df_trigger_recent_prior_withLabels = pd.concat(
        [df_lat_known, df_lat_unknown], ignore_index=True
    )
    logger.info(
        "After laterality recovery: patients=%d, studies=%d, records=%d",
        df_trigger_recent_prior_withLabels["PATIENTID"].nunique(),
        df_trigger_recent_prior_withLabels["STUDYID"].nunique(),
        len(df_trigger_recent_prior_withLabels),
    )
    _vc(df_trigger_recent_prior_withLabels["Laterality_Biopsy"], "Laterality_Biopsy (post-recovery)")

    # ----------------------------------------------------------------- #
    # Build study-level screening frame for matching                    #
    # ----------------------------------------------------------------- #
    df_screening_labels_studyLevel = df_screening_labels[[
        "STUDYID", "PATIENTID", "PATIENTAGE", "SEX", "RACE",
        "MANUFACTURER_DICOM", "MG_modality",
        "COMPLETED_DTTM_SHIFTED", "REGION", "FACILITY",
        "BI-RADS Category", "Breast Density",
        "has_valid_views_L", "has_valid_views_R",
        "valid_images_L", "valid_images_R",
    ]].copy().drop_duplicates(subset=["STUDYID", "PATIENTID"])

    # Hard guard: one record per STUDYID, no STUDYID assigned to multiple patients.
    studyid_patient_counts = df_screening_labels_studyLevel.groupby(
        "STUDYID"
    )["PATIENTID"].nunique()
    studyids_multi = studyid_patient_counts[studyid_patient_counts > 1].index
    if len(studyids_multi):
        logger.info(
            "Dropping %d STUDYIDs with multiple patients in screening_labels_studyLevel.",
            len(studyids_multi),
        )
        df_screening_labels_studyLevel = df_screening_labels_studyLevel[
            ~df_screening_labels_studyLevel["STUDYID"].isin(studyids_multi)
        ]

    # COMPLETED_DTTM_SHIFTED → STUDYDATE (the date column build_longitudinal_frame
    # expects on the longitudinal side).
    df_screening_labels_studyLevel = df_screening_labels_studyLevel.rename(
        columns={"COMPLETED_DTTM_SHIFTED": "STUDYDATE"}
    ).dropna(subset=["STUDYDATE"])
    logger.info(
        "df_screening_labels_studyLevel: records=%d (post STUDYDATE drop)",
        len(df_screening_labels_studyLevel),
    )

    # ----------------------------------------------------------------- #
    # Build cases index frame (one row per biopsy event per patient)    #
    # ----------------------------------------------------------------- #
    df_path_index = df_trigger_recent_prior_withLabels[[
        "PATIENTID", "STUDYID", "linked_study_date", "biopsy_study_id",
        "Laterality_Biopsy", "Cancer", "Cancer_Description", "subtype",
        "BI-RADS Category", "Breast Density", "PATIENTAGE", "SEX", "RACE",
        "MANUFACTURER_DICOM", "MG_modality", "REGION", "FACILITY",
        "has_valid_views_L", "has_valid_views_R",
        "valid_images_L", "valid_images_R",
    ]].copy()

    df_screening_labels_studyLevel = df_screening_labels_studyLevel[[
        "PATIENTID", "STUDYID", "STUDYDATE", "BI-RADS Category", "Breast Density",
        "PATIENTAGE", "SEX", "RACE", "MANUFACTURER_DICOM", "MG_modality",
        "REGION", "FACILITY",
        "has_valid_views_L", "has_valid_views_R", "valid_images_L", "valid_images_R",
    ]].copy()

    assert df_path_index["PATIENTID"].nunique() == df_path_index["STUDYID"].nunique(), (
        "df_path_index has multiple studies per patient or multiple records per (PATIENTID, STUDYID)."
    )
    assert df_path_index["STUDYID"].nunique() == len(df_path_index), (
        "df_path_index has more than one record per STUDYID."
    )
    assert (
        df_screening_labels_studyLevel["STUDYID"].nunique()
        == len(df_screening_labels_studyLevel)
    ), "df_screening_labels_studyLevel has more than one record per STUDYID."

    # MANUFACTURER_CATEGORY (bucketed) derived from image-meta MANUFACTURER_DICOM
    # on both index and longitudinal sides — always image-meta-derived for consistency.
    df_path_index["MANUFACTURER_CATEGORY"] = df_path_index["MANUFACTURER_DICOM"].apply(map_manufacturer)
    df_screening_labels_studyLevel["MANUFACTURER_CATEGORY"] = (
        df_screening_labels_studyLevel["MANUFACTURER_DICOM"].apply(map_manufacturer)
    )

    # ----------------------------------------------------------------- #
    # Bilateral-only restriction                                        #
    # ----------------------------------------------------------------- #
    # Require both lateralities (CC + MLO each) on every study that enters
    # the longitudinal frame. This guarantees that any matched control's
    # studies have the same image lateralities available as the case's,
    # without per-study laterality matching downstream.
    n_path_pre = len(df_path_index)
    df_path_index = df_path_index[
        df_path_index["has_valid_views_L"] & df_path_index["has_valid_views_R"]
    ].copy()
    logger.info(
        "Bilateral filter on cases (df_path_index): %d → %d rows (%d dropped)",
        n_path_pre, len(df_path_index), n_path_pre - len(df_path_index),
    )

    n_scr_pre = len(df_screening_labels_studyLevel)
    df_screening_labels_studyLevel = df_screening_labels_studyLevel[
        df_screening_labels_studyLevel["has_valid_views_L"]
        & df_screening_labels_studyLevel["has_valid_views_R"]
    ].copy()
    logger.info(
        "Bilateral filter on screening pool: %d → %d rows (%d dropped)",
        n_scr_pre, len(df_screening_labels_studyLevel),
        n_scr_pre - len(df_screening_labels_studyLevel),
    )

    # ----------------------------------------------------------------- #
    # Cases longitudinal frame                                          #
    # ----------------------------------------------------------------- #
    df_longitudinal = build_longitudinal_frame(
        df_path_index,
        df_screening_labels_studyLevel,
        index_date_col="linked_study_date",
        longitudinal_date_col="STUDYDATE",
    )
    df_longitudinal = df_longitudinal.sort_values(
        ["PATIENTID", "exam_order"]
    ).reset_index(drop=True)
    df_longitudinal.to_parquet(output_dir / "cases_longitudinal_unfiltered.parquet")
    logger.info(
        "df_longitudinal (cases, unfiltered): records=%d, patients=%d, studies=%d",
        len(df_longitudinal),
        df_longitudinal["PATIENTID"].nunique(),
        df_longitudinal["STUDYID"].nunique(),
    )

    df_longitudinal_2priors = filter_annual_priors(
        df_longitudinal, min_days=min_days, max_days=max_days, min_consecutive=2
    )
    logger.info(
        "df_longitudinal_2priors: records=%d, patients=%d",
        len(df_longitudinal_2priors),
        df_longitudinal_2priors["PATIENTID"].nunique(),
    )
    _vc(df_longitudinal_2priors["Cancer_Description"], "Cancer_Description (final cohort)")
    logger.info(
        "Length of consecutive prior chains (cases):\n%s",
        df_longitudinal_2priors.groupby("PATIENTID")["exam_order"].max().value_counts(),
    )

    # ----------------------------------------------------------------- #
    # Controls longitudinal frame                                       #
    # ----------------------------------------------------------------- #
    df_controls = df_screening_labels_studyLevel[
        ~df_screening_labels_studyLevel["PATIENTID"].isin(df_path_index["PATIENTID"])
    ].copy()
    logger.info(
        "df_controls (pure screening pool): patients=%d, records=%d",
        df_controls["PATIENTID"].nunique(), len(df_controls),
    )

    # Pseudo-index: most recent screening study per patient.
    df_controls["STUDYDATE"] = parse_dates(df_controls["STUDYDATE"], "STUDYDATE")
    df_controls = df_controls.dropna(subset=["STUDYDATE"])
    df_controls = df_controls.sort_values(
        ["PATIENTID", "STUDYDATE", "STUDYID"], ascending=[True, False, True]
    )
    df_controls["exam_order"] = df_controls.groupby("PATIENTID").cumcount()
    df_controls_index = df_controls[df_controls["exam_order"] == 0].copy()
    assert (
        df_controls_index["PATIENTID"].nunique()
        == df_controls_index["STUDYID"].nunique()
    ), "df_controls_index has duplicate (PATIENTID, STUDYID) records."
    logger.info(
        "df_controls_index: patients=%d, studies=%d",
        df_controls_index["PATIENTID"].nunique(),
        df_controls_index["STUDYID"].nunique(),
    )

    df_controls_longitudinal = build_longitudinal_frame(
        df_controls_index, df_controls,
        index_date_col="STUDYDATE", longitudinal_date_col="STUDYDATE",
    )
    df_controls_longitudinal.to_parquet(output_dir / "controls_longitudinal_unfiltered.parquet")
    logger.info(
        "df_controls_longitudinal (unfiltered): records=%d, patients=%d, studies=%d",
        len(df_controls_longitudinal),
        df_controls_longitudinal["PATIENTID"].nunique(),
        df_controls_longitudinal["STUDYID"].nunique(),
    )

    df_controls_longitudinal_2priors = filter_annual_priors(
        df_controls_longitudinal, min_days=min_days, max_days=max_days, min_consecutive=2,
    )
    logger.info(
        "df_controls_longitudinal_2priors: records=%d, patients=%d, studies=%d",
        len(df_controls_longitudinal_2priors),
        df_controls_longitudinal_2priors["PATIENTID"].nunique(),
        df_controls_longitudinal_2priors["STUDYID"].nunique(),
    )

    # ----------------------------------------------------------------- #
    # Save                                                              #
    # ----------------------------------------------------------------- #
    cases_out = output_dir / f"cases_longitudinal_2priorsminimum{suffix}.parquet"
    ctrls_out = output_dir / f"controls_longitudinal_2priorsminimum{suffix}.parquet"
    df_longitudinal_2priors.to_parquet(cases_out)
    df_controls_longitudinal_2priors.to_parquet(ctrls_out)
    logger.info("Wrote %s", cases_out)
    logger.info("Wrote %s", ctrls_out)

    # Final regression-check counts.
    logger.info(
        "Length of consecutive prior chains (controls):\n%s",
        df_controls_longitudinal_2priors.groupby("PATIENTID")["exam_order"].max().value_counts(),
    )
    logger.info(
        "Length of consecutive prior chains (cases):\n%s",
        df_longitudinal_2priors.groupby("PATIENTID")["exam_order"].max().value_counts(),
    )
    logger.info(
        "Valid age fraction (controls): %.4f",
        df_controls_longitudinal_2priors["PATIENTAGE"].notna().sum()
        / len(df_controls_longitudinal_2priors),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Curate longitudinal cohort (cases + controls).")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to write longitudinal parquets. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--min-days", type=int, default=335,
        help="Lower bound on consecutive-exam interval, in days. "
             "Default 335 matches the augmented manuscript cohort; the "
             "prior tight window used 365.",
    )
    parser.add_argument(
        "--max-days", type=int, default=425,
        help="Upper bound on consecutive-exam interval, in days. "
             "Default 425 preserves the original tight window; raise (e.g. 800) to admit "
             "patients who skipped a year.",
    )
    parser.add_argument(
        "--include-mesh-path-proven", action=argparse.BooleanOptionalAction, default=True,
        help="Augment the biopsied cohort with mesh-path-proven studies "
             "from the cancer-classification CSV. Adds disjoint patients "
             "(overlap with the default trigger/recent_prior cohort is "
             "dropped). Output suffix gets `_mesh` component. Default True "
             "(manuscript cohort); pass --no-include-mesh-path-proven to "
             "reproduce the pre-augmentation cohort.",
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
        output_dir=args.output_dir,
        min_days=args.min_days,
        max_days=args.max_days,
        include_mesh_path_proven=args.include_mesh_path_proven,
    )
