"""BI-RADS-0 laterality recovery from gemini-flash + manual annotation tables.

Ported from `mammo_data_processing.utils_read_datasets.get_birads0_with_laterality_from_annotations`
to remove the upstream `import polars` (sibling repo loads polars at module import).

Returns a per-(STUDYID, IMAGELATERALITY) frame restricted to BI-RADS=0 rows,
with IMAGELATERALITY normalized to 'L'/'R' (and 'both' rows duplicated as
one L row + one R row).
"""
from __future__ import annotations

import json
import logging

import pandas as pd

logger = logging.getLogger(__name__)


# Default annotation table paths — frozen here for reproducibility. The
# corresponding files in /path/to/datasets are point-in-time exports; pinning
# the paths keeps step 01 deterministic across reruns.
DEFAULT_ANNOTATION_PATH = (
    "/path/to/datasets/mammo/mammo_hoppr/labels/birads_breast_density_mesh/"
    "annotation_birads_breast_density_07_23_2025.csv"
)
DEFAULT_MAMMO_ENTITIES_PATH = (
    "/path/to/datasets/mammo/mammo_hoppr/labels/birads_breast_density_mesh/"
    "initial_br0_ann_JB_mammo_entities_converted07302025.parquet"
)
DEFAULT_MAMMO_ENTITIES_PATH2 = (
    "/path/to/datasets/mammo/mammo_hoppr/labels/birads_breast_density_mesh/"
    "mammo_entities_converted_new_br0_lateralities08222025.parquet"
)
DEFAULT_DIAGNOSTIC_NEGATIVE_PATH = (
    "/path/to/datasets/mammo/mammo_hoppr/labels/birads_breast_density_mesh/"
    "modified_screening_reports_using_followup_diagnostic_cleaned_and_added_birads_20260402.csv"
)


def get_birads0_with_laterality_from_annotations(
    annotation_path: str = DEFAULT_ANNOTATION_PATH,
    mammo_entities_path: str = DEFAULT_MAMMO_ENTITIES_PATH,
    mammo_entities_path2: str = DEFAULT_MAMMO_ENTITIES_PATH2,
    correct_birads0_diagnostic_negative_screenings: bool = False,
    diagnostic_negative_path: str = DEFAULT_DIAGNOSTIC_NEGATIVE_PATH,
) -> pd.DataFrame:
    """Extract BI-RADS-0 annotations with laterality from the gemini-flash and
    manual-annotation tables.

    When ``correct_birads0_diagnostic_negative_screenings=True``, BI-RADS=0
    rows are upgraded to the BI-RADS recorded in a follow-up diagnostic
    report (per the ``diagnostic_negative_path`` CSV) when one exists.

    Returns a frame with columns: STUDYID, BI-RADS, IMAGELATERALITY, plus
    any auxiliary columns from the source tables.
    """
    df_annotation = pd.read_csv(annotation_path)
    df_annotation = df_annotation.loc[df_annotation["CREATED_BY"] != "ALIG"]

    df_gemini = pd.read_parquet(mammo_entities_path, engine="pyarrow")
    df_gemini = df_gemini[df_gemini["CREATED_BY"] == "gemini_flash_2.5"]

    df_gemini2 = pd.read_parquet(mammo_entities_path2, engine="pyarrow")
    df_gemini2 = df_gemini2[df_gemini2["CREATED_BY"] == "gemini_flash_2.5"]

    df_annotation = df_annotation[["STUDY_ID", "ANNOTATION_ENTRY"]]
    df_gemini = df_gemini[["STUDY_ID", "ANNOTATION_ENTRY"]]
    df_gemini2 = df_gemini2[["STUDY_ID", "ANNOTATION_ENTRY"]]

    df = pd.concat([df_gemini2, df_gemini, df_annotation], ignore_index=True)
    df = df.drop_duplicates(subset=["STUDY_ID", "ANNOTATION_ENTRY"])
    df = df.rename(columns={"STUDY_ID": "STUDYID"}).reset_index(drop=True)

    df["ANNOTATION_ENTRY"] = df["ANNOTATION_ENTRY"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else x
    )
    df["BI-RADS"] = df["ANNOTATION_ENTRY"].apply(
        lambda x: x.get("bi_rads") if isinstance(x, dict) else None
    )
    df["IMAGELATERALITY"] = df["ANNOTATION_ENTRY"].apply(
        lambda x: x.get("laterality") if isinstance(x, dict) else None
    )
    df = df.drop(columns=["ANNOTATION_ENTRY"])

    df_br0 = df.loc[df["BI-RADS"] == "0"].copy()
    df_br0["BI-RADS"] = df_br0["BI-RADS"].astype(int)
    logger.info(
        "BI-RADS-0 annotations before splitting 'both' into L/R:\n%s",
        df_br0["IMAGELATERALITY"].value_counts(),
    )

    # Split bilateral 'both' rows into one L row and one R row.
    df_both = df_br0[df_br0["IMAGELATERALITY"] == "both"]
    df_left = df_both.copy()
    df_left["IMAGELATERALITY"] = "left"
    df_right = df_both.copy()
    df_right["IMAGELATERALITY"] = "right"
    df_br0 = pd.concat(
        [df_br0[df_br0["IMAGELATERALITY"] != "both"], df_left, df_right],
        ignore_index=True,
    )
    df_br0["IMAGELATERALITY"] = (
        df_br0["IMAGELATERALITY"]
        .str.replace("right", "R", regex=True)
        .str.replace("left", "L", regex=True)
    )
    df_br0 = df_br0.sort_values("STUDYID").reset_index(drop=True)
    logger.info(
        "BI-RADS-0 annotations after L/R split:\n%s",
        df_br0["IMAGELATERALITY"].value_counts(),
    )

    if correct_birads0_diagnostic_negative_screenings:
        logger.info(
            "Upgrading BI-RADS=0 rows to follow-up diagnostic BI-RADS where available."
        )
        df_diag = pd.read_csv(diagnostic_negative_path)
        df_diag = df_diag.rename(columns={"BIRADS_LATERALITY": "IMAGELATERALITY"})
        df_br0 = df_br0.merge(
            df_diag[["STUDYID", "IMAGELATERALITY", "BI-RADS Category"]].rename(
                columns={"BI-RADS Category": "BI-RADS_new"}
            ),
            on=["STUDYID", "IMAGELATERALITY"],
            how="left",
        )
        df_br0["BI-RADS"] = df_br0["BI-RADS_new"].combine_first(df_br0["BI-RADS"])
        df_br0 = df_br0.drop(columns=["BI-RADS_new"])
        logger.info("BI-RADS distribution after upgrade:\n%s",
                    df_br0["BI-RADS"].value_counts(dropna=False))

    return df_br0
