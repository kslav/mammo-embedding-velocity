"""MG_modality classifier — Synth2D vs FFDM from SERIESDESCRIPTION.

Two functions:
- ``classify_image_modality`` (per-image): port of
  ``synth2d_vs_trad2d`` from ``mammo_data_processing/utils.py``. Adds an
  ``MG_modality`` column with values ``{"Synth2d", "FFDM"}`` based on
  regex matches against ``SERIESDESCRIPTION``. Optional ``include_tomo``
  flag also adds an ``is_tomo`` column.

- ``study_level_mg_modality`` (per-study): collapses the per-image
  classification to one row per STUDYID with ``MG_modality ∈
  {"Synth2d", "FFDM", "S2d_FFDM"}``. The third value tags studies where
  both Synth2d and FFDM images coexist within the same exam.

Pulled into this codebase so the cohort pipeline doesn't have to import
the sibling ``mammo_data_processing`` repo (which pulls polars at
module load — see ``utils/birads0_laterality.py`` for the analogous
case). Patterns are kept verbatim from the source so any pull from
upstream stays mergeable.
"""

from __future__ import annotations

import logging
import re

import pandas as pd

logger = logging.getLogger(__name__)

# Synth2D detection patterns (Hologic: C-View, S-View; GE: V-Preview, 2D_PROC).
# Verbatim from `mammo_data_processing.utils.synth2d_vs_trad2d`.
_SYNTHETIC_PATTERNS = [
    r"c[\s\-]?view",           # matches "C-View", "C View", "CView"
    r"\bintelligent\b",
    r"v[\s\-]?preview",        # matches "V-Preview", "V Preview", "VPreview"
    r"s[\s\-]?view",
    r"\bsynth(?:e)?sized?\b",  # matches "synthesized", "synthetic" (non-capturing)
    r"\b2D_PROC\b",
]
_SYNTHETIC_REGEX = re.compile("|".join(_SYNTHETIC_PATTERNS), flags=re.IGNORECASE)

_TOMO_PATTERNS = [
    r"\btomosynthesis\b",
    r"\bdbt\b",
    r"\b3[\s\-]?d\b",
    r"^\s*nan\s*$",
]
_TOMO_REGEX = re.compile("|".join(_TOMO_PATTERNS), flags=re.IGNORECASE)


def classify_image_modality(
    df: pd.DataFrame, include_tomo: bool = False
) -> pd.DataFrame:
    """Add ``MG_modality`` column (per image) based on ``SERIESDESCRIPTION``.

    Parameters
    ----------
    df : DataFrame
        Image-level rows. Must contain ``SERIESDESCRIPTION``.
    include_tomo : bool
        If True, also add an ``is_tomo`` bool column flagging tomosynthesis
        / DBT / 3D-tagged images. Default False (the manuscript filters
        to standard screening before this point).

    Returns
    -------
    DataFrame
        Copy of ``df`` with an added ``MG_modality`` column of
        ``{"Synth2d", "FFDM"}`` strings (per image).
    """
    if "SERIESDESCRIPTION" not in df.columns:
        raise ValueError("classify_image_modality requires column SERIESDESCRIPTION")

    out = df.copy()
    is_synth = out["SERIESDESCRIPTION"].str.contains(_SYNTHETIC_REGEX, na=False)
    out["MG_modality"] = is_synth.map({True: "Synth2d", False: "FFDM"})

    if include_tomo:
        out["is_tomo"] = out["SERIESDESCRIPTION"].str.contains(_TOMO_REGEX, na=False)
        logger.info("MG_modality + is_tomo image-level counts:")
        logger.info("\n%s", out["MG_modality"].value_counts(dropna=False).to_string())
        logger.info("\n%s", out["is_tomo"].value_counts(dropna=False).to_string())
    else:
        logger.info("MG_modality image-level counts:\n%s",
                    out["MG_modality"].value_counts(dropna=False).to_string())
    return out


def study_level_mg_modality(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-image ``MG_modality`` to one row per ``STUDYID``.

    Output values:
      - ``"FFDM"``      — every image in the study is FFDM
      - ``"Synth2d"``   — every image in the study is Synth2d
      - ``"S2d_FFDM"``  — both modalities present within the study

    Useful as a study-level batch label for downstream harmonization
    or descriptive tables. Requires ``classify_image_modality`` to
    have been called first (so ``MG_modality`` is on the input).
    """
    needed = {"STUDYID", "MG_modality"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"study_level_mg_modality missing columns: {missing}")

    # Per-study set of distinct modalities seen across images.
    per_study = (
        df.drop_duplicates(subset=["STUDYID", "MG_modality"])
        .groupby("STUDYID")["MG_modality"]
        .apply(lambda s: frozenset(s.dropna()))
        .reset_index(name="_modality_set")
    )

    def _label(s: frozenset) -> str:
        if s == frozenset(["FFDM"]):
            return "FFDM"
        if s == frozenset(["Synth2d"]):
            return "Synth2d"
        if s == frozenset(["FFDM", "Synth2d"]):
            return "S2d_FFDM"
        # Any unexpected set (empty or unknown) falls back to "Unknown" so
        # the column is well-defined for every STUDYID downstream consumers
        # see.
        return "Unknown"

    per_study["MG_modality"] = per_study["_modality_set"].apply(_label)
    out = per_study[["STUDYID", "MG_modality"]]
    logger.info("MG_modality study-level counts:\n%s",
                out["MG_modality"].value_counts(dropna=False).to_string())
    return out
