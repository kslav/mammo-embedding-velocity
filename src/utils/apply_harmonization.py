#!/usr/bin/env python3
"""
Apply harmonization steps 2 (baseline subtraction) and/or 3 (per-dim
z-score) to a precomputed per-exam embedding parquet.

Step 1 (per-image L2 norm) cannot be applied here because it operates on
individual `.pt` files BEFORE mean-pooling. To use step 1, re-run
04a/04b with `--harm-l2-norm`.

Use this script to iterate quickly on baseline/z-score harmonization
without paying the ~10 minute cost of re-aggregating from raw .pt files.

Typical workflow:
  1. Run 04a or 04b once (no flags) to produce exam_embeddings.parquet
     under {output_dir}/{breast,study}/.
  2. Run this script:
       python src/apply_harmonization.py \\
           --in-dir  /path/to/baseline/run \\
           --out-dir /path/to/harmonized/run \\
           --subtract-baseline --zscore-dim
  3. Re-run 04a/04b with `--exam-parquet` pointing at the new
     {breast,study}/exam_embeddings.parquet under --out-dir to recompute
     the statistics (much faster than --embeddings-dir from scratch).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Insert src/ on sys.path so `from utils.cohort_embeddings` resolves whether
# this script is invoked as `python src/utils/apply_harmonization.py` (from
# repo root) or `python apply_harmonization.py` (from src/utils/).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.cohort_embeddings import apply_post_harmonization  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in-dir", type=Path, required=True,
                        help="Dir containing {breast,study}/exam_embeddings.parquet "
                             "produced by 04a or 04b (with no harm flags).")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Dir to write transformed parquets to. "
                             "Same {breast,study}/ subfolder layout as in-dir.")
    parser.add_argument("--granularities", nargs="+",
                        default=["breast", "study"],
                        choices=["breast", "study", "breast_both_sides"])
    parser.add_argument("--subtract-baseline", action="store_true",
                        help="Step 2: subtract patient's earliest-prior embedding.")
    parser.add_argument("--zscore-dim", action="store_true",
                        help="Step 3: per-dim z-score (fit on index rows).")
    parser.add_argument("--combat-manuf", action="store_true",
                        help="Step 4: ComBat batch-correction on "
                             "MANUFACTURER_CATEGORY, preserving group + is_index "
                             "+ time_to_index_years as biological covariates.")
    parser.add_argument("--combat-method",
                        choices=["standard", "longitudinal", "opnested"],
                        default="standard",
                        help="When --combat-manuf is set: 'standard' (default) "
                             "uses cross-sectional NeuroComBat; 'longitudinal' "
                             "uses longCombat (Beer 2020) with a per-patient "
                             "random intercept (requires Python ≥3.10; run "
                             "from .venv, not .venv_mammoclip); 'opnested' "
                             "runs OPNested-ComBat (Horng 2022): tries all "
                             "permutations of --batch-cols and picks the "
                             "one that minimizes residual A-D k-sample "
                             "rejections. Uses our patched cross-sectional "
                             "ComBat internally.")
    parser.add_argument("--batch-cols", nargs="+", default=["MANUFACTURER_CATEGORY"],
                        help="Batch label column(s) for ComBat. "
                             "standard/longitudinal take exactly one column "
                             "(default MANUFACTURER_CATEGORY; the special value "
                             "COMBINED_BATCH computes MANUFACTURER_CATEGORY + '_' + "
                             "MG_modality on the fly). opnested takes two or more "
                             "columns to harmonize on sequentially (e.g. "
                             "MANUFACTURER_CATEGORY MG_modality) — order does not "
                             "matter, all permutations are evaluated. Whenever a "
                             "batch label is MG_modality or COMBINED_BATCH, also "
                             "pass --cases-csv/--controls-csv so MG_modality can "
                             "be joined onto the exam parquet.")
    parser.add_argument("--cases-csv", type=Path, default=None,
                        help="Matched cases CSV to source MG_modality from — "
                             "required when a batch label is COMBINED_BATCH or "
                             "MG_modality.")
    parser.add_argument("--controls-csv", type=Path, default=None,
                        help="Matched controls CSV to source MG_modality from — "
                             "required when a batch label is COMBINED_BATCH or "
                             "MG_modality.")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="Only used when --combat-method longitudinal: "
                             "number of parallel workers for the per-feature "
                             "MixedLM loop. 1 (default) runs serially. Values "
                             "> 1 activate the ProcessPoolExecutor-based "
                             "_patched_long_combat; sensible values 8–32.")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not (args.subtract_baseline or args.zscore_dim or args.combat_manuf):
        logger.warning("No flags set — output will be identical to input.")

    # Validate --batch-cols arity against the combat method.
    if args.combat_manuf:
        if args.combat_method == "opnested":
            if len(args.batch_cols) < 2:
                parser.error("--combat-method opnested requires 2+ --batch-cols "
                             "(e.g. MANUFACTURER_CATEGORY MG_modality)")
        elif len(args.batch_cols) != 1:
            parser.error(f"--combat-method {args.combat_method} takes exactly one "
                         f"--batch-cols entry, got {args.batch_cols}")

    # Load MG_modality once from raw CSVs if any batch label needs it
    # (COMBINED_BATCH is composed from it; MG_modality uses it directly).
    mg_lookup = None
    needs_mg = args.combat_manuf and any(
        c in ("COMBINED_BATCH", "MG_modality") for c in args.batch_cols
    )
    if needs_mg:
        if args.cases_csv is None or args.controls_csv is None:
            parser.error("MG_modality-based harmonization requires --cases-csv and --controls-csv")
        logger.info("loading MG_modality from %s and %s", args.cases_csv, args.controls_csv)
        cases = pd.read_csv(args.cases_csv, usecols=["PATIENTID", "STUDYID", "MG_modality"], low_memory=False)
        ctrls = pd.read_csv(args.controls_csv, usecols=["PATIENTID", "STUDYID", "MG_modality"], low_memory=False)
        mg_lookup = (pd.concat([cases, ctrls], ignore_index=True)
                       .drop_duplicates(["PATIENTID", "STUDYID"]))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for gran in args.granularities:
        in_path = args.in_dir / gran / "exam_embeddings.parquet"
        if not in_path.is_file():
            logger.warning("missing %s, skipping %s", in_path, gran)
            continue
        logger.info("=== %s ===", gran)
        logger.info("loading %s", in_path)
        exam = pd.read_parquet(in_path)
        logger.info("rows in: %d, patients: %d",
                    len(exam), exam["PATIENTID"].nunique())

        batch_col = args.batch_cols[0]  # scalar path (standard/longitudinal)
        # Ensure MG_modality is attached whenever needed.
        if args.combat_manuf and mg_lookup is not None and "MG_modality" not in exam.columns:
            exam = exam.merge(mg_lookup, on=["PATIENTID", "STUDYID"], how="left")
            missing = exam["MG_modality"].isna().sum()
            if missing:
                logger.warning("MG_modality missing on %d rows after merge; treated as 'Unknown'", missing)
                exam["MG_modality"] = exam["MG_modality"].fillna("Unknown")
        if args.combat_manuf and "COMBINED_BATCH" in args.batch_cols:
            exam["COMBINED_BATCH"] = (
                exam["MANUFACTURER_CATEGORY"].astype(str) + "_" + exam["MG_modality"].astype(str)
            )
            logger.info("COMBINED_BATCH counts: %s", exam["COMBINED_BATCH"].value_counts().to_dict())

        # Route OPNested-ComBat to its dedicated helper (multi-batch).
        if args.combat_manuf and args.combat_method == "opnested":
            # Optional pre-4 harmonization steps still apply (baseline/zscore).
            from utils.cohort_embeddings import (  # noqa: E402
                apply_baseline_subtraction, apply_zscore, apply_opnested_combat,
            )
            work = exam
            if args.subtract_baseline:
                work = apply_baseline_subtraction(work)
            if args.zscore_dim:
                work = apply_zscore(work)
            out, order_info = apply_opnested_combat(
                work, batch_cols=list(args.batch_cols),
            )
            # Persist the order-selection diagnostic per-granularity.
            gran_out = args.out_dir / gran
            gran_out.mkdir(parents=True, exist_ok=True)
            order_json = gran_out / "opnested_order.json"
            import json
            with open(order_json, "w") as f:
                json.dump(order_info, f, indent=2, default=str)
            logger.info("wrote %s (chosen order: %s)", order_json, order_info["chosen_order"])
        else:
            out = apply_post_harmonization(
                exam,
                subtract_baseline=args.subtract_baseline,
                zscore_dim=args.zscore_dim,
                combat_manuf=args.combat_manuf,
                combat_method=args.combat_method,
                batch_col=batch_col,
                n_jobs=args.n_jobs,
            )
        logger.info("rows out: %d, patients: %d",
                    len(out), out["PATIENTID"].nunique())

        gran_out = args.out_dir / gran
        gran_out.mkdir(parents=True, exist_ok=True)
        out_path = gran_out / "exam_embeddings.parquet"
        out.to_parquet(out_path)
        logger.info("wrote %s", out_path)

    logger.info("Done. Use --exam-parquet-dir on 04a/04b to consume the transformed "
                "parquets (one per granularity under %s).", args.out_dir)


if __name__ == "__main__":
    main()
