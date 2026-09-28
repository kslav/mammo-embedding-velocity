"""Measure η² (between-group / total trace) for arbitrary categorical labels
on a precomputed exam_embeddings.parquet. Lightweight — no .pt re-loads.

Used as a fast post-hoc check after 04a/04b have run, e.g. to confirm
ComBat removed manufacturer variance while preserving Cancer / is_index.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def variance_decomposition(X: np.ndarray, labels: np.ndarray) -> dict:
    X = X.astype(np.float64)
    grand = X.mean(axis=0, keepdims=True)
    ss_total = float(((X - grand) ** 2).sum())
    ss_between = 0.0
    for lab in np.unique(labels):
        mask = labels == lab
        n = int(mask.sum())
        mu = X[mask].mean(axis=0, keepdims=True)
        ss_between += n * float(((mu - grand) ** 2).sum())
    eta2 = ss_between / ss_total if ss_total > 0 else float("nan")
    return {"eta2": eta2, "ss_total": ss_total, "ss_between": ss_between}


def permute_eta2(X: np.ndarray, labels: np.ndarray, n_perm: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    obs = variance_decomposition(X, labels)["eta2"]
    null = np.empty(n_perm)
    for i in range(n_perm):
        null[i] = variance_decomposition(X, rng.permutation(labels))["eta2"]
    p = (float((null >= obs).sum()) + 1) / (n_perm + 1)
    return {
        "eta2_observed": obs,
        "eta2_null_mean": float(null.mean()),
        "eta2_null_p95": float(np.quantile(null, 0.95)),
        "pvalue": p,
    }


def report(parquet: Path, label_cols: list[str], n_perm: int,
           min_levels: int = 2, drop_singletons: bool = True) -> dict:
    df = pd.read_parquet(parquet)
    logger.info("Loaded %s: %d rows, %d patients",
                parquet, len(df), df["PATIENTID"].nunique())

    X = np.stack(df["embedding"].to_numpy()).astype(np.float64)
    out: dict = {"n_rows": len(df), "n_patients": int(df["PATIENTID"].nunique())}
    for col in label_cols:
        if col not in df.columns:
            logger.warning("  %s: not in columns; skipping", col)
            continue
        labels = df[col].astype(str).to_numpy()
        levels = pd.Series(labels).value_counts()
        if drop_singletons:
            keep = levels[levels >= 30].index.tolist()
            mask = pd.Series(labels).isin(keep).to_numpy()
            X_sub, labs_sub = X[mask], labels[mask]
        else:
            X_sub, labs_sub = X, labels
        if len(np.unique(labs_sub)) < min_levels:
            logger.warning("  %s: <%d levels after singleton drop; skipping",
                           col, min_levels)
            continue
        var = variance_decomposition(X_sub, labs_sub)
        rec = {"levels": dict(levels), "eta2": var["eta2"]}
        if n_perm > 0:
            rec.update(permute_eta2(X_sub, labs_sub, n_perm=n_perm, seed=0))
        out[col] = rec
        logger.info("  %s: η²=%.4f%s  levels=%s", col, rec["eta2"],
                    f", p={rec['pvalue']:.4g}" if "pvalue" in rec else "",
                    {k: int(v) for k, v in dict(levels).items()})
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-dir", type=Path, required=True,
                   help="Dir with {breast,study}/exam_embeddings.parquet (pre-ComBat).")
    p.add_argument("--combat-dir", type=Path, default=None,
                   help="Optional dir with post-ComBat parquets for side-by-side.")
    p.add_argument("--granularities", nargs="+", default=["breast", "study"])
    # `group` is patient-level (malignant / biopsy_negative / control), set by
    # build_group_map in cohort_embeddings; it already carries the case/control
    # contrast that `Cancer` would, but without the NaN-on-priors mess.
    p.add_argument("--label-cols", nargs="+",
                   default=["MANUFACTURER_CATEGORY", "is_index", "group"])
    p.add_argument("--n-perm", type=int, default=1000)
    p.add_argument("--out-json", type=Path, default=None)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    summary: dict = {}
    for gran in args.granularities:
        logger.info("=== %s — baseline ===", gran)
        base = report(
            args.baseline_dir / gran / "exam_embeddings.parquet",
            args.label_cols, args.n_perm,
        )
        summary[f"{gran}/baseline"] = base
        if args.combat_dir is not None:
            logger.info("=== %s — combat ===", gran)
            combat = report(
                args.combat_dir / gran / "exam_embeddings.parquet",
                args.label_cols, args.n_perm,
            )
            summary[f"{gran}/combat"] = combat

    if args.out_json is None:
        args.out_json = (args.combat_dir or args.baseline_dir) / "eta2_on_parquet.json"
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    logger.info("Wrote %s", args.out_json)


if __name__ == "__main__":
    main()
