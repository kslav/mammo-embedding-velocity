"""Benchmark parallel longitudinal ComBat against the saved serial baseline.

Uses the 200-patient MII subset. Compares:
  - byte-level output equality (should be exact match under float32 packing)
  - wall time speedup vs saved serial baseline (4.5 min on the loaded cohort)

Reads the baseline .npz saved by the earlier serial run at
`/path/to/scratch/longcombat_baseline_mii_200pt.npz`.
"""
import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.cohort_embeddings import apply_longitudinal_combat_manuf


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--baseline",
        type=Path,
        default=Path("/path/to/scratch/longcombat_baseline_mii_200pt.npz"),
    )
    p.add_argument(
        "--in-parquet",
        type=Path,
        default=Path(
            "/path/to/scratch/drift_cohort/"
            "drift_analysis_augmented_335_425_mii/breast/exam_embeddings.parquet"
        ),
    )
    p.add_argument("--n-patients", type=int, default=200)
    p.add_argument("--n-jobs", type=int, default=16)
    args = p.parse_args()

    print(f"loading full parquet: {args.in_parquet}")
    df = pd.read_parquet(args.in_parquet)
    print(f"  {len(df)} rows, {df['PATIENTID'].nunique()} patients")

    pids = sorted(df["PATIENTID"].unique())[: args.n_patients]
    sub = df[df["PATIENTID"].isin(pids)].reset_index(drop=True)
    sub_X = np.stack(sub["embedding"].to_numpy())
    print(f"subset: {len(sub)} rows, {len(pids)} patients, {sub_X.shape[1]} features")

    print(f"\nrunning apply_longitudinal_combat_manuf(n_jobs={args.n_jobs})")
    t0 = time.time()
    out = apply_longitudinal_combat_manuf(sub, n_jobs=args.n_jobs)
    elapsed = time.time() - t0
    out_X = np.stack(out["embedding"].to_numpy())

    print(f"\nWALL TIME: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  per-feature: {1000*elapsed/sub_X.shape[1]:.0f} ms/feature")

    print(f"\noutput hash: {hashlib.sha256(out_X.tobytes()).hexdigest()[:16]}")

    if args.baseline.exists():
        print(f"\ncomparing to saved baseline: {args.baseline}")
        with np.load(args.baseline) as npz:
            baseline_X = npz["output"]
            # wall_seconds was saved as a 1-element array; extract the scalar
            baseline_wall = float(np.asarray(npz["wall_seconds"]).ravel()[0])
        print(f"  baseline shape: {baseline_X.shape}, wall {baseline_wall:.1f}s")
        if baseline_X.shape != out_X.shape:
            print(f"  !! shape mismatch — cannot compare byte-level")
            return
        max_diff = float(np.abs(baseline_X.astype(np.float64) - out_X.astype(np.float64)).max())
        mean_diff = float(np.abs(baseline_X.astype(np.float64) - out_X.astype(np.float64)).mean())
        print(f"  max abs diff (parallel vs saved serial): {max_diff:.2e}")
        print(f"  mean abs diff: {mean_diff:.2e}")
        print(f"  identical? {np.array_equal(baseline_X, out_X)}")
        speedup = baseline_wall / elapsed
        print(f"\nSPEEDUP: {speedup:.2f}× (baseline serial {baseline_wall:.1f}s -> "
              f"parallel {elapsed:.1f}s at n_jobs={args.n_jobs})")
    else:
        print(f"\nno baseline at {args.baseline}; skipping comparison")


if __name__ == "__main__":
    main()
