"""Project a per-exam embedding parquet into a truncated PCA basis.

Exploratory: fit PCA (unsupervised, on all exams) on a precomputed
{breast,study}/exam_embeddings.parquet, keep the leading PCs explaining a target
fraction of variance, replace the `embedding` column with the truncated PC
scores, and write a new parquet in the same layout so 04b can run on it
(`--exam-parquet-dir <out> --emb-dim <n_pcs>`).

PCA is fit on the full set of exam embeddings and applied to every row; it does
not use the case/control label, so there is no supervised leakage. All non-
embedding columns are carried through unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", type=Path, required=True,
                    help="Dir with {gran}/exam_embeddings.parquet (already "
                         "L2-normed + long-ComBat).")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--granularities", nargs="+", default=["breast"])
    ap.add_argument("--var-target", type=float, default=0.90,
                    help="Cumulative explained-variance fraction to retain.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for gran in args.granularities:
        exam = pd.read_parquet(args.in_dir / gran / "exam_embeddings.parquet")
        X = np.stack(exam["embedding"].values).astype(np.float64)
        n, d = X.shape

        # Full PCA (mean-centered by sklearn), then pick k for the variance target.
        pca = PCA(svd_solver="full", random_state=0).fit(X)
        cum = np.cumsum(pca.explained_variance_ratio_)
        k = int(np.searchsorted(cum, args.var_target) + 1)
        k = min(k, d)
        scores = pca.transform(X)[:, :k].astype(np.float32)

        out = exam.copy()
        out["embedding"] = list(scores)  # one k-vector per row
        gran_out = args.out_dir / gran
        gran_out.mkdir(parents=True, exist_ok=True)
        out.to_parquet(gran_out / "exam_embeddings.parquet")

        summary[gran] = {"n_rows": n, "orig_dim": d, "n_pcs": k,
                         "var_retained": float(cum[k - 1]),
                         "var_target": args.var_target}
        print(f"[{gran}] {n} rows, {d}-d -> {k} PCs "
              f"({cum[k-1]*100:.1f}% variance retained)")

    (args.out_dir / "pca_truncation.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {args.out_dir}/pca_truncation.json")


if __name__ == "__main__":
    main()
