"""
Unit tests for the three harmonization transforms in
src/utils/cohort_embeddings.py.

These run on synthetic embeddings — no real .pt loads, no SLURM. The goal
is to verify the *math* of each step in isolation, plus the all-three
combined path. Real-data smoke testing happens implicitly during
04a/04b runs.

Run with:
    .venv_mammoclip/bin/python -m pytest src/diagnostics/test_harmonization.py -v
or for a quick check without pytest:
    .venv_mammoclip/bin/python src/diagnostics/test_harmonization.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make src/ importable when running directly from the repo root.
# parents[0]=src/diagnostics, parents[1]=src.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.cohort_embeddings import (  # noqa: E402
    apply_baseline_subtraction,
    apply_post_harmonization,
    apply_zscore,
)


# ----------------------------- fixtures ----------------------------- #


def _toy_exam_table(emb_dim: int = 8, n_patients: int = 4, seed: int = 0) -> pd.DataFrame:
    """
    Build a deterministic per-exam table with 3 exams per patient
    (exam_order 0=index, 1=first prior, 2=second prior). 4 patients
    total, half labeled `case` and half `control`.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_patients):
        pid = f"P{p:02d}"
        group = "case" if p < n_patients // 2 else "control"
        for k in (0, 1, 2):
            emb = rng.standard_normal(emb_dim).astype(np.float32) * (3.0 + p)
            rows.append({
                "PATIENTID": pid,
                "STUDYID": f"{pid}_S{k}",
                "exam_order": k,
                "is_index": (k == 0),
                "date_diff": (np.nan if k == 0 else 380.0),
                "group": group,
                "embedding": emb,
            })
    return pd.DataFrame(rows)


# ----------------------------- step 1: L2 norm ----------------------------- #


def test_l2_norm_per_image():
    """
    Step 1 is applied inside `aggregate_exam_embeddings` per-image, so we
    test the math directly: each per-image vector divided by its norm
    must have unit length, and zero vectors must stay zero.
    """
    rng = np.random.default_rng(1)
    embs = rng.standard_normal((10, 32)).astype(np.float32) * 5.0
    embs = np.vstack([embs, np.zeros((1, 32), dtype=np.float32)])  # add a zero vector

    normed = []
    for e in embs:
        n = float(np.linalg.norm(e))
        if n > 0:
            normed.append(e / n)
        else:
            normed.append(e)
    normed = np.stack(normed)

    # Non-zero rows have norm 1
    norms = np.linalg.norm(normed, axis=1)
    np.testing.assert_allclose(norms[:-1], 1.0, atol=1e-5)
    # Zero row stays zero
    np.testing.assert_array_equal(normed[-1], np.zeros(32))


def test_l2_norm_pool_is_not_unit():
    """
    The mean of unit vectors is generally NOT a unit vector — it's the
    centroid on/inside the unit ball. Confirm so callers don't expect
    unit-length per-EXAM embeddings after l2_norm + pool.
    """
    rng = np.random.default_rng(2)
    unit_vecs = rng.standard_normal((4, 32)).astype(np.float32)
    unit_vecs = unit_vecs / np.linalg.norm(unit_vecs, axis=1, keepdims=True)
    pooled = unit_vecs.mean(axis=0)
    norm = float(np.linalg.norm(pooled))
    assert 0.0 < norm < 1.0, f"pooled norm should be in (0, 1), got {norm}"


# ----------------------------- step 2: baseline subtraction ----------------------------- #


def test_baseline_subtraction_zero_at_baseline():
    """After baseline subtraction the highest-exam_order row per patient
    must be the zero vector."""
    df = _toy_exam_table()
    out = apply_baseline_subtraction(df)
    for pid, g in out.groupby("PATIENTID"):
        baseline_row = g.loc[g["exam_order"].idxmax()]
        np.testing.assert_allclose(baseline_row["embedding"], 0.0, atol=1e-6)


def test_baseline_subtraction_preserves_displacements():
    """
    The whole motivation for using baseline subtraction is that
    displacements between consecutive exams are INVARIANT under it.
    """
    df = _toy_exam_table()
    out = apply_baseline_subtraction(df)

    for pid in df["PATIENTID"].unique():
        before = df[df["PATIENTID"] == pid].sort_values("exam_order").reset_index(drop=True)
        after = out[out["PATIENTID"] == pid].sort_values("exam_order").reset_index(drop=True)
        for i in range(len(before) - 1):
            disp_before = before.loc[i, "embedding"] - before.loc[i + 1, "embedding"]
            disp_after = after.loc[i, "embedding"] - after.loc[i + 1, "embedding"]
            np.testing.assert_allclose(disp_before, disp_after, atol=1e-6,
                err_msg=f"displacement mismatch for {pid} interval {i}")


def test_baseline_subtraction_drops_lonely_patients():
    """A patient with only one retained exam can't be baseline-subtracted
    (no baseline distinct from the row); they should be dropped with
    a warning, not crash."""
    df = _toy_exam_table()
    df = df[~((df["PATIENTID"] == "P00") & (df["exam_order"].isin([1, 2])))]
    out = apply_baseline_subtraction(df)
    assert "P00" not in out["PATIENTID"].values, "lonely patient should be dropped"
    assert {"P01", "P02", "P03"}.issubset(out["PATIENTID"].unique()), \
        "non-lonely patients should be retained"


# ----------------------------- step 3: per-dim z-score ----------------------------- #


def test_zscore_index_rows_have_unit_variance():
    """After z-score, the index rows (the population we fit on) have
    per-dimension mean ≈ 0 and *sample* std ≈ 1 (ddof=1, the production
    setting since the statistical-correctness pass on 2026-05-22)."""
    df = _toy_exam_table()
    out = apply_zscore(df)
    idx = out[out["is_index"]].copy()
    X = np.stack(idx["embedding"].values).astype(np.float64)
    # Mean ≈ 0, sample-std ≈ 1 per dim. Use loose tolerances since
    # n=2 cases + 2 controls = 4 samples.
    np.testing.assert_allclose(X.mean(axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(X.std(axis=0, ddof=1), 1.0, atol=1e-5)


def test_zscore_uses_index_fit_for_non_index():
    """Non-index rows are transformed with the SAME mean/std fit on
    index rows, not refit. Verify by reconstructing manually using
    sample-std (ddof=1) to match the production setting."""
    df = _toy_exam_table()
    X_all = np.stack(df["embedding"].values).astype(np.float64)
    idx_mask = df["is_index"].astype(bool).to_numpy()
    mu = X_all[idx_mask].mean(axis=0)
    sigma = X_all[idx_mask].std(axis=0, ddof=1)
    sigma = np.where(sigma > 0, sigma, 1.0)
    expected = (X_all - mu) / sigma

    out = apply_zscore(df)
    actual = np.stack(out["embedding"].values).astype(np.float64)
    np.testing.assert_allclose(actual, expected, atol=1e-5)


def test_zscore_handles_dead_dims():
    """A constant dimension (zero variance) should not produce NaN/inf;
    the implementation replaces sigma=0 with 1 to fall through cleanly."""
    df = _toy_exam_table(emb_dim=4)
    # Force dim 0 to be constant across index rows
    new_embs = []
    for _, row in df.iterrows():
        e = row["embedding"].copy()
        if row["is_index"]:
            e[0] = 7.0
        new_embs.append(e)
    df = df.copy()
    df["embedding"] = new_embs

    out = apply_zscore(df)
    Z = np.stack(out["embedding"].values).astype(np.float64)
    assert np.isfinite(Z).all(), "z-score produced non-finite values on a dead dim"


# ----------------------------- combined ----------------------------- #


def test_apply_post_harmonization_compose_order():
    """
    `apply_post_harmonization` with both flags should equal
    apply_zscore(apply_baseline_subtraction(df)) — baseline first,
    then z-score (matches the documented composition order).
    """
    df = _toy_exam_table()
    expected = apply_zscore(apply_baseline_subtraction(df))
    actual = apply_post_harmonization(df, subtract_baseline=True, zscore_dim=True)

    # Same patients survive
    assert set(actual["PATIENTID"]) == set(expected["PATIENTID"])
    # Same embeddings (sorted to align)
    expected = expected.sort_values(["PATIENTID", "exam_order"]).reset_index(drop=True)
    actual = actual.sort_values(["PATIENTID", "exam_order"]).reset_index(drop=True)
    Xe = np.stack(expected["embedding"].values)
    Xa = np.stack(actual["embedding"].values)
    np.testing.assert_allclose(Xa, Xe, atol=1e-5)


def test_apply_post_harmonization_no_flags_is_identity():
    """With both flags off, `apply_post_harmonization` returns the input
    unchanged (modulo a copy)."""
    df = _toy_exam_table()
    out = apply_post_harmonization(df)
    assert (out["PATIENTID"].values == df["PATIENTID"].values).all()
    Xe = np.stack(df["embedding"].values)
    Xa = np.stack(out["embedding"].values)
    np.testing.assert_allclose(Xa, Xe)


# ----------------------------- main (lightweight runner) ----------------------------- #


if __name__ == "__main__":
    # Allow running without pytest: just call each test function and report.
    import inspect
    tests = [(name, fn) for name, fn in inspect.getmembers(sys.modules[__name__],
                                                          inspect.isfunction)
             if name.startswith("test_")]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  ✓ {name}")
        except Exception as e:
            print(f"  ✗ {name}: {type(e).__name__}: {e}")
            failed.append(name)
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed"
          + (f" — failed: {failed}" if failed else ""))
    sys.exit(1 if failed else 0)
