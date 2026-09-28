"""Unit tests for the ComBat harmonization step (apply_combat_manuf).

Tests on synthetic embeddings — no real .pt loads, no SLURM. The goal is
to verify:
  1. ComBat removes the planted batch (manufacturer) shift
  2. Preserved biological covariates (Cancer, is_index) survive correction
  3. Tiny batches are passed through unchanged (no NeuroComBat fit)
  4. Auto-computation of time_to_index_years from date_diff cumsum

Run with:
    .venv_mammoclip/bin/python -m pytest src/diagnostics/test_combat_harmonization.py -v
or:
    .venv_mammoclip/bin/python src/diagnostics/test_combat_harmonization.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.cohort_embeddings import (  # noqa: E402
    _compute_time_to_index_years,
    apply_combat_manuf,
)


# ----------------------------- fixtures ----------------------------- #


def _planted_batch_table(
    emb_dim: int = 16,
    n_patients_per_batch: int = 35,
    n_batches: int = 3,
    n_exams_per_patient: int = 3,
    cancer_signal_strength: float = 0.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Synthetic per-exam table with a planted batch (manufacturer) shift.

    Each batch gets its own additive offset across all dims. We assign
    Cancer=1 to half the patients within each batch so cancer is uncorrelated
    with batch (the design ComBat is built for). When cancer_signal_strength>0
    we also add a separate cancer-specific direction in feature space —
    that's what we want preserved.
    """
    rng = np.random.default_rng(seed)
    batch_offsets = rng.standard_normal((n_batches, emb_dim)) * 5.0  # big planted shifts
    cancer_dir = rng.standard_normal(emb_dim)
    cancer_dir /= np.linalg.norm(cancer_dir)

    rows = []
    for b in range(n_batches):
        for p in range(n_patients_per_batch):
            pid = f"B{b}P{p:03d}"
            cancer = int(p % 2 == 0)  # 50/50 within each batch
            for k in range(n_exams_per_patient):
                emb = rng.standard_normal(emb_dim).astype(np.float64)
                emb += batch_offsets[b]                       # planted batch shift
                emb += cancer * cancer_signal_strength * cancer_dir
                rows.append({
                    "PATIENTID": pid,
                    "STUDYID": f"{pid}_S{k}",
                    "MANUFACTURER_CATEGORY": f"Manuf{b}",
                    "Cancer": cancer,
                    "is_index": (k == 0),
                    "exam_order": k,
                    "date_diff": (np.nan if k == 0 else 380.0),
                    "embedding": emb.astype(np.float32),
                })
    return pd.DataFrame(rows)


def _eta2_between(X: np.ndarray, labels: np.ndarray) -> float:
    """Multivariate eta-squared analog."""
    grand = X.mean(axis=0, keepdims=True)
    ss_total = float(((X - grand) ** 2).sum())
    ss_between = 0.0
    for lab in np.unique(labels):
        m = labels == lab
        n = int(m.sum())
        mu = X[m].mean(axis=0, keepdims=True)
        ss_between += n * float(((mu - grand) ** 2).sum())
    return ss_between / ss_total if ss_total > 0 else float("nan")


# ----------------------------- batch-removal tests ----------------------------- #


def test_combat_removes_planted_batch_shift():
    """η² for MANUFACTURER_CATEGORY drops sharply after ComBat."""
    df = _planted_batch_table(seed=1)
    X_pre = np.stack(df["embedding"].to_numpy()).astype(np.float64)
    eta_pre = _eta2_between(X_pre, df["MANUFACTURER_CATEGORY"].to_numpy())

    out = apply_combat_manuf(df)
    X_post = np.stack(out["embedding"].to_numpy()).astype(np.float64)
    eta_post = _eta2_between(X_post, out["MANUFACTURER_CATEGORY"].to_numpy())

    assert eta_pre > 0.4, f"planted batch should produce large η²; got {eta_pre:.3f}"
    assert eta_post < 0.05, (
        f"ComBat should drive η² toward zero; got pre={eta_pre:.3f}, post={eta_post:.3f}"
    )
    assert eta_post < eta_pre / 5, "ComBat should reduce η² by >5×"


def test_combat_returns_same_shape_and_columns():
    df = _planted_batch_table(seed=2)
    out = apply_combat_manuf(df)
    assert len(out) == len(df)
    assert set(out.columns) >= set(df.columns), "all input columns should survive"
    # Embedding column still ndarrays of the same dim
    e0 = np.asarray(out["embedding"].iloc[0])
    e_in = np.asarray(df["embedding"].iloc[0])
    assert e0.shape == e_in.shape


def test_combat_passes_through_tiny_batches():
    """Rows whose batch has fewer than min_batch_size are passed through
    unchanged; the larger batches get harmonized."""
    df = _planted_batch_table(seed=3)
    # Add a tiny 5-row batch that should be passed through.
    rng = np.random.default_rng(99)
    extra_rows = []
    for k in range(5):
        emb = (rng.standard_normal(16) * 5.0 + 100.0).astype(np.float32)  # far-away mean
        extra_rows.append({
            "PATIENTID": f"TINY_P{k}", "STUDYID": f"TINY_S{k}",
            "MANUFACTURER_CATEGORY": "TinyManuf",
            "Cancer": 0, "is_index": True, "exam_order": 0,
            "date_diff": np.nan, "embedding": emb,
        })
    df_aug = pd.concat([df, pd.DataFrame(extra_rows)], ignore_index=True)
    pre_tiny = np.stack(
        df_aug.loc[df_aug["MANUFACTURER_CATEGORY"] == "TinyManuf", "embedding"].to_numpy()
    )

    out = apply_combat_manuf(df_aug)
    post_tiny = np.stack(
        out.loc[out["MANUFACTURER_CATEGORY"] == "TinyManuf", "embedding"].to_numpy()
    )
    np.testing.assert_allclose(post_tiny, pre_tiny, atol=1e-6,
                               err_msg="tiny batch should not be harmonized")


# ----------------------------- biological signal preservation ----------------------------- #


def test_combat_preserves_cancer_signal():
    """Plant a Cancer-specific direction; verify the Cancer η² survives
    (or is at least not crushed) after ComBat."""
    df = _planted_batch_table(cancer_signal_strength=8.0, seed=4)
    X_pre = np.stack(df["embedding"].to_numpy()).astype(np.float64)
    eta_cancer_pre = _eta2_between(X_pre, df["Cancer"].to_numpy())
    eta_batch_pre = _eta2_between(X_pre, df["MANUFACTURER_CATEGORY"].to_numpy())

    out = apply_combat_manuf(df)
    X_post = np.stack(out["embedding"].to_numpy()).astype(np.float64)
    eta_cancer_post = _eta2_between(X_post, out["Cancer"].to_numpy())
    eta_batch_post = _eta2_between(X_post, out["MANUFACTURER_CATEGORY"].to_numpy())

    # Batch effect goes way down
    assert eta_batch_post < eta_batch_pre / 5, (
        f"batch should be reduced; pre={eta_batch_pre:.3f} post={eta_batch_post:.3f}"
    )
    # Cancer effect is preserved within ~2× of pre (some loss is OK; what we don't
    # want is ComBat *flattening* Cancer along with batch).
    assert eta_cancer_post >= eta_cancer_pre * 0.5, (
        f"Cancer signal should not be destroyed; pre={eta_cancer_pre:.3f} "
        f"post={eta_cancer_post:.3f}"
    )


# ----------------------------- time_to_index_years auto-compute ----------------------------- #


def test_compute_time_to_index_years():
    """Cumulative date_diff per patient, sorted by exam_order ascending,
    divided by 365.25. Index gets 0; deeper priors get larger positive values."""
    df = _planted_batch_table(n_patients_per_batch=2, n_batches=1, seed=5)
    df = df.sort_values(["PATIENTID", "exam_order"]).reset_index(drop=True)
    tti = _compute_time_to_index_years(df)
    df["time_to_index_years"] = tti

    for pid, g in df.groupby("PATIENTID"):
        g = g.sort_values("exam_order")
        ttis = g["time_to_index_years"].tolist()
        # exam_order=0 → 0 years
        assert ttis[0] == 0.0, f"{pid}: index time_to_index_years should be 0, got {ttis[0]}"
        # Monotonic non-decreasing as we go back in time
        assert all(b >= a for a, b in zip(ttis, ttis[1:])), \
            f"{pid}: time_to_index_years should be non-decreasing across exam_order"
        # Each step ≈ 380 / 365.25 ≈ 1.04 years (the planted date_diff)
        for i in range(1, len(ttis)):
            assert abs((ttis[i] - ttis[i - 1]) - (380.0 / 365.25)) < 1e-6


def test_combat_auto_computes_time_to_index_years():
    """If time_to_index_years isn't a column, apply_combat_manuf should
    compute it from date_diff cumsum without erroring."""
    df = _planted_batch_table(seed=6)
    assert "time_to_index_years" not in df.columns
    out = apply_combat_manuf(df)
    # Function returns a copy with possibly an added column; either way the
    # call should succeed and not raise.
    assert len(out) == len(df)


# ----------------------------- main runner ----------------------------- #


if __name__ == "__main__":
    import inspect
    tests = [(n, fn) for n, fn in inspect.getmembers(sys.modules[__name__],
                                                     inspect.isfunction)
             if n.startswith("test_")]
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
