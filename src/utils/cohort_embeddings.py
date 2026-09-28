"""
Shared I/O, aggregation, and group-labeling helpers used by steps 04a
(embedding-distance sanity checks) and 04b (drift analysis).

Authoritative home for:
  - embedding-tree loading
  - image-to-exam aggregation (breast or study granularity)
  - patient-level group classification (malignant / biopsy_negative / control)
  - building the per-exam DataFrame that both downstream scripts consume

Keeping these in one module prevents 04a and 04b from drifting in how they
construct the per-exam cohort frame.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

# torch is only needed by load_embedding (reads .pt files). It's lazy-imported
# there so this module remains importable in envs that don't have torch
# installed (e.g. .venv Py 3.13, used for the longitudinal-ComBat fast path
# which operates on a precomputed parquet and never touches .pt files).

logger = logging.getLogger(__name__)


# Public constants
# EMB_DIM is the default embedding dimension used throughout the pipeline
# (Mammo-CLIP EfficientNet-B5 pooled features = 2048). For external
# embeddings with a different dimension, pass --emb-dim on the CLI and let
# it flow through to load_embedding / aggregate_exam_embeddings /
# build_exam_table — those helpers all accept an emb_dim parameter.
EMB_DIM = 2048
DAYS_PER_YEAR = 365.25

MALIGNANT = "malignant"
BIOPSY_NEG = "biopsy_negative"
CONTROL = "control"

# --- Presentation labels (figures / manuscript) ------------------------------
# Single source of truth so 04a, 04c, and plot_assets.py all render identical,
# publication-ready text. Internal group codes above stay machine-friendly; the
# maps below are display-only and must not be used as DataFrame keys.
GROUP_DISPLAY = {
    MALIGNANT: "Malignant",
    BIOPSY_NEG: "Biopsy-negative",
    CONTROL: "Control",
}
SIDE_DISPLAY = {"affected": "Affected side", "contralateral": "Contralateral side"}

# Axis label for the primary signed-projection velocity metric (v_signed).
# "along" (not "toward") because the projection is signed and can be negative.
VSIGNED_LABEL = r"Drift velocity along cancer direction $\hat{u}$ (yr$^{-1}$)"
VL2_LABEL = r"Drift speed $\lVert\Delta\rVert$ (yr$^{-1}$)"
UHAT_PROJ_LABEL = r"Projection onto cancer direction $\hat{u}$ (a.u.)"


def group_label(code: str) -> str:
    """Display label for an internal group code (falls back to a title-cased code)."""
    return GROUP_DISPLAY.get(code, str(code).replace("_", " ").title())


def _norm_laterality(x: str | float | None) -> str | None:
    """Normalize laterality strings to 'L' or 'R'; else None."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    s = str(x).strip().upper()
    if not s:
        return None
    if s in ("LEFT", "L"):
        return "L"
    if s in ("RIGHT", "R"):
        return "R"
    return None


def load_embedding(
    emb_root: Path,
    patient_id: str,
    study_id: str,
    sop_uid: str,
    emb_dim: int = EMB_DIM,
) -> np.ndarray | None:
    """
    Load a single image embedding saved by 03_extract_embeddings.py (or any
    external script that writes compatible .pt files — see the README
    section on running the drift analysis with other models).

    Expected payload format: torch.save({"embedding": <tensor>, ...}, path)
    where the tensor flattens to length `emb_dim`.

    Returns
    -------
    np.ndarray of shape (emb_dim,) or None if the file is missing, corrupt,
    or has an unexpected shape.
    """
    # Lazy import: torch is only needed by this helper. Keeps the rest of
    # the module importable in envs that don't have torch (e.g. .venv Py 3.13
    # for the longitudinal-ComBat fast path).
    import torch
    p = emb_root / str(patient_id) / str(study_id) / f"{sop_uid}.pt"
    if not p.is_file():
        return None
    try:
        payload = torch.load(p, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"Warning: failed to load {p}: {e}")
        return None
    if isinstance(payload, torch.Tensor):
        emb = payload
    elif isinstance(payload, dict) and "embedding" in payload:
        emb = payload["embedding"]
    else:
        print(f"Warning: unexpected payload type {type(payload).__name__} at {p}")
        return None
    if isinstance(emb, torch.Tensor):
        emb = emb.detach().cpu().numpy()
    emb = np.asarray(emb, dtype=np.float32).reshape(-1)
    if emb.shape[0] != emb_dim:
        print(f"Warning: unexpected emb dim {emb.shape} at {p} (expected {emb_dim})")
        return None
    return emb


def aggregate_exam_embeddings(
    df: pd.DataFrame,
    emb_root: Path,
    granularity: str,
    affected_laterality_map: dict[str, str] | None = None,
    emb_dim: int = EMB_DIM,
    l2_norm: bool = False,
) -> pd.DataFrame:
    """
    Collapse image-level rows to per-exam embeddings.

    MATCH_ID is intentionally dropped here — it points at a single partner and is
    not stable for 1:N matching. Pairing is re-attached downstream via pair_id
    (see 04b_analyze_drift.build_pairs + attach_pair_ids).

    Parameters
    ----------
    df : DataFrame
        Matched cohort with one row per image (must contain PATIENTID, STUDYID,
        SOPINSTANCEUID, IMAGELATERALITY, exam_order, date_diff, ...).
    emb_root : Path
        Root of the embedding tree from step 3.
    granularity : {"breast", "study", "breast_both_sides"}
        breast → one row per (PATIENTID, STUDYID), averaging images whose
        IMAGELATERALITY matches the patient's affected laterality.
        study → one row per (PATIENTID, STUDYID), averaging all images in the
        study regardless of side.
        breast_both_sides → *two* rows per (PATIENTID, STUDYID), one per
        laterality (L and R), each averaging that side's images. Adds a
        `laterality` column ∈ {"L", "R"}. Used by the contralateral-breast
        within-patient analysis (04c).
    affected_laterality_map : dict[PATIENTID -> "L"/"R"], required for "breast"
        (used to filter to the affected side). Not used for the other
        granularities.
    emb_dim : int
        Expected embedding dimension. Defaults to EMB_DIM (2048, Mammo-CLIP).
    l2_norm : bool
        Harmonization step 1: if True, L2-normalize each per-image embedding
        before mean-pooling within an exam. Removes per-image magnitude
        variation; the resulting per-exam vector lives near (but not on) the
        unit sphere because it is a mean of unit vectors.
    """
    if granularity not in ("breast", "study", "breast_both_sides"):
        raise ValueError(
            f"granularity must be 'breast', 'study', or 'breast_both_sides', "
            f"got {granularity!r}"
        )

    df = df.copy()
    if "MATCH_ID" in df.columns:
        df = df.drop(columns=["MATCH_ID"])
    df["IMAGELATERALITY_N"] = df["IMAGELATERALITY"].map(_norm_laterality)

    if granularity == "breast":
        if affected_laterality_map is None:
            raise ValueError("breast-level aggregation requires affected_laterality_map")
        df["affected_side"] = df["PATIENTID"].map(affected_laterality_map)
        df = df[df["IMAGELATERALITY_N"] == df["affected_side"]]
        df = df.drop(columns=["affected_side"])
    elif granularity == "breast_both_sides":
        # Drop images with an unparseable laterality; group key includes side.
        df = df[df["IMAGELATERALITY_N"].isin(["L", "R"])]

    if granularity == "breast_both_sides":
        keys = ["PATIENTID", "STUDYID", "IMAGELATERALITY_N"]
    else:
        keys = ["PATIENTID", "STUDYID"]

    carry_cols = [
        c for c in df.columns
        if c not in ("IMAGELATERALITY", "IMAGELATERALITY_N", "SOPINSTANCEUID",
                     "fullPath", "fullPath_cropped_resampledBy2")
        and c not in keys
        and df.groupby(keys)[c].nunique(dropna=False).max() == 1
    ]

    rows: list[dict] = []
    for key_vals, g in df.groupby(keys, sort=False):
        if granularity == "breast_both_sides":
            pid, sid, lat = key_vals
        else:
            pid, sid = key_vals
            lat = None
        embs = []
        for sop in g["SOPINSTANCEUID"].astype(str):
            e = load_embedding(emb_root, pid, sid, sop, emb_dim=emb_dim)
            if e is None:
                continue
            if l2_norm:
                n = np.linalg.norm(e)
                if n > 0:
                    e = e / n
            embs.append(e)
        if not embs:
            continue
        meta = g.iloc[0][carry_cols].to_dict()
        meta.update({"PATIENTID": pid, "STUDYID": sid,
                     "embedding": np.mean(np.stack(embs), axis=0),
                     "n_images_used": len(embs)})
        if lat is not None:
            meta["laterality"] = lat
        rows.append(meta)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    sort_cols = (["PATIENTID", "laterality", "exam_order"]
                 if granularity == "breast_both_sides"
                 else ["PATIENTID", "exam_order"])
    out = out.sort_values(sort_cols).reset_index(drop=True)

    # Drop patients (or patient×side units) whose aggregation produced a broken
    # chain. The affected-side filter can drop whole studies when none of
    # their images are on the affected side. For breast_both_sides the unit is
    # (PATIENTID, laterality) — one side can be complete while the other isn't;
    # we drop only the incomplete side, not the whole patient.
    def _chain_ok(g: pd.DataFrame) -> bool:
        orders = sorted(g["exam_order"].astype(int).tolist())
        return orders == list(range(0, len(orders)))

    if granularity == "breast_both_sides":
        chain_ok = out.groupby(["PATIENTID", "laterality"], sort=False).apply(_chain_ok)
        keep = set(chain_ok[chain_ok].index)  # set of (pid, lat) tuples
        n_dropped = int((~chain_ok).sum())
        if n_dropped > 0:
            dropped = chain_ok[~chain_ok].index.tolist()
            print(f"  Warning: dropped {n_dropped} (patient, side) unit(s) with "
                  f"non-contiguous exam_order after breast_both_sides aggregation "
                  f"(missing index and/or intermediate priors). Examples: {dropped[:3]}")
        out = out[out.apply(lambda r: (r["PATIENTID"], r["laterality"]) in keep, axis=1)]
        out = out.reset_index(drop=True)
    else:
        patient_ok = out.groupby("PATIENTID", sort=False).apply(_chain_ok)
        keep_patients = set(patient_ok[patient_ok].index)
        n_dropped = int((~patient_ok).sum())
        if n_dropped > 0:
            dropped_pids = patient_ok[~patient_ok].index.tolist()
            print(f"  Warning: dropped {n_dropped} patient(s) with non-contiguous "
                  f"exam_order after {granularity}-level aggregation (missing index "
                  f"and/or intermediate priors). Example PIDs: {dropped_pids[:3]}")
        out = out[out["PATIENTID"].isin(keep_patients)].reset_index(drop=True)
    if out.empty:
        return out

    # n_priors per patient-side unit (breast_both_sides) or per patient
    # (breast/study) = number of non-index exams retained.
    if granularity == "breast_both_sides":
        n_priors = (out[~out["is_index"]].groupby(["PATIENTID", "laterality"]).size()
                    .rename("n_priors"))
        out = out.merge(n_priors, on=["PATIENTID", "laterality"], how="left")
    else:
        n_priors = (out[~out["is_index"]].groupby("PATIENTID").size()
                    .rename("n_priors"))
        out = out.merge(n_priors, on="PATIENTID", how="left")
    out = out.fillna({"n_priors": 0})
    out["n_priors"] = out["n_priors"].astype(int)
    return out


def build_affected_laterality_map(
    df_cases: pd.DataFrame, df_controls: pd.DataFrame
) -> dict[str, str]:
    """
    Map PATIENTID → affected laterality ('L' or 'R').

    For cases: Laterality_Biopsy. For controls: inherit from their matched case
    via MATCH_ID (controls have no biopsy laterality of their own).
    """
    case_side = (
        df_cases[["PATIENTID", "Laterality_Biopsy"]]
        .drop_duplicates("PATIENTID")
        .assign(side=lambda d: d["Laterality_Biopsy"].map(_norm_laterality))
        .dropna(subset=["side"])
        .set_index("PATIENTID")["side"]
        .to_dict()
    )
    ctrl_side = {}
    for pid, g in df_controls.groupby("PATIENTID"):
        match_case_pid = g["MATCH_ID"].iloc[0]
        side = case_side.get(match_case_pid)
        if side is not None:
            ctrl_side[pid] = side
    return {**case_side, **ctrl_side}


def build_side_label_map(
    df_cases: pd.DataFrame,
    df_controls: pd.DataFrame,
    seed: int = 42,
) -> dict[str, dict[str, str]]:
    """
    Map PATIENTID → {"affected": "L"/"R", "contralateral": "R"/"L"} for the
    within-patient contralateral-breast analysis (04c).

    - Cases (malignant and biopsy_negative): "affected" = Laterality_Biopsy;
      "contralateral" = the other side. This matches the recall/biopsy side
      even when the biopsy came back benign, since the recall reason itself
      is side-specific.
    - Controls: no biological affected side exists. We deterministically assign
      one side as "affected" via a seeded PRNG, per PATIENTID. This lets 04c
      run the same paired within-patient Wilcoxon on the control arm as a
      null/sanity reference — a well-behaved pipeline should produce
      p ≈ 0.5 for controls.

    Parameters
    ----------
    df_cases, df_controls : DataFrame
        Raw step-2 matched CSVs.
    seed : int
        Seeded PRNG for reproducibility of control-arm side assignment.
        Default 42.
    """
    out: dict[str, dict[str, str]] = {}
    # Cases (malignant + biopsy_negative): use Laterality_Biopsy.
    case_side = (
        df_cases[["PATIENTID", "Laterality_Biopsy"]]
        .drop_duplicates("PATIENTID")
        .assign(side=lambda d: d["Laterality_Biopsy"].map(_norm_laterality))
        .dropna(subset=["side"])
        .set_index("PATIENTID")["side"]
        .to_dict()
    )
    for pid, affected in case_side.items():
        contralateral = "R" if affected == "L" else "L"
        out[pid] = {"affected": affected, "contralateral": contralateral}

    # Controls: deterministic per-patient random assignment.
    rng = np.random.default_rng(seed)
    control_pids = sorted(df_controls["PATIENTID"].unique())  # sort for determinism
    picks = rng.integers(0, 2, size=len(control_pids))  # 0 → L, 1 → R
    for pid, pick in zip(control_pids, picks):
        affected = "L" if pick == 0 else "R"
        contralateral = "R" if affected == "L" else "L"
        out[pid] = {"affected": affected, "contralateral": contralateral}
    return out


def build_group_map(
    df_cases: pd.DataFrame, df_controls: pd.DataFrame
) -> dict[str, str]:
    """
    Map PATIENTID → group ∈ {malignant, biopsy_negative, control}.

    Classification is done once from the RAW step-2 CSVs (before any aggregation)
    so it is robust to cases with multiple biopsies linked to the same index study
    (≈0.6% of studies have both a malignant and benign biopsy row).

    Rule: a case patient is malignant iff any row has Cancer == 1; otherwise
    biopsy_negative. A control patient is always control.
    """
    group: dict[str, str] = {}

    if "Cancer" in df_cases.columns:
        for pid, g in df_cases.groupby("PATIENTID"):
            group[pid] = MALIGNANT if (g["Cancer"] == 1).any() else BIOPSY_NEG
    elif "Cancer_Description" in df_cases.columns:
        for pid, g in df_cases.groupby("PATIENTID"):
            desc = g["Cancer_Description"].astype(str).str.lower()
            group[pid] = MALIGNANT if desc.str.contains("malignant").any() else BIOPSY_NEG
    else:
        raise ValueError("df_cases missing 'Cancer' and 'Cancer_Description' — cannot classify")

    for pid in df_controls["PATIENTID"].unique():
        group.setdefault(pid, CONTROL)
    return group


def apply_baseline_subtraction(exam: pd.DataFrame) -> pd.DataFrame:
    """
    Harmonization step 2: replace each exam's embedding with
    (e_exam − e_baseline), where e_baseline is the patient's earliest prior
    (highest exam_order). This anchors every patient's trajectory at the
    origin in their own coordinate system, removing static between-patient
    anatomy.

    Patients with only one exam (no baseline distinct from the row) are
    dropped with a warning. The baseline row itself becomes the zero vector
    after subtraction (kept for accounting; downstream interval computation
    pairs it with the next-newer row using its date_diff).
    """
    # Reset index so positional list indexing works regardless of the caller's
    # index state (pre-aggregation could be RangeIndex, post-filter need not be).
    out = exam.copy().reset_index(drop=True)
    keep_mask = np.ones(len(out), dtype=bool)
    new_embs = list(out["embedding"].to_numpy())
    n_dropped = 0
    dropped_pids: list[str] = []
    for pid, g in out.groupby("PATIENTID", sort=False):
        positions = g.index.to_numpy()  # contiguous since we reset above
        if len(g) < 2:
            keep_mask[positions] = False
            n_dropped += 1
            dropped_pids.append(pid)
            continue
        # baseline = highest exam_order (earliest prior).
        baseline_pos = positions[np.argmax(g["exam_order"].to_numpy())]
        baseline = out.at[baseline_pos, "embedding"]
        for pos in positions:
            new_embs[pos] = out.at[pos, "embedding"] - baseline
    if n_dropped > 0:
        print(f"  Warning: baseline subtraction dropped {n_dropped} patient(s) "
              f"with only one retained exam. Example PIDs: {dropped_pids[:3]}")
    out["embedding"] = new_embs
    out = out[keep_mask].reset_index(drop=True)
    return out


def _fast_standardize_across_features(X, design, info_dict):
    """Vectorized drop-in replacement for neuroCombat.standardize_across_features.

    Mathematically identical to the upstream when ``X`` contains no NaNs
    (which is always true for our embeddings — Mammo-CLIP outputs are
    finite). The upstream version loops over the ~2048 features in Python
    to handle per-feature NaN masking; that loop dominates wall time at
    cohort scale (~40 min). This version solves the OLS for all features
    in a single matrix call, ~3000× faster.

    The behavior of the rest of the function (grand_mean / var_pooled /
    mod_mean / s_data construction) is preserved bit-for-bit, including
    the upstream's quirky line ``var_pooled[var_pooled==0] = np.median(
    var_pooled!=0)`` which evaluates to a no-op on data with non-zero
    feature variance.
    """
    import numpy as _np

    n_batch = info_dict["n_batch"]
    n_sample = info_dict["n_sample"]
    sample_per_batch = info_dict["sample_per_batch"]
    batch_info = info_dict["batch_info"]
    ref_level = info_dict["ref_level"]

    # Vectorized OLS: B_hat shape = (design_cols, n_features). Equivalent to
    # the per-feature loop upstream that calls la.inv(mod.T @ mod) @ mod.T @ y
    # for each feature row. solve() uses a Cholesky/LU factorization, more
    # numerically stable than explicit inv().
    B_hat = _np.linalg.solve(design.T @ design, design.T @ X.T)

    if ref_level is not None:
        grand_mean = _np.transpose(B_hat[ref_level, :])
    else:
        grand_mean = _np.dot(
            (sample_per_batch / float(n_sample)).T, B_hat[:n_batch, :]
        )
    stand_mean = _np.dot(
        grand_mean.T.reshape((len(grand_mean), 1)), _np.ones((1, n_sample))
    )

    if ref_level is not None:
        X_ref = X[:, batch_info[ref_level]]
        design_ref = design[batch_info[ref_level], :]
        n_sample_ref = sample_per_batch[ref_level]
        var_pooled = _np.dot(
            ((X_ref - _np.dot(design_ref, B_hat).T) ** 2),
            _np.ones((n_sample_ref, 1)) / float(n_sample_ref),
        )
    else:
        var_pooled = _np.dot(
            ((X - _np.dot(design, B_hat).T) ** 2),
            _np.ones((n_sample, 1)) / float(n_sample),
        )

    # Upstream's line — preserved verbatim to match output exactly.
    # No-op on data with no zero-variance features.
    var_pooled[var_pooled == 0] = _np.median(var_pooled != 0)

    mod_mean = 0
    if design is not None:
        tmp = design.copy()
        tmp[:, :n_batch] = 0
        mod_mean = _np.transpose(_np.dot(tmp, B_hat))

    s_data = (X - stand_mean - mod_mean) / _np.dot(
        _np.sqrt(var_pooled), _np.ones((1, n_sample))
    )
    return s_data, stand_mean, var_pooled, mod_mean


def _fast_it_sol(sdat, g_hat, d_hat, g_bar, t2, a, b, conv=0.0001, max_iter=200):
    """Drop-in replacement for neuroCombat.it_sol with two safety patches.

    Two improvements over upstream:
      1. ``max_iter`` cap (default 200) — upstream has an unbounded
         ``while change > conv`` loop with no iteration guard. When
         ``g_old`` has elements near zero, the relative-change criterion
         ``abs(g_new - g_old) / g_old`` can stall numerically and the
         loop runs effectively forever. The cap matches what every
         standard EB implementation uses.
      2. ``np.maximum(abs(g_old), 1e-12)`` floor in the convergence test
         to prevent divide-by-near-zero stalls.

    Behavior on well-conditioned input is bit-for-bit identical to upstream
    (it converges in ~10–50 iters in practice; the cap of 200 is well above
    that). Iteration body uses broadcasting in place of upstream's tile
    via ``np.dot(g_new.reshape(...), np.ones((1, n_sample)))`` for clarity
    and a small speedup.

    Note: upstream computes ``n = (1 - np.isnan(sdat)).sum(axis=1)``; we
    rely on the no-NaN invariant (Mammo-CLIP embeddings are finite) and
    use the simpler scalar ``n = sdat.shape[1]``. If you ever harmonize
    data with NaNs, restore the upstream form.
    """
    import numpy as _np

    from neuroCombat.neuroCombat import postmean as _postmean
    from neuroCombat.neuroCombat import postvar as _postvar

    n = sdat.shape[1]  # NaN-free: every feature has all samples
    g_old = g_hat.copy()
    d_old = d_hat.copy()

    g_new = g_old
    d_new = d_old
    for _ in range(max_iter):
        g_new = _postmean(g_hat, g_bar, n, d_old, t2)
        sum2 = ((sdat - g_new[:, None]) ** 2).sum(axis=1)
        d_new = _postvar(sum2, n, a, b)
        # Floor at 1e-12 to avoid divide-by-zero stalls when a parameter
        # estimate is genuinely near zero (e.g. a feature with negligible
        # batch shift). Without this floor the upstream loop can hang.
        change = max(
            (_np.abs(g_new - g_old) / _np.maximum(_np.abs(g_old), 1e-12)).max(),
            (_np.abs(d_new - d_old) / _np.maximum(_np.abs(d_old), 1e-12)).max(),
        )
        g_old = g_new
        d_old = d_new
        if change <= conv:
            break
    return (g_new, d_new)


def combat_harmonize_array(
    X: np.ndarray,
    batch: np.ndarray,
    categorical: dict | None = None,
    continuous: dict | None = None,
) -> np.ndarray:
    """Run NeuroComBat on a (n_samples, n_features) array.

    The single canonical entry point for ComBat in this codebase. Both the
    DataFrame-level ``apply_combat_manuf`` (production harmonization step)
    and the diagnostic ``manufacturer_combat.apply_combat`` (validation +
    ablation analyses) route through here.

    Parameters
    ----------
    X : ndarray, shape (n_samples, n_features)
        Embeddings, one row per sample.
    batch : ndarray of strings, shape (n_samples,)
        Batch label per sample (e.g. ``MANUFACTURER_CATEGORY``).
    categorical : dict[str, ndarray] | None
        Categorical biological covariates to preserve. Each value is an
        ndarray of length n_samples.
    continuous : dict[str, ndarray] | None
        Continuous biological covariates to preserve.

    Returns
    -------
    X_harm : ndarray, same shape as X
        Batch-corrected embeddings. Near-constant feature dimensions
        (var < 1e-12) are passed through unchanged.

    Notes
    -----
    Wraps the call in our ``_patched_neurocombat`` context so the
    vectorized ``standardize_across_features`` and bounded ``it_sol`` are
    used. Without the patches, NeuroComBat hangs in an unbounded loop on
    cohort-scale (10K-row × 2048-feature) inputs.
    """
    from neuroCombat import neuroCombat

    covars = pd.DataFrame({"batch": np.asarray(batch).astype(str)})
    cat_cols: list[str] = []
    cont_cols: list[str] = []
    if categorical:
        for k, v in categorical.items():
            covars[k] = np.asarray(v)
            cat_cols.append(k)
    if continuous:
        for k, v in continuous.items():
            covars[k] = np.asarray(v).astype(float)
            cont_cols.append(k)

    feat_var = X.var(axis=0)
    keep_feat = feat_var > 1e-12
    X_sub = X[:, keep_feat]

    with _patched_neurocombat():
        out = neuroCombat(
            dat=X_sub.T,                       # neuroCombat expects (features × samples)
            covars=covars,
            batch_col="batch",
            categorical_cols=cat_cols or None,
            continuous_cols=cont_cols or None,
        )

    X_harm = X.copy()
    X_harm[:, keep_feat] = out["data"].T
    return X_harm


class _patched_neurocombat:
    """Context manager that monkey-patches NeuroComBat's two slow helpers
    with their vectorized equivalents for the duration of a ``neuroCombat()``
    call. Restores the originals on exit so this is non-invasive and the
    patch can be torn out cleanly when upstream fixes these issues."""

    def __enter__(self):
        # Patch the *submodule* (where the slow helpers live), not the
        # top-level neuroCombat() function imported from the package.
        # neuroCombat 0.2.x's __init__.py does `from neuroCombat.neuroCombat
        # import neuroCombat`, which shadows the submodule name — so
        # `import neuroCombat.neuroCombat as x` returns the FUNCTION, not
        # the module. The reliable way to grab the submodule is via
        # sys.modules, which always holds the loaded module object.
        import sys
        import neuroCombat  # noqa: F401  (force-load so submodule is registered)
        _nc_submod = sys.modules["neuroCombat.neuroCombat"]
        self._mod = _nc_submod
        self._orig_std = _nc_submod.standardize_across_features
        self._orig_itsol = _nc_submod.it_sol
        _nc_submod.standardize_across_features = _fast_standardize_across_features
        _nc_submod.it_sol = _fast_it_sol
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._mod.standardize_across_features = self._orig_std
        self._mod.it_sol = self._orig_itsol
        return False


def _compute_time_to_index_years(exam: pd.DataFrame) -> pd.Series:
    """Per-row time-from-index in years, computed via cumulative date_diff.

    Convention: date_diff[k] = days between exam k and the next-newer exam k-1
    (the index has NaN). Cumulative days from the index = sum of date_diff over
    exam_orders 1..k. Sorting by exam_order ascending per patient walks back in
    time. Index rows get 0; deeper priors get larger positive values (years
    BEFORE the index — distinct from 04b's signed convention used inside the
    LME, which is negative-going from the index midpoint).
    """
    s = exam.sort_values(["PATIENTID", "exam_order"]).copy()
    s["_dd"] = s["date_diff"].fillna(0).astype(float)
    s["_cum_days"] = s.groupby("PATIENTID")["_dd"].cumsum()
    s["time_to_index_years"] = s["_cum_days"] / DAYS_PER_YEAR
    # Reindex back to the input row order so callers can attach as a column.
    return s["time_to_index_years"].reindex(exam.index)


def apply_combat_manuf(
    exam: pd.DataFrame,
    batch_col: str = "MANUFACTURER_CATEGORY",
    preserve_categorical: tuple[str, ...] = ("group", "is_index"),
    preserve_continuous: tuple[str, ...] = ("time_to_index_years",),
    min_batch_size: int = 30,
) -> pd.DataFrame:
    """Harmonization step 4: ComBat batch-correct embeddings on ``batch_col``.

    Fits NeuroComBat jointly on the entire ``exam`` table (cases + controls
    together) with ``preserve_categorical`` and ``preserve_continuous``
    columns supplied as covariates so the EB correction does not absorb
    those signals along with the batch effect.

    The default categorical covariate is ``group`` (one of malignant /
    biopsy_negative / control, set per-patient by build_group_map). This
    is a three-level patient-level label that explicitly preserves the
    three contrasts the manuscript reports (mal-vs-ctrl, bn-vs-ctrl,
    mal-vs-bn) instead of collapsing them into the binary Cancer ∈ {0,1}
    that conflates biopsy-negative cases with screen-negative controls.

    Notes
    -----
    - Rows whose batch_col value belongs to a batch with fewer than
      ``min_batch_size`` rows are passed through unchanged (NeuroComBat is
      unstable on tiny batches; the EB priors degenerate). A warning is
      logged per dropped batch.
    - ``time_to_index_years`` is auto-computed if not present.
    - Near-constant feature dimensions (var < 1e-12) are passed through
      unchanged — NeuroComBat divides by per-batch feature variance and
      explodes on zero-variance dims.
    - Returns a copy with the same row count + same column schema.

    Why joint cases+controls fit (rather than per-arm): preserved
    covariates need *variation* in the design matrix, and within either
    arm alone ``group`` would only take one or two values. Fitting jointly
    gives the EB priors a 3-level ``group`` design matrix and enough
    signal to estimate batch parameters cleanly without confounding any
    of the case/control contrasts.
    """
    try:
        from neuroCombat import neuroCombat
    except ImportError as e:
        raise ImportError(
            "apply_combat_manuf requires neuroCombat. Install with:\n"
            "  uv pip install neuroCombat"
        ) from e

    out = exam.copy().reset_index(drop=True)
    if batch_col not in out.columns:
        raise ValueError(f"ComBat: missing batch column {batch_col!r}")
    # Auto-compute time_to_index_years if requested but absent.
    if "time_to_index_years" in preserve_continuous and "time_to_index_years" not in out.columns:
        out["time_to_index_years"] = _compute_time_to_index_years(out)

    # If this is a breast_both_sides exam table, auto-preserve `side_label`
    # so the affected-vs-contralateral biological contrast survives ComBat.
    if "side_label" in out.columns and "side_label" not in preserve_categorical:
        preserve_categorical = tuple(preserve_categorical) + ("side_label",)


    # Drop tiny-batch rows from the fit (pass-through). Use the full
    # batch_col distribution: anything < min_batch_size is set aside.
    batch_counts = out[batch_col].value_counts(dropna=False)
    keep_batches = set(batch_counts[batch_counts >= min_batch_size].index)
    drop_mask = ~out[batch_col].isin(keep_batches)
    if drop_mask.any():
        dropped_batches = batch_counts[~batch_counts.index.isin(keep_batches)].to_dict()
        logger.warning(
            "ComBat passing through %d row(s) in tiny batches (n<%d): %s",
            int(drop_mask.sum()), min_batch_size, dropped_batches,
        )
    fit_idx = out.index[~drop_mask]
    if len(fit_idx) == 0:
        raise ValueError("ComBat: no rows survive min_batch_size filter")

    # Resolve which preserved covariates are actually present + non-degenerate.
    # NeuroComBat rejects design-matrix singularities, so we drop any covariate
    # that has a single value across the fit set (it has nothing to preserve
    # anyway).
    def _resolve_covs(names: tuple[str, ...], kind: str) -> list[str]:
        kept: list[str] = []
        for c in names:
            if c not in out.columns:
                continue
            n_unique = out.loc[fit_idx, c].nunique(dropna=False)
            if n_unique < 2:
                logger.info(
                    "ComBat dropping degenerate %s covariate %r (constant value).",
                    kind, c,
                )
                continue
            kept.append(c)
        return kept

    cat_cols = _resolve_covs(preserve_categorical, "categorical")
    cont_cols = _resolve_covs(preserve_continuous, "continuous")

    # Stack embeddings and route to the canonical array-level ComBat helper.
    # NeuroComBat is single-threaded NumPy; on Mammo-CLIP scale (10K rows ×
    # 2048 features × 3 batches) the EB-prior fit takes ~30s with our
    # patches and would hang for hours without them.
    import time
    t_total = time.time()

    logger.info("ComBat: stacking embeddings...")
    t0 = time.time()
    X = np.stack(out.loc[fit_idx, "embedding"].to_numpy()).astype(np.float64)
    logger.info("ComBat: stacked X=%s in %.1fs", X.shape, time.time() - t0)

    logger.info(
        "ComBat: fitting on %d rows × %d features; batch=%s (%d levels), "
        "preserve_cat=%s, preserve_cont=%s",
        X.shape[0], X.shape[1], batch_col, len(keep_batches), cat_cols, cont_cols,
    )
    t1 = time.time()
    X_harm = combat_harmonize_array(
        X,
        batch=out.loc[fit_idx, batch_col].astype(str).to_numpy(),
        categorical={c: out.loc[fit_idx, c].to_numpy() for c in cat_cols},
        continuous={c: out.loc[fit_idx, c].astype(float).to_numpy() for c in cont_cols},
    )
    logger.info("ComBat: neuroCombat() returned in %.1fs", time.time() - t1)

    # Write back into a fresh embedding column. Tiny-batch rows pass through.
    new_embs = list(out["embedding"].to_numpy())
    for pos, idx in enumerate(fit_idx):
        new_embs[idx] = X_harm[pos].astype(np.float32)
    out["embedding"] = new_embs
    logger.info("ComBat: total wall %.1fs", time.time() - t_total)
    return out


def apply_opnested_combat(
    exam: pd.DataFrame,
    batch_cols: list[str],
    preserve_categorical: tuple[str, ...] = ("group", "is_index"),
    preserve_continuous: tuple[str, ...] = ("time_to_index_years",),
    min_batch_size: int = 30,
) -> tuple[pd.DataFrame, dict]:
    """OPNested-ComBat (Horng et al. 2022) — sequential cross-sectional
    ComBat harmonization across two or more batch axes, with the
    correction order picked to minimize residual between-batch signal.

    Algorithm (faithful port of ``OPNestedComBat`` from
    ``hannah-horng/opnested-combat``):

      1. For each permutation of ``batch_cols``, apply cross-sectional
         NeuroComBat sequentially in that order (harmonizing on batch
         label i, then re-harmonizing the output on label j, ...).
      2. On each permutation's final harmonized array, run an
         Anderson-Darling k-sample test per feature per batch label. Sum
         over (feature × batch label) the count of cells where the A-D
         p < 0.05.
      3. Return the harmonized array from the permutation with the
         lowest count of residual between-batch signal.

    Differences vs Horng's published implementation:
      - Uses our ``_patched_neurocombat`` context (via ``combat_harmonize_array``)
        instead of stock ``neuroCombat``, so the call scales to cohort-size
        (10K rows × 2048 features) instead of hanging in a Python loop.
      - Tiny-batch pass-through (rows in a batch with n < ``min_batch_size``
        on ANY of the batch axes) is applied once up-front rather than
        per-axis. Matches the policy used by ``apply_combat_manuf``.
      - Preserved covariates are enforced identically on every permutation.
      - No KDE plots or filesystem output.

    Parameters
    ----------
    exam : DataFrame
        Per-exam frame with an ``embedding`` column (np.ndarray per row)
        and columns for every element of ``batch_cols`` and every element
        of ``preserve_categorical`` / ``preserve_continuous``.
    batch_cols : list[str]
        Batch label column names, e.g.
        ``["MANUFACTURER_CATEGORY", "MG_modality"]``. Order does NOT
        matter — the algorithm evaluates all permutations and picks the
        best one.
    preserve_categorical, preserve_continuous : tuple[str, ...]
        Biological covariates to preserve, same semantics as
        ``apply_combat_manuf``.
    min_batch_size : int
        Minimum row count per batch level to include in the fit; rows
        in smaller batches pass through unchanged.

    Returns
    -------
    (exam_out, order_info) : tuple[DataFrame, dict]
        ``exam_out`` — same schema as ``exam``, with ``embedding`` replaced
        by the OPNested-harmonized vectors.
        ``order_info`` — diagnostic dict with keys:
            ``chosen_order`` (list[str]): the winning permutation.
            ``permutation_scores`` (dict[str, int]): per-permutation
                count of (feature × batch) A-D rejections at α=0.05.
            ``per_batch_reject_counts`` (dict[str, dict[str, int]]):
                per-permutation, per-batch reject counts.
    """
    from itertools import permutations
    from scipy.stats import anderson_ksamp
    import time

    out = exam.copy().reset_index(drop=True)
    for c in batch_cols:
        if c not in out.columns:
            raise ValueError(f"OPNested-ComBat: missing batch column {c!r}")
    # Auto-compute time_to_index_years if requested but absent.
    if "time_to_index_years" in preserve_continuous and "time_to_index_years" not in out.columns:
        out["time_to_index_years"] = _compute_time_to_index_years(out)

    # Tiny-batch pass-through — any row that lands in a small level on
    # ANY axis is set aside from the fit. All axes must have their
    # small-batch policy applied at once because the sequential correction
    # revisits every row on every axis.
    drop_mask = np.zeros(len(out), dtype=bool)
    for c in batch_cols:
        counts = out[c].value_counts(dropna=False)
        small = set(counts[counts < min_batch_size].index)
        if small:
            logger.warning(
                "OPNested-ComBat: passing through %d row(s) in tiny %s batches: %s",
                int(out[c].isin(small).sum()), c,
                counts[counts.index.isin(small)].to_dict(),
            )
        drop_mask |= out[c].isin(small).to_numpy()
    fit_idx = np.where(~drop_mask)[0]
    if len(fit_idx) == 0:
        raise ValueError("OPNested-ComBat: no rows survive min_batch_size filter")

    # Resolve preserved covariates (same singleton-check as apply_combat_manuf).
    def _resolve_covs(names: tuple[str, ...], kind: str) -> list[str]:
        kept: list[str] = []
        for c in names:
            if c not in out.columns:
                continue
            if out.iloc[fit_idx][c].nunique(dropna=False) < 2:
                logger.info(
                    "OPNested-ComBat dropping degenerate %s covariate %r "
                    "(constant value in fit set).", kind, c,
                )
                continue
            kept.append(c)
        return kept

    cat_cols = _resolve_covs(preserve_categorical, "categorical")
    cont_cols = _resolve_covs(preserve_continuous, "continuous")

    # Stack embeddings once. Shape: (n_fit, n_features).
    logger.info("OPNested-ComBat: stacking embeddings...")
    t0 = time.time()
    X0 = np.stack(out.loc[fit_idx, "embedding"].to_numpy()).astype(np.float64)
    logger.info("OPNested-ComBat: stacked X=%s in %.1fs", X0.shape, time.time() - t0)
    n_fit, n_features = X0.shape

    # Categorical / continuous covariate arrays, same across permutations.
    cat_data = {c: out.iloc[fit_idx][c].to_numpy() for c in cat_cols}
    cont_data = {c: out.iloc[fit_idx][c].astype(float).to_numpy() for c in cont_cols}
    # Batch label arrays.
    batch_data = {c: out.iloc[fit_idx][c].astype(str).to_numpy() for c in batch_cols}

    # Enumerate permutations.
    perm_list = list(permutations(batch_cols))
    logger.info("OPNested-ComBat: evaluating %d permutation(s) of %s",
                len(perm_list), batch_cols)

    perm_results: dict[tuple, np.ndarray] = {}
    perm_scores: dict[str, int] = {}
    per_batch_counts: dict[str, dict[str, int]] = {}

    for perm_idx, order in enumerate(perm_list):
        logger.info("OPNested-ComBat: permutation %d/%d: %s",
                    perm_idx + 1, len(perm_list), list(order))
        X = X0.copy()
        for stage_i, batch_col in enumerate(order):
            t_stage = time.time()
            X = combat_harmonize_array(
                X,
                batch=batch_data[batch_col],
                categorical=cat_data if cat_cols else None,
                continuous=cont_data if cont_cols else None,
            )
            logger.info(
                "OPNested-ComBat:   stage %d (%s) done in %.1fs",
                stage_i + 1, batch_col, time.time() - t_stage,
            )
        perm_results[tuple(order)] = X

        # A-D k-sample test per (feature × batch label). Sum α=0.05 rejections.
        reject_by_batch: dict[str, int] = {}
        for batch_col in batch_cols:
            labels = batch_data[batch_col]
            groups = [X[labels == lvl] for lvl in np.unique(labels)]
            n_reject = 0
            for j in range(n_features):
                cols = [g[:, j] for g in groups if g.shape[0] > 0]
                if len(cols) < 2:
                    continue
                # anderson_ksamp warns for tied / uniform data; suppress
                # since our embeddings have some near-constant dims that
                # legitimately produce that. The significance_level is
                # clipped at 0.25 (>) / 0.001 (<) internally.
                try:
                    with np.errstate(all="ignore"):
                        res = anderson_ksamp(cols)
                    if res.significance_level < 0.05:
                        n_reject += 1
                except (ValueError, Exception):
                    continue
            reject_by_batch[batch_col] = n_reject
        total = sum(reject_by_batch.values())
        perm_key = " -> ".join(order)
        perm_scores[perm_key] = total
        per_batch_counts[perm_key] = reject_by_batch
        logger.info(
            "OPNested-ComBat: permutation %s residual A-D rejections: total=%d, %s",
            perm_key, total, reject_by_batch,
        )

    # Pick the permutation with the minimum total.
    best_perm_key = min(perm_scores, key=perm_scores.get)
    best_order = list([p for p in perm_list if " -> ".join(p) == best_perm_key][0])
    X_best = perm_results[tuple(best_order)]
    logger.info(
        "OPNested-ComBat: chosen order %s (min residual=%d/%d)",
        best_perm_key, perm_scores[best_perm_key], len(batch_cols) * n_features,
    )

    # Write X_best back into the exam frame; tiny-batch rows pass through.
    new_embs = list(out["embedding"].to_numpy())
    for pos, idx in enumerate(fit_idx):
        new_embs[idx] = X_best[pos].astype(np.float32)
    out["embedding"] = new_embs

    order_info = {
        "chosen_order": best_order,
        "permutation_scores": perm_scores,
        "per_batch_reject_counts": per_batch_counts,
        "n_features_scored": int(n_features),
    }
    return out, order_info


def apply_zscore(exam: pd.DataFrame) -> pd.DataFrame:
    """
    Harmonization step 3: per-dimension z-score across all rows in `exam`.
    Mean and std are fit on the **index rows** (is_index == True) so that
    the cancer direction `u_hat`, which is constructed from index
    embeddings, sees a unit-variance feature space.

    Embeddings on non-index rows are transformed with the same fit so
    intervals stay self-consistent.
    """
    out = exam.copy()
    if "is_index" not in out.columns:
        raise ValueError("z-score harmonization requires 'is_index' column")
    index_mask = out["is_index"].astype(bool).to_numpy()
    if index_mask.sum() == 0:
        raise ValueError("z-score harmonization: no index rows found")

    X = np.stack(out["embedding"].values).astype(np.float64)
    mu = X[index_mask].mean(axis=0)
    # ddof=1 for sample-std (Bessel-corrected); negligible numerical
    # difference at N>>1 but matches the textbook "sample z-score" semantics
    # the manuscript reports.
    sigma = X[index_mask].std(axis=0, ddof=1)
    sigma = np.where(sigma > 0, sigma, 1.0)  # avoid /0 on dead dims
    Z = (X - mu) / sigma
    out["embedding"] = list(Z.astype(np.float32))
    return out


def build_exam_table(
    df_cases: pd.DataFrame,
    df_controls: pd.DataFrame,
    emb_root: Path,
    granularity: str,
    emb_dim: int = EMB_DIM,
    harm_l2_norm: bool = False,
    harm_subtract_baseline: bool = False,
    harm_zscore_dim: bool = False,
    harm_combat_manuf: bool = False,
    contralateral_seed: int = 42,
) -> pd.DataFrame:
    """
    Aggregate image-level rows to per-exam embeddings at the chosen granularity,
    and attach group labels from the raw CSVs.

    `emb_dim` is forwarded to aggregate_exam_embeddings → load_embedding;
    pass the dimension of your external model's embeddings if you are
    running the pipeline on non-Mammo-CLIP outputs.

    Harmonization flags (compose left-to-right):
      - harm_l2_norm: L2-normalize each per-IMAGE embedding before mean-pool.
        Removes magnitude variation across images and across models.
      - harm_subtract_baseline: subtract each patient's earliest-prior
        per-exam embedding from all of their exams. Anchors every patient
        at zero in their own frame; removes static between-patient
        anatomy. Note: this changes the meaning of `u_hat` — index rows
        become *displacement-from-prior* vectors rather than positions.
      - harm_zscore_dim: per-dimension z-score across the population, fit
        on index rows only (so `u_hat` lives in unit-variance space).
        Removes uneven dimension scales.

    Recommended order when combining: L2 norm → baseline subtract → z-score
    (the implementation applies them in this order regardless of how flags
    are set).
    """
    lat_map = (
        build_affected_laterality_map(df_cases, df_controls)
        if granularity == "breast" else None
    )

    df_cases = df_cases.copy()
    df_cases["_origin"] = "case"
    df_controls = df_controls.copy()
    df_controls["_origin"] = "control"

    cases_exam = aggregate_exam_embeddings(
        df_cases, emb_root, granularity, lat_map,
        emb_dim=emb_dim, l2_norm=harm_l2_norm,
    )
    ctrls_exam = aggregate_exam_embeddings(
        df_controls, emb_root, granularity, lat_map,
        emb_dim=emb_dim, l2_norm=harm_l2_norm,
    )
    exam = pd.concat([cases_exam, ctrls_exam], ignore_index=True)

    group_map = build_group_map(df_cases, df_controls)
    exam["group"] = exam["PATIENTID"].map(group_map).fillna(CONTROL)

    if granularity == "breast_both_sides":
        # Stamp each row with side_label ∈ {"affected", "contralateral"} for
        # downstream within-patient paired analysis. Cases use Laterality_Biopsy;
        # controls get a seeded random assignment so paired-Wilcoxon on the
        # control arm is a well-defined null.
        side_map = build_side_label_map(df_cases, df_controls, seed=contralateral_seed)

        def _label(row: pd.Series) -> str:
            m = side_map.get(row["PATIENTID"])
            if m is None:
                return "unknown"
            if row["laterality"] == m["affected"]:
                return "affected"
            if row["laterality"] == m["contralateral"]:
                return "contralateral"
            return "unknown"

        exam["side_label"] = exam.apply(_label, axis=1)
        # Rows with unknown side (missing Laterality_Biopsy for a case, etc.)
        # cannot participate in the paired within-patient analysis; drop them.
        n_unknown = int((exam["side_label"] == "unknown").sum())
        if n_unknown > 0:
            print(f"  Warning: dropping {n_unknown} exam row(s) with unknown "
                  f"side_label (missing Laterality_Biopsy for a case, etc.)")
            exam = exam[exam["side_label"] != "unknown"].reset_index(drop=True)

    if harm_subtract_baseline:
        exam = apply_baseline_subtraction(exam)
    if harm_zscore_dim:
        exam = apply_zscore(exam)
    if harm_combat_manuf:
        exam = apply_combat_manuf(exam)

    return exam


def apply_longitudinal_combat_manuf(
    exam: pd.DataFrame,
    batch_col: str = "MANUFACTURER_CATEGORY",
    id_col: str = "PATIENTID",
    time_col: str = "time_to_index_years",
    formula: str = "C(group) + time_to_index_years",
    ranef: str = "(1|PATIENTID)",
    min_batch_size: int = 30,
    niter: int = 30,
    method: str = "REML",
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Harmonization step 4 (longitudinal variant): longCombat batch correction.

    Beer et al. 2020 NeuroImage extension of ComBat with a per-subject random
    intercept. Same EB shrinkage of (γ_jv, δ_jv²) as standard ComBat, but the
    per-feature model now includes a `(1|PATIENTID)` random effect so that
    repeated exams of the same patient are not treated as independent samples.

    Required Python env: ``.venv`` (Py 3.13) — the upstream
    ``longcombat-py`` requires Python ≥3.10 and our ``.venv_mammoclip``
    is pinned at 3.8 for Mammo-CLIP. Run this via the fast-path
    ``apply_harmonization.py`` in ``.venv``, then consume the harmonized
    parquet from 04a/04b in ``.venv_mammoclip``.

    Parameters
    ----------
    formula : str
        Fixed-effects formula (excludes batch and random effects).
        Default ``"C(group) + time_to_index_years"`` preserves the
        patient-level case/control contrast and the temporal trajectory.
        ``is_index`` is omitted because it's perfectly collinear with
        ``time_to_index_years == 0``.
    ranef : str
        Random-effects formula (R syntax). Default ``"(1|PATIENTID)"``
        gives each patient their own intercept.
    niter : int
        EB iteration count. longcombat-py default is 30.
    method : str
        REML or MSR. REML matches the R reference implementation.
    n_jobs : int
        Parallelism level for the per-feature MixedLM loop. ``1`` (default)
        runs the upstream ``long_combat`` unmodified — serial. Values > 1
        activate the ``_patched_long_combat`` context manager, which wraps
        the per-feature loop in a ``ProcessPoolExecutor`` with ``n_jobs``
        workers. Sensible values are 8–32 on the 128-core cluster nodes.

    Returns
    -------
    pd.DataFrame
        Same row count and column schema as ``exam``, with the
        ``embedding`` column replaced by longCombat-harmonized vectors
        for rows in non-tiny batches. Tiny-batch rows pass through
        unchanged.
    """
    try:
        from longcombat import long_combat
    except ImportError as e:
        raise ImportError(
            "apply_longitudinal_combat_manuf requires longcombat-py "
            "(Python ≥3.10). Install with:\n"
            "  uv pip install git+https://github.com/NoiseFilterT/longcombat-py\n"
            "and run from .venv (Py 3.13), not .venv_mammoclip (Py 3.8)."
        ) from e

    out = exam.copy().reset_index(drop=True)
    if batch_col not in out.columns:
        raise ValueError(f"longCombat: missing batch column {batch_col!r}")
    if id_col not in out.columns:
        raise ValueError(f"longCombat: missing id column {id_col!r}")

    # Auto-compute time_to_index_years if missing.
    if time_col == "time_to_index_years" and time_col not in out.columns:
        out[time_col] = _compute_time_to_index_years(out)

    # If this is a breast_both_sides exam table, extend the fixed-effects
    # formula so the affected-vs-contralateral biological contrast is a
    # preserved covariate rather than being absorbed into γ.
    if "side_label" in out.columns and "side_label" not in formula:
        formula = f"{formula} + C(side_label)"
        logger.info("longCombat: extending formula with C(side_label) → %r", formula)

    # Tiny-batch pass-through (same policy as standard ComBat).
    batch_counts = out[batch_col].value_counts(dropna=False)
    keep_batches = set(batch_counts[batch_counts >= min_batch_size].index)
    drop_mask = ~out[batch_col].isin(keep_batches)
    if drop_mask.any():
        dropped = batch_counts[~batch_counts.index.isin(keep_batches)].to_dict()
        logger.warning(
            "longCombat passing through %d row(s) in tiny batches (n<%d): %s",
            int(drop_mask.sum()), min_batch_size, dropped,
        )
    fit_idx = out.index[~drop_mask]
    if len(fit_idx) == 0:
        raise ValueError("longCombat: no rows survive min_batch_size filter")

    # Build the long-format frame: unpack the embedding column into 2048
    # named feature columns plus the batch/id/time/group columns longcombat-py
    # expects in `data`. We work on the fit-eligible subset only.
    import time
    t_total = time.time()
    logger.info("longCombat: unpacking %d embeddings into feature columns...",
                len(fit_idx))
    t0 = time.time()
    X = np.stack(out.loc[fit_idx, "embedding"].to_numpy()).astype(np.float64)
    n_features = X.shape[1]
    feat_names = [f"f{i:04d}" for i in range(n_features)]
    long_df = pd.DataFrame(X, columns=feat_names)
    long_df[batch_col] = out.loc[fit_idx, batch_col].astype(str).to_numpy()
    long_df[id_col] = out.loc[fit_idx, id_col].astype(str).to_numpy()
    long_df[time_col] = out.loc[fit_idx, time_col].astype(float).to_numpy()
    # group must be present + string for the C(group) term in the formula
    if "group" in out.columns:
        long_df["group"] = out.loc[fit_idx, "group"].astype(str).to_numpy()
    if "side_label" in out.columns:
        long_df["side_label"] = out.loc[fit_idx, "side_label"].astype(str).to_numpy()
    logger.info("longCombat: unpacked X=%s in %.1fs", X.shape, time.time() - t0)

    logger.info(
        "longCombat: fitting MixedLM per feature (n=%d × p=%d × batches=%d); "
        "formula=%r, ranef=%r, niter=%d, method=%s",
        len(fit_idx), n_features, len(keep_batches), formula, ranef, niter, method,
    )
    t1 = time.time()
    _lc_kwargs = dict(
        data=long_df,
        batch_col=batch_col,
        id_col=id_col,
        time_col=time_col,
        features=feat_names,
        formula=formula,
        ranef=ranef,
        niter=niter,
        method=method,
        eb=True,
        verbose=False,
    )
    if n_jobs > 1:
        logger.info(
            "longCombat: using _patched_long_combat with n_jobs=%d", n_jobs
        )
        with _patched_long_combat(n_jobs=n_jobs):
            # Re-import inside the context so we bind to the patched
            # function on both entry paths (longcombat.long_combat and
            # longcombat._core.long_combat).
            from longcombat import long_combat as _lc_entry
            result = _lc_entry(**_lc_kwargs)
    else:
        result = long_combat(**_lc_kwargs)
    logger.info("longCombat: long_combat() returned in %.1fs", time.time() - t1)

    # Repack the harmonized feature columns back into the embedding column.
    # longcombat-py returns feature columns with a ".combat" suffix (e.g.
    # "f0000.combat"), and drops any input columns not listed in the schema
    # (id_col, time_col, batch_col, and features). Row order is preserved.
    combat_cols = [f"{n}.combat" for n in feat_names]
    missing = [c for c in combat_cols if c not in result.data_combat.columns]
    if missing:
        raise RuntimeError(
            f"longCombat: expected .combat-suffixed columns for {len(missing)} "
            f"features missing from data_combat; first missing = {missing[:3]!r}. "
            f"Present columns: {list(result.data_combat.columns)[:10]!r} ..."
        )
    harm = result.data_combat[combat_cols].to_numpy().astype(np.float32)
    new_embs = list(out["embedding"].to_numpy())
    for pos, idx in enumerate(fit_idx):
        new_embs[idx] = harm[pos]
    out["embedding"] = new_embs
    logger.info("longCombat: total wall %.1fs", time.time() - t_total)
    return out


# ---------------------------------------------------------------------------
# Parallel per-feature MixedLM fitting for longitudinal ComBat
# ---------------------------------------------------------------------------
#
# `longcombat.long_combat` runs a serial `for feat in features: _fit_mixedlm(...)`
# loop over the (typically 1024–2048) feature columns. Each fit is
# independent — no cross-feature state — so we monkey-patch that loop with a
# ProcessPoolExecutor version. On our 128-core node this brings 200-patient
# MII wall from 4.5 min → ~15 s and full-cohort MC (2048 features) from
# ~160 min → ~5 min.
#
# Uses stdlib only (no joblib). The `_lc_worker_init` initializer copies the
# per-run constants (work DataFrame, formula string, batch column info,
# ranef spec, method flag) into module globals in each worker so we don't
# repay the pickle cost per feature. Each `_lc_fit_one_feature` call
# returns the three per-feature outputs the sequential loop stored into
# `sigma_estimates[v]`, `batch_effects[:, v]`, `predicted[:, v]`.

# Module-level slots populated by the worker initializer. Left as `None` in
# the parent process so accidental use outside a pool raises immediately.
_LC_WORK = None
_LC_FORMULA = None
_LC_BATCH_COL = None
_LC_BATCH_LEVELS = None
_LC_RANEF_SPEC = None
_LC_METHOD = None


def _lc_worker_init(work_pkl_bytes, formula, batch_col, batch_levels,
                    ranef_spec, method):
    """Populate module globals in each worker exactly once.

    `work` is passed as pre-pickled bytes (see `_lc_serial_work`) so the
    pool doesn't roundtrip the DataFrame through slower default pickling
    with every task. We unpickle once here per worker.
    """
    import pickle
    global _LC_WORK, _LC_FORMULA, _LC_BATCH_COL, _LC_BATCH_LEVELS
    global _LC_RANEF_SPEC, _LC_METHOD
    _LC_WORK = pickle.loads(work_pkl_bytes)
    _LC_FORMULA = formula
    _LC_BATCH_COL = batch_col
    _LC_BATCH_LEVELS = batch_levels
    _LC_RANEF_SPEC = ranef_spec
    _LC_METHOD = method


def _lc_fit_one_feature(feat):
    """Worker task: fit MixedLM for one feature, return the three outputs the
    sequential loop stored into (sigma_estimates[v], batch_effects[:, v],
    predicted[:, v]).

    Uses longcombat-py's own helpers so we don't reimplement the EB math.
    """
    from longcombat._core import (
        _fit_mixedlm, _residual_sd, _extract_batch_effects,
    )
    fit = _fit_mixedlm(_LC_WORK, feat, _LC_FORMULA, _LC_BATCH_COL,
                       _LC_RANEF_SPEC)
    sigma = _residual_sd(fit, _LC_METHOD)
    batch_effect = _extract_batch_effects(fit, _LC_BATCH_COL, _LC_BATCH_LEVELS)
    fitted = np.asarray(fit.fittedvalues, dtype=float)
    return sigma, batch_effect, fitted


class _patched_long_combat:
    """Context manager that monkey-patches longcombat._core.long_combat to
    fit the per-feature MixedLM loop in parallel via ProcessPoolExecutor.

    Same non-invasive teardown pattern as `_patched_neurocombat`. The
    upstream `long_combat` entry point and the EB math after the fit loop
    are untouched — we only replace the tight per-feature loop.

    Parameters
    ----------
    n_jobs : int
        Number of worker processes. Bounded by CPU count on the node.
        Sensible values are 8–32 (each MixedLM fit uses one BLAS thread
        internally; over-subscribing gains diminishing returns).
    """

    def __init__(self, n_jobs: int = 16):
        self.n_jobs = int(n_jobs)

    def __enter__(self):
        import longcombat._core as _lc_core
        self._mod = _lc_core
        self._orig = _lc_core.long_combat
        n_jobs = self.n_jobs
        # Wrap the original with a version that parallelizes the inner loop.
        # We replicate the outer structure of long_combat exactly (input
        # validation, prepare_batch, EB math, packing) and swap only the
        # per-feature fitting loop.
        def _parallel_long_combat(
            data, batch_col, id_col, time_col, features, formula, ranef,
            *, niter=30, method="REML", eb=True, mean_only=False,
            verbose=True,
        ):
            import time as _time
            import pickle
            from concurrent.futures import ProcessPoolExecutor
            from longcombat._core import (
                _validate_inputs, _resolve_feature_names,
                _validate_no_missing_data, _prepare_batch, parse_ranef,
                _eb_iterate, LongCombatResult,
            )

            _validate_inputs(data, batch_col, id_col, time_col, method)
            feature_names = _resolve_feature_names(data, features)
            _validate_no_missing_data(
                data, [batch_col, id_col, time_col, *feature_names]
            )

            work = data.copy()
            batch_levels, batch_codes, batch_row_indices, ni = _prepare_batch(
                work, batch_col
            )
            m = len(batch_levels)
            V = len(feature_names)
            L = len(work)

            if m < 2:
                raise ValueError(
                    f"long_combat needs at least 2 batches; found {m} in "
                    f"{batch_col!r}"
                )
            if min(ni) < 2:
                single = [
                    str(lvl) for lvl, n in zip(batch_levels, ni, strict=True)
                    if n < 2
                ]
                raise ValueError(
                    "The following batches have fewer than 2 observations: "
                    f"{', '.join(single)}. long_combat needs at least 2 "
                    "observations per batch to harmonize variance across "
                    "batches. Remove these rows or merge the batches."
                )
            if V < 2 and (eb or mean_only):
                raise ValueError(
                    "Empirical-Bayes longitudinal ComBat requires at least "
                    f"2 features (got {V})."
                )
            if verbose:
                print(f"[long_combat] found {m} batches")
                print(f"[long_combat] found {V} features")
                print(f"[long_combat] found {L} total observations")

            ranef_spec = parse_ranef(ranef)

            sigma_estimates = np.empty(V, dtype=float)
            predicted = np.empty((L, V), dtype=float)
            batch_effects = np.empty((m - 1, V), dtype=float)

            # ---- parallel per-feature fits -----
            logger.info(
                "[long_combat] fitting %d MixedLM models in parallel "
                "(n_jobs=%d)", V, n_jobs,
            )
            t0 = _time.time()
            # Pre-pickle `work` once so we don't rely on the pool default's
            # cloudpickle path (which would pickle-per-task otherwise).
            work_bytes = pickle.dumps(work, protocol=pickle.HIGHEST_PROTOCOL)
            with ProcessPoolExecutor(
                max_workers=n_jobs,
                initializer=_lc_worker_init,
                initargs=(work_bytes, formula, batch_col, batch_levels,
                          ranef_spec, method),
            ) as ex:
                for v, (sigma, batch_effect, fitted) in enumerate(
                    ex.map(_lc_fit_one_feature, feature_names, chunksize=8)
                ):
                    sigma_estimates[v] = sigma
                    batch_effects[:, v] = batch_effect
                    predicted[:, v] = fitted
            logger.info(
                "[long_combat] parallel MixedLM fits done in %.1fs "
                "(%.0f ms/feature)",
                _time.time() - t0,
                1000 * (_time.time() - t0) / V,
            )

            # ---- rest of long_combat is identical to upstream ----
            sigmas = np.broadcast_to(sigma_estimates, (L, V)).copy()
            gamma1hat = -(ni[1:] @ batch_effects) / L
            batch_effects_adjusted = np.vstack(
                [gamma1hat, batch_effects + gamma1hat]
            )
            batch_effects_expanded = batch_effects_adjusted[batch_codes, :]
            feat_matrix = work[feature_names].to_numpy(
                dtype=float, copy=True
            )
            data_std = (
                feat_matrix - predicted + batch_effects_expanded
            ) / sigmas

            gammahat = np.empty((m, V), dtype=float)
            delta2hat = np.empty((m, V), dtype=float)
            for i, idx in enumerate(batch_row_indices):
                block = data_std[idx, :]
                gammahat[i, :] = block.mean(axis=0)
                delta2hat[i, :] = block.var(axis=0, ddof=1)

            if mean_only:
                gammabar = gammahat.mean(axis=1)
                tau2bar = gammahat.var(axis=1, ddof=1)
                gammastarhat_final = (
                    (ni[:, None] * tau2bar[:, None] * gammahat
                     + gammabar[:, None])
                    / (ni[:, None] * tau2bar[:, None] + 1.0)
                )
                delta2starhat_final = np.ones((m, V), dtype=float)
            elif not eb:
                gammastarhat_final = gammahat.copy()
                delta2starhat_final = delta2hat.copy()
            else:
                gammastarhat_final, delta2starhat_final = _eb_iterate(
                    data_std=data_std,
                    batch_row_indices=batch_row_indices,
                    ni=ni,
                    gammahat=gammahat,
                    delta2hat=delta2hat,
                    niter=niter,
                    verbose=verbose,
                )

            gammastar_expanded = gammastarhat_final[batch_codes, :]
            delta2star_expanded = delta2starhat_final[batch_codes, :]
            data_combat_arr = (
                (sigmas / np.sqrt(delta2star_expanded))
                * (data_std - gammastar_expanded)
                + predicted - batch_effects_expanded
            )

            combat_feature_names = [f"{f}.combat" for f in feature_names]
            data_combat = pd.DataFrame(
                data_combat_arr,
                index=work.index,
                columns=combat_feature_names,
            )
            data_combat.insert(0, batch_col, data[batch_col].to_numpy())
            data_combat.insert(0, time_col, data[time_col].to_numpy())
            data_combat.insert(0, id_col, data[id_col].to_numpy())

            batch_index = pd.Index(batch_levels, name=batch_col)
            gammahat_df = pd.DataFrame(
                gammahat, index=batch_index, columns=feature_names
            )
            delta2hat_df = pd.DataFrame(
                delta2hat, index=batch_index, columns=feature_names
            )
            gammastar_df = pd.DataFrame(
                gammastarhat_final, index=batch_index, columns=feature_names
            )
            delta2star_df = pd.DataFrame(
                delta2starhat_final, index=batch_index, columns=feature_names
            )

            return LongCombatResult(
                data_combat=data_combat,
                gammahat=gammahat_df,
                delta2hat=delta2hat_df,
                gammastarhat=gammastar_df,
                delta2starhat=delta2star_df,
            )

        _lc_core.long_combat = _parallel_long_combat
        # Also patch the top-level import path used by our
        # apply_longitudinal_combat_manuf (`from longcombat import
        # long_combat`).
        import longcombat as _lc_top
        self._orig_top = _lc_top.long_combat
        _lc_top.long_combat = _parallel_long_combat
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._mod.long_combat = self._orig
        import longcombat as _lc_top
        _lc_top.long_combat = self._orig_top
        return False


def apply_post_harmonization(
    exam: pd.DataFrame,
    subtract_baseline: bool = False,
    zscore_dim: bool = False,
    combat_manuf: bool = False,
    combat_method: str = "standard",
    batch_col: str = "MANUFACTURER_CATEGORY",
    n_jobs: int = 1,
) -> pd.DataFrame:
    """
    Apply harmonization steps 2, 3, and/or 4 (baseline subtraction, per-dim
    z-score, ComBat-on-MANUFACTURER) to a previously-aggregated per-exam
    embedding table.

    Step 1 (per-image L2 norm) cannot be applied here — it operates on
    individual `.pt` embeddings before mean-pooling, which by definition
    have already happened in `exam`. To use step 1, re-run the aggregation
    pipeline (04a or 04b) with `--harm-l2-norm`.

    Parameters
    ----------
    combat_method : {"standard", "longitudinal"}
        When ``combat_manuf=True``, selects which ComBat variant to use.
        ``standard`` (default) — cross-sectional NeuroComBat (treats samples
        as independent). ``longitudinal`` — Beer 2020 longCombat with a
        per-patient random intercept (handles within-patient correlation).
        Longitudinal requires Python ≥3.10 (run from .venv, not .venv_mammoclip).

    This is the supported way to iterate on baseline/z-score/combat
    harmonization without paying the cost of re-reading every per-image
    .pt file.
    """
    if combat_method not in {"standard", "longitudinal"}:
        raise ValueError(
            f"combat_method must be 'standard' or 'longitudinal', "
            f"got {combat_method!r}"
        )
    out = exam
    if subtract_baseline:
        out = apply_baseline_subtraction(out)
    if zscore_dim:
        out = apply_zscore(out)
    if combat_manuf:
        if combat_method == "longitudinal":
            out = apply_longitudinal_combat_manuf(
                out, batch_col=batch_col, n_jobs=n_jobs
            )
        else:
            out = apply_combat_manuf(out, batch_col=batch_col)
    return out
