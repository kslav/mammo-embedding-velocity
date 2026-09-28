"""Modality-stable within-patient contralateral results, to mirror Fig 5B.

Reuses the Fig-5B modality-stable PID set (patients whose MG modality is fixed
across their whole chain) and the committed contralateral interval CSVs (which
carry per-interval v_signed and PATIENTID) plus the headline exam parquets (for
u_hat position). Emits, per model (headline L2 + long-ComBat):

  (A) per-interval paired Wilcoxon (affected > contralateral), iv1-4  -> Table 5
  (B) per-exam-order u_hat-position difference (affected - contralateral) -> Table S8

restricted to modality-stable patients, for malignant / biopsy_negative / control.
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR as D, SCRATCH_DIR as SCRATCH  # noqa: E402
CONTRA = {
    "Mammo-CLIP": "drift_analysis_augmented_335_425_mammoclip_l_contralat_combat_long",
    "HOPPR": "drift_analysis_augmented_335_425_hoppr_l_contralat_combat_long",
    "MedImageInsight": "drift_analysis_augmented_335_425_mii_l_contralat_combat_long",
    "BiomedCLIP": "drift_analysis_augmented_335_425_biomedclip_l_contralat_combat_long",
}
ARMS = ["malignant", "biopsy_negative", "control"]
MAXIV = 4


def modstable_pids():
    c = pd.read_csv(D / "matched_cases_Density_nPriors_d335_d425_mesh_modstable.csv",
                    usecols=["PATIENTID"])
    k = pd.read_csv(D / "matched_controls_Density_nPriors_d335_d425_mesh_modstable.csv",
                    usecols=["PATIENTID"])
    return set(c["PATIENTID"]) | set(k["PATIENTID"])


def holm(ps):
    ps = list(ps)
    idx = [i for i, p in enumerate(ps) if p is not None and not np.isnan(p)]
    if not idx:
        return [np.nan] * len(ps)
    corr = multipletests([ps[i] for i in idx], method="holm")[1]
    out = [np.nan] * len(ps)
    for j, i in enumerate(idx):
        out[i] = corr[j]
    return out


def per_interval(df, keep):
    """(A) affected>contralateral paired Wilcoxon per interval, Holm across iv."""
    d = df[df["PATIENTID"].isin(keep)]
    rows = []
    for iv in range(1, MAXIV + 1):
        s = d[d["interval_idx"] == iv]
        piv = s.pivot_table(index="PATIENTID", columns="side_label",
                            values="v_signed", aggfunc="mean")
        piv = piv.dropna(subset=["affected", "contralateral"])
        diff = (piv["affected"] - piv["contralateral"]).to_numpy()
        n = len(diff)
        if n < 10 or np.allclose(diff, 0):
            rows.append({"iv": iv, "n": n, "p_raw": np.nan})
        else:
            rows.append({"iv": iv, "n": n,
                         "p_raw": wilcoxon(diff, alternative="greater").pvalue})
    for r, pc in zip(rows, holm([r["p_raw"] for r in rows])):
        r["p_holm"] = pc
    return rows


def per_order_position(parquet, keep, uhat):
    """(B) affected-contralateral u_hat position per exam_order, Holm across order."""
    df = pd.read_parquet(parquet)
    df = df[df["exam_order"] <= MAXIV].copy()
    df["proj"] = np.stack(df["embedding"].values) @ uhat
    df["proj_z"] = (df["proj"] - df["proj"].mean()) / df["proj"].std()
    d = df[df["PATIENTID"].isin(keep)]
    rows = []
    for o in range(MAXIV + 1):
        s = d[d["exam_order"] == o]
        piv = s.pivot_table(index="PATIENTID", columns="side_label",
                            values="proj_z", aggfunc="mean").dropna(
                            subset=["affected", "contralateral"])
        diff = (piv["affected"] - piv["contralateral"]).to_numpy()
        n = len(diff)
        if n < 10 or np.allclose(diff, 0):
            rows.append({"order": o, "n": n, "median": float(np.median(diff)) if n else np.nan, "p_raw": np.nan})
        else:
            rows.append({"order": o, "n": n, "median": float(np.median(diff)),
                         "p_raw": wilcoxon(diff, alternative="greater").pvalue})
    for r, pc in zip(rows, holm([r["p_raw"] for r in rows])):
        r["p_holm"] = pc
    return rows


def uhat_affected(parquet):
    df = pd.read_parquet(parquet)
    idx = df[(df["is_index"]) & (df["side_label"] == "affected")]
    m = np.stack(idx.loc[idx["group"] == "malignant", "embedding"].values).mean(0)
    c = np.stack(idx.loc[idx["group"] == "control", "embedding"].values).mean(0)
    u = m - c
    return u / np.linalg.norm(u)


def main():
    keep = modstable_pids()
    print(f"modstable PIDs: {len(keep)}\n")
    for model, sub in CONTRA.items():
        base = SCRATCH / sub / "breast_both_sides"
        parquet = base / "exam_embeddings.parquet"
        u = uhat_affected(parquet)
        print(f"===== {model} =====")
        # (A) per-interval, per arm
        for arm in ARMS:
            f = base.parent / "breast_both_sides" / f"contralateral_{arm}_intervals.csv"
            df = pd.read_csv(f)
            rows = per_interval(df, keep)
            cells = " ".join(f"iv{r['iv']}={r['p_holm']:.2g}(n{r['n']})" if not np.isnan(r['p_holm']) else f"iv{r['iv']}=NA(n{r['n']})" for r in rows)
            print(f"  [A iv] {arm:16s} {cells}")
        # (B) per-order position, per arm
        for arm in ARMS:
            rows = per_order_position(parquet, keep & set(pd.read_csv(base.parent/'breast_both_sides'/f'contralateral_{arm}_intervals.csv',usecols=['PATIENTID'])['PATIENTID']), u)
            cells = " ".join(f"o{r['order']}={r['median']:+.3f}/{r['p_holm']:.2g}(n{r['n']})" if not np.isnan(r['p_holm']) else f"o{r['order']}={r['median']:+.3f}/NA(n{r['n']})" for r in rows)
            print(f"  [B pos] {arm:16s} {cells}")
        print()


if __name__ == "__main__":
    main()
