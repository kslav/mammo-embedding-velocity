"""Malignant-vs-benign within-patient side-asymmetry, modality-stable subgroup,
per interval (iv1-4), one-sided Mann-Whitney U (malignant > benign) on the
affected-minus-contralateral v_signed. Holm-corrected across iv1-4 within each
cohort. Companion computation for supplement Table (mvb-within), modality-stable
column extended beyond iv2. Reads the committed contralateral interval CSVs
(PATIENTID retained in scratch; here we join modstable PIDs from the matched
CSVs). Prints per-interval p/n for all four models.
"""
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

S = Path("/path/to/scratch/drift_cohort")
D = Path("/path/to/data/FM_Mammo/hoppr_data/breast_cancer/drift_analysis")
CONTRA = {
    "Mammo-CLIP": "drift_analysis_augmented_335_425_mammoclip_l_contralat_combat_long",
    "HOPPR": "drift_analysis_augmented_335_425_hoppr_l_contralat_combat_long",
    "MedImageInsight": "drift_analysis_augmented_335_425_mii_l_contralat_combat_long",
    "BiomedCLIP": "drift_analysis_augmented_335_425_biomedclip_l_contralat_combat_long",
}
MAXIV = 4


def modstable_pids():
    c = pd.read_csv(D / "matched_cases_Density_nPriors_d335_d425_mesh_modstable.csv",
                    usecols=["PATIENTID"])
    k = pd.read_csv(D / "matched_controls_Density_nPriors_d335_d425_mesh_modstable.csv",
                    usecols=["PATIENTID"])
    return set(c["PATIENTID"]) | set(k["PATIENTID"])


def holm(ps):
    idx = [i for i, p in enumerate(ps) if p == p]
    out = [np.nan] * len(ps)
    if idx:
        c = multipletests([ps[i] for i in idx], method="holm")[1]
        for j, i in enumerate(idx):
            out[i] = c[j]
    return out


def main():
    keep = modstable_pids()
    for model, sub in CONTRA.items():
        base = S / sub / "breast_both_sides"
        frames = [pd.read_csv(base / f"contralateral_{a}_intervals.csv")
                  for a in ("malignant", "biopsy_negative")]
        df = pd.concat(frames, ignore_index=True)
        df = df[df["PATIENTID"].isin(keep)]
        piv = df.pivot_table(index=["PATIENTID", "group", "interval_idx"],
                             columns="side_label", values="v_signed",
                             aggfunc="mean").reset_index()
        piv["asym"] = piv["affected"] - piv["contralateral"]
        rows = []
        for iv in range(1, MAXIV + 1):
            s = piv[piv["interval_idx"] == iv]
            a = s.loc[s["group"] == "malignant", "asym"].dropna().to_numpy()
            b = s.loc[s["group"] == "biopsy_negative", "asym"].dropna().to_numpy()
            p = (mannwhitneyu(a, b, alternative="greater").pvalue
                 if len(a) >= 10 and len(b) >= 10 else np.nan)
            rows.append((iv, len(a), len(b), p))
        ph = holm([r[3] for r in rows])
        cells = " ".join(
            f"iv{r[0]}(nm{r[1]}/nb{r[2]})={pc:.2g}" if pc == pc else f"iv{r[0]}(nm{r[1]}/nb{r[2]})=n/a"
            for r, pc in zip(rows, ph))
        print(f"{model:16s} {cells}")


if __name__ == "__main__":
    main()
