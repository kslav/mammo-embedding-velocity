"""Pathology-subtype subgroup analysis (invasive vs DCIS) for the malignant arm,
reusing committed headline (L2 + longCombat) per-interval velocities. No re-extraction.

Between-patient: paired one-sided Wilcoxon (malignant > matched control) on v_signed
per interval, within each subtype group, Holm-corrected across intervals 1-4.
Within-patient: paired one-sided Wilcoxon (biopsied > contralateral) on v_signed
per interval, within each subtype group, Holm-corrected.
Subtype is per malignant patient; controls inherit their matched case's subtype.
"""
import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_DIR as D, SCRATCH_DIR as B, BIOMEDCLIP_SCRATCH_DIR as E  # noqa: E402
MAXIV = 4

INVASIVE = {"idc", "carcinoma_nos", "ilc", "invasive_nos", "mucinous",
            "papillary_carcinoma", "metaplastic", "adenocarcinoma", "mixed_ductal_lobular"}
DCIS = {"dcis_nos", "dcis_intermediate", "dcis_high_grade", "dcis_low_grade"}

BETWEEN = {
    "Mammo-CLIP": B / "drift_analysis_augmented_335_425_l_combat_long",
    "HOPPR": B / "drift_analysis_augmented_335_425_hoppr_l_combat_long",
    "MedImageInsight": B / "drift_analysis_augmented_335_425_mii_l_combat_long",
    "BiomedCLIP": E / "drift_analysis_biomedclip_l_combat_long",
}
CONTRA = {
    "Mammo-CLIP": B / "drift_analysis_augmented_335_425_mammoclip_l_contralat_combat_long",
    "HOPPR": B / "drift_analysis_augmented_335_425_hoppr_l_contralat_combat_long",
    "MedImageInsight": B / "drift_analysis_augmented_335_425_mii_l_contralat_combat_long",
    "BiomedCLIP": B / "drift_analysis_augmented_335_425_biomedclip_l_contralat_combat_long",
}


def subtype_map():
    ca = pd.read_csv(D / "matched_cases_Density_nPriors_d335_d425_mesh.csv",
                     usecols=["PATIENTID", "Cancer", "subtype"], low_memory=False)
    mal = ca[ca["Cancer"] == 1].drop_duplicates("PATIENTID")
    m = {}
    for pid, s in zip(mal["PATIENTID"], mal["subtype"]):
        if s in INVASIVE:
            m[pid] = "invasive"
        elif s in DCIS:
            m[pid] = "dcis"
    return m


def pid_set(fname):
    return set(pd.read_csv(D / fname, usecols=["PATIENTID"])["PATIENTID"])


MGMOD_P2 = pid_set("matched_cases_Density_nPriors_d335_d425_mesh_mgmod_p2.csv")  # between: modality-matched
MODSTABLE = pid_set("matched_cases_Density_nPriors_d335_d425_mesh_modstable.csv")  # within: modality-stable


def holm(ps):
    idx = [i for i, p in enumerate(ps) if p == p]
    out = [np.nan] * len(ps)
    if idx:
        c = multipletests([ps[i] for i in idx], method="holm")[1]
        for j, i in enumerate(idx):
            out[i] = c[j]
    return out


def between(model, smap, keep=None):
    df = pd.read_csv(BETWEEN[model] / "breast" / "breast_malignant_vs_control_intervals.csv")
    # subtype attaches to the malignant patient; propagate to each pair via pair_id
    mal = df[df["group"] == "malignant"][["pair_id", "PATIENTID"]].drop_duplicates()
    if keep is not None:
        mal = mal[mal["PATIENTID"].isin(keep)]
    mal["sub"] = mal["PATIENTID"].map(smap)
    pair_sub = dict(zip(mal["pair_id"], mal["sub"]))
    df["sub"] = df["pair_id"].map(pair_sub)
    res = {}
    for grp in ("invasive", "dcis"):
        sub = df[df["sub"] == grp]
        ps, ns = [], []
        for iv in range(1, MAXIV + 1):
            s = sub[sub["interval_idx"] == iv]
            piv = s.pivot_table(index="pair_id", columns="group", values="v_signed", aggfunc="mean")
            piv = piv.dropna(subset=["malignant", "control"])
            d = (piv["malignant"] - piv["control"]).to_numpy()
            p = wilcoxon(d, alternative="greater").pvalue if len(d) >= 10 else np.nan
            ps.append(p); ns.append(len(d))
        res[grp] = (ns, holm(ps))
    return res


def within(model, smap, keep=None):
    df = pd.read_csv(CONTRA[model] / "breast_both_sides" / "contralateral_malignant_intervals.csv")
    if keep is not None:
        df = df[df["PATIENTID"].isin(keep)]
    df["sub"] = df["PATIENTID"].map(smap)
    res = {}
    for grp in ("invasive", "dcis"):
        sub = df[df["sub"] == grp]
        ps, ns = [], []
        for iv in range(1, MAXIV + 1):
            s = sub[sub["interval_idx"] == iv]
            piv = s.pivot_table(index="PATIENTID", columns="side_label", values="v_signed", aggfunc="mean")
            piv = piv.dropna(subset=["affected", "contralateral"])
            d = (piv["affected"] - piv["contralateral"]).to_numpy()
            p = wilcoxon(d, alternative="greater").pvalue if len(d) >= 10 else np.nan
            ps.append(p); ns.append(len(d))
        res[grp] = (ns, holm(ps))
    return res


def fmt(ns, ps):
    return "  ".join(f"iv{i+1}(n={ns[i]})={ps[i]:.2g}" if ps[i]==ps[i] else f"iv{i+1}(n={ns[i]})=n/a"
                     for i in range(MAXIV))


def main():
    smap = subtype_map()
    from collections import Counter
    print("subtype group sizes (malignant patients):", Counter(smap.values()), "\n")
    passes = [
        ("BETWEEN-PATIENT (malignant vs control), FULL cohort", between, None),
        ("BETWEEN-PATIENT, MODALITY-MATCHED (mgmod_p2)", between, MGMOD_P2),
        ("WITHIN-PATIENT (biopsied vs contralateral), FULL", within, None),
        ("WITHIN-PATIENT, MODALITY-STABLE", within, MODSTABLE),
    ]
    for analysis, fn, keep in passes:
        print(f"===== {analysis} =====")
        for model in BETWEEN:
            r = fn(model, smap, keep=keep)
            for grp in ("invasive", "dcis"):
                ns, ps = r[grp]
                print(f"  {model:16s} {grp:9s} {fmt(ns, ps)}")
        print()


if __name__ == "__main__":
    main()
