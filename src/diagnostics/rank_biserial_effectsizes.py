"""Rank-biserial effect sizes (and AUC for the unpaired tables) beside every
per-interval Wilcoxon p in the manuscript.

Co-author asked for a standardized effect size next to each Wilcoxon p. This
script recomputes the published Holm p for each cell (verification that our row
subset is right) and, next to it, the matched-pair rank-biserial r for the
paired tables or the rank-biserial r + AUC for the two unpaired Mann-Whitney
tables. One value per p.

Tables covered (same subsetting as their generators, verified by p-match):
  main   Table 2  tab:primary          between mal-vs-ctrl, breast   [swap median->r]
  main   Table 4  tab:contralat        within, full                 [add r]
  main   Table 5  tab:contralat-stable within, modality-stable       [add r]
  main   Table 6  tab:subgroup         race strata, between          [add r]
  supp   Table S5 tab:study            study-level between           [add r]
  supp   Table S7 tab:bnprimary        between BN-vs-ctrl, breast     [swap median->r]
  supp   Table S14 tab:mvb-between     unpaired MWU mal-vs-benign     [add r + AUC]
  supp   Table S15 tab:mvb-within      unpaired MWU asym mal-vs-ben   [add r + AUC]
  supp   Table S16 tab:subtype-between between by subtype            [add r]
  supp   Table S17 tab:subtype-within  within by subtype             [add r]

Paired r (Kerby 2014 matched-pairs rank-biserial): on the nonzero paired diffs
d, r = (W+ - W-)/(W+ + W-) with W+/W- the summed ranks of |d| for d>0 / d<0.
Unpaired r/AUC: AUC = U1/(n_a*n_b) with U1 = MWU statistic for the first sample
(one-sided a>b); r = 2*AUC - 1.

Reads only committed results/ CSVs (per-model velocity intervals, which carry
the de-identified PATIENTID, plus cohort_membership.csv for subgroup membership),
so it reproduces every r/AUC from a repo-only path. Light (<1 min); run via
slurm_jobs/diagnostics/slurm_diagnostic.sh (VENV=mammoclip).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon, mannwhitneyu
from statsmodels.stats.multitest import multipletests

REPO = Path("/path/to/repo")
R = REPO / "results"
MODELS = ["Mammo-CLIP", "HOPPR", "MedImageInsight", "BiomedCLIP"]
MAXIV = 4

# All inputs are committed under results/: per-model velocity intervals (which
# now carry the de-identified PATIENTID) and the cohort membership table, so the
# script reproduces every r/AUC from a repo-only path.
MODELKEY = {"Mammo-CLIP": "mammoclip", "HOPPR": "hoppr",
            "MedImageInsight": "mii", "BiomedCLIP": "biomedclip"}
BETWEEN = {m: R / f"{MODELKEY[m]}_l_combat_long" for m in MODELS}
CONTRA = {m: R / f"{MODELKEY[m]}_l_contralat_combat_long" for m in MODELS}

# committed de-identified cohort table -> subgroup-membership sets
# (build_cohort_membership.py). Between-patient subsets key on the case PATIENTID.
_COH = pd.read_csv(R / "cohort_membership.csv", low_memory=False)
_PP = _COH.drop_duplicates("PATIENTID")


def _flag(col: str, cohort: str | None = None) -> set:
    d = _PP if cohort is None else _PP[_PP["cohort"] == cohort]
    return set(d[d[col]]["PATIENTID"])


# depth-adaptive modality-matched sets (case PATIENTIDs), one per interval depth
MGMOD = {1: _flag("mgmod_p2", "case"), 2: _flag("mgmod_p2", "case"),
         3: _flag("mgmod_p3", "case"), 4: _flag("mgmod_p4", "case")}
RACEMATCH = _flag("racematch", "case")
RACEMISMATCH = _flag("racemismatch", "case")
MODSTABLE = _flag("modstable")               # cases OR controls (within-patient)
MODSTABLE_CASES = _flag("modstable", "case")
FFDM = _flag("ffdm_all")


def subtype_map() -> dict:
    mal = _PP[(_PP["cohort"] == "case") & (_PP["Cancer"] == 1)]
    inv = {"idc", "carcinoma_nos", "ilc", "invasive_nos", "mucinous",
           "papillary_carcinoma", "metaplastic", "adenocarcinoma", "mixed_ductal_lobular"}
    dcis = {"dcis_nos", "dcis_intermediate", "dcis_high_grade", "dcis_low_grade"}
    m = {}
    for pid, s in zip(mal["PATIENTID"], mal["subtype"]):
        if s in inv:
            m[pid] = "invasive"
        elif s in dcis:
            m[pid] = "dcis"
    return m


# ---- effect sizes -----------------------------------------------------------
def rb_paired(d: np.ndarray) -> float:
    """Matched-pairs rank-biserial on paired diffs d (sign = one-sided direction)."""
    nz = d[d != 0]
    if nz.size == 0:
        return np.nan
    ranks = rankdata(np.abs(nz))
    T = ranks.sum()
    return float((ranks[nz > 0].sum() - ranks[nz < 0].sum()) / T)


def p_paired(d: np.ndarray) -> float:
    return wilcoxon(d, alternative="greater").pvalue if (d != 0).sum() >= 10 else np.nan


def rb_auc_unpaired(a: np.ndarray, b: np.ndarray):
    """(p, r, AUC) for one-sided MWU a>b."""
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 10 or len(b) < 10:
        return np.nan, np.nan, np.nan
    u1 = mannwhitneyu(a, b, alternative="greater").statistic
    auc = u1 / (len(a) * len(b))
    return mannwhitneyu(a, b, alternative="greater").pvalue, 2 * auc - 1, auc


def holm(ps: list[float]) -> list[float]:
    idx = [i for i, p in enumerate(ps) if p == p]
    out = [np.nan] * len(ps)
    if idx:
        c = multipletests([ps[i] for i in idx], method="holm")[1]
        for j, i in enumerate(idx):
            out[i] = c[j]
    return out


# ---- paired d-vector builders (mirror each table's generator) ---------------
def d_between(df: pd.DataFrame, case_group: str, iv: int, keep=None) -> np.ndarray:
    if keep is not None:
        good = set(df[(df.group == case_group) & df.PATIENTID.isin(keep)].pair_id)
        df = df[df.pair_id.isin(good)]
    s = df[df.interval_idx == iv]
    piv = s.pivot_table(index="pair_id", columns="group", values="v_signed", aggfunc="mean")
    piv = piv.dropna(subset=[case_group, "control"])
    return (piv[case_group] - piv["control"]).to_numpy()


def d_within(df: pd.DataFrame, iv: int, keep=None) -> np.ndarray:
    if keep is not None:
        df = df[df.PATIENTID.isin(keep)]
    s = df[df.interval_idx == iv]
    per = s.groupby(["PATIENTID", "side_label"])["v_signed"].mean().reset_index()
    aff = per[per.side_label == "affected"][["PATIENTID", "v_signed"]]
    con = per[per.side_label == "contralateral"][["PATIENTID", "v_signed"]]
    m = aff.merge(con, on="PATIENTID", suffixes=("_a", "_c"))
    return (m.v_signed_a - m.v_signed_c).to_numpy()


# ---- per-table drivers ------------------------------------------------------
def hdr(title):
    print(f"\n{'='*78}\n{title}\n{'='*78}")


def paired_column(dvecs: list[np.ndarray]) -> list[tuple]:
    """dvecs indexed iv1..iv4 -> [(n, holm_p, r), ...]."""
    raw = [p_paired(d) for d in dvecs]
    hp = holm(raw)
    return [(len(d), hp[i], rb_paired(d)) for i, d in enumerate(dvecs)]


def fmt_paired(cells):
    return "  ".join(
        f"iv{i+1}(n={n}) p={p:.2g} r={r:+.2f}" if p == p else f"iv{i+1}(n={n}) p=n/a r={r:+.2f}"
        for i, (n, p, r) in enumerate(cells))


def run_between(title, csv_name, case_group, mm=False, gran="breast"):
    hdr(title)
    for model in MODELS:
        df = pd.read_csv(BETWEEN[model] / gran / csv_name)
        full = paired_column([d_between(df, case_group, iv) for iv in range(1, MAXIV + 1)])
        print(f"[{model}] FULL   {fmt_paired(full)}")
        if mm:
            mmc = paired_column([d_between(df, case_group, iv, keep=MGMOD[iv])
                                 for iv in range(1, MAXIV + 1)])
            print(f"[{model}] Mm     {fmt_paired(mmc)}")


def run_within(title, keep, arms=("malignant", "biopsy_negative", "control")):
    hdr(title)
    for model in MODELS:
        for arm in arms:
            df = pd.read_csv(CONTRA[model] / "breast_both_sides"
                             / f"contralateral_{arm}_intervals.csv")
            col = paired_column([d_within(df, iv, keep=keep) for iv in range(1, MAXIV + 1)])
            print(f"[{model}] {arm:15s} {fmt_paired(col)}")


def run_race():
    hdr("Table 6  tab:subgroup  (race strata, between mal-vs-ctrl, breast)")
    for model in MODELS:
        df = pd.read_csv(BETWEEN[model] / "breast" / "breast_malignant_vs_control_intervals.csv")
        for name, keep in [("race-match", RACEMATCH), ("race-mismatch", RACEMISMATCH)]:
            col = paired_column([d_between(df, "malignant", iv, keep=keep)
                                 for iv in range(1, MAXIV + 1)])
            print(f"[{model}] {name:14s} {fmt_paired(col)}")


def run_subtype():
    smap = subtype_map()
    from collections import Counter
    print("\nsubtype sizes:", Counter(smap.values()))
    # between
    hdr("Table S16  tab:subtype-between  (between mal-vs-ctrl by subtype)")
    for model in MODELS:
        df = pd.read_csv(BETWEEN[model] / "breast" / "breast_malignant_vs_control_intervals.csv")
        malp = df[df.group == "malignant"][["pair_id", "PATIENTID"]].drop_duplicates()
        pair_sub = dict(zip(malp.pair_id, malp.PATIENTID.map(smap)))
        df["sub"] = df.pair_id.map(pair_sub)
        for grp in ("invasive", "dcis"):
            sub = df[df["sub"] == grp]
            for label, keep in [("Full", None), ("Mm", MGMOD[1])]:
                dv = [d_between(sub, "malignant", iv, keep=keep) for iv in range(1, MAXIV + 1)]
                print(f"[{model}] {grp:9s} {label:4s} {fmt_paired(paired_column(dv))}")
    # within
    hdr("Table S17  tab:subtype-within  (within biopsied-vs-contra by subtype)")
    for model in MODELS:
        df = pd.read_csv(CONTRA[model] / "breast_both_sides" / "contralateral_malignant_intervals.csv")
        df["sub"] = df.PATIENTID.map(smap)
        for grp in ("invasive", "dcis"):
            sub = df[df["sub"] == grp]
            for label, keep in [("Full", None), ("Stable", MODSTABLE_CASES)]:
                dv = [d_within(sub, iv, keep=keep) for iv in range(1, MAXIV + 1)]
                print(f"[{model}] {grp:9s} {label:6s} {fmt_paired(paired_column(dv))}")


def run_mvb_between():
    hdr("Table S14  tab:mvb-between  (unpaired MWU mal>benign, v per interval)")
    for model in MODELS:
        df = pd.read_csv(BETWEEN[model] / "breast" / "breast_malignant_vs_biopsy_negative_intervals.csv")
        for label, keep in [("Full", None), ("FFDM", FFDM)]:
            d = df if keep is None else df[df.PATIENTID.isin(keep)]
            raw, cells = [], []
            for iv in range(1, MAXIV + 1):
                s = d[d.interval_idx == iv]
                a = s.loc[s.group == "malignant", "v_signed"].to_numpy()
                b = s.loc[s.group == "biopsy_negative", "v_signed"].to_numpy()
                p, r, auc = rb_auc_unpaired(a, b)
                raw.append(p); cells.append((len(a[~np.isnan(a)]), len(b[~np.isnan(b)]), r, auc))
            hp = holm(raw)
            txt = "  ".join(
                f"iv{i+1}(nm{na}/nb{nb}) p={hp[i]:.2g} r={r:+.2f} AUC={auc:.2f}"
                if hp[i] == hp[i] else f"iv{i+1}(nm{na}/nb{nb}) p=n/a"
                for i, (na, nb, r, auc) in enumerate(cells))
            print(f"[{model}] {label:4s} {txt}")


def _asym(frames_dir: Path, keep=None) -> pd.DataFrame:
    frames = [pd.read_csv(frames_dir / f"contralateral_{a}_intervals.csv")
              for a in ("malignant", "biopsy_negative")]
    df = pd.concat(frames, ignore_index=True)
    if keep is not None:
        df = df[df.PATIENTID.isin(keep)]
    piv = df.pivot_table(index=["PATIENTID", "group", "interval_idx"],
                         columns="side_label", values="v_signed", aggfunc="mean").reset_index()
    piv["asym"] = piv["affected"] - piv["contralateral"]
    return piv


def run_mvb_within():
    hdr("Table S15  tab:mvb-within  (unpaired MWU asym mal>benign)")
    for model in MODELS:
        base = CONTRA[model] / "breast_both_sides"
        for label, keep in [("Full", None), ("Modstable", MODSTABLE)]:
            piv = _asym(base, keep=keep)
            raw, cells = [], []
            for iv in range(1, MAXIV + 1):
                s = piv[piv.interval_idx == iv]
                a = s.loc[s.group == "malignant", "asym"].to_numpy()
                b = s.loc[s.group == "biopsy_negative", "asym"].to_numpy()
                p, r, auc = rb_auc_unpaired(a, b)
                raw.append(p); cells.append((len(a[~np.isnan(a)]), len(b[~np.isnan(b)]), r, auc))
            hp = holm(raw)
            txt = "  ".join(
                f"iv{i+1}(nm{na}/nb{nb}) p={hp[i]:.2g} r={r:+.2f} AUC={auc:.2f}"
                if hp[i] == hp[i] else f"iv{i+1}(nm{na}/nb{nb}) p=n/a"
                for i, (na, nb, r, auc) in enumerate(cells))
            print(f"[{model}] {label:9s} {txt}")


def main():
    run_between("Table 2  tab:primary  (between mal-vs-ctrl, breast, Full+Mm)",
                "breast_malignant_vs_control_intervals.csv", "malignant", mm=True)
    run_between("Table S7  tab:bnprimary  (between BN-vs-ctrl, breast, Full+Mm)",
                "breast_biopsy_negative_vs_control_intervals.csv", "biopsy_negative", mm=True)
    run_between("Table S5  tab:study  (study-level between mal-vs-ctrl, Full+Mm)",
                "study_malignant_vs_control_intervals.csv", "malignant", mm=True, gran="study")
    run_within("Table 4  tab:contralat  (within, FULL cohort)", keep=None)
    run_within("Table 5  tab:contralat-stable  (within, MODALITY-STABLE)", keep=MODSTABLE)
    run_race()
    run_subtype()
    run_mvb_between()
    run_mvb_within()


if __name__ == "__main__":
    main()
