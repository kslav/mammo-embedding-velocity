"""Consolidate subgroup drift stats (MG_modality + race) into one tidy CSV.

Rows = (subgroup, model, l2norm, harm_variant, granularity, comparison).
Columns capture the paired-Wilcoxon + LME result (the analysis these
match/mismatch questions are about — case-vs-control pairs). The unpaired
malignant-vs-biopsy-negative MWU is intentionally excluded: subgroup
matching is defined on case/control pairs, so it does not apply to that
within-cases contrast, and 04b computes it on the full cohort regardless
of the pair filter.

Subgroups covered:
  - MG_modality match through prior_{2,3,4} + mismatch (depth-2 discordant).
  - Race match / mismatch (5-bin scheme; race was not a matching criterion).
"""
import json
from pathlib import Path
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import SCRATCH_DIR as BASE  # noqa: E402
OUT = Path("results/subgroup_numbers.csv")

MODELS = [("mammoclip", ""), ("mammoclip", "_l"), ("hoppr", "_hoppr"),
          ("hoppr", "_hoppr_l"), ("mii", "_mii"), ("mii", "_mii_l")]
# (harm_variant label, scratch dir suffix)
HARMS = [("baseline", ""), ("cs_combat", "_combat_group"), ("long_combat", "_combat_long")]
# (subgroup label, subdir template)
SUBGROUPS = [("mgmod_match_p2", "subgroup_mgmod_p2"),
             ("mgmod_match_p3", "subgroup_mgmod_p3"),
             ("mgmod_match_p4", "subgroup_mgmod_p4"),
             ("mgmod_mismatch_p2", "subgroup_mgmod_p2_mismatch"),
             ("race_match", "subgroup_racematch"),
             ("race_mismatch", "subgroup_racemismatch")]
COMPARISONS = ["malignant_vs_control", "biopsy_negative_vs_control"]
GRANS = ["breast", "study"]

rows = []
for model, msuf in MODELS:
    l2 = "L2" if msuf.endswith("_l") else "raw"
    for harm_name, hsuf in HARMS:
        for sg_name, sg_dir in SUBGROUPS:
            d = BASE / f"drift_analysis_augmented_335_425{msuf}{hsuf}" / sg_dir
            summ = d / "summary.json"
            if not summ.is_file():
                continue
            s = json.loads(summ.read_text())
            for gran in GRANS:
                for comp in COMPARISONS:
                    e = s.get(f"{gran}__{comp}", {})
                    pw = e.get("patient_wilcoxon", {})
                    lme = e.get("lme", {})
                    rows.append({
                        "subgroup": sg_name,
                        "model": model,
                        "l2norm": l2,
                        "harm_variant": harm_name,
                        "granularity": gran,
                        "comparison": comp,
                        "n_pairs": pw.get("n_pairs"),
                        "wilcoxon_p": pw.get("pvalue"),
                        "median_diff": pw.get("median_diff"),
                        "lme_interaction_beta": lme.get("interaction_beta"),
                        "lme_interaction_p": lme.get("interaction_pvalue"),
                    })

df = pd.DataFrame(rows)
OUT.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUT, index=False)
print(f"Wrote {OUT}: {len(df)} rows")
print(df.groupby(["subgroup","comparison"]).size().to_string())
