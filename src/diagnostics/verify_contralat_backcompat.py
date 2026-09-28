import sys
sys.path.insert(0, "/path/to/repo/src")
import numpy as np
import pandas as pd
from pathlib import Path
from utils.cohort_embeddings import build_exam_table

CASES = Path("/path/to/data/FM_Mammo/hoppr_data/breast_cancer/drift_analysis/matched_cases_Density_nPriors_d335_d425_mesh.csv")
CTRLS = Path("/path/to/data/FM_Mammo/hoppr_data/breast_cancer/drift_analysis/matched_controls_Density_nPriors_d335_d425_mesh.csv")
EMB = Path("/path/to/scratch/drift_cohort/embeddings_mii")
REF = Path("/path/to/scratch/drift_cohort/drift_analysis_augmented_335_425_mii/breast/exam_embeddings.parquet")

df_cases = pd.read_csv(CASES)
df_ctrls = pd.read_csv(CTRLS)
print("Re-aggregating MII breast (this branch's code)...")
new = build_exam_table(df_cases, df_ctrls, EMB, granularity="breast", emb_dim=1024)
print(f"  new: {len(new)} rows, {new['PATIENTID'].nunique()} patients")

ref = pd.read_parquet(REF)
print(f"  ref: {len(ref)} rows, {ref['PATIENTID'].nunique()} patients")

# Align on (PATIENTID, STUDYID) and compare embeddings elementwise.
key = ["PATIENTID", "STUDYID"]
merged = new[key + ["embedding"]].merge(
    ref[key + ["embedding"]], on=key, suffixes=("_new", "_ref"), how="inner"
)
print(f"  merged: {len(merged)} common rows (unmatched new: {len(new) - len(merged)}, ref: {len(ref) - len(merged)})")

max_abs = 0.0
mean_abs = 0.0
n = len(merged)
for _, row in merged.iterrows():
    a = np.asarray(row["embedding_new"], dtype=np.float64)
    b = np.asarray(row["embedding_ref"], dtype=np.float64)
    diff = np.abs(a - b)
    max_abs = max(max_abs, diff.max())
    mean_abs += diff.mean() / n
print(f"  max |Δ| across all elements: {max_abs:.3e}")
print(f"  mean |Δ| across all elements: {mean_abs:.3e}")
print("  PASS" if max_abs < 1e-4 else "  FAIL — regression!")
