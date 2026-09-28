"""Build the example-views supplement figure.

Four patient rows x four view columns (LCC, LMLO, RCC, RMLO):

    row 1: malignant case          (index study)
    row 2: its matched control
    row 3: benign (biopsy-negative) case
    row 4: its matched control

The four patients are chosen by src/diagnostics/select_example_views.py — the
MG-modality-matched malignant and benign cases whose index projection onto u_hat
sits farthest above their matched control, constrained to density-matched,
age-similar examples with a standard 2L+2R index view set. This shows,
qualitatively, what "far apart on u_hat at the index" looks like: the case rows
should read as more suspicious than their matched-control rows.

Images are the preprocessed breast-cropped .jph (fullPath_cropped_resampledBy2),
the same source step 03 embeds. CC vs MLO is inferred per image from the
pectoral-muscle signature (MLO carries more tissue toward the top of the frame),
since the catalogue carries laterality but not view. Reads the selection CSV +
the matched cohort CSVs (per-image laterality + paths).
"""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

DRIFT_CSV_DIR = Path(
    "/path/to/data/FM_Mammo/hoppr_data/breast_cancer/drift_analysis")
CASES = DRIFT_CSV_DIR / "matched_cases_Density_nPriors_d335_d425_mesh.csv"
CONTROLS = DRIFT_CSV_DIR / "matched_controls_Density_nPriors_d335_d425_mesh.csv"

ROW_LABELS = {
    "malignant_case": "Malignant case",
    "malignant_control": "Matched control",
    "benign_case": "Benign (biopsy-neg) case",
    "benign_control": "Matched control",
}
ROW_ORDER = ["malignant_case", "malignant_control", "benign_case", "benign_control"]
COL_ORDER = ["LCC", "LMLO", "RCC", "RMLO"]


def _load(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        a = np.array(Image.open(BytesIO(f.read()))).astype(np.float32)
    return (a - a.min()) / (a.max() - a.min() + 1e-8)


def _is_mlo(img: np.ndarray, laterality: str) -> float:
    """Score how MLO-like an image is (higher = more MLO). In an MLO view the
    pectoral muscle and axillary tissue extend to the very top edge of the frame,
    whereas a CC view's tissue sits lower with a near-empty top edge. The mean
    intensity of the top ~10% of rows separates the two cleanly and robustly
    (validated on the selected studies; the older chest-wall-corner heuristic
    misordered some pairs). ``laterality`` is unused but kept for the call site."""
    h = img.shape[0]
    return float(img[: max(1, h // 10)].mean())


def _assign_views(rows: pd.DataFrame) -> dict:
    """Map the 4 image rows of a study to LCC/LMLO/RCC/RMLO by laterality + the
    MLO score (higher score = MLO within each side)."""
    out = {}
    for lat in ("L", "R"):
        side = rows[rows["IMAGELATERALITY"] == lat]
        scored = sorted(
            ((_is_mlo(_load(r["fullPath_cropped_resampledBy2"]), lat), r)
             for _, r in side.iterrows()),
            key=lambda t: t[0])
        # lowest score = CC, highest = MLO
        if len(scored) >= 2:
            out[f"{lat}CC"] = scored[0][1]
            out[f"{lat}MLO"] = scored[-1][1]
        elif len(scored) == 1:
            out[f"{lat}CC"] = scored[0][1]
    return out


def build(selection_csv: Path, out_path: Path, also_png: bool = False) -> None:
    sel = pd.read_csv(selection_csv)
    alld = pd.concat([pd.read_csv(CASES, low_memory=False),
                      pd.read_csv(CONTROLS, low_memory=False)], ignore_index=True)

    fig, axes = plt.subplots(4, 4, figsize=(8.5, 10.5))
    for ri, role in enumerate(ROW_ORDER):
        s = sel[sel["role"] == role].iloc[0]
        study_rows = alld[(alld["STUDYID"] == s["STUDYID"]) & (alld["is_index"])]
        views = _assign_views(study_rows)
        affected = s["affected_side"] if pd.notna(s["affected_side"]) else None
        for ci, view in enumerate(COL_ORDER):
            ax = axes[ri, ci]
            ax.set_xticks([]); ax.set_yticks([])
            if view in views:
                # aspect="equal" (imshow default) preserves the true pixel aspect
                # ratio so the breast is not stretched to fill the cell.
                ax.imshow(_load(views[view]["fullPath_cropped_resampledBy2"]),
                          cmap="gray")
                # mark the affected side for cases with a colored spine
                if affected and view.startswith(affected):
                    for sp in ax.spines.values():
                        sp.set_color("#d62728"); sp.set_linewidth(2.5)
                else:
                    for sp in ax.spines.values():
                        sp.set_color("0.7"); sp.set_linewidth(0.6)
            else:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center",
                        transform=ax.transAxes, color="0.6")
            if ri == 0:
                ax.set_title(view, fontsize=11, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(ROW_LABELS[role], fontsize=9.5, fontweight="bold")
    fig.tight_layout(h_pad=0.4, w_pad=0.3)
    fig.savefig(out_path, bbox_inches="tight")
    if also_png:
        fig.savefig(out_path.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--selection-csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--also-png", action="store_true")
    args = ap.parse_args()
    build(args.selection_csv, args.out, args.also_png)


if __name__ == "__main__":
    main()
