# mammo-embedding-velocity

Code companion for the study of **embedding velocity** in screening mammography:
the rate at which frozen foundation-model embeddings move toward a learned
"cancer direction" in the years *before* a breast biopsy, measured in a matched
longitudinal case–control cohort across four foundation models.

> **Paper:** Slavkova et al., *Foundation model embeddings capture pre-diagnostic changes on screening mammograms.* arXiv:2609.26605. https://arxiv.org/abs/2609.26605

> **Scope.** This repository is **code only.** It contains no patient data, no
> image embeddings, and no model weights. To run the pipeline you supply your own
> cohort tables, images, and public model checkpoints — see
> [Expected inputs](#expected-inputs). All paths in the code and in `config.yaml`
> are placeholders (`/path/to/...`) that you fill in; see [Configuration](#configuration).

---

## Repository layout

```
src/
  01_curate_longitudinal_cohort.py    # build the longitudinal cohort (index + priors)
  02_curate_case_control_cohort.py    # 1:1 case–control matching
  03_extract_embeddings.py            # frozen foundation-model embeddings per image (GPU)
  04a_analyze_embedding_distances.py  # group separation, permutation null, projections, t-SNE
  04b_analyze_drift.py                # embedding-velocity stats (LOO cancer direction, Wilcoxon, LME)
  04c_analyze_contralateral_drift.py  # within-patient (biopsied vs. contralateral) design
  utils/                              # shared helpers (aggregation, cancer direction, harmonization, plotting)
  diagnostics/                        # supplementary analyses (effect sizes, subgroups, confounder checks, ...)
  config.py                           # loads config.yaml (or $MAMMO_CONFIG)
config.yaml                           # central path config you fill in
requirements.txt                      # curation environment (steps 01–02)
requirements_mammoclip.txt            # extraction + analysis environment (steps 03, 04a–c)
```

## Pipeline overview

| Step | Script | Consumes | Produces |
|------|--------|----------|----------|
| 1 | `01_curate_longitudinal_cohort.py` | cohort/RIS table | longitudinal frame (one row per study, `is_index`, `exam_order`, `date_diff`) |
| 2 | `02_curate_case_control_cohort.py` | longitudinal frame | matched case/control tables (`MATCH_ID`) + image paths |
| 3 | `03_extract_embeddings.py` | matched tables + images + a model checkpoint | one `.pt` embedding per image, saved as `{out}/{PATIENTID}/{STUDYID}/{SOPINSTANCEUID}.pt` |
| 4a | `04a_analyze_embedding_distances.py` | embeddings | centroid separation, label-permutation null, `û` projections, t-SNE |
| 4b | `04b_analyze_drift.py` | embeddings + matched tables | per-interval signed **embedding velocity**, leave-one-out cancer direction `û`, Wilcoxon + LME |
| 4c | `04c_analyze_contralateral_drift.py` | embeddings + matched tables | within-patient biopsied-vs-contralateral velocity |

`src/diagnostics/` holds the supplementary analyses that back the paper's
supplement (rank-biserial effect sizes, invasive-vs-DCIS subtype subgroup,
race/modality/calendar-year confounder checks, harmonization ablations, PCA
probe, etc.). Each is a standalone script with an `argparse` CLI.

---

## Expected inputs

### 1. Cohort / RIS table
A study-level dataframe (one row per `PATIENTID` × `STUDYID`, or per image with
the study columns repeated). Required / expected columns:

| Column | Meaning |
|--------|---------|
| `PATIENTID`, `STUDYID` | de-identified patient / study identifiers |
| `STUDYDATE` | exam date (`%Y%m%d`), used to order priors and compute intervals |
| `BI-RADS Category` | index assessment (0 → biopsy trigger; 1/2 → screen-negative) |
| `Breast Density` | matching variable (A–D) |
| `PATIENTAGE` | matching variable (±3 y) |
| `Laterality_Biopsy` | biopsied side (defines breast-level analysis) |
| `Cancer`, `Cancer_Description`, `subtype` | outcome + pathology (malignant/benign, invasive/DCIS, …) |
| `SEX`, `RACE`, `REGION` | demographics / subgroup + confounder variables |
| `MANUFACTURER_*`, `MG_modality` | acquisition (FFDM vs. Synth2D); harmonization / stratification |
| `SOPINSTANCEUID` | join key to the image catalogue |

Cohort rules (density/BI-RADS filtering, the ≥2-annual-priors consecutive-chain
rule, case/control definitions) are implemented in `01`/`02`; see the docstrings
and `--help`.

### 2. DICOM / image catalogue
A table (e.g. parquet) mapping `SOPINSTANCEUID` → a path to the **preprocessed**
image (the code expects fields like `fullPath_cropped_resampledBy2`). Step 3
joins on `SOPINSTANCEUID` to locate each image.

### 3. Model checkpoints
The four encoders in the paper. Provide your own for the three public models;
HOPPR is proprietary and not distributed here.

| Model | Availability |
|-------|--------------|
| Mammo-CLIP | public (see the Mammo-CLIP release) |
| MedImageInsight | public (Microsoft) |
| BiomedCLIP | public (Microsoft, PMC-15M) |
| HOPPR | **proprietary / internal — not publicly available**; its results are reported in the paper but cannot be reproduced externally |

### 4. Paths / config
Fill in `config.yaml` (every value is a `/path/to/...` placeholder). Analysis and
diagnostic scripts read it through `src/config.py`; pipeline steps also accept
`--input-dir` / `--output-dir` / `--exam-parquet-dir`-style flags (see each
script's `--help`). See [Configuration](#configuration) for details.

---

## Environments

Two Python environments are used, because the embedding-extraction /
Mammo-CLIP stack pins older dependencies:

- **Curation (steps 01–02):** Python 3.10+ — `requirements.txt`
- **Extraction + analysis (03, 04a–c, diagnostics):** Python 3.8, PyTorch 2.2.x — `requirements_mammoclip.txt`

Create each with `venv`, `uv`, or `conda` and install the matching requirements
file. Pin versions to your platform as needed.

## Running

Every script is a plain CLI:

```bash
python src/01_curate_longitudinal_cohort.py --help
python src/04b_analyze_drift.py --help
```

The original study ran these on an HPC cluster via a job scheduler, but the
scripts themselves have no scheduler dependency.

## Configuration

All data, scratch, checkpoint, and output paths are centralized in
[`config.yaml`](config.yaml) at the repo root. Edit it with your paths, or point
the `MAMMO_CONFIG` environment variable at your own file. Scripts load it through
[`src/config.py`](src/config.py):

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts under src/diagnostics/
from config import DATA_DIR, SCRATCH_DIR, RESULTS_DIR, CHECKPOINTS
```

Keys: `data_dir` (cohort tables), `scratch_dir` (analysis outputs / embeddings),
`biomedclip_scratch_dir`, `results_dir` (repo-relative), `dicom_catalogue`,
`source_tables_dir`, and `checkpoints` (per model).

**Coverage.** The reproduction-core diagnostics (effect sizes, cohort-membership
build, subtype subgroup, malignant-vs-benign, region pseudonymization) and the
loader are wired to `config.yaml`; the numbered pipeline steps take paths via CLI
flags. Some exploratory diagnostics under `src/diagnostics/` still carry inline
`/path/to/...` placeholders that map 1:1 to the same config keys — set them to
your values as needed.

## Reproducibility notes

- **Two tiers.** The drift / embedding-velocity statistics (per-interval Wilcoxon,
  rank-biserial effect sizes, LME) reproduce from the per-exam velocity tables
  that step 4 emits. The embedding-*vector* analyses (centroid separation,
  variance decomposition, PCA probe) require the raw embeddings from step 3.
- Results in the paper come from a single data source; external replication needs
  your own cohort plus the public model checkpoints above.

## Citation

If you use this code, please cite:

> Slavkova, K. P., Brattain, E., Gowd, A., Pattnaik, A., Delbrouck, J.-B.,
> Morgan, M., Bauml, J., Abderezaei, J., & Siddiqui, K. (2026). *Foundation
> model embeddings capture pre-diagnostic changes on screening mammograms.*
> arXiv:2609.26605. https://arxiv.org/abs/2609.26605

A machine-readable citation is in [`CITATION.cff`](CITATION.cff).

## License

This repository is licensed under the Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0).
See [`LICENSE`](LICENSE) or <https://creativecommons.org/licenses/by-nc-sa/4.0/>.

## Disclaimer

Research code, provided as-is, for reproducibility and methodological
transparency. **Not for clinical use.**
