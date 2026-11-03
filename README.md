# Multimodal classification of schizophrenia (reward fMRI, fitness, gut-metabolic modules)

Analysis code for the manuscript "Multimodal classification of schizophrenia ..." (Translational Psychiatry, submission 2026TP002154R).

## Contents
- `python_scripts/` - machine-learning pipeline (run in numerical order): preprocessing, Random Forest + TreeSHAP, Elastic Net (nested CV), feature overlap, modality ablation, stability, group descriptives and sensitivity analyses, cross-modality coupling, revision figures.
- `R_scripts/` - earlier R analyses (PCA, a-priori partial correlations, normality tests). File paths inside them must be adapted.
- `requirements.txt` - package versions used for the revision analyses.

## Data
No data are included. The individual-level clinical, imaging and metagenomic data are sensitive patient data without consent for public deposition. De-identified data may be requested from the corresponding author (Oliver Grimm, University Hospital Frankfurt). The scripts expect the master table `data_combined_analysis/gbm_abundance_global_clr.xlsx` and the intermediate files written by `01_preprocess.py` (`data/`), relative to the working directory.

## Notes
- Cross-validation: 5 outer stratified folds (random_state=42); 3 inner folds for Elastic Net hyper-parameters.
- Python 3.11, Linux.
- License: add one (e.g. MIT) before release.
