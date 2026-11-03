"""
01_preprocess.py
================
Load and preprocess multimodal data for SZ vs. CTRL classification.

AENDERUNGEN gegenueber alter Version:
- KEIN StandardScaler hier (kein Data Leakage)
- KEINE Konfound-Regression hier (kein Data Leakage)
- Beide Schritte passieren innerhalb der CV-Folds in 02_ und 03_
- NEU: pasas.csv gespeichert fuer alle 67 Probanden (CTRL + SZ)

Outputs (saved to ../data/):
  X_raw.npy          Rohe Feature-Matrix (118 Features), NICHT skaliert
  covariates.npy     [age, gender] fuer Konfound-Regression in CV
  y.npy              Binaere Labels (0=CTRL, 1=SZ/SZ_out)
  feature_names.json Liste der 118 Feature-Namen
  panss.csv          PANSS N/P/G/Total fuer SZ-Patienten (n=34)
  pasas.csv          PAS + SAS fuer ALLE Probanden (n=67)
  subject_ids.json   Subject-IDs in Zeilenreihenfolge
  modality_map.json  Feature-Index-zu-Modalitaet Mapping
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

# ── Pfade ──────────────────────────────────────────────────────────────────
BASE       = Path(__file__).parent.parent
DATA_DIR   = BASE / "data"
EXCEL_PATH = Path("data_combined_analysis/gbm_abundance_global_clr.xlsx")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ── Spalten-Definitionen ───────────────────────────────────────────────────
MICRO_COLS_IDX = slice(1, 104)

FMRI_COLS = [
    "roi_mean ACC",
    "roi_mean antIns",
    "roi_mean Putamen",
    "roi_mean Nacc",
    "roi_mean Ncau",
]

FITNESS_COLS = [
    "VO_max",
    "Jump",
    "Hand_mean",
    "Sum_Skinfolds",
    "Brozek_BF_age",
    "WHR",
    "WHtR",
    "bmi",
    "IPAQ_Tot_Phys_Activity_MET.minutes_week",
    "Average_sitting_per_day_Minutes",
]

IPAQ_COL       = "IPAQ_Tot_Phys_Activity_MET.minutes_week"
COVARIATE_COLS = ["age", "gender"]
PANSS_COLS     = ["N_score_PANSS", "P_score_PANSS", "G_score_PANSS", "PANSS_total"]
PASAS_COLS     = ["PAS", "SAS", "WHODAS", "CDSS"]


def load_raw():
    df = pd.read_excel(EXCEL_PATH, sheet_name="gbm_abundance_global_clr")
    print(f"Geladen: {df.shape[0]} Probanden x {df.shape[1]} Spalten")
    return df


def build_labels(df):
    y = np.where(df["group"].str.strip() == "CTRL", 0, 1)
    print(f"Labels: {(y==0).sum()} CTRL, {(y==1).sum()} SZ/SZ_out")
    return y.astype(np.int32)


def preprocess_features(df):
    micro_data  = df.iloc[:, MICRO_COLS_IDX].values.astype(float) / 1e15
    micro_names = list(df.columns[MICRO_COLS_IDX])

    fmri_data  = df[FMRI_COLS].values.astype(float)
    fmri_names = FMRI_COLS

    fitness_raw           = df[FITNESS_COLS].copy()
    fitness_raw[IPAQ_COL] = np.log1p(fitness_raw[IPAQ_COL].values)
    fitness_data          = fitness_raw.values.astype(float)
    fitness_names         = FITNESS_COLS

    X_raw         = np.hstack([micro_data, fmri_data, fitness_data])
    feature_names = micro_names + fmri_names + fitness_names

    assert X_raw.shape[1] == 118, f"Erwartet 118 Features, erhalten {X_raw.shape[1]}"
    print(f"Feature-Matrix: {X_raw.shape}  (roh, noch NICHT skaliert)")
    return X_raw, feature_names


def main():
    df                   = load_raw()
    y                    = build_labels(df)
    X_raw, feature_names = preprocess_features(df)
    covariates           = df[COVARIATE_COLS].values.astype(float)
    subject_ids          = df["Column1"].tolist()

    np.save(DATA_DIR / "X_raw.npy",      X_raw)
    np.save(DATA_DIR / "covariates.npy", covariates)
    np.save(DATA_DIR / "y.npy",          y)

    with open(DATA_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    with open(DATA_DIR / "subject_ids.json", "w") as f:
        json.dump(subject_ids, f, indent=2)

    modality_map = {
        "microbiome": list(range(0,   103)),
        "fmri":       list(range(103, 108)),
        "fitness":    list(range(108, 118)),
    }
    with open(DATA_DIR / "modality_map.json", "w") as f:
        json.dump(modality_map, f, indent=2)

    # PANSS - nur SZ (n=34)
    sz_mask  = y == 1
    panss_df = df.loc[sz_mask, ["Column1", "group"] + PANSS_COLS].copy()
    panss_df.rename(columns={"Column1": "subject_id"}, inplace=True)
    panss_df.to_csv(DATA_DIR / "panss.csv", index=False)
    print(f"PANSS gespeichert: {sz_mask.sum()} SZ-Probanden")

    # PAS/SAS - alle 67 Probanden (NEU)
    pasas_df = df[["Column1", "group"] + PASAS_COLS].copy()
    pasas_df.rename(columns={"Column1": "subject_id"}, inplace=True)
    pasas_df.to_csv(DATA_DIR / "pasas.csv", index=False)
    print(f"PAS/SAS gespeichert: {len(pasas_df)} Probanden (CTRL + SZ)")

    print("\n── Preprocessing abgeschlossen ─────────────────────────────────")
    print(f"  X_raw shape (nicht skaliert): {X_raw.shape}")
    print(f"  covariates shape:             {covariates.shape}")
    print(f"  y shape:                      {y.shape}")
    print(f"  Outputs in: {DATA_DIR}")
    print("\n  HINWEIS: Scaling + Konfound-Regression passieren jetzt")
    print("  INNERHALB der CV-Folds in 02_classify_rf.py und 03_classify_elasticnet.py")


if __name__ == "__main__":
    main()
