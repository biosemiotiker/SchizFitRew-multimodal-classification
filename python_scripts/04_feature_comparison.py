"""
04_feature_comparison.py
========================
Vergleich Top-Features RF (SHAP) vs. Elastic Net (Koeffizienten).
Berechnet Overlap, Modalitaets-Breakdown, PANSS- und
PAS/SAS/WHODAS/CDSS-Korrelationen.

Outputs (../results/):
  feature_overlap.csv
  panss_shap_corr.csv
  psychosocial_shap_corr.csv   <- PAS + SAS + WHODAS + CDSS kombiniert
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests

BASE        = Path(__file__).parent.parent
DATA_DIR    = BASE / "data"
SHAP_DIR    = BASE / "results" / "shap_values"
COEF_DIR    = BASE / "results" / "coefficients"
RESULTS_DIR = BASE / "results"
RESULTS_DIR.mkdir(exist_ok=True)

TOP_N = 20

# ── Label-Dictionary ──────────────────────────────────────────────────────────
FEATURE_LABELS = {
    "roi_mean ACC":                              "Anterior Cingulate Cortex",
    "roi_mean antIns":                           "Anterior Insular Cortex",
    "roi_mean Putamen":                          "Putamen",
    "roi_mean Nacc":                             "Nucleus Accumbens",
    "roi_mean Ncau":                             "Caudate Nucleus",
    "VO_max":                                    "VO2max",
    "Jump":                                      "Jump",
    "Hand_mean":                                 "Handgrip Strength",
    "Sum_Skinfolds":                             "Skinfolds Sum",
    "Brozek_BF_age":                             "Body Fat Percentage",
    "WHR":                                       "WHR",
    "WHtR":                                      "WHtR",
    "bmi":                                       "BMI",
    "IPAQ_Tot_Phys_Activity_MET.minutes_week":   "Physical Activity",
    "Average_sitting_per_day_Minutes":           "Sedentary Behavior",
}


def get_label(feature_name: str) -> str:
    return FEATURE_LABELS.get(feature_name, feature_name)


def get_modality(feature_name: str, modality_map: dict, feature_names: list) -> str:
    idx = feature_names.index(feature_name)
    for mod, indices in modality_map.items():
        if idx in indices:
            return mod
    return "unknown"


def apply_fdr(df: pd.DataFrame, p_col: str = "p_value") -> pd.DataFrame:
    _, p_fdr, _, _ = multipletests(df[p_col].values, method="fdr_bh")
    df["p_value_fdr"]        = p_fdr
    df["significant_fdr"]    = p_fdr < 0.05
    df["significant_uncorr"] = df[p_col] < 0.05
    return df


def run_correlations(shap_matrix, feature_names, top20_names, top20_idx,
                     outcome_df, outcome_cols, valid_subjects, subject_ids,
                     label: str) -> pd.DataFrame:
    valid_idx  = [subject_ids.index(s) for s in valid_subjects]
    shap_valid = shap_matrix[valid_idx]
    vals_valid = outcome_df.loc[valid_subjects, outcome_cols].values.astype(float)

    print(f"\n{label}: {len(valid_subjects)} Probanden")
    rows = []
    for pi, pc in enumerate(outcome_cols):
        col      = vals_valid[:, pi]
        nan_mask = ~np.isnan(col)
        for fi, fname in zip(top20_idx, top20_names):
            shap_col = shap_valid[nan_mask, fi]
            p_clean  = col[nan_mask]
            if len(p_clean) < 5:
                continue
            rho, pval = stats.spearmanr(shap_col, p_clean)
            rows.append({
                "feature":       fname,
                "display_label": get_label(fname),
                "scale":         pc,
                "spearman_rho":  rho,
                "p_value":       pval,
                "n":             int(nan_mask.sum()),
            })

    corr_df = pd.DataFrame(rows)
    corr_df = apply_fdr(corr_df)

    notable_uncorr = corr_df[corr_df["significant_uncorr"]].sort_values("p_value")
    notable_fdr    = corr_df[corr_df["significant_fdr"]].sort_values("p_value_fdr")

    print(f"  Signifikant (unkorrigiert p<0.05): {len(notable_uncorr)}")
    if len(notable_uncorr) > 0:
        print(notable_uncorr[["display_label", "scale", "spearman_rho",
                               "p_value", "p_value_fdr"]].to_string(index=False))
    print(f"  Signifikant (FDR-korrigiert q<0.05): {len(notable_fdr)}")
    if len(notable_fdr) > 0:
        print(notable_fdr[["display_label", "scale", "spearman_rho",
                            "p_value", "p_value_fdr"]].to_string(index=False))
    else:
        print("  → Keine Korrelation ueberlebt FDR-Korrektur")

    return corr_df


def main():
    # ── Daten laden ──────────────────────────────────────────────────────────
    with open(DATA_DIR / "feature_names.json") as f:
        feature_names = json.load(f)
    with open(DATA_DIR / "modality_map.json") as f:
        modality_map = json.load(f)
    with open(DATA_DIR / "subject_ids.json") as f:
        subject_ids = json.load(f)

    y           = np.load(DATA_DIR / "y.npy")
    shap_matrix = np.load(SHAP_DIR / "shap.npy")
    shap_imp    = pd.read_csv(SHAP_DIR / "shap_importance.csv")
    enet_coef   = pd.read_csv(COEF_DIR / "enet_coefs.csv")

    # ── Top-20 beider Modelle ─────────────────────────────────────────────────
    top_shap  = set(shap_imp.head(TOP_N)["feature"].tolist())
    top_enet  = set(enet_coef.head(TOP_N)["feature"].tolist())
    overlap   = top_shap & top_enet
    only_shap = top_shap - top_enet
    only_enet = top_enet - top_shap

    print(f"Top-{TOP_N} RF(SHAP) ∩ Elastic Net: {len(overlap)} Features")
    print(f"Nur in RF: {len(only_shap)}")
    print(f"Nur in Elastic Net: {len(only_enet)}")
    print(f"Ueberlappende Features: {sorted(overlap)}")

    for label, feat_set in [("RF (SHAP)", top_shap), ("Elastic Net", top_enet)]:
        counts = {"microbiome": 0, "fmri": 0, "fitness": 0}
        for f in feat_set:
            mod = get_modality(f, modality_map, feature_names)
            if mod in counts:
                counts[mod] += 1
        print(f"\n  Modalitaeten — {label}: {counts}")

    # ── Overlap-Tabelle speichern ─────────────────────────────────────────────
    all_features = top_shap | top_enet
    rows = []
    for feat in all_features:
        mod       = get_modality(feat, modality_map, feature_names)
        shap_rank = shap_imp[shap_imp["feature"] == feat]["rank"].values
        enet_rank = enet_coef[enet_coef["feature"] == feat]["rank"].values
        rows.append({
            "feature":       feat,
            "display_label": get_label(feat),
            "modality":      mod,
            "in_both":       feat in overlap,
            "shap_rank":     int(shap_rank[0]) if len(shap_rank) > 0 else None,
            "enet_rank":     int(enet_rank[0]) if len(enet_rank) > 0 else None,
            "mean_abs_shap": float(shap_imp[shap_imp["feature"] == feat]["mean_abs_shap"].values[0])
                             if feat in top_shap else None,
            "abs_mean_coef": float(enet_coef[enet_coef["feature"] == feat]["abs_mean_coef"].values[0])
                             if feat in top_enet else None,
        })

    overlap_df = pd.DataFrame(rows).sort_values(
        ["in_both", "shap_rank"], ascending=[False, True]
    ).reset_index(drop=True)
    overlap_df.to_csv(RESULTS_DIR / "feature_overlap.csv", index=False)
    print(f"\nOverlap-Tabelle gespeichert: feature_overlap.csv")

    # ── Top-20 SHAP Feature-Indizes ──────────────────────────────────────────
    top20_names = shap_imp.head(TOP_N)["feature"].tolist()
    top20_idx   = [feature_names.index(f) for f in top20_names]

    sz_mask = y == 1
    sz_subs = [subject_ids[i] for i in range(len(y)) if sz_mask[i]]

    # ── PANSS-Korrelation (nur SZ, n=34) ─────────────────────────────────────
    panss_df   = pd.read_csv(DATA_DIR / "panss.csv").set_index("subject_id")
    panss_cols = ["N_score_PANSS", "P_score_PANSS", "G_score_PANSS", "PANSS_total"]
    valid_sz   = [s for s in sz_subs if s in panss_df.index
                  and not panss_df.loc[s, panss_cols].isna().all()]

    panss_corr = run_correlations(
        shap_matrix, feature_names, top20_names, top20_idx,
        panss_df, panss_cols, valid_sz, subject_ids,
        label="PANSS-Korrelation (nur SZ)"
    )
    panss_corr.to_csv(RESULTS_DIR / "panss_shap_corr.csv", index=False)
    print("panss_shap_corr.csv gespeichert")

    # ── PAS/SAS/WHODAS/CDSS-Korrelation (alle 67 Probanden) ──────────────────
    pasas_df      = pd.read_csv(DATA_DIR / "pasas.csv").set_index("subject_id")
    psychosocial_cols = ["PAS", "SAS", "WHODAS", "CDSS"]

    # Fehlende Spalten aus Excel nachladen
    missing_cols = [c for c in psychosocial_cols if c not in pasas_df.columns]
    if missing_cols:
        print(f"INFO: {missing_cols} nicht in pasas.csv – lade aus Excel")
        excel_df = pd.read_excel(
            Path("data_combined_analysis/gbm_abundance_global_clr.xlsx"),
            sheet_name="gbm_abundance_global_clr"
        ).set_index("subject_id")
        for c in missing_cols:
            if c in excel_df.columns:
                pasas_df[c] = excel_df[c]
            else:
                print(f"WARNING: '{c}' nicht gefunden – wird uebersprungen")
                psychosocial_cols.remove(c)

    valid_all = [s for s in subject_ids if s in pasas_df.index
                 and not pasas_df.loc[s, psychosocial_cols].isna().all()]

    psychosocial_corr = run_correlations(
        shap_matrix, feature_names, top20_names, top20_idx,
        pasas_df, psychosocial_cols, valid_all, subject_ids,
        label="PAS/SAS/WHODAS/CDSS-Korrelation (alle Probanden)"
    )
    psychosocial_corr.to_csv(RESULTS_DIR / "psychosocial_shap_corr.csv", index=False)
    print("psychosocial_shap_corr.csv gespeichert")

    print("\nFeature-Vergleich abgeschlossen.")
    print(f"  Outputs in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
