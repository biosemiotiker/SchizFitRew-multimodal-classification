"""
03_classify_elasticnet.py
=========================
Elastic Net logistic regression classification (SZ vs. CTRL).

AENDERUNGEN gegenueber alter Version:
- Liest X_raw.npy + covariates.npy statt X_A/X_B
- Konfound-Regression und Scaling innerhalb der CV-Folds (kein Data Leakage)
- Gleiches Label-Dictionary wie 02_classify_rf.py

Inputs (from ../data/):
  X_raw.npy, covariates.npy, y.npy, feature_names.json

Outputs (../results/):
  cv_metrics/enet_metrics.csv
  cv_metrics/enet_summary.csv
  coefficients/enet_coefs.csv
  coefficients/enet_roc_data.pkl
"""

import json
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    matthews_corrcoef, confusion_matrix, roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Pfade ──────────────────────────────────────────────────────────────────
BASE        = Path(__file__).parent.parent
DATA_DIR    = BASE / "data"
METRICS_DIR = BASE / "results" / "cv_metrics"
COEF_DIR    = BASE / "results" / "coefficients"
METRICS_DIR.mkdir(parents=True, exist_ok=True)
COEF_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameter-Suchraum ────────────────────────────────────────────────
N_FOLDS   = 5
L1_RATIOS = [0.1, 0.5, 0.9]
C_VALUES  = [0.001, 0.01, 0.1, 1.0, 10.0]

# ── Label-Dictionary fuer Figures (publikationsreif) ──────────────────────
FEATURE_LABELS = {
    # fMRI
    "roi_mean ACC":     "Anterior Cingulate Cortex",
    "roi_mean antIns":  "Anterior Insular Cortex",
    "roi_mean Putamen": "Putamen",
    "roi_mean Nacc":    "Nucleus Accumbens",
    "roi_mean Ncau":    "Caudate Nucleus",
    # Fitness
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


# ── Custom Transformer: Konfound-Regression ────────────────────────────────
class ConfoundRegressor(BaseEstimator, TransformerMixin):
    """
    Regressiert Kovariaten (Alter, Geschlecht) aus Feature-Matrix heraus.
    Wird INNERHALB jedes CV-Folds gefittet – kein Data Leakage.
    """
    def __init__(self):
        self.lr_ = None

    def fit(self, X, covariates):
        self.lr_ = LinearRegression()
        self.lr_.fit(covariates, X)
        return self

    def transform(self, X, covariates):
        return X - self.lr_.predict(covariates)


# ── Hilfsfunktion: Metriken ────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_prob) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    return {
        "auroc":       roc_auc_score(y_true, y_prob),
        "accuracy":    accuracy_score(y_true, y_pred),
        "f1_macro":    f1_score(y_true, y_pred, average="macro"),
        "mcc":         matthews_corrcoef(y_true, y_pred),
        "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else np.nan,
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
    }


# ── Haupt-CV-Funktion ──────────────────────────────────────────────────────
def run_cv(X_raw: np.ndarray, covariates: np.ndarray,
           y: np.ndarray, feature_names: list):

    print("\n=== Elastic Net (Konfound-Regression + Scaling innerhalb CV) ===")

    skf          = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_metrics = []
    coef_matrix  = []
    roc_data     = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_raw, y)):

        X_train_raw = X_raw[train_idx]
        X_test_raw  = X_raw[test_idx]
        cov_train   = covariates[train_idx]
        cov_test    = covariates[test_idx]
        y_train     = y[train_idx]
        y_test      = y[test_idx]

        # 1. Konfound-Regression: NUR auf Trainingsdaten fitten
        cr = ConfoundRegressor()
        cr.fit(X_train_raw, cov_train)
        X_train_res = cr.transform(X_train_raw, cov_train)
        X_test_res  = cr.transform(X_test_raw,  cov_test)

        # 2. Scaling: NUR auf Trainingsdaten fitten
        scaler      = StandardScaler()
        X_train     = scaler.fit_transform(X_train_res)
        X_test      = scaler.transform(X_test_res)

        # 3. Innere CV: Hyperparameter-Suche (nur auf Trainingsdaten)
        best_auroc = -np.inf
        best_model = None
        inner_cv   = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        for l1r in L1_RATIOS:
            for C in C_VALUES:
                model = LogisticRegression(
                    penalty="elasticnet", solver="saga",
                    l1_ratio=l1r, C=C,
                    class_weight="balanced",
                    max_iter=10000, random_state=42,
                )
                inner_aurocs = []
                for tr, va in inner_cv.split(X_train, y_train):
                    model.fit(X_train[tr], y_train[tr])
                    prob = model.predict_proba(X_train[va])[:, 1]
                    try:
                        inner_aurocs.append(roc_auc_score(y_train[va], prob))
                    except ValueError:
                        inner_aurocs.append(0.5)
                mean_inner = np.mean(inner_aurocs)
                if mean_inner > best_auroc:
                    best_auroc = mean_inner
                    best_model = (l1r, C)

        # 4. Bestes Modell auf vollem Trainingsfold trainieren
        l1r_best, C_best = best_model
        model = LogisticRegression(
            penalty="elasticnet", solver="saga",
            l1_ratio=l1r_best, C=C_best,
            class_weight="balanced",
            max_iter=10000, random_state=42,
        )
        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        metrics               = compute_metrics(y_test, y_pred, y_prob)
        metrics["fold"]       = fold + 1
        metrics["best_l1r"]   = l1r_best
        metrics["best_C"]     = C_best
        fold_metrics.append(metrics)
        coef_matrix.append(model.coef_[0])

        n_nonzero = np.sum(model.coef_[0] != 0)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data.append({"fold": fold+1, "fpr": fpr.tolist(),
                         "tpr": tpr.tolist(), "auc": metrics["auroc"]})

        print(
            f"  Fold {fold+1}: AUROC={metrics['auroc']:.3f}  "
            f"Acc={metrics['accuracy']:.3f}  MCC={metrics['mcc']:.3f}  "
            f"l1_ratio={l1r_best}  C={C_best}  "
            f"non-zero coefs={n_nonzero}/{len(feature_names)}"
        )

    # ── Zusammenfassung ────────────────────────────────────────────────────
    df_folds     = pd.DataFrame(fold_metrics)
    numeric_cols = ["auroc", "accuracy", "f1_macro", "mcc", "sensitivity", "specificity"]
    summary      = df_folds[numeric_cols].agg(["mean", "std"]).T
    summary.columns = ["mean", "std"]

    df_folds.to_csv(METRICS_DIR / "enet_metrics.csv", index=False)
    summary.to_csv(METRICS_DIR  / "enet_summary.csv")

    print("\n  Zusammenfassung (mean+/-std):")
    for col in numeric_cols:
        print(f"    {col:12s}: {summary.loc[col,'mean']:.3f} +/- {summary.loc[col,'std']:.3f}")

    # ── Koeffizienten ──────────────────────────────────────────────────────
    coef_arr       = np.array(coef_matrix)
    mean_coefs     = coef_arr.mean(axis=0)
    std_coefs      = coef_arr.std(axis=0)
    n_zero_per_fold = (coef_arr == 0).sum(axis=1)
    print(f"\n  Sparsity check — zero coefs per fold: {n_zero_per_fold.tolist()}")

    display_labels = [get_label(f) for f in feature_names]

    coef_df = pd.DataFrame({
        "feature":          feature_names,
        "display_label":    display_labels,
        "mean_coef":        mean_coefs,
        "std_coef":         std_coefs,
        "abs_mean_coef":    np.abs(mean_coefs),
        "n_folds_nonzero":  (coef_arr != 0).sum(axis=0),
    }).sort_values("abs_mean_coef", ascending=False).reset_index(drop=True)
    coef_df["rank"] = range(1, len(coef_df) + 1)

    coef_df.to_csv(COEF_DIR / "enet_coefs.csv", index=False)

    with open(COEF_DIR / "enet_roc_data.pkl", "wb") as f:
        pickle.dump(roc_data, f)

    print("\n  Top-10 Features (|mean coefficient|):")
    print(coef_df[["rank","display_label","mean_coef","abs_mean_coef"]].head(10).to_string(index=False))

    return coef_df


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    X_raw      = np.load(DATA_DIR / "X_raw.npy")
    covariates = np.load(DATA_DIR / "covariates.npy")
    y          = np.load(DATA_DIR / "y.npy")

    with open(DATA_DIR / "feature_names.json") as f:
        feature_names = json.load(f)

    run_cv(X_raw, covariates, y, feature_names)

    print("\nElastic Net Klassifikation abgeschlossen.")


if __name__ == "__main__":
    main()
