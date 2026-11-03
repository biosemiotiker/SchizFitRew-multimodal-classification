"""
02_classify_rf.py
=================
Random Forest classification (SZ vs. CTRL) with 5-fold stratified CV + SHAP.

AENDERUNGEN gegenueber alter Version:
- Liest X_raw.npy (nicht skaliert) + covariates.npy statt X_A/X_B
- Konfound-Regression und Scaling passieren INNERHALB der CV-Folds (kein Data Leakage)
- Custom ConfoundRegressor als sklearn-kompatibler Transformer
- Label-Dictionary fuer publikationsreife Figure-Beschriftungen

Inputs (from ../data/):
  X_raw.npy, covariates.npy, y.npy, feature_names.json

Outputs (../results/):
  cv_metrics/rf_metrics.csv        per-fold metrics
  cv_metrics/rf_summary.csv        mean+/-std summary
  shap_values/shap.npy             (n_subjects, n_features)
  shap_values/shap_importance.csv  mean |SHAP| per feature
  shap_values/roc_data.pkl         ROC curves per fold
"""

import json
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    matthews_corrcoef, confusion_matrix, roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import shap

warnings.filterwarnings("ignore")

# ── Pfade ──────────────────────────────────────────────────────────────────
BASE        = Path(__file__).parent.parent
DATA_DIR    = BASE / "data"
METRICS_DIR = BASE / "results" / "cv_metrics"
SHAP_DIR    = BASE / "results" / "shap_values"
METRICS_DIR.mkdir(parents=True, exist_ok=True)
SHAP_DIR.mkdir(parents=True, exist_ok=True)

# ── Random Forest Parameter ────────────────────────────────────────────────
RF_PARAMS = dict(
    n_estimators=500,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
N_FOLDS = 5

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
    """Gibt publikationsreifen Namen zurueck, sonst Original."""
    return FEATURE_LABELS.get(feature_name, feature_name)


# ── Custom Transformer: Konfound-Regression ────────────────────────────────
class ConfoundRegressor(BaseEstimator, TransformerMixin):
    """
    Regressiert Kovariaten (Alter, Geschlecht) aus Feature-Matrix heraus.
    fit() und transform() erhalten die Kovariaten als separates Argument.
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

    print("\n=== Random Forest (Konfound-Regression + Scaling innerhalb CV) ===")

    skf          = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_metrics = []
    shap_matrix  = np.zeros((len(y), X_raw.shape[1]))
    fold_probs   = np.zeros(len(y))
    fold_preds   = np.zeros(len(y), dtype=int)
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
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train_res)
        X_test  = scaler.transform(X_test_res)

        # 3. Random Forest trainieren
        rf = RandomForestClassifier(**RF_PARAMS)
        rf.fit(X_train, y_train)

        y_prob = rf.predict_proba(X_test)[:, 1]
        y_pred = rf.predict(X_test)

        metrics         = compute_metrics(y_test, y_pred, y_prob)
        metrics["fold"] = fold + 1
        fold_metrics.append(metrics)

        # 4. SHAP auf Testdaten
        explainer = shap.TreeExplainer(rf)
        sv        = explainer.shap_values(X_test)
        if isinstance(sv, list):
            sv = sv[1]
        elif isinstance(sv, np.ndarray) and sv.ndim == 3:
            sv = sv[:, :, 1]
        shap_matrix[test_idx] = sv
        fold_probs[test_idx]  = y_prob
        fold_preds[test_idx]  = y_pred

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_data.append({"fold": fold+1, "fpr": fpr.tolist(),
                         "tpr": tpr.tolist(), "auc": metrics["auroc"]})

        print(
            f"  Fold {fold+1}: AUROC={metrics['auroc']:.3f}  "
            f"Acc={metrics['accuracy']:.3f}  MCC={metrics['mcc']:.3f}  "
            f"Sens={metrics['sensitivity']:.3f}  Spec={metrics['specificity']:.3f}"
        )

    # ── Zusammenfassung ────────────────────────────────────────────────────
    df_folds     = pd.DataFrame(fold_metrics)
    numeric_cols = ["auroc", "accuracy", "f1_macro", "mcc", "sensitivity", "specificity"]
    summary      = df_folds[numeric_cols].agg(["mean", "std"]).T
    summary.columns = ["mean", "std"]

    df_folds.to_csv(METRICS_DIR / "rf_metrics.csv", index=False)
    summary.to_csv(METRICS_DIR  / "rf_summary.csv")

    print("\n  Zusammenfassung (mean+/-std):")
    for col in numeric_cols:
        print(f"    {col:12s}: {summary.loc[col,'mean']:.3f} +/- {summary.loc[col,'std']:.3f}")

    # Sanity Check
    fi_sum = rf.feature_importances_.sum()
    print(f"\n  Sanity check — feature_importances_.sum() = {fi_sum:.6f} (sollte 1.0 sein)")

    # ── SHAP Importance ────────────────────────────────────────────────────
    mean_abs_shap  = np.abs(shap_matrix).mean(axis=0)
    display_labels = [get_label(f) for f in feature_names]

    shap_importance = pd.DataFrame({
        "feature":       feature_names,
        "display_label": display_labels,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    shap_importance["rank"] = range(1, len(shap_importance) + 1)

    np.save(SHAP_DIR / "shap.npy",       shap_matrix)
    np.save(SHAP_DIR / "fold_probs.npy", fold_probs)
    np.save(SHAP_DIR / "fold_preds.npy", fold_preds)
    shap_importance.to_csv(SHAP_DIR / "shap_importance.csv", index=False)

    with open(SHAP_DIR / "roc_data.pkl", "wb") as f:
        pickle.dump(roc_data, f)

    print("\n  Top-10 Features (mean |SHAP|):")
    print(shap_importance[["rank","display_label","mean_abs_shap"]].head(10).to_string(index=False))

    return shap_matrix, shap_importance


# ── Permutations-Baseline ──────────────────────────────────────────────────
def permutation_baseline(X_raw: np.ndarray, covariates: np.ndarray,
                         y: np.ndarray, n_perm: int = 1000) -> np.ndarray:
    """Shuffled Labels -> AUROC sollte ~0.5 sein.

    Aktualisiert 2026-07: n_perm von 20 auf 1000 erhoeht, um einen
    belastbaren empirischen p-Wert zu erlauben (statt nur eines groben
    Sanity-Checks). Gibt jetzt das volle Array der Permutations-AUROCs
    zurueck (statt nur des Mittelwerts), damit main() Perzentile/p-Wert
    berechnen kann. RNG-Sequenz (RandomState(0), sequenzielle
    .permutation(y)-Aufrufe) ist identisch zur alten Version, sodass die
    ersten 20 Werte mit der frueheren n=20-Baseline uebereinstimmen.
    """
    rng    = np.random.RandomState(0)
    aurocs = []
    skf    = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=0)

    for _ in range(n_perm):
        y_perm      = rng.permutation(y)
        fold_aurocs = []
        for train_idx, test_idx in skf.split(X_raw, y_perm):
            cr = ConfoundRegressor()
            cr.fit(X_raw[train_idx], covariates[train_idx])
            X_tr = cr.transform(X_raw[train_idx], covariates[train_idx])
            X_te = cr.transform(X_raw[test_idx],  covariates[test_idx])

            scaler = StandardScaler()
            X_tr   = scaler.fit_transform(X_tr)
            X_te   = scaler.transform(X_te)

            rf = RandomForestClassifier(**RF_PARAMS)
            rf.fit(X_tr, y_perm[train_idx])
            prob = rf.predict_proba(X_te)[:, 1]
            fold_aurocs.append(roc_auc_score(y_perm[test_idx], prob))
        aurocs.append(np.mean(fold_aurocs))

    return np.array(aurocs, dtype=float)


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    X_raw      = np.load(DATA_DIR / "X_raw.npy")
    covariates = np.load(DATA_DIR / "covariates.npy")
    y          = np.load(DATA_DIR / "y.npy")

    with open(DATA_DIR / "feature_names.json") as f:
        feature_names = json.load(f)

    run_cv(X_raw, covariates, y, feature_names)

    print("\n=== Permutations-Baseline (1000 Permutationen) ===")
    perm_aurocs = permutation_baseline(X_raw, covariates, y, n_perm=1000)
    perm_mean   = float(perm_aurocs.mean())
    perm_std    = float(perm_aurocs.std())
    print(f"  Permutierter AUROC (Mittelwert): {perm_mean:.3f} +/- {perm_std:.3f} (erwartet: ~0.5)")

    rf_summary_path = METRICS_DIR / "rf_summary.csv"
    observed_auroc  = pd.read_csv(rf_summary_path, index_col=0).loc["auroc", "mean"]
    n_perm          = len(perm_aurocs)
    n_geq           = int(np.sum(perm_aurocs >= observed_auroc))
    empirical_p     = (n_geq + 1) / (n_perm + 1)
    print(f"  Beobachteter AUROC: {observed_auroc:.3f}  |  empirischer p-Wert: {empirical_p:.4f}")

    pd.DataFrame([{
        "permutation_auroc_mean": perm_mean,
        "permutation_auroc_std": perm_std,
        "n_permutations": n_perm,
        "observed_auroc": observed_auroc,
        "empirical_p_value": empirical_p,
        "n_perm_geq_observed": n_geq,
        "permutation_auroc_p95": float(np.percentile(perm_aurocs, 95)),
        "permutation_auroc_p99": float(np.percentile(perm_aurocs, 99)),
    }]).to_csv(METRICS_DIR / "permutation_baseline.csv", index=False)

    pd.DataFrame({"permutation_auroc": perm_aurocs}).to_csv(
        METRICS_DIR / "permutation_baseline_1000.csv", index=False
    )

    print("\nRandom Forest Klassifikation abgeschlossen.")


if __name__ == "__main__":
    main()
