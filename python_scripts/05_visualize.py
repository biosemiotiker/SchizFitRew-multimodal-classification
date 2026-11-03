"""
05_visualize.py
===============
Erstellt alle Figures fuer den ML-Analyse-Report.

AENDERUNGEN gegenueber alter Version:
- Aktualisierte Pfade (keine Varianten A/B mehr)
- Publikationsreife Labels via FEATURE_LABELS Dictionary
- Neues Farbschema (publikationsfreundlich, druckstabil)
- PAS/SAS-Korrelations-Heatmap neu hinzugefuegt

Outputs (../results/figures/):
  fig1_roc_curves.png
  fig2_shap_beeswarm.png
  fig3_shap_modality.png
  fig4_enet_coefficients.png
  fig5_venn_overlap.png
  fig6_panss_heatmap.png
  fig6b_pasas_heatmap.png   <- NEU
  fig7_group_boxplots.png
"""

import json
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
from matplotlib_venn import venn2

warnings.filterwarnings("ignore")

# ── Pfade ──────────────────────────────────────────────────────────────────
BASE        = Path(__file__).parent.parent
DATA_DIR    = BASE / "data"
SHAP_DIR    = BASE / "results" / "shap_values"
COEF_DIR    = BASE / "results" / "coefficients"
RESULTS_DIR = BASE / "results"
FIG_DIR     = BASE / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── Farbschema ─────────────────────────────────────────────────────────────
COLORS = {
    "ctrl":        "#f0a500",   # warmes Gold-Gelb
    "sz":          "#2c7bb6",   # Blau
    "fmri":        "#e8432d",   # Rotorange
    "fitness":     "#5a9e6f",   # Gruen
    "microbiome":  "#6b3a2a",   # Braun
    "shap_high":   "#d62728",   # Rot
    "shap_low":    "#7b5ea7",   # Lila
}

MODALITY_COLORS = {
    "fmri":       COLORS["fmri"],
    "fitness":    COLORS["fitness"],
    "microbiome": COLORS["microbiome"],
}

# ── Label-Dictionary ───────────────────────────────────────────────────────
FEATURE_LABELS = {
    "roi_mean ACC":     "Anterior Cingulate Cortex",
    "roi_mean antIns":  "Anterior Insular Cortex",
    "roi_mean Putamen": "Putamen",
    "roi_mean Nacc":    "Nucleus Accumbens",
    "roi_mean Ncau":    "Caudate Nucleus",
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

def get_label(name: str) -> str:
    return FEATURE_LABELS.get(name, name)

SAVE_KW = dict(dpi=300, bbox_inches="tight")


# ── Fig 1: ROC Curves ──────────────────────────────────────────────────────
def plot_roc(roc_path, title, out_path):
    with open(roc_path, "rb") as f:
        roc_data = pickle.load(f)

    fig, ax = plt.subplots(figsize=(5, 5))
    aucs = []
    for fold_data in roc_data:
        ax.plot(fold_data["fpr"], fold_data["tpr"],
                color=COLORS["sz"], alpha=0.4, lw=1.2)
        aucs.append(fold_data["auc"])

    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"{title}\nMean AUC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f}", fontsize=12)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    sns.despine(ax=ax)
    fig.savefig(out_path, **SAVE_KW)
    plt.close(fig)
    print(f"  Gespeichert: {out_path.name}")


# ── Fig 2: SHAP Beeswarm ───────────────────────────────────────────────────
def plot_shap_beeswarm(shap_matrix, X_raw, feature_names, shap_imp, top_n=20):
    top_features = shap_imp.head(top_n)["feature"].tolist()
    top_idx      = [feature_names.index(f) for f in top_features]
    top_labels   = [get_label(f) for f in top_features]

    shap_top = shap_matrix[:, top_idx]
    X_top    = X_raw[:, top_idx]

    cmap = LinearSegmentedColormap.from_list(
        "shap_cmap", [COLORS["shap_low"], "#f7f7f7", COLORS["shap_high"]]
    )

    fig, ax = plt.subplots(figsize=(9, 8))
    for i, (feat_idx, label) in enumerate(zip(range(top_n), top_labels)):
        shap_vals = shap_top[:, feat_idx]
        feat_vals = X_top[:, feat_idx]
        feat_norm = (feat_vals - feat_vals.min()) / (feat_vals.ptp() + 1e-9)

        jitter = np.random.RandomState(feat_idx).uniform(-0.3, 0.3, len(shap_vals))
        colors = cmap(feat_norm)
        ax.scatter(shap_vals, np.full_like(shap_vals, top_n - 1 - i) + jitter,
                   c=colors, s=18, alpha=0.8, linewidths=0)

    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_labels[::-1], fontsize=9)
    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("SHAP Value", fontsize=11)
    ax.set_title(f"SHAP Feature Importance – Top {top_n}", fontsize=12)

    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.04)
    cb.set_label("Feature Value\n(scaled)", fontsize=9)
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["Low", "High"])

    sns.despine(ax=ax)
    out = FIG_DIR / "fig2_shap_beeswarm.png"
    fig.savefig(out, **SAVE_KW)
    plt.close(fig)
    print(f"  Gespeichert: {out.name}")


# ── Fig 3: SHAP nach Modalitaet ────────────────────────────────────────────
def plot_shap_modality(shap_imp, modality_map, feature_names, top_n=20):
    top_df = shap_imp.head(top_n).copy()

    def get_mod(feat):
        idx = feature_names.index(feat)
        for mod, indices in modality_map.items():
            if idx in indices:
                return mod
        return "unknown"

    top_df["modality"]      = top_df["feature"].apply(get_mod)
    top_df["display_label"] = top_df["feature"].apply(get_label)
    top_df["color"]         = top_df["modality"].map(MODALITY_COLORS)

    fig, ax = plt.subplots(figsize=(8, 7))
    bars = ax.barh(range(top_n), top_df["mean_abs_shap"].values[::-1],
                   color=top_df["color"].values[::-1], edgecolor="white", height=0.7)

    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_df["display_label"].values[::-1], fontsize=9)
    ax.set_xlabel("Mean |SHAP Value|", fontsize=11)
    ax.set_title(f"Feature Importance by Modality – Top {top_n}", fontsize=12)

    legend_handles = [
        mpatches.Patch(color=MODALITY_COLORS["fmri"],       label="fMRI"),
        mpatches.Patch(color=MODALITY_COLORS["fitness"],    label="Fitness"),
        mpatches.Patch(color=MODALITY_COLORS["microbiome"], label="Microbiome"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9)
    sns.despine(ax=ax)

    out = FIG_DIR / "fig3_shap_modality.png"
    fig.savefig(out, **SAVE_KW)
    plt.close(fig)
    print(f"  Gespeichert: {out.name}")


# ── Fig 4: Elastic Net Koeffizienten ───────────────────────────────────────
def plot_enet_coefs(coef_df, top_n=20):
    top_df = coef_df.head(top_n).copy()
    top_df["display_label"] = top_df["feature"].apply(get_label)
    top_df_sorted = top_df.sort_values("mean_coef")

    colors = [COLORS["sz"] if c > 0 else COLORS["ctrl"]
              for c in top_df_sorted["mean_coef"]]

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.barh(range(top_n), top_df_sorted["mean_coef"].values,
            xerr=top_df_sorted["std_coef"].values,
            color=colors, edgecolor="white", height=0.7,
            error_kw=dict(ecolor="gray", capsize=3, lw=1))

    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_df_sorted["display_label"].values, fontsize=9)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Mean Coefficient (± SD)", fontsize=11)
    ax.set_title(f"Elastic Net Coefficients – Top {top_n}", fontsize=12)

    legend_handles = [
        mpatches.Patch(color=COLORS["sz"],   label="Higher in SZ"),
        mpatches.Patch(color=COLORS["ctrl"], label="Higher in HC"),
    ]
    ax.legend(handles=legend_handles, fontsize=9)
    sns.despine(ax=ax)

    out = FIG_DIR / "fig4_enet_coefficients.png"
    fig.savefig(out, **SAVE_KW)
    plt.close(fig)
    print(f"  Gespeichert: {out.name}")


# ── Fig 5: Venn Diagram ────────────────────────────────────────────────────
def plot_venn(shap_imp, coef_df, top_n=20):
    top_shap = set(shap_imp.head(top_n)["feature"].tolist())
    top_enet = set(coef_df.head(top_n)["feature"].tolist())

    fig, ax = plt.subplots(figsize=(6, 5))
    v = venn2([top_shap, top_enet],
              set_labels=("RF (SHAP)", "Elastic Net"),
              set_colors=(COLORS["fmri"], COLORS["microbiome"]),
              alpha=0.6, ax=ax)
    ax.set_title(f"Feature Overlap: Top-{top_n} RF vs. Elastic Net", fontsize=12)

    out = FIG_DIR / "fig5_venn_overlap.png"
    fig.savefig(out, **SAVE_KW)
    plt.close(fig)
    print(f"  Gespeichert: {out.name}")


# ── Fig 6: PANSS Heatmap ───────────────────────────────────────────────────
def plot_corr_heatmap(corr_csv, top_n, scale_col, title, out_name,
                      scale_labels=None):
    corr_df   = pd.read_csv(corr_csv)
    top_feats = corr_df["feature"].unique()[:top_n]
    scales    = corr_df[scale_col].unique()

    pivot_rho = corr_df.pivot(index="feature", columns=scale_col,
                               values="spearman_rho").loc[top_feats]
    pivot_p   = corr_df.pivot(index="feature", columns=scale_col,
                               values="significant_uncorr").loc[top_feats]
    pivot_fdr = corr_df.pivot(index="feature", columns=scale_col,
                               values="significant_fdr").loc[top_feats]

    display_labels = [get_label(f) for f in pivot_rho.index]
    if scale_labels:
        pivot_rho.columns = scale_labels
        pivot_p.columns   = scale_labels
        pivot_fdr.columns = scale_labels

    cmap = LinearSegmentedColormap.from_list(
        "corr_cmap", [COLORS["shap_low"], "#f7f7f7", COLORS["shap_high"]]
    )

    fig, ax = plt.subplots(figsize=(6, 8))
    sns.heatmap(pivot_rho.astype(float), ax=ax, cmap=cmap,
                center=0, vmin=-0.6, vmax=0.6,
                linewidths=0.5, linecolor="white",
                yticklabels=display_labels,
                cbar_kws={"label": "Spearman ρ", "shrink": 0.6})

    # Sterne fuer unkorrigierte Signifikanz, Raute fuer FDR
    for i, feat in enumerate(pivot_rho.index):
        for j, scale in enumerate(pivot_rho.columns):
            sig_uncorr = pivot_p.loc[feat, scale]
            sig_fdr    = pivot_fdr.loc[feat, scale]
            if sig_fdr:
                ax.text(j + 0.5, i + 0.5, "◆", ha="center", va="center",
                        fontsize=10, color="white")
            elif sig_uncorr:
                ax.text(j + 0.5, i + 0.5, "*", ha="center", va="center",
                        fontsize=12, color="white")

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=30)
    ax.tick_params(axis="y", labelsize=8)

    fig.text(0.01, 0.01,
             "* p < 0.05 (uncorrected)   ◆ q < 0.05 (FDR-corrected)",
             fontsize=7, color="gray")

    out = FIG_DIR / out_name
    fig.savefig(out, **SAVE_KW)
    plt.close(fig)
    print(f"  Gespeichert: {out.name}")


# ── Fig 7: Gruppenboxplots Top-5 ───────────────────────────────────────────
def plot_group_boxplots(X_raw, y, feature_names, shap_imp, top_n=5):
    top_features = shap_imp.head(top_n)["feature"].tolist()
    top_labels   = [get_label(f) for f in top_features]
    top_idx      = [feature_names.index(f) for f in top_features]

    ctrl_mask = y == 0
    sz_mask   = y == 1

    fig, axes = plt.subplots(1, top_n, figsize=(14, 4), sharey=False)
    for i, (feat_idx, label) in enumerate(zip(top_idx, top_labels)):
        vals     = X_raw[:, feat_idx]
        mean_v   = vals.mean()
        std_v    = vals.std()
        z_vals   = (vals - mean_v) / (std_v + 1e-9)

        ctrl_vals = z_vals[ctrl_mask]
        sz_vals   = z_vals[sz_mask]

        axes[i].boxplot([ctrl_vals, sz_vals],
                        patch_artist=True,
                        boxprops=dict(facecolor="white"),
                        medianprops=dict(color="black", lw=2),
                        whiskerprops=dict(color="gray"),
                        capprops=dict(color="gray"),
                        flierprops=dict(marker="o", markersize=3,
                                        markerfacecolor="gray", alpha=0.5))

        # Punkte
        for vals_grp, x_pos, color in [
            (ctrl_vals, 1, COLORS["ctrl"]),
            (sz_vals,   2, COLORS["sz"]),
        ]:
            jitter = np.random.RandomState(i).uniform(-0.12, 0.12, len(vals_grp))
            axes[i].scatter(np.full(len(vals_grp), x_pos) + jitter,
                            vals_grp, color=color, s=18, alpha=0.7, zorder=3)

        axes[i].set_xticks([1, 2])
        axes[i].set_xticklabels(["HC", "SZ"], fontsize=9)
        axes[i].set_title(label, fontsize=9, wrap=True)
        axes[i].set_ylabel("z-score" if i == 0 else "", fontsize=9)
        sns.despine(ax=axes[i])

    fig.suptitle("Group Distributions – Top 5 SHAP Features", fontsize=12, y=1.02)
    out = FIG_DIR / "fig7_group_boxplots.png"
    fig.savefig(out, **SAVE_KW)
    plt.close(fig)
    print(f"  Gespeichert: {out.name}")


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    # Daten laden
    X_raw      = np.load(DATA_DIR / "X_raw.npy")
    y          = np.load(DATA_DIR / "y.npy")
    shap_matrix = np.load(SHAP_DIR / "shap.npy")

    with open(DATA_DIR / "feature_names.json") as f:
        feature_names = json.load(f)
    with open(DATA_DIR / "modality_map.json") as f:
        modality_map = json.load(f)

    shap_imp = pd.read_csv(SHAP_DIR / "shap_importance.csv")
    coef_df  = pd.read_csv(COEF_DIR / "enet_coefs.csv")

    print("\n=== Figures erstellen ===")

    # Fig 1: ROC Curves
    plot_roc(SHAP_DIR / "roc_data.pkl",
             "Random Forest – ROC Curves (5-fold CV)",
             FIG_DIR / "fig1_roc_rf.png")
    plot_roc(COEF_DIR / "enet_roc_data.pkl",
             "Elastic Net – ROC Curves (5-fold CV)",
             FIG_DIR / "fig1_roc_enet.png")

    # Fig 2: SHAP Beeswarm
    plot_shap_beeswarm(shap_matrix, X_raw, feature_names, shap_imp)

    # Fig 3: SHAP nach Modalitaet
    plot_shap_modality(shap_imp, modality_map, feature_names)

    # Fig 4: Elastic Net Koeffizienten
    plot_enet_coefs(coef_df)

    # Fig 5: Venn Diagram
    plot_venn(shap_imp, coef_df)

    # Fig 6a: PANSS Heatmap
    plot_corr_heatmap(
        RESULTS_DIR / "panss_shap_corr.csv",
        top_n=20, scale_col="scale",
        title="SHAP–PANSS Correlations (SZ only, n=34)\n* uncorrected  ◆ FDR-corrected",
        out_name="fig6_panss_heatmap.png",
        scale_labels=["PANSS-N", "PANSS-P", "PANSS-G", "PANSS-Total"],
    )

    # Fig 6b: PAS/SAS Heatmap (NEU)
    plot_corr_heatmap(
        RESULTS_DIR / "pasas_shap_corr.csv",
        top_n=20, scale_col="scale",
        title="SHAP–Anhedonia Correlations (n=67)\n* uncorrected  ◆ FDR-corrected",
        out_name="fig6b_pasas_heatmap.png",
        scale_labels=["PAS", "SAS"],
    )

    # Fig 7: Gruppenboxplots
    plot_group_boxplots(X_raw, y, feature_names, shap_imp)

    print(f"\nAlle Figures gespeichert in: {FIG_DIR}")


if __name__ == "__main__":
    main()
