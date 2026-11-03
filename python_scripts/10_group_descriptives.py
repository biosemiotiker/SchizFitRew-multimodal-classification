"""
10_group_descriptives.py — Revision 2026TP002154, AP4 (R2 #6, R3) + AP3 (R1 #4, R2 #4)

AP4  Table S1: group descriptives for all 118 features (raw scale; microbiome = CLR)
     CTRL / SZ mean (SD), Welch t, p, Hedges g, BH-FDR within modality,
     plus age/sex-adjusted group effect (OLS, HC3) with FDR.
AP3  Within SZ: partial Spearman (age, sex) of every feature with CPZ equivalents and
     cumulative lifetime dose; FDR within modality; robust features flagged.
     Clozapine/olanzapine (ATC N05AH02/N05AH03) vs other antipsychotics for robust features.
"""
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings("ignore")

BASE = Path(__file__).parent.parent
OUT = BASE / "results" / "descriptives"; OUT.mkdir(parents=True, exist_ok=True)
X = np.load(BASE / "data/X_raw.npy"); COV = np.load(BASE / "data/covariates.npy")
y = np.load(BASE / "data/y.npy").astype(int)
FN = json.load(open(BASE / "data/feature_names.json")); IDS = json.load(open(BASE / "data/subject_ids.json"))
MOD = json.load(open(BASE / "data/modality_map.json"))
modality = {FN[i]: m for m, ii in MOD.items() for i in ii}
ov = pd.read_csv(BASE / "results/feature_overlap.csv").set_index("feature")
robust = set(ov.index[ov.in_both])
LAB = {"roi_mean ACC": "Anterior cingulate cortex", "roi_mean antIns": "Anterior insula",
       "roi_mean Putamen": "Putamen", "roi_mean Nacc": "Nucleus accumbens", "roi_mean Ncau": "Caudate nucleus",
       "VO_max": "VO2max", "Jump": "Standing long jump", "Hand_mean": "Handgrip strength",
       "Sum_Skinfolds": "Sum of skinfolds", "Brozek_BF_age": "Body fat (%)", "WHR": "Waist-hip ratio",
       "WHtR": "Waist-height ratio", "bmi": "BMI",
       "IPAQ_Tot_Phys_Activity_MET.minutes_week": "Physical activity (MET-min/week)",
       "Average_sitting_per_day_Minutes": "Sitting time (min/day)"}
lab = lambda f: LAB.get(f, f.replace("..", " (").replace(".", " ").strip())

# ── AP4 ──────────────────────────────────────────────────────────────────────
rows = []
for i, f in enumerate(FN):
    a, b = X[y == 0, i], X[y == 1, i]
    if np.ptp(X[:, i]) < 1e-9:   # constant module (not detected in any sample) -> no test
        rows.append(dict(modality=modality[f], feature=f, label=lab(f), robust_feature=f in robust,
                         CTRL_mean=a.mean(), CTRL_sd=0.0, SZ_mean=b.mean(), SZ_sd=0.0, note="constant (not detected)"))
        continue
    t, p = stats.ttest_ind(b, a, equal_var=False)
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    g = (b.mean() - a.mean()) / sp * (1 - 3 / (4 * (len(a) + len(b)) - 9))
    D = sm.add_constant(np.column_stack([y, COV]))
    fit = sm.OLS(X[:, i], D).fit(cov_type="HC3")
    rows.append(dict(modality=modality[f], feature=f, label=lab(f), robust_feature=f in robust,
                     CTRL_mean=a.mean(), CTRL_sd=a.std(ddof=1), SZ_mean=b.mean(), SZ_sd=b.std(ddof=1),
                     welch_t=t, p=p, hedges_g=g, adj_beta_group=fit.params[1], adj_p=fit.pvalues[1], note=""))
S1 = pd.DataFrame(rows)
for col in ["p", "adj_p"]:
    ok_ = S1[col].notna()
    S1.loc[ok_, "q_" + col] = S1[ok_].groupby("modality")[col].transform(lambda v: multipletests(v, method="fdr_bh")[1])
S1["modality"] = pd.Categorical(S1.modality, ["fmri", "fitness", "microbiome"])
S1 = S1.sort_values(["modality", "p"])

# ── AP3 ──────────────────────────────────────────────────────────────────────
M = pd.read_excel(BASE.parent / "data_combined_analysis/gbm_abundance_global_clr.xlsx",
                  sheet_name="gbm_abundance_global_clr").set_index("Column1").loc[IDS]
sz = y == 1
def presid(v, C):
    D = np.column_stack([np.ones(len(v)), C]); return v - D @ np.linalg.lstsq(D, v, rcond=None)[0]
med_rows = []
for dose in ["cpzeq", "cum_ltd"]:
    dv = pd.to_numeric(M[dose], errors="coerce").values
    ok = sz & ~np.isnan(dv)
    for i, f in enumerate(FN):
        if np.ptp(X[:, i]) < 1e-9: continue
        r, p = stats.pearsonr(presid(stats.rankdata(X[ok, i]), COV[ok]), presid(stats.rankdata(dv[ok]), COV[ok]))
        med_rows.append(dict(dose=dose, n=ok.sum(), modality=modality[f], feature=f, label=lab(f),
                             robust_feature=f in robust, partial_rho=r, p=p))
MED = pd.DataFrame(med_rows)
MED["q_within_modality"] = MED.groupby(["dose", "modality"]).p.transform(lambda v: multipletests(v, method="fdr_bh")[1])
rb = MED.robust_feature
MED.loc[rb, "q_robust_family"] = MED[rb].groupby("dose").p.transform(lambda v: multipletests(v, method="fdr_bh")[1])

atc = (M["atc_code"].astype(str) + " " + M["atc_code2"].astype(str)).str.upper()
clo_ola = atc.str.contains("N05AH02|N05AH03").values & sz
other = sz & ~clo_ola
co_rows = []
for f in sorted(robust):
    i = FN.index(f)
    u, p = stats.mannwhitneyu(X[clo_ola, i], X[other, i])
    co_rows.append(dict(feature=f, label=lab(f), n_clo_ola=clo_ola.sum(), n_other=other.sum(),
                        median_clo_ola=np.median(X[clo_ola, i]), median_other=np.median(X[other, i]), U=u, p=p))
CO = pd.DataFrame(co_rows); CO["q"] = multipletests(CO.p, method="fdr_bh")[1]

with pd.ExcelWriter(OUT / "TableS1_group_descriptives.xlsx") as w:
    S1.to_excel(w, sheet_name="S1_group_descriptives", index=False)
    MED.to_excel(w, sheet_name="S_medication_within_SZ", index=False)
    CO.to_excel(w, sheet_name="S_clozapine_olanzapine", index=False)
S1.to_csv(OUT / "TableS1_group_descriptives.csv", index=False)
MED.to_csv(OUT / "medication_feature_assoc.csv", index=False)
CO.to_csv(OUT / "clozapine_olanzapine_robust.csv", index=False)

print("S1: q_p<.05 per modality:", S1[S1.q_p < .05].groupby("modality").size().to_dict(),
      "| adj q<.05:", S1[S1.q_adj_p < .05].groupby("modality").size().to_dict())
print(S1[(S1.q_p < .05) | S1.robust_feature][["modality", "label", "CTRL_mean", "SZ_mean", "hedges_g", "p", "q_p", "q_adj_p", "robust_feature"]].round(3).to_string())
print("\nMedication: q_within_modality<.05:", MED[MED.q_within_modality < .05][["dose", "label", "partial_rho", "p"]].round(3).to_string())
print(MED[rb].sort_values("p").head(8)[["dose", "n", "label", "partial_rho", "p", "q_robust_family"]].round(3).to_string())
print("\nClo/Ola:", CO.round(3).to_string())
