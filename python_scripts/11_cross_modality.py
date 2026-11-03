"""
11_cross_modality.py  —  Revision 2026TP002154, AP5 (R1 #2, R2 #7, R3)

Direct test of cross-modality coupling (microbiome <-> reward fMRI <-> fitness).

Residualisation: features regressed on age + sex (+ group dummy for pooled analyses),
so pooled coupling reflects WITHIN-group covariation, not shared case-control differences.

A  Block level (PLS-SVD on cross-correlation matrix):
   statistics = first singular value (d1) and mean squared cross-correlation (msc),
   permutation null (5000; rows of one block shuffled, within group for pooled).
   Pairs: Microbiome-fMRI, Microbiome-Fitness, fMRI-Fitness (positive control).
   Samples: pooled (group-residualised), SZ, CTRL.
B  Feature level: Spearman correlations of all cross-modality pairs; BH-FDR per block pair
   + a-priori family (5 robust microbiome x 7 robust fMRI/fitness features) with FDR.
C  Moderation: y ~ group * microbiome + age + sex for the a-priori family (interaction term).
"""
import json, itertools, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
warnings.filterwarnings("ignore")

BASE = Path(__file__).parent.parent
OUT = BASE / "results" / "cross_modality"; OUT.mkdir(parents=True, exist_ok=True)
X = np.load(BASE / "data/X_raw.npy"); COV = np.load(BASE / "data/covariates.npy")
y = np.load(BASE / "data/y.npy").astype(int)
FN = json.load(open(BASE / "data/feature_names.json"))
MOD = json.load(open(BASE / "data/modality_map.json"))
ov = pd.read_csv(BASE / "results/feature_overlap.csv")
ROB_MIC = ov[(ov.in_both) & (ov.modality == "microbiome")].feature.tolist()
ROB_OTH = ov[(ov.in_both) & (ov.modality != "microbiome")].feature.tolist()
idx = {f: i for i, f in enumerate(FN)}
N_PERM = 5000; rng = np.random.default_rng(42)


def resid(Xs, covs):
    D = np.column_stack([np.ones(len(Xs)), covs])
    B, *_ = np.linalg.lstsq(D, Xs, rcond=None)
    R = Xs - D @ B
    return (R - R.mean(0)) / R.std(0, ddof=1)


def rank(A):
    return np.apply_along_axis(stats.rankdata, 0, A)


def block_stats(A, B):
    C = (A.T @ B) / (len(A) - 1)
    return np.linalg.svd(C, compute_uv=False)[0], np.mean(C ** 2)


def perm_block(A, B, groups):
    d1, msc = block_stats(A, B)
    nd, nm = np.empty(N_PERM), np.empty(N_PERM)
    for p in range(N_PERM):
        perm = np.arange(len(A))
        for g in np.unique(groups):
            ii = np.where(groups == g)[0]; perm[ii] = rng.permutation(ii)
        nd[p], nm[p] = block_stats(A, B[perm])
    return dict(d1=d1, p_d1=(1 + (nd >= d1).sum()) / (N_PERM + 1),
                msc=msc, p_msc=(1 + (nm >= msc).sum()) / (N_PERM + 1),
                msc_null_mean=nm.mean())


samples = {"pooled (group-adjusted)": (np.ones(len(y), bool), True),
           "SZ": (y == 1, False), "CTRL": (y == 0, False)}
blocks = {"Microbiome": MOD["microbiome"], "fMRI": MOD["fmri"], "Fitness": MOD["fitness"]}
pairs = [("Microbiome", "fMRI"), ("Microbiome", "Fitness"), ("fMRI", "Fitness")]

blk_rows, feat_rows = [], []
for sname, (mask, adj_group) in samples.items():
    covs = np.column_stack([COV[mask], y[mask]]) if adj_group else COV[mask]
    Z = resid(rank(X[mask]), covs)           # rank-based (Spearman-type) after residualisation
    g = y[mask]
    for a, b in pairs:
        r = perm_block(Z[:, blocks[a]], Z[:, blocks[b]], g)
        blk_rows.append(dict(sample=sname, n=mask.sum(), block_pair=f"{a}–{b}",
                             n_pairs=len(blocks[a]) * len(blocks[b]), **r))
        for i in blocks[a]:
            for j in blocks[b]:
                rho, p = stats.pearsonr(Z[:, i], Z[:, j])
                feat_rows.append(dict(sample=sname, block_pair=f"{a}–{b}", feat_a=FN[i], feat_b=FN[j],
                                      rho=rho, p=p,
                                      a_priori=(FN[i] in ROB_MIC and FN[j] in ROB_OTH)))
blk = pd.DataFrame(blk_rows)
feat = pd.DataFrame(feat_rows)
feat["q_fdr_blockpair"] = feat.groupby(["sample", "block_pair"]).p.transform(
    lambda p: multipletests(p, method="fdr_bh")[1])
ap = feat.a_priori
feat.loc[ap, "q_fdr_apriori"] = feat[ap].groupby("sample").p.transform(
    lambda p: multipletests(p, method="fdr_bh")[1])

# C  moderation
mod_rows = []
for m in ROB_MIC:
    for o in ROB_OTH:
        d = pd.DataFrame(dict(yv=stats.zscore(X[:, idx[o]]), mic=stats.zscore(X[:, idx[m]]),
                              grp=y, age=COV[:, 0], sex=COV[:, 1]))
        d["int"] = d.grp * d.mic
        fit = sm.OLS(d.yv, sm.add_constant(d[["grp", "mic", "int", "age", "sex"]])).fit(cov_type="HC3")
        mod_rows.append(dict(microbiome=m, outcome=o, b_mic=fit.params["mic"], p_mic=fit.pvalues["mic"],
                             b_interaction=fit.params["int"], p_interaction=fit.pvalues["int"]))
mod = pd.DataFrame(mod_rows)
mod["q_interaction"] = multipletests(mod.p_interaction, method="fdr_bh")[1]
mod["q_mic"] = multipletests(mod.p_mic, method="fdr_bh")[1]

blk.to_csv(OUT / "block_coupling_pls.csv", index=False)
feat.to_csv(OUT / "cross_modality_correlation.csv", index=False)
mod.to_csv(OUT / "moderation_group_x_microbiome.csv", index=False)

print(blk.round(4).to_string())
print("\nFDR<0.05 per block pair:\n", feat[feat.q_fdr_blockpair < .05].groupby(["sample", "block_pair"]).size())
print("\nA-priori family, min q per sample:\n", feat[ap].groupby("sample").q_fdr_apriori.min().round(3))
print(feat[ap & (feat.p < .05)].sort_values("p")[["sample", "feat_a", "feat_b", "rho", "p", "q_fdr_apriori"]].round(3).to_string())
print("\nModeration (sorted):\n", mod.sort_values("p_interaction").head(8).round(3).to_string())
