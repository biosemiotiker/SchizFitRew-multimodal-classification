"""
08_stability.py — Revision 2026TP002154, AP2 (R1 #3, R2 #1)

Same CV scheme as 07 (rep 0 = original seed-42 split; reps 1..50 seeds 1000+r),
same preprocessing (age/sex regression + scaling fitted on training fold only).

A  Feature-ranking stability (full 118-feature model):
   per repeat, RF ranking by mean |SHAP| (class SZ) over pooled out-of-fold test samples,
   ENet ranking by mean |coef| over the 5 fold models (+ selection frequency coef != 0).
   -> top-20 membership frequency, median rank + IQR, per-repeat RF/ENet top-20 overlap.
B  Null distribution for the RF/ENet top-20 overlap:
   (i) hypergeometric (independent random lists), (ii) label permutation (n = 1000, same pipeline).
C  Reduced models (R2 #1), evaluated over the same 51 repeats:
   - fMRI + Fitness + microbiome PCA (5 PCs, fitted on training fold)
   - nested univariate top-k selection (k = 10, 20; Welch t on training fold) from all 118 features
"""
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy import stats
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

BASE = Path(__file__).parent.parent
OUT = BASE / "results" / "stability"; OUT.mkdir(parents=True, exist_ok=True)
X = np.load(BASE / "data/X_raw.npy"); COV = np.load(BASE / "data/covariates.npy")
Y = np.load(BASE / "data/y.npy").astype(int)
FN = json.load(open(BASE / "data/feature_names.json")); MOD = json.load(open(BASE / "data/modality_map.json"))
P = X.shape[1]; N_FOLDS, N_REP, N_PERM, TOP = 5, 51, 1000, 20
L1_RATIOS = [0.1, 0.5, 0.9]; C_VALUES = [0.001, 0.01, 0.1, 1, 10]
MIC, NONMIC = MOD["microbiome"], sorted(MOD["fmri"] + MOD["fitness"])


def prep(tr, te, cols=None):
    cols = np.arange(P) if cols is None else np.asarray(cols)
    Xtr, Xte = X[np.ix_(tr, cols)], X[np.ix_(te, cols)]
    lr = LinearRegression().fit(COV[tr], Xtr)
    Xtr, Xte = Xtr - lr.predict(COV[tr]), Xte - lr.predict(COV[te])
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xte)


def rf():
    return RandomForestClassifier(n_estimators=500, max_features="sqrt", class_weight="balanced",
                                  random_state=42, n_jobs=1)


def enet_fit(Xtr, ytr):
    best, bp = -np.inf, None
    inner = StratifiedKFold(3, shuffle=True, random_state=42)
    for l1r in L1_RATIOS:
        for C in C_VALUES:
            a = []
            for i_tr, i_va in inner.split(Xtr, ytr):
                if len(np.unique(ytr[i_tr])) < 2: continue
                m = LogisticRegression(l1_ratio=l1r, C=C, solver="saga", class_weight="balanced",
                                       max_iter=10000, random_state=42).fit(Xtr[i_tr], ytr[i_tr])
                a.append(roc_auc_score(ytr[i_va], m.predict_proba(Xtr[i_va])[:, 1]))
            if np.mean(a) > best: best, bp = np.mean(a), (C, l1r)
    return LogisticRegression(l1_ratio=bp[1], C=bp[0], solver="saga", class_weight="balanced",
                              max_iter=10000, random_state=42).fit(Xtr, ytr)


def rank_desc(v, rng):
    """rank 1 = largest; ties (e.g. many zero ENet coefs) broken at random."""
    order = np.lexsort((rng.random(len(v)), -v))
    r = np.empty(len(v), int); r[order] = np.arange(1, len(v) + 1); return r


def importance_rep(y, seed, rng):
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
    shap_abs, coef_abs, nz = np.zeros(P), np.zeros(P), np.zeros(P)
    for tr, te in skf.split(X, y):
        Xtr, Xte = prep(tr, te)
        m = rf().fit(Xtr, y[tr])
        sv = shap.TreeExplainer(m).shap_values(Xte)
        sv = sv[1] if isinstance(sv, list) else (sv[..., 1] if sv.ndim == 3 else sv)
        shap_abs += np.abs(sv).sum(0)
        e = enet_fit(Xtr, y[tr]); c = e.coef_.ravel()
        coef_abs += np.abs(c) / N_FOLDS; nz += (c != 0) / N_FOLDS
    shap_abs /= len(y)
    return shap_abs, coef_abs, nz, rank_desc(shap_abs, rng), rank_desc(coef_abs, rng)


def stab_job(rep):
    seed = 42 if rep == 0 else 1000 + rep
    s, c, nz, rr, re = importance_rep(Y, seed, np.random.default_rng(rep))
    return dict(rep=rep, shap=s, coef=c, nz=nz, rank_rf=rr, rank_en=re)


def perm_job(p):
    rng = np.random.default_rng(10_000 + p)
    yp = rng.permutation(Y)
    *_, rr, re = importance_rep(yp, 42, rng)
    return ((rr <= TOP) & (re <= TOP)).sum()


def reduced_job(rep):
    seed = 42 if rep == 0 else 1000 + rep
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
    out = []
    for f, (tr, te) in enumerate(skf.split(X, Y)):
        Xtr, Xte = prep(tr, te)
        mic_i = [i for i in range(P) if i in set(MIC)]; non_i = [i for i in range(P) if i in set(NONMIC)]
        pca = PCA(5, random_state=42).fit(Xtr[:, mic_i])
        variants = {"fMRI + Fitness + Microbiome 5 PCs":
                    (np.column_stack([Xtr[:, non_i], pca.transform(Xtr[:, mic_i])]),
                     np.column_stack([Xte[:, non_i], pca.transform(Xte[:, mic_i])]))}
        t = np.abs(stats.ttest_ind(Xtr[Y[tr] == 1], Xtr[Y[tr] == 0], equal_var=False).statistic)
        t = np.nan_to_num(t)
        for k in (10, 20):
            sel = np.argsort(-t)[:k]
            variants[f"Nested top-{k} (univariate)"] = (Xtr[:, sel], Xte[:, sel])
        for name, (a, b) in variants.items():
            for model in ("RF", "ENet"):
                m = rf().fit(a, Y[tr]) if model == "RF" else enet_fit(a, Y[tr])
                out.append(dict(rep=rep, fold=f, model=model, subset=name,
                                auroc=roc_auc_score(Y[te], m.predict_proba(b)[:, 1])))
    return out


if __name__ == "__main__":
    # A
    st = Parallel(n_jobs=60)(delayed(stab_job)(r) for r in range(N_REP))
    RR = np.array([d["rank_rf"] for d in st]); RE = np.array([d["rank_en"] for d in st])
    NZ = np.array([d["nz"] for d in st])
    modality = {FN[i]: m for m, ii in MOD.items() for i in ii}
    ov0 = pd.read_csv(BASE / "results/feature_overlap.csv").set_index("feature")
    S = pd.DataFrame(dict(feature=FN, modality=[modality[f] for f in FN],
                          orig_robust=[bool(ov0.in_both.get(f, False)) for f in FN],
                          rf_top20_freq=(RR <= TOP).mean(0), rf_median_rank=np.median(RR, 0),
                          rf_rank_q25=np.percentile(RR, 25, 0), rf_rank_q75=np.percentile(RR, 75, 0),
                          enet_top20_freq=(RE <= TOP).mean(0), enet_median_rank=np.median(RE, 0),
                          enet_rank_q25=np.percentile(RE, 25, 0), enet_rank_q75=np.percentile(RE, 75, 0),
                          enet_selection_freq=NZ.mean(0),
                          both_top20_freq=((RR <= TOP) & (RE <= TOP)).mean(0)))
    S = S.sort_values("rf_median_rank")
    S.to_csv(OUT / "stability_selection_frequency.csv", index=False)
    np.save(OUT / "ranks_rf.npy", RR); np.save(OUT / "ranks_enet.npy", RE)
    overlap_rep = ((RR <= TOP) & (RE <= TOP)).sum(1)

    # B
    obs = int(ov0.in_both.sum())
    hyper = stats.hypergeom(P, TOP, TOP)
    null = np.array(Parallel(n_jobs=100)(delayed(perm_job)(p) for p in range(N_PERM)))
    pd.DataFrame(dict(perm=np.arange(N_PERM), overlap=null)).to_csv(OUT / "overlap_null.csv", index=False)
    B = dict(observed_overlap_original=obs, overlap_rep0=int(overlap_rep[0]),
             overlap_reps_median=float(np.median(overlap_rep)),
             overlap_reps_range=f"{overlap_rep.min()}-{overlap_rep.max()}",
             hypergeom_expected=hyper.mean(), hypergeom_p=hyper.sf(obs - 1),
             perm_null_mean=null.mean(), perm_null_p95=np.percentile(null, 95),
             perm_p=(1 + (null >= obs).sum()) / (N_PERM + 1))
    json.dump(B, open(OUT / "overlap_test.json", "w"), indent=2, default=float)

    # C
    red = pd.DataFrame([r for rr in Parallel(n_jobs=100)(delayed(reduced_job)(r) for r in range(N_REP)) for r in rr])
    red.to_csv(OUT / "reduced_models_fold_metrics.csv", index=False)
    abl = pd.read_csv(BASE / "results/ablation/ablation_fold_metrics.csv")
    allm = pd.concat([abl[["rep", "fold", "model", "subset", "auroc"]], red])
    rep_auc = allm.groupby(["model", "subset", "rep"]).auroc.mean().reset_index()
    D = rep_auc.groupby(["model", "subset"]).auroc.agg(
        mean="mean", sd="std", p2_5=lambda v: np.percentile(v, 2.5), p97_5=lambda v: np.percentile(v, 97.5),
        orig_split=lambda v: v.iloc[0]).reset_index()
    D.to_csv(OUT / "repeated_cv_distribution.csv", index=False)

    print(json.dumps(B, indent=1, default=float))
    print(S.head(25)[["feature", "modality", "orig_robust", "rf_top20_freq", "rf_median_rank",
                      "enet_top20_freq", "enet_selection_freq", "both_top20_freq"]].round(2).to_string())
    print(S[S.orig_robust][["feature", "rf_top20_freq", "enet_top20_freq", "both_top20_freq"]].round(2).to_string())
    print(D.round(3).to_string())
