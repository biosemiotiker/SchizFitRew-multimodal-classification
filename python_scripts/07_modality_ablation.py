"""
07_modality_ablation.py  —  Revision 2026TP002154, AP1 (R1 #1, R2 #3)

Leave-one-modality-out / modality-combination comparison.
All 7 modality subsets x {RF, Elastic Net} are evaluated on IDENTICAL outer folds.
Repeat 0 uses StratifiedKFold(5, shuffle, random_state=42) = the original split of 02/03.
Repeats 1..N_REP-1 use seeds 1000+r for inference on differences between subsets.

Preprocessing identical to 02/03: confound regression (age, sex) + StandardScaler fitted
on the training fold only. RF hyperparameters fixed a priori; ENet grid via inner 3-fold CV
(selection metric: mean inner AUROC).

Inference on paired AUROC differences (subset vs. full model):
  corrected repeated k-fold t-test (Nadeau & Bengio 2003; Bouckaert & Frank 2004),
  computed on per-fold AUROC differences across all repeats x folds.
"""
import json, warnings, itertools
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

warnings.filterwarnings("ignore")
BASE = Path(__file__).parent.parent
OUT = BASE / "results" / "ablation"; OUT.mkdir(parents=True, exist_ok=True)

X = np.load(BASE / "data/X_raw.npy"); COV = np.load(BASE / "data/covariates.npy")
y = np.load(BASE / "data/y.npy").astype(int)
MOD = json.load(open(BASE / "data/modality_map.json"))
N_FOLDS, N_REP = 5, 51          # repeat 0 = original split
L1_RATIOS = [0.1, 0.5, 0.9]; C_VALUES = [0.001, 0.01, 0.1, 1, 10]

SUBSETS = {}
names = {"microbiome": "Microbiome", "fmri": "fMRI", "fitness": "Fitness"}
for k in (1, 2, 3):
    for combo in itertools.combinations(["microbiome", "fmri", "fitness"], k):
        SUBSETS[" + ".join(names[c] for c in combo)] = sorted(sum((MOD[c] for c in combo), []))
FULL = "Microbiome + fMRI + Fitness"


def prep(tr, te, cols):
    Xtr, Xte = X[np.ix_(tr, cols)], X[np.ix_(te, cols)]
    lr = LinearRegression().fit(COV[tr], Xtr)
    Xtr, Xte = Xtr - lr.predict(COV[tr]), Xte - lr.predict(COV[te])
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xte)


def enet(C, l1r):
    return LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=l1r, C=C,
                              class_weight="balanced", max_iter=10000, random_state=42)


def fit_predict(model, Xtr, ytr, Xte):
    if model == "RF":
        m = RandomForestClassifier(n_estimators=500, max_features="sqrt",
                                   class_weight="balanced", random_state=42, n_jobs=1)
        m.fit(Xtr, ytr)
    else:
        best, bp = -np.inf, None
        inner = StratifiedKFold(3, shuffle=True, random_state=42)
        for l1r in L1_RATIOS:
            for C in C_VALUES:
                a = []
                for i_tr, i_va in inner.split(Xtr, ytr):
                    mm = enet(C, l1r).fit(Xtr[i_tr], ytr[i_tr])
                    a.append(roc_auc_score(ytr[i_va], mm.predict_proba(Xtr[i_va])[:, 1]))
                if np.mean(a) > best: best, bp = np.mean(a), (C, l1r)
        m = enet(*bp).fit(Xtr, ytr)
    return m.predict_proba(Xte)[:, 1], m.predict(Xte)


def run_one(rep, model, subset):
    seed = 42 if rep == 0 else 1000 + rep
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
    cols, rows = SUBSETS[subset], []
    for f, (tr, te) in enumerate(skf.split(X, y)):
        Xtr, Xte = prep(tr, te, cols)
        p, yhat = fit_predict(model, Xtr, y[tr], Xte)
        tn, fp, fn, tp = confusion_matrix(y[te], yhat, labels=[0, 1]).ravel()
        rows.append(dict(rep=rep, fold=f, model=model, subset=subset, n_features=len(cols),
                         auroc=roc_auc_score(y[te], p), accuracy=accuracy_score(y[te], yhat),
                         sensitivity=tp / (tp + fn), specificity=tn / (tn + fp)))
    return rows


if __name__ == "__main__":
    jobs = [(r, m, s) for r in range(N_REP) for m in ("RF", "ENet") for s in SUBSETS]
    res = Parallel(n_jobs=100, verbose=5)(delayed(run_one)(*j) for j in jobs)
    df = pd.DataFrame([row for rr in res for row in rr])
    df.to_csv(OUT / "ablation_fold_metrics.csv", index=False)

    # Summary: original split (rep 0) and repeated CV (all reps)
    agg = lambda d: d.groupby(["model", "subset", "n_features"])[
        ["auroc", "accuracy", "sensitivity", "specificity"]].agg(["mean", "std"])
    s0 = agg(df[df.rep == 0]); s0.columns = [f"orig_{a}_{b}" for a, b in s0.columns]
    rep_auc = df.groupby(["model", "subset", "rep"]).auroc.mean().reset_index()
    sR = rep_auc.groupby(["model", "subset"]).auroc.agg(
        rep_auroc_mean="mean", rep_auroc_sd="std",
        rep_auroc_p2_5=lambda v: np.percentile(v, 2.5), rep_auroc_p97_5=lambda v: np.percentile(v, 97.5))

    # Corrected repeated k-fold t-test: subset vs. FULL, per model
    n_te = len(y) / N_FOLDS; n_tr = len(y) - n_te; k = N_FOLDS * N_REP
    tests = []
    for m in ("RF", "ENet"):
        full = df[(df.model == m) & (df.subset == FULL)].set_index(["rep", "fold"]).auroc
        for s in SUBSETS:
            if s == FULL: continue
            d = (full - df[(df.model == m) & (df.subset == s)].set_index(["rep", "fold"]).auroc).values
            var = np.var(d, ddof=1)
            t = d.mean() / np.sqrt((1 / k + n_te / n_tr) * var) if var > 0 else np.nan
            p = 2 * stats.t.sf(abs(t), df=k - 1)
            tests.append(dict(model=m, comparison=f"Full vs {s}", delta_auroc_full_minus_subset=d.mean(),
                              sd_delta=np.sqrt(var), t_corrected=t, p_corrected=p,
                              frac_reps_full_better=(full.groupby("rep").mean() >
                                                     df[(df.model == m) & (df.subset == s)].groupby("rep").auroc.mean()).mean()))
    tests = pd.DataFrame(tests)
    from statsmodels.stats.multitest import multipletests
    tests["q_fdr_within_model"] = tests.groupby("model").p_corrected.transform(
        lambda p: multipletests(p, method="fdr_bh")[1])
    summ = s0.join(sR).reset_index()
    summ.to_csv(OUT / "ablation_summary.csv", index=False)
    tests.to_csv(OUT / "ablation_tests_vs_full.csv", index=False)
    print(summ[["model", "subset", "n_features", "orig_auroc_mean", "orig_auroc_std",
                "rep_auroc_mean", "rep_auroc_p2_5", "rep_auroc_p97_5"]].round(3).to_string())
    print(tests.round(4).to_string())
