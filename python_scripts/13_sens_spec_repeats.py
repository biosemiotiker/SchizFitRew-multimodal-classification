"""
13_sens_spec_repeats.py — Revision 2026TP002154, R2 #5 (sensitivity/specificity distributions)

Same 51 repeats and fold seeds as 08_stability.py (rep 0 = seed 42, reps 1..50 = seeds 1000+r),
same preprocessing (age/sex regression + scaling fitted on training fold only), same RF settings
(500 trees, sqrt features, balanced class weights, random_state=42).
Per repeat: metrics computed per outer fold at the default threshold 0.5 and averaged over the 5 folds. Summary: mean, SD, 2.5/97.5 percentiles.
"""
from pathlib import Path
import json, warnings
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix
warnings.filterwarnings("ignore")

BASE = Path(__file__).parent.parent
OUT = BASE / "results" / "stability"; OUT.mkdir(parents=True, exist_ok=True)
X = np.load(BASE / "data/X_raw.npy"); COV = np.load(BASE / "data/covariates.npy")
Y = np.load(BASE / "data/y.npy").astype(int)
N_FOLDS, N_REP = 5, 51

def prep(tr, te):
    lr = LinearRegression().fit(COV[tr], X[tr])
    Xtr, Xte = X[tr] - lr.predict(COV[tr]), X[te] - lr.predict(COV[te])
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xte)

def rep_job(rep):
    """Fold-wise metrics averaged over the 5 folds, as in 08_stability.py (repeated_cv_distribution)."""
    seed = 42 if rep == 0 else 1000 + rep
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
    fold = []
    for tr, te in skf.split(X, Y):
        Xtr, Xte = prep(tr, te)
        m = RandomForestClassifier(n_estimators=500, max_features="sqrt", class_weight="balanced",
                                   random_state=42, n_jobs=1).fit(Xtr, Y[tr])
        pr = m.predict_proba(Xte)[:, 1]; yt = Y[te]
        tn, fp, fn, tp = confusion_matrix(yt, (pr >= 0.5).astype(int), labels=[0, 1]).ravel()
        fold.append(dict(auroc=roc_auc_score(yt, pr), sensitivity=tp / (tp + fn),
                         specificity=tn / (tn + fp), accuracy=(tp + tn) / len(yt)))
    f = pd.DataFrame(fold).mean()
    return dict(rep=rep, **f.to_dict())

if __name__ == "__main__":
    rows = Parallel(n_jobs=8)(delayed(rep_job)(r) for r in range(N_REP))
    df = pd.DataFrame(rows); df.to_csv(OUT / "repeated_cv_sens_spec.csv", index=False)
    summ = []
    for c in ["auroc", "sensitivity", "specificity", "accuracy"]:
        v = df[c].values
        summ.append(dict(metric=c, mean=v.mean(), sd=v.std(ddof=1),
                         p2_5=np.percentile(v, 2.5), p97_5=np.percentile(v, 97.5)))
    pd.DataFrame(summ).to_csv(OUT / "repeated_cv_sens_spec_summary.csv", index=False)
    print(pd.DataFrame(summ).round(3).to_string(index=False))
