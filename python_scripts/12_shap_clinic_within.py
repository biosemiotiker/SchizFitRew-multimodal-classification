"""12_shap_clinic_within.py — Revision AP7 (R2 #7, R3).
Korreliert SHAP-Werte der RF-Top-20 mit PAS/SAS/WHODAS/CDSS nur innerhalb SZ (bisher alle 67),
vergleicht mit gepoolter und gruppenadjustierter Variante; PANSS war bereits nur SZ (Kontrolle).
Zusaetzlich: Summe der mittleren |SHAP| als Kontext."""
import json,numpy as np,pandas as pd
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests
B=Path(__file__).resolve().parents[1];D=B/'data';R=B/'results';O=R/'shap_clinic';O.mkdir(exist_ok=True)
S=np.load(R/'shap_values/shap.npy');ids=json.load(open(D/'subject_ids.json'));fn=json.load(open(D/'feature_names.json'))
imp=pd.read_csv(R/'shap_values/shap_importance.csv');top=imp.head(20).feature.tolist()
pa=pd.read_csv(D/'pasas.csv').set_index('subject_id');pn=pd.read_csv(D/'panss.csv').set_index('subject_id')
grp=pa.group.reindex(ids).astype(str).str.startswith('SZ').astype(int).values
def corr(scale_df,cols,mask,label,adj=False):
    rows=[]
    for c in cols:
        v=scale_df[c].reindex(ids).astype(float).values
        for f in top:
            s=S[:,fn.index(f)];m=mask&~np.isnan(v)
            if adj:  # partial Spearman: Raenge, Gruppe residualisiert
                X=np.c_[np.ones(m.sum()),grp[m]]
                r_=lambda z:z-X@np.linalg.lstsq(X,z,rcond=None)[0]
                rho,p=stats.pearsonr(r_(stats.rankdata(s[m])),r_(stats.rankdata(v[m])))
            else: rho,p=stats.spearmanr(s[m],v[m])
            rows.append(dict(analysis=label,feature=f,scale=c,rho=rho,p=p,n=int(m.sum())))
    d=pd.DataFrame(rows);d['q_fdr']=multipletests(d.p,method='fdr_bh')[1];return d
sz=grp==1;allm=np.ones(len(ids),bool);ps=['PAS','SAS','WHODAS','CDSS'];pc=['P_score_PANSS','N_score_PANSS','G_score_PANSS','PANSS_total']
out=pd.concat([corr(pa,ps,allm,'psychosocial_pooled_orig'),corr(pa,ps,allm,'psychosocial_pooled_groupadj',True),
               corr(pa,ps,sz,'psychosocial_within_SZ'),corr(pn,pc,sz,'panss_within_SZ')])
out.to_csv(O/'shap_clinic_correlations.csv',index=False)
summ=out.groupby('analysis').agg(n_tests=('p','size'),n_p05=('p',lambda x:(x<.05).sum()),n_fdr=('q_fdr',lambda x:(x<.05).sum()),max_abs_rho=('rho',lambda x:x.abs().max()))
summ.to_csv(O/'shap_clinic_summary.csv')
ctx=pd.DataFrame(dict(sum_mean_abs_shap_all118=[imp.mean_abs_shap.sum()],sum_top20=[imp.head(20).mean_abs_shap.sum()],share_top20=[imp.head(20).mean_abs_shap.sum()/imp.mean_abs_shap.sum()]))
ctx.to_csv(O/'shap_sum_context.csv',index=False)
print(summ.round(3).to_string());print(ctx.round(4).to_string())
print(out[out.p<.05].sort_values(['analysis','p']).round(3).to_string())
print('SZ n',sz.sum(),'nonmissing',{c:int(pa[c].reindex(ids)[sz].notna().sum()) for c in ps})
