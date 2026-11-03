
import json,numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg');import matplotlib.pyplot as plt
from pathlib import Path
B=Path(__file__).resolve().parents[1];R=B/'results';D=B/'data'
O=R/'revision_figures';O.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300})
LAB={'roi_mean Nacc':'Nucleus accumbens','roi_mean Ncau':'Caudate nucleus','roi_mean Putamen':'Putamen','roi_mean ACC':'Anterior cingulate cortex',
'Hand_mean':'Handgrip strength','Sum_Skinfolds':'Sum of skinfolds','Brozek_BF_age':'Body fat (%)','Average_sitting_per_day_Minutes':'Sedentary behaviour',
'WHtR':'Waist-to-height ratio','bmi':'BMI','VO_max':'VO$_2$max','WHR':'Waist-to-hip ratio','cysteine.degradation.I':'Cysteine degradation I','leucine.degradation':'Leucine degradation',
'trehalose.degradation':'Trehalose degradation','methionine.degradation.II':'Methionine degradation II','ethanol.production.I':'Ethanol production I',
'ribose.degradation':'Ribose degradation','allose.degradation':'Allose degradation','xylose.degradation':'Xylose degradation'}
UNIT={'fmri':'β (a.u.)','fitness':'raw units','microbiome':'CLR'}
YL={'Hand_mean':'kg','Sum_Skinfolds':'mm','Brozek_BF_age':'%','Average_sitting_per_day_Minutes':'sitting, min/day'}
X=np.load(D/'X_raw.npy');fn=json.load(open(D/'feature_names.json'));ids=json.load(open(D/'subject_ids.json'))
grp=pd.read_csv(D/'pasas.csv').set_index('subject_id').group.reindex(ids).astype(str).str.startswith('SZ').values
S1=pd.read_csv(R/'descriptives/TableS1_group_descriptives.csv').set_index('feature')
ov=pd.read_csv(R/'feature_overlap.csv');feats=ov[ov.in_both].feature.tolist()
# ---- Fig 11
C={'CTRL':'#7f7f7f','SZ':'#2a5caa'}
fig,axs=plt.subplots(3,4,figsize=(7.2,6.2));rng=np.random.default_rng(1)
for ax,f in zip(axs.flat,feats):
    v=X[:,fn.index(f)];d=[v[~grp],v[grp]]
    bp=ax.boxplot(d,widths=.5,showfliers=False,medianprops=dict(color='k'),patch_artist=True)
    for b,c in zip(bp['boxes'],C.values()): b.set_facecolor(c);b.set_alpha(.35)
    for i,(dd,c) in enumerate(zip(d,C.values())): ax.scatter(i+1+rng.uniform(-.13,.13,len(dd)),dd,s=6,color=c,lw=0)
    r=S1.loc[f];ax.set_xticks([1,2],['HC','SZ'])
    qs='q < 0.001' if r.q_p<.001 else (f'q = {r.q_p:.3f}' if r.q_p<.01 else f'q = {r.q_p:.2f}')
    ax.set_title(f"{LAB[f]}\ng = {r.hedges_g:+.2f}, {qs}",fontsize=7.5)
    ax.set_ylabel(YL.get(f,UNIT[r.modality]),fontsize=7)
fig.tight_layout();fig.savefig(O/'fig11_group_distributions.png');fig.savefig(O/'fig11_group_distributions.svg')
# ---- Fig 13: PANSS five factor within SZ
P=pd.read_csv(R/'shap_clinic/shap_panss5factor_within_SZ.csv')
order=pd.read_csv(R/'shap_values/shap_importance.csv').head(20).feature.tolist()
facs=['Positive','Negative','Disorganized','Excited','Depressed']
facs=[f for f in facs if f in set(P.factor)] or list(dict.fromkeys(P.factor))
M=P.pivot(index='feature',columns='factor',values='rho').reindex(index=order,columns=facs)
Pm=P.pivot(index='feature',columns='factor',values='p').reindex(index=order,columns=facs)
fig,ax=plt.subplots(figsize=(4.6,5.8))
im=ax.imshow(M.values,cmap='RdBu_r',vmin=-.45,vmax=.45,aspect='auto')
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        if Pm.values[i,j]<.05: ax.text(j,i,'•',ha='center',va='center')
ax.set_xticks(range(len(facs)),facs,rotation=35,ha='right');ax.set_yticks(range(len(order)),[LAB.get(f,f) for f in order])
cb=fig.colorbar(im,ax=ax,shrink=.6);cb.set_label('Spearman ρ (SHAP × factor)')
ax.set_title('PANSS five factors\npatients (n = 34)',fontsize=8.5,loc='left')
fig.text(.02,.01,'• p < 0.05 uncorrected; none survive FDR (q < 0.05).',fontsize=6.5)
fig.tight_layout(rect=(0,.03,1,1));fig.savefig(O/'fig13_shap_panss5.png');fig.savefig(O/'fig13_shap_panss5.svg')
# ---- S-Confound
Cc=pd.read_csv(R/'shap_clinic/shap_clinic_correlations.csv');sc=['PAS','SAS','WHODAS','CDSS']
pan=[('psychosocial_pooled_orig','a  Pooled, n = 67\n    (original)'),('psychosocial_pooled_groupadj','b  Pooled, adjusted\n    for diagnosis'),('psychosocial_within_SZ','c  Patients only\n    n = 34')]
fig,axs=plt.subplots(1,3,figsize=(8.2,5.6),sharey=True)
for ax,(a,t) in zip(axs,pan):
    d=Cc[Cc.analysis==a];M=d.pivot(index='feature',columns='scale',values='rho').reindex(index=order,columns=sc)
    Pm=d.pivot(index='feature',columns='scale',values='p').reindex(index=order,columns=sc)
    im=ax.imshow(M.values,cmap='RdBu_r',vmin=-.45,vmax=.45,aspect='auto')
    for i in range(M.shape[0]):
        for j in range(4):
            if Pm.values[i,j]<.05: ax.text(j,i,'•',ha='center',va='center')
    ax.set_xticks(range(4),sc,rotation=35,ha='right');ax.set_title(t,fontsize=8,loc='left')
axs[0].set_yticks(range(len(order)),[LAB.get(f,f) for f in order])
cb=fig.colorbar(im,ax=axs,shrink=.5);cb.set_label('Spearman ρ (SHAP × scale)')
fig.text(.02,.01,'• p < 0.05 uncorrected; no correlation survives FDR (q < 0.05) in any panel.',fontsize=6.5)
fig.savefig(O/'FigS_confound_shap_clinic.png',bbox_inches='tight');fig.savefig(O/'FigS_confound_shap_clinic.svg',bbox_inches='tight')
print(sorted(set(P.factor)), Cc.groupby('analysis').q_fdr.min().round(3).to_dict(), P.q_fdr.min().round(3), P.rho.abs().max().round(3))
