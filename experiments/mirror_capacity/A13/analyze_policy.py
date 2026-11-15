# Copyright 2026 Unjuno
# SPDX-License-Identifier: Apache-2.0
"""Posthoc A13 policy/optimizer analysis, never used for selecting checkpoints."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
ROOT=Path(__file__).resolve().parent


def true_teacher_coordinates(fam,w):
    # Replicate only A11 teacher initialization for POSTHOC latent-code interpretation.
    tag={'ramp':0,'tent':1,'sine':2}[fam]
    rng=np.random.default_rng(417211+w*19937+tag*170981)
    if fam=='tent':
        center=rng.uniform(-.65,.65,24).astype('float32')
        width=rng.uniform(.20,.46,24).astype('float32')
        return {'center':center[16:],'width':width[16:]}
    if fam=='sine':
        freq=rng.uniform(.75,2.5,24).astype('float32')
        phase=rng.uniform(-np.pi,np.pi,24).astype('float32')
        scale=rng.uniform(.65,1.15,24).astype('float32')
        return {'frequency':freq[16:],'phase':phase[16:],'amplitude':scale[16:]}
    raise ValueError(fam)


def leave_one_out_r2(codes,target):
    pred=np.zeros(len(target))
    for i in range(len(target)):
        idx=np.arange(len(target))!=i
        model=LinearRegression().fit(codes[idx],target[idx]);pred[i]=model.predict(codes[i:i+1])[0]
    denom=((target-target.mean())**2).sum()
    return float(1-((target-pred)**2).sum()/denom) if denom>0 else float('nan')


def run():
    p=pd.read_csv(ROOT/'main/policy_diagnostics.csv')
    a=pd.read_csv(ROOT/'main/audit.csv')
    worlds=pd.read_csv(ROOT/'main/world_means.csv')
    f=pd.read_csv(ROOT/'followup/world_means.csv')
    score=pd.concat([worlds.rename(columns={'support_mse_reloaded':'support_mse'}),f],ignore_index=True)
    policy=[]
    for (family,world,seed,method),g in p.groupby(['family','world','seed','method']):
        assert len(g)==8
        g=g.sort_values('task')
        codes=g[['es_c0','es_c1']].to_numpy()
        for field,latent in true_teacher_coordinates(family,int(world)).items():
            policy.append(dict(family=family,world=int(world),seed=int(seed),method=method,
                               latent=field,loocv_linear_r2=leave_one_out_r2(codes,latent)))
    pd.DataFrame(policy).to_csv(ROOT/'main/policy_latent_posthoc.csv',index=False)
    traces=[]
    for path in (ROOT/'main/traces').glob('*.json'):
        z=json.loads(path.read_text())
        for row in z['es']:
            tr=row['es']
            best=tr[-1]['best_support_mse']
            first=min((r['generation'] for r in tr if r['best_support_mse']<=best*1.02),default=30)
            traces.append(dict(family=z['family'],method=z['method'],world=z['world'],seed=z['seed'],task=row['task'],
                               sigma0=float(np.mean(tr[0]['sigma'])),sigma30=float(np.mean(tr[-1]['sigma'])),
                               first_within_2pct_final=int(first),
                               population_mse_start=tr[0]['population_best_mse'],final=best))
    d=pd.DataFrame(traces)
    d.to_csv(ROOT/'main/evolution_diagnostics.csv',index=False)
    summary={'task_counts':len(p),'world_counts':worlds.groupby(['family','method','algorithm']).size().to_dict() if False else {},
             'es_task_support_wins_over_adam':{},'es_task_support_wins_over_random':{},'code_jacobian_median':{},
             'es_trajectory':{},'code_latent_linear_loocv_r2':{},'method_comparison':{}}
    for fam in ['tent','sine']:
     for meth in ['mirror','shift','film']:
      k=f'{fam}:{meth}'
      q=p.query('family==@fam and method==@meth')
      summary['es_task_support_wins_over_adam'][k]=int((q.es_support_mse<q.adam_support_mse).sum())
      summary['es_task_support_wins_over_random'][k]=int((q.es_support_mse<q.random_support_mse).sum())
      summary['code_jacobian_median'][k]=dict(smin=float(q.jac_s_min.median()),smax=float(q.jac_s_max.median()))
      tr=d.query('family==@fam and method==@meth')
      summary['es_trajectory'][k]=dict(median_final_sigma=float(tr.sigma30.median()),median_first_near_final=int(tr.first_within_2pct_final.median()),
                                      tasks_near_final_by_generation_15=int((tr.first_within_2pct_final<=15).sum()),tasks=48)
      lat=pd.DataFrame(policy).query('family==@fam and method==@meth')
      summary['code_latent_linear_loocv_r2'][k]={field:float(x.loocv_linear_r2.median()) for field,x in lat.groupby('latent')}
      summary['method_comparison'][k]={}
      for alg in ('adam','adam4','random','sobol','es'):
       t=score.query('family==@fam and method==@meth and algorithm==@alg')
       summary['method_comparison'][k][alg]={
           'worlds':len(t),'iid_nrmse_mean':float(t.iid_nrmse.mean()),'ood_nrmse_mean':float(t.ood_nrmse.mean()),
           'support_mse_mean':float(t.support_mse.mean()),'worlds_joint_pass':int(t.joint_quality_pass.sum())}
    with (ROOT/'main/posthoc_policy_summary.json').open('w') as out:json.dump(summary,out,ensure_ascii=False,indent=2)
    print(json.dumps(summary,ensure_ascii=False,indent=2)[:20000])

if __name__=='__main__':run()
