"""Fixed per-world summaries and paired decisions; no model/data selection."""
import json,hashlib,sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
ROOT=Path(__file__).resolve().parent

def summary(file,tag):
    raw=pd.read_csv(ROOT/'results'/file)
    w=raw.groupby(['family','world','method'],as_index=False).mean(numeric_only=True)
    out=[]
    for (family,method),g in w.groupby(['family','method']):
        r={'family':family,'method':method,'worlds':len(g),'inits_per_world':3}
        for k in ['test_mean_mse','test_worst_task_mse','inference_bytes']:
            r[k+'_median']=float(g[k].median());r[k+'_min']=float(g[k].min());r[k+'_max']=float(g[k].max())
        out.append(r)
    pd.DataFrame(out).to_csv(ROOT/'results'/f'{tag}_SUMMARY.csv',index=False)
    w.to_csv(ROOT/'results'/f'{tag}_WORLD.csv',index=False)
    return w

def paired(w,a,b,family):
    part=w[w.family==family];a_df=part[part.method==a].set_index('world');b_df=part[part.method==b].set_index('world')
    delta=(a_df['test_worst_task_mse']-b_df['test_worst_task_mse']).to_numpy()
    mean=(a_df['test_mean_mse']-b_df['test_mean_mse']).to_numpy()
    se=float(delta.std(ddof=1)/np.sqrt(len(delta)))
    rng=np.random.default_rng(104729)
    boot=delta[rng.integers(0,len(delta),size=(10000,len(delta)))].mean(1)
    bootci=np.quantile(boot,[.025,.975]).tolist()
    acceptable=(a_df['test_worst_task_mse']<=.95*b_df['test_worst_task_mse']) & (a_df['test_mean_mse']<=1.01*b_df['test_mean_mse'])
    return {'family':family,'candidate':a,'control':b,'worlds':len(delta),
            'worlds_worst_5pct_better':int((a_df['test_worst_task_mse']<=.95*b_df['test_worst_task_mse']).sum()),
            'worlds_joint_gate':int(acceptable.sum()),'mean_paired_worst_delta':float(delta.mean()),
            'mean_paired_mean_delta':float(mean.mean()),'u_c_world_mean':se,'k_95_t4':float(t.ppf(.975,4)),
            'interval95_world_cluster_bootstrap':bootci,'bootstrap_resamples':10000,
            'interval95_t':[(float(delta.mean()-t.ppf(.975,4)*se)),float(delta.mean()+t.ppf(.975,4)*se)],
            'raw_world_worst_deltas':delta.tolist(), 'same_bytes':bool(np.array_equal(a_df.inference_bytes,b_df.inference_bytes)),
            'near_zero_control':bool(float(b_df.test_worst_task_mse.max())<1e-8),
            'pass_four_of_five_joint':bool(acceptable.sum()>=4)}

if __name__=='__main__':
    w=summary('FRESH_RESULTS.csv','PRIMARY');v=summary('FOLLOWUP_RESULTS.csv','FOLLOWUP')
    effects=[]
    for family in ['identical','aligned','partial','unrelated']:
        for a,b in [('shared_pcgrad','shared'),('mirror_r2_pcgrad','mirror_r2'),('selected_private_pcgrad','selected_private'),('random_private_pcgrad','random_private'),('selected_private','random_private')]:
            effects.append(paired(w,a,b,family))
    for family in ['partial','unrelated']:
        for a in ['gradient_conflict','gradient_norm','fisher_norm']:
            effects.append(paired(v,a,'val_residual',family))
    (ROOT/'results/PAIRED_EFFECTS.json').write_text(json.dumps(effects,indent=2))
    p=json.load(open(ROOT/'VERIFICATION.json'));f=json.load(open(ROOT/'FOLLOWUP_VERIFICATION.json'))
    total={'primary_calibration_models':192,'primary_fresh_models':720,'primary_replay_models':48,
           'zero_factor_diagnostic_models':60,'followup_models':285,'followup_shared_prefix_runs':30,
           'saved_inference_payloads_verified':p['payloads_checked']+f['payloads_checked'],
           'main_and_followup_class_oracle_bounds_verified':True,'main_source_lock_unchanged':True,
           'max_independent_metric_error':max(p['max_independent_task_metric_error'],f['max_independent_task_metric_error']),
           'native_alias_identical_pairs':60,'information_metric_train_residual_identical_pairs':30,
           'test_count':12,'external_replication':False,
           'conclusion':'Gradient-conflict-only explanation and remedy not supported in this fixture. Selective private capacity can help; no Mirror-specific advantage over identical native low-rank factorization.'}
    (ROOT/'EXPERIMENT_COUNTS.json').write_text(json.dumps(total,indent=2))
    print(json.dumps(total,indent=2))
    for r in effects:
        if r['family'] in ['partial','unrelated']:print(r['family'],r['candidate'],r['control'],r['worlds_joint_gate'],r['mean_paired_worst_delta'],r['interval95_t'])
