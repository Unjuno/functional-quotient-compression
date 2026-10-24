"""Frozen preregistered gates applied to MA-1177 results. No training or fresh-based retuning."""
import csv,json,hashlib,math
from collections import defaultdict
from pathlib import Path
import statistics as st
from run import ROOT,RESULTS,file_sha


def read_rows(path):
    with path.open(newline='',encoding='utf-8') as f:
        return list(csv.DictReader(f))

def asfloat(x,key):
    return float(x[key])

def mean(rs,key):
    return st.mean(float(r[key]) for r in rs)

def main():
    records=read_rows(RESULTS/'RESULTS_CORE.csv')
    assert len(records)==120
    hard=[r for r in records if r['lane']=='hard']
    dist=[r for r in records if r['lane']=='distill']
    assert len(hard)==72 and len(dist)==48
    hmap={(r['world'],r['seed'],r['kind'],r['beta'],r['loss']):r for r in hard}
    dmap={(r['world'],r['seed'],r['kind'],r['eps'],r['loss']):r for r in dist}
    worlds=sorted({r['world'] for r in hard}); seeds=sorted({int(r['seed']) for r in hard}); kinds=sorted({r['kind'] for r in hard})
    paired_h=[]
    for world in worlds:
      for seed in seeds:
       for kind in kinds:
        for beta in ('2.5','12.0'):
         a=hmap[(world,str(seed),kind,beta,'standard_ce')]
         b=hmap[(world,str(seed),kind,beta,'fixed_ce')]
         old=float(a['fresh_ce_fp64']);new=float(b['fresh_ce_fp64'])
         paired_h.append({'world':world,'seed':seed,'kind':kind,'beta':float(beta),'old':old,'fixed':new,
             'absolute_delta_fixed_minus_old':new-old,
             'relative_gain':(old-new)/old,
             'old_train_saturated_fraction':float(a['train_saturated_gt16p6']),
             'fixed_train_saturated_fraction':float(b['train_saturated_gt16p6'])})
    mirror_pairs=[r for r in paired_h if r['kind']=='mirror' and r['beta']==12.0]
    film_map={(r['world'],r['seed']):r for r in paired_h if r['kind']=='film' and r['beta']==12.0}
    hard_success=sum(p['relative_gain']>=.05 for p in mirror_pairs)
    mirror_specific=sum(p['relative_gain']-film_map[(p['world'],p['seed'])]['relative_gain']>=.05 for p in mirror_pairs)
    paired_d=[]
    for world in worlds:
      for seed in seeds:
       for kind in ('mirror','film'):
        for eps in ('1e-08','1e-12'):
         # Robust to CSV float format since input file pandas vs writer python repr differ.
         epsnum=float(eps)
         def lookup(loss):
             return next(r for r in dist if r['world']==world and int(r['seed'])==seed and r['kind']==kind and float(r['eps'])==epsnum and r['loss']==loss)
         a,b=lookup('standard_soft_ce'),lookup('fixed_soft_ce')
         paired_d.append({'world':world,'seed':seed,'kind':kind,'eps':epsnum,
             'old_kl':float(a['fresh_teacher_student_fp64_kl']),
             'fixed_kl':float(b['fresh_teacher_student_fp64_kl']),
             'old_top1_mismatch':float(a['fresh_top1_mismatch']),
             'fixed_top1_mismatch':float(b['fresh_top1_mismatch']),
             'old_displacement':float(a['max_param_abs_displacement']),
             'fixed_displacement':float(b['max_param_abs_displacement']),
             'old_init_grad':float(a['initial_equal_logit_grad_norm']),
             'fixed_init_grad':float(b['initial_equal_logit_grad_norm'])})
    mirror_dist=[p for p in paired_d if p['kind']=='mirror' and p['eps']==1e-8]
    dist_success=sum(p['old_kl']-p['fixed_kl']>1e-9 and p['fixed_init_grad']==0 for p in mirror_dist)
    h_mirror_aligned=[r for r in hard if r['world']=='householder_aligned' and r['kind']=='mirror' and float(r['beta'])==12.0]
    old_mean=mean([r for r in h_mirror_aligned if r['loss']=='standard_ce'],'fresh_ce_fp64')
    fixed_mean=mean([r for r in h_mirror_aligned if r['loss']=='fixed_ce'],'fresh_ce_fp64')
    d_c=[p['old_kl']-p['fixed_kl'] for p in mirror_dist]
    posthoc=read_rows(ROOT/'exploratory/NEAR_TEACHER_DEV_ONLY.csv')
    posthoc_rows=[]
    for kind in ('mirror','film'):
      for sigma in (0.0001,0.01):
       for loss in ('old','fixed'):
        matching=[r for r in posthoc if r['kind']==kind and float(r['init_noise_sigma'])==sigma and r['loss']==loss]
        posthoc_rows.append({'kind':kind,'noise_sigma':sigma,'loss':loss,'dev_kl_initial_mean':mean(matching,'initial_dev_teacher_student_kl'),'dev_kl_final_mean':mean(matching,'final_dev_teacher_student_kl'),'dev_top1_mismatch_mean':mean(matching,'dev_top1_mismatch')})
    audit=json.loads((RESULTS/'NUMERIC_AUDIT.json').read_text())
    numeric_pass=(all(row['CE_standard_target_gradient']==0 and row['CE_fixed_target_gradient']<0 and abs(row['CE_fixed_gradient_sum'])<1e-12 for row in audit['dtype_cliff_tests'])
                  and all(row['n_sat']>0 and row['standard_target_grad']==0 and row['fixed_target_grad']<0 for row in audit['actual_trained_Mirror_logit_test']))
    summary={
       'id':'MA-1177', 'record_counts':{'hard':len(hard),'distill':len(dist),'posthoc_DEV_only':len(posthoc)},
       'numeric_pass':numeric_pass,
       'gates':{'numeric':numeric_pass,'mirror_hard_ce_gain_5pct_pairs':hard_success,'mirror_hard_ce_gate':hard_success>=4,
             'mirror_specific_advantage_5percentagepoints_pairs':mirror_specific,'mirror_specific_gate':mirror_specific>=4,
             'mirror_distill_eps1e8_positive_1e9_pairs':dist_success,'mirror_distill_gate':dist_success>=5,
             'architecture_adoption':False},
       'key_means':{'aligned_mirror_hard_CE12_standard_fresh_CE':old_mean,'aligned_mirror_hard_CE12_fixed_fresh_CE':fixed_mean,
           'max_hard_fresh_CE_abs_delta':max(abs(p['absolute_delta_fixed_minus_old']) for p in paired_h),
           'max_hard_fresh_CE_rel_delta':max(abs(p['relative_gain']) for p in paired_h),
           'mirror_distill_eps1e8_old_fresh_KL_mean':st.mean(p['old_kl'] for p in mirror_dist),
           'mirror_distill_eps1e8_fixed_fresh_KL_mean':st.mean(p['fixed_kl'] for p in mirror_dist),
           'mirror_distill_eps1e8_paired_gain_sd':st.stdev(d_c),'mirror_distill_eps1e8_paired_gain_u_c':st.stdev(d_c)/math.sqrt(len(d_c)),
           'mirror_distill_eps1e8_old_top1_mismatch_mean':st.mean(p['old_top1_mismatch'] for p in mirror_dist),
           'mirror_distill_eps1e8_fixed_top1_mismatch_mean':st.mean(p['fixed_top1_mismatch'] for p in mirror_dist),
           'mirror_hard_aligned_high_train_saturated_fraction_mean':st.mean(p['old_train_saturated_fraction'] for p in mirror_pairs if p['world']=='householder_aligned')},
       'paired_hard':paired_h,'paired_distill':paired_d,'exploratory_near_teacher_dev_only':posthoc_rows,
       'cpu_training_times_by_loss':{lane+'_'+l:mean([r for r in records if r['lane']==lane and r['loss']==l],'wall_s') for lane,losses in [('hard',('standard_ce','fixed_ce')),('distill',('standard_soft_ce','fixed_soft_ce'))] for l in losses},
       'methodology':'No hyperparameters were adjusted after opening fresh/OOD; posthoc near-transfer is explicitly DEV-only and exploratory, not part of prereg gates. Synthetic 12-key last-token conditional-attention classification and exact-copy distillation, NOT real grokking, causal LM or evidence of superior Mirror capacity.'}
    out=RESULTS/'ANALYSIS.json';out.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('ANALYSIS rows',summary['record_counts'],'gates',summary['gates'],'key',summary['key_means'])
    return summary
if __name__=='__main__':main()
