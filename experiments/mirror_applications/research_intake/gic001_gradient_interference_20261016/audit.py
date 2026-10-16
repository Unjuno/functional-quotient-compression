"""Post-training independent arithmetic checks, world-level analysis and runtime audit."""
from __future__ import annotations
import hashlib,itertools,json,time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import BSpline
from gic import CFG,ROOT,Model,encode,load_model,make_world,train

OUT=ROOT/'evidence'

def reference_optimum(c,h,rank,ids):
    # Symmetric square root, independently of the core's Cholesky whitening.
    eig,vec=np.linalg.eigh(h);root=(vec*np.sqrt(eig))@vec.T
    inv=(vec/np.sqrt(eig))@vec.T
    keep=[i for i in range(len(c)) if i not in ids];result=c.copy()
    z=c[keep]@root;avg=z.mean(0);u,s,v=np.linalg.svd(z-avg,full_matrices=False)
    result[keep]=(avg+(u[:,:rank]*s[:rank])@v[:rank])@inv
    return result

def reference_population(c,truth):
    # Evaluate BSpline objects, not the core's quadratic-form loss computation.
    k=c.shape[1];edges=np.linspace(-1,1,k-2);v,w=np.polynomial.legendre.leggauss(8)
    nodes=np.concatenate([(a+b)/2+(b-a)*v/2 for a,b in zip(edges[:-1],edges[1:])])
    weights=np.concatenate([w*(b-a)/4 for a,b in zip(edges[:-1],edges[1:])])
    knots=np.r_[np.repeat(-1.,4),edges[1:-1],np.repeat(1.,4)]
    error=BSpline(knots,(c-truth).T,3,axis=0)(nodes)
    return weights@(error**2)

def comparison(df,regime,candidate,control,mean_required=False):
    key=['world','init'];a=df[(df.regime==regime)&(df.method==candidate)].set_index(key)
    b=df[(df.regime==regime)&(df.method==control)].set_index(key)
    ratios=pd.DataFrame({'mean_reduction':1-a.population_mean/b.population_mean,
                         'worst_reduction':1-a.population_worst/b.population_worst})
    m=ratios.groupby('world').median()
    m['pass']=(m.worst_reduction>=.05)&(m.mean_reduction>=(.05 if mean_required else -.01))
    return {'candidate':candidate,'control':control,'regime':regime,'worlds_passing':int(m['pass'].sum()),
            'verdict':'PASS_SCOPED' if int(m['pass'].sum())>=4 else 'FAIL',
            'world_details':[{k:(bool(v) if k=='pass' else float(v)) for k,v in row.items()}|{'world':int(i)} for i,row in m.iterrows()]}

def require_single_thread():
    from threadpoolctl import threadpool_info
    pools=threadpool_info()
    if any(p['num_threads']!=1 for p in pools):
        raise RuntimeError('Run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 before Python starts')
    return pools

def main():
    pools=require_single_thread()
    (OUT/'BENCHMARK_THREADPOOLS.json').write_text(json.dumps(pools,indent=2)+'\n')
    lock=json.loads((ROOT/'PRE_RUN_LOCK.json').read_text())
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in lock['files'].items())
    rows=[];cache={};max_error=0.; max_floor_error=0.;max_train_oracle_error=0.; count=0
    for stage,worlds,seeds in [('development',CFG['development_worlds'],CFG['development_init_seeds']),('fresh',CFG['fresh_worlds'],CFG['fresh_init_seeds'])]:
        files=list((OUT/stage).glob('*.json'))
        expected=set(itertools.product(CFG['regimes'],worlds,seeds,CFG['methods']))
        observed=set()
        for path in files:
            row=json.loads(path.read_text());obs=(row['regime'],row['world'],row['init'],row['method'])
            assert obs in expected and obs not in observed;observed.add(obs)
            payload=path.with_suffix('.bin');model=load_model(payload);raw=payload.read_bytes()
            assert raw==encode(model) and hashlib.sha256(raw).hexdigest()==row['payload_sha256']
            assert len(raw)==row['payload_bytes']==48+8*row['learned_scalars']+4*len(model.ids)
            key=(row['world'],row['regime'])
            if key not in cache:cache[key]=make_world(*key)
            world=cache[key];coef=model.coeff();pop=reference_population(coef,world['truth'])
            tr=((world['train']['B']@coef.T-world['train']['Y'])**2).mean(0)
            te=((world['test']['B']@coef.T-world['test']['Y'])**2).mean(0)
            actual={'population_mean':pop.mean(),'population_worst':pop.max(),'train_mean':tr.mean(),'test_mean':te.mean(),'test_worst':te.max()}
            err=max(abs(actual[k]-row[k]) for k in actual);max_error=max(max_error,err)
            assert err<1e-10
            if model.kind=='independent':pop_opt=world['truth']
            else:pop_opt=reference_optimum(world['truth'],world['population_H'],model.r,model.ids)
            floor=reference_population(pop_opt,world['truth']).mean()
            max_floor_error=max(max_floor_error,abs(floor-row['population_floor']))
            assert abs(floor-row['population_floor'])<1e-10 and pop.mean()>=floor-1e-10
            train_ls=np.linalg.lstsq(world['train']['B'],world['train']['Y'],rcond=None)[0].T
            train_opt=train_ls if model.kind=='independent' else reference_optimum(train_ls,world['train']['stats']['H'],model.r,model.ids)
            empirical=((world['train']['B']@train_opt.T-world['train']['Y'])**2).mean()
            max_train_oracle_error=max(max_train_oracle_error,abs(empirical-row['oracle_train_mean']))
            assert abs(empirical-row['oracle_train_mean'])<1e-10
            row['stage']=stage;rows.append(row);count+=1
        assert observed==expected
    df=pd.DataFrame(rows);fresh=df[df.stage=='fresh'].copy()
    for _,g in fresh[fresh.method.str.startswith('private')].groupby(['world','init','regime']):
        assert g.preallocation_parent_sha256.nunique()==1
    flat=df.drop(columns=['trace']).copy()
    for col in flat.columns:
        if any(isinstance(v,list) for v in flat[col]):flat[col]=flat[col].map(lambda x:json.dumps(x))
    flat.to_csv(OUT/'RESULTS_CORE.csv',index=False,float_format='%.17g')
    cols=['population_mean','population_worst','population_floor','train_optimum_gap','gradient_conflict_tail','cancellation_tail','payload_bytes','train_seconds']
    world_summary=fresh.groupby(['regime','method','world'])[cols].median().reset_index()
    world_summary.to_csv(OUT/'WORLD_MEDIANS.csv',index=False,float_format='%.17g')
    summary=world_summary.groupby(['regime','method'])[cols].agg(['median','min','max'])
    summary.columns=['_'.join(c) for c in summary.columns]
    summary=summary.reset_index();summary.to_csv(OUT/'SUMMARY.csv',index=False,float_format='%.12g')
    outcomes={'H2_pcgrad':[], 'H3':comparison(fresh,'partial','private_conflict','rank2',True)}
    for regime in CFG['regimes']:
        for candidate,control in [('shared_pcgrad','shared'),('rank2_pcgrad','rank2')]:
            outcomes['H2_pcgrad'].append(comparison(fresh,regime,candidate,control))
    outcomes['H4_vs_loss']=comparison(fresh,'partial','private_conflict','private_loss')
    outcomes['H4_vs_random']=comparison(fresh,'partial','private_conflict','private_random')
    c1=outcomes['H4_vs_loss']['world_details'];c2=outcomes['H4_vs_random']['world_details']
    both=sum(a['pass'] and b['pass'] for a,b in zip(c1,c2));outcomes['H4']={'worlds_passing_both':int(both),'verdict':'PASS_SCOPED' if both>=4 else 'FAIL'}
    outcomes['native_rank2_alias']='exact same function/parameters as ordinary shared low-rank spline dictionary; no Mirror-specific novelty'
    allocation=[]
    for world in CFG['fresh_worlds']:
        for init in CFG['fresh_init_seeds']:
            g=fresh[(fresh.world==world)&(fresh.init==init)&(fresh.regime=='partial')]
            truth=set(cache[(world,'partial')]['exceptions']);record={'world':world,'init':init,'true_exceptions':sorted(truth)}
            for _,r in g[g.method.str.startswith('private')].iterrows():record[r.method]=r.selected_ids
            allocation.append(record)
    outcomes['partial_allocation']=allocation
    replay=[]
    world=make_world(86001,'partial')
    for method in CFG['methods']:
        model,new=train(method,world,86001,0)
        old=fresh[(fresh.world==86001)&(fresh.init==0)&(fresh.regime=='partial')&(fresh.method==method)].iloc[0]
        digest=hashlib.sha256(encode(model)).hexdigest()
        assert digest==old.payload_sha256
        replay.append({'method':method,'sha256':digest,'exact':True})
    timings=[]
    for method in CFG['methods']:
        r=fresh[(fresh.world==86001)&(fresh.init==0)&(fresh.regime=='partial')&(fresh.method==method)].iloc[0]
        model=load_model(OUT/'fresh'/r.payload_name)
        for batch in (1,64):
            x=np.linspace(-.95,.95,batch)
            for _ in range(30):model.predict(x)
            samples=[]
            for _ in range(5):
                start=time.perf_counter_ns()
                for _ in range(100):model.predict(x)
                samples.append((time.perf_counter_ns()-start)/100/1e6)
            timings.append({'method':method,'batch':batch,'median_ms':float(np.median(samples)),
                            'min_ms':float(min(samples)),'max_ms':float(max(samples)),
                            'effective_coefficient_workspace_bytes':8*8*12,'basis_workspace_bytes':8*batch*12})
    pd.DataFrame(timings).to_csv(OUT/'RUNTIME.csv',index=False)
    verification={'source_lock_match':True,'total_main_models':count,'fresh_models':len(fresh),'development_models':44,
                  'total_main_optimizer_updates':count*CFG['updates'],'payload_roundtrips':count,
                  'max_metric_abs_error':float(max_error),'max_population_floor_abs_error':float(max_floor_error),
                  'max_empirical_oracle_abs_error':float(max_train_oracle_error),'shared_parent_hash_groups':60,
                  'replayed_training_models':replay,'audit_backend':'independent BSpline-object integration and symmetric-root SVD',
                  'external_replication':False,'runtime_clock_locked':False,'training_used_concurrent_workers':True,
                  'interruption':'First fresh shell command hit tool timeout after 120 finished fits; completed payloads verified, missing independent fits rerun. No source/protocol changes.'}
    (OUT/'VERIFICATION.json').write_text(json.dumps(verification,indent=2,allow_nan=False)+'\n')
    (OUT/'DECISION.json').write_text(json.dumps(outcomes,indent=2,allow_nan=False)+'\n')
    print(json.dumps(verification,indent=2)); print(summary[['regime','method','population_mean_median','population_worst_median','population_floor_median','payload_bytes_median']].to_string(index=False))
    print(json.dumps({k:v for k,v in outcomes.items() if k not in ('partial_allocation','H2_pcgrad')},indent=2))

if __name__=='__main__':main()
