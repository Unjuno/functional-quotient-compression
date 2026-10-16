"""Post-run independent payload reconstruction, replay, and declared diagnostics.

This verifier is not a training dependency and does not change FRESH_LOCK.
"""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'))
import core,run


def independent_predict(mode,p,x):
    """Separate interpolation/evaluation path: np.interp, not training hats."""
    if mode=='independent': curves=p['C']
    elif mode=='shared': curves=np.tile(p['c0'],(8,1))
    elif mode=='film': curves=p['a'][:,None]*p['c0']+p['b'][:,None]
    else:
        curves=p['c0']+np.einsum('tr,kr->tk',p['M'],p['D'])
        if 'P' in p:
            for i,t in enumerate(p['ids']):curves[t]+=p['P'][i]
    return np.stack([np.interp(x,np.linspace(-1,1,17),c) for c in curves],axis=1)


def verify():
    lock=json.loads((ROOT/'results/FRESH_LOCK.json').read_text())
    assert lock['source_hashes']==run.source_hashes()
    all_rows=pd.concat([pd.read_csv(ROOT/'results/CALIBRATION_RESULTS.csv'),pd.read_csv(ROOT/'results/FRESH_RESULTS.csv')],ignore_index=True)
    fresh=all_rows[all_rows.stage=='fresh']; maxerr=0.; checked=0
    worlds={}
    for row in all_rows.to_dict('records'):
        key=(row['family'],int(row['world']))
        if key not in worlds:worlds[key]=core.make_world(*key)
        w=worlds[key];path=ROOT/row['payload']
        assert path.stat().st_size==row['inference_bytes']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['payload_sha256']
        mode,p=core.load_model(path)
        y=independent_predict(mode,p,w['xtest'])
        errors=np.mean((y-w['Ytest'])**2,axis=0)
        maxerr=max(maxerr,float(np.max(np.abs(errors-[row[f'test_task_{t}'] for t in range(8)]))))
        checked+=1
    assert maxerr<1e-12
    assert fresh.train_minus_oracle.min()>=-1e-9
    alias=0
    for _,g in fresh.groupby(['family','world','init']):
        a=g[g.method=='mirror_r2'].iloc[0];b=g[g.method=='native_lowrank'].iloc[0]
        assert a.payload_sha256==b.payload_sha256
        assert a.test_mean_mse==b.test_mean_mse
        priv=g[g.method.str.contains('private')]
        assert priv.prefix_hash.nunique()==1
        alias+=1
    replay=[]
    for family in core.FAMILIES:
        w=core.make_world(family,86111)
        for method in json.loads((ROOT/'PROTOCOL.json').read_text())['methods']:
            p,row,_=run.fit(w,method,1,lock['learning_rates'][method])
            old=fresh[(fresh.family==family)&(fresh.world==86111)&(fresh['init']==1)&(fresh.method==method)].iloc[0]
            assert row['state_hash']==old.state_hash
            assert abs(row['test_mean_mse']-old.test_mean_mse)<1e-14
            path=ROOT/'artifacts/replay.gic';core.save_model(path,method,p)
            assert hashlib.sha256(path.read_bytes()).hexdigest()==old.payload_sha256
            replay.append({'family':family,'method':method,'state_exact':True,'payload_exact':True})
    report={'source_lock_unchanged':True,'payloads_checked':checked,'fresh_payloads':len(fresh),'max_independent_task_metric_error':maxerr,
            'fresh_replayed_models':len(replay),'fresh_replay_state_and_payload_exact':True,'native_alias_pairs_exact':alias,
            'all_private_prefix_pairs_exact':True,'min_train_minus_empirical_oracle':float(fresh.train_minus_oracle.min()),
            'fresh_worlds':sorted(fresh.world.unique().tolist()),'fresh_initializations':sorted(fresh['init'].unique().tolist()),
            'same_session_audit_not_external_replication':True,'all_checks_pass':True}
    (ROOT/'VERIFICATION.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


def traps():
    rows=[]
    for world in [86111,86112,86113,86114,86115]:
        w=core.make_world('aligned',world)
        mean=core.oracle(w['Btrain'],w['Ytrain'],0)[0]
        for init in [1,2,3]:
            for zero in [True,False]:
                for surgery in [False,True]:
                    p=core.initialize('mirror_r2',8,17,init);p['c0'][:]=mean;p['M'][:]=0
                    if zero:p['D'][:]=0
                    initial=core.gradients('mirror_r2',p,w['H'],w['Q'])
                    initial_diag=core.diagnostics(initial[1]);initial_D=np.linalg.norm(initial[0]['D']);initial_M=np.linalg.norm(initial[0]['M'])
                    opt=core.Adam(p,.03);rng=np.random.default_rng(np.random.SeedSequence([world,init,800]))
                    for step in range(600):
                        g,s,_=core.gradients('mirror_r2',p,w['H'],w['Q'])
                        if surgery:g=core.project_into(g,s,'mirror_r2',rng)
                        opt.update(p,g)
                    err=run.metrics(w,'mirror_r2',p,'test')
                    rows.append({'world':world,'init':init,'both_factors_zero':zero,'pcgrad':surgery,
                                 'test_mean_mse':float(err.mean()),'test_worst_mse':float(err.max()),
                                 'initial_mean_shared_grad_norm':float(np.linalg.norm(initial[1].mean(0))),
                                 'initial_individual_shared_grad_norm':initial_diag['mean_grad_norm'],
                                 'initial_D_grad_norm':float(initial_D),'initial_M_grad_norm':float(initial_M),
                                 'final_D_norm':float(np.linalg.norm(p['D'])),'final_M_norm':float(np.linalg.norm(p['M']))})
    run.write_csv(ROOT/'results/ZERO_FACTOR_DIAGNOSTIC.csv',rows)
    print(pd.DataFrame(rows).groupby(['both_factors_zero','pcgrad'])[['test_mean_mse','test_worst_mse','final_D_norm','final_M_norm']].agg(['median','min','max']).to_string())
    # Analytic two-task common-scalar test and paid one-coordinate-per-task escape.
    theta=0.;m=np.zeros(2);target=np.array([-1.,1.])
    for _ in range(200):m-=.1*(m-target)
    toy={'shared_theta_final':theta,'shared_each_task_half_squared_loss':[.5,.5],
         'opposed_pcgrad_update':core.pcgrad(np.array([[1.],[-1.]]),np.random.default_rng(1)).tolist(),
         'private_coordinates_final':m.tolist(),'private_each_task_half_squared_loss':(.5*(m-target)**2).tolist(),
         'interpretation':'private adds paid degrees of freedom; gradient surgery alone cannot violate the shared class minimum'}
    (ROOT/'results/ANALYTIC_TOY.json').write_text(json.dumps(toy,indent=2))


def benchmark():
    df=pd.read_csv(ROOT/'results/FRESH_RESULTS.csv');rows=[]
    for row in df[(df.world==86111)&(df['init']==1)&(df.family=='aligned')].to_dict('records'):
        mode,p=core.load_model(ROOT/row['payload'])
        for batch in [1,64]:
            x=np.linspace(-.99,.99,batch)
            def forward():return core.basis(x,17)@core.coefficients(mode,p).T
            for _ in range(100):forward()
            groups=[]
            for _ in range(7):
                t=time.perf_counter_ns()
                for __ in range(500):forward()
                groups.append((time.perf_counter_ns()-t)/500/1e6)
            rows.append({'method':row['method'],'batch':batch,'tasks_per_call':8,'median_ms':float(np.median(groups)),
                         'min_ms':float(np.min(groups)),'max_ms':float(np.max(groups)),
                         'warmup':100,'groups':7,'calls_per_group':500,'inference_bytes':row['inference_bytes']})
    run.write_csv(ROOT/'results/BENCHMARK.csv',rows)
    print(pd.DataFrame(rows).pivot(index='method',columns='batch',values='median_ms').to_string())

if __name__=='__main__':
    verify();traps();benchmark()
