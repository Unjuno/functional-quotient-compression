"""GIC001-B: prospective new-world allocation comparison, same primary core."""
from __future__ import annotations
import hashlib,json,sys,time
from pathlib import Path
from typing import Any
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'source'))
import core,run


def scores(p:dict[str,np.ndarray],w:dict[str,Any]) -> dict[str,np.ndarray]:
    _,g,h=core.gradients('mirror_r2',p,w['H'],w['Q'])
    pair=g@g.T
    return {'gradient_conflict':np.maximum(-pair,0).sum(1),
            'gradient_norm':np.sum(g*g,axis=1),
            'fisher_norm':np.sum(h*np.linalg.solve(w['H'],h.T).T,axis=1)/4,
            'train_residual':run.metrics(w,'mirror_r2',p,'train'),
            'val_residual':run.metrics(w,'mirror_r2',p,'val')}


def top2(a:np.ndarray) -> list[int]:
    return np.argsort(-a,kind='stable')[:2].tolist()


def hashes() -> dict[str,str]:
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [ROOT/'FOLLOWUP_PROTOCOL.json',ROOT/'followup.py',ROOT/'followup_tests.py',ROOT/'source/core.py',ROOT/'source/run.py']}


def execute():
    cfg=json.loads((ROOT/'FOLLOWUP_PROTOCOL.json').read_text())
    lock=json.loads((ROOT/'results/FOLLOWUP_LOCK.json').read_text())
    assert lock['hashes']==hashes()
    primary=json.loads((ROOT/'results/FRESH_LOCK.json').read_text())
    assert primary['source_hashes']==run.source_hashes()
    rows=[];score_rows=[];prefix_hashes={};max_alias_error=0.
    for world in cfg['fresh_worlds']:
        for family in cfg['families']:
            w=core.make_world(family,world)
            for init in cfg['initializations']:
                p,_,_=run.fit(w,'mirror_r2',init,.03,steps=150)
                prefix_hashes[f'{world}_{family}_{init}']=run.state_hash(p)
                sc=scores(p,w);max_alias_error=max(max_alias_error,float(np.max(np.abs(sc['train_residual']-sc['fisher_norm']))))
                assert top2(sc['train_residual'])==top2(sc['fisher_norm'])
                for t in range(8):score_rows.append({'world':world,'family':family,'init':init,'task':t,**{k:float(v[t]) for k,v in sc.items()}})
                for policy in cfg['policies']:
                    if policy=='oracle_private' and family!='partial':continue
                    if policy=='oracle_private':ids=w['private_truth']
                    elif policy=='random':ids=sorted(np.random.default_rng(np.random.SeedSequence([world,init,901])).choice(8,2,replace=False).tolist())
                    else:ids=top2(sc[policy])
                    q={k:v.copy() for k,v in p.items()};q['P']=np.zeros((2,17));q['ids']=np.array(ids,dtype=np.int64)
                    opt=core.Adam(q,.03);start=time.perf_counter()
                    for _ in range(450):
                        g,_,_=core.gradients('selected_private',q,w['H'],w['Q']);opt.update(q,g)
                    tr=run.metrics(w,'selected_private',q,'train');te=run.metrics(w,'selected_private',q,'test')
                    oc=core.oracle(w['Btrain'],w['Ytrain'],2,ids)
                    path=ROOT/'artifacts'/f'followup_{world}_{family}_{init}_{policy}.gic'
                    size=core.save_model(path,'selected_private',q)
                    rows.append({'world':world,'family':family,'init':init,'method':policy,'private_ids':','.join(map(str,ids)),
                                 'overlap':len(set(ids)&set(w['private_truth'])) if family=='partial' else None,
                                 'test_mean_mse':float(te.mean()),'test_worst_task_mse':float(te.max()),'train_mse':float(tr.mean()),
                                 'oracle_train_mse':float(np.mean((w['Btrain']@oc.T-w['Ytrain'])**2)),
                                 'inference_bytes':size,'parameters':101,'steps':600,'suffix_seconds':time.perf_counter()-start,
                                 'prefix_hash':run.state_hash(p),'payload':str(path.relative_to(ROOT)),
                                 'payload_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                                 **{f'test_task_{t}':float(te[t]) for t in range(8)}})
                for method in cfg['references']:
                    q,r,_=run.fit(w,method,init,.03)
                    path=ROOT/'artifacts'/f'followup_{world}_{family}_{init}_{method}.gic'
                    r['inference_bytes']=core.save_model(path,method,q);r['payload']=str(path.relative_to(ROOT))
                    r['payload_sha256']=hashlib.sha256(path.read_bytes()).hexdigest();rows.append(r)
        run.write_csv(ROOT/'results/FOLLOWUP_RESULTS.csv',rows);run.write_csv(ROOT/'results/ALLOCATION_SCORES.csv',score_rows)
        print(json.dumps({'world_finished':world,'rows':len(rows)}),flush=True)
    verify_followup(rows,max_alias_error)


def verify_followup(rows,max_alias_error):
    from verify import independent_predict
    count=0;maxerr=0.;alias=0
    for r in rows:
        path=ROOT/r['payload'];assert path.stat().st_size==r['inference_bytes']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==r['payload_sha256']
        mode,p=core.load_model(path);w=core.make_world(r['family'],r['world'])
        err=np.mean((independent_predict(mode,p,w['xtest'])-w['Ytest'])**2,axis=0)
        maxerr=max(maxerr,float(np.max(np.abs(err-[r[f'test_task_{t}'] for t in range(8)]))))
        assert r['train_mse']>=r['oracle_train_mse']-1e-9;count+=1
    for world in [86211,86212,86213,86214,86215]:
        for family in ['partial','unrelated']:
            for init in [1,2,3]:
                a=next(r for r in rows if r['world']==world and r['family']==family and r['init']==init and r['method']=='fisher_norm')
                b=next(r for r in rows if r['world']==world and r['family']==family and r['init']==init and r['method']=='train_residual')
                assert a['payload_sha256']==b['payload_sha256'];alias+=1
    assert maxerr<1e-12 and max_alias_error<1e-10
    report={'payloads_checked':count,'max_independent_task_metric_error':maxerr,'information_metric_vs_train_residual_score_error':max_alias_error,
            'fisher_train_residual_payload_alias_pairs':alias,'original_source_lock_unchanged':True,'followup_hash_lock_unchanged':True,
            'all_checks_pass':True,'external_replication':False}
    (ROOT/'FOLLOWUP_VERIFICATION.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':execute()
