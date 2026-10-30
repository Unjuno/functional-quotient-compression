"""MA-1182 execution-only scheduling adapter. Uses *identical* frozen train_one/score/verify functions."""
from __future__ import annotations
import argparse,csv,hashlib,json,platform,os,time,statistics
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from multiprocessing import get_context
import torch
from run import CONFIG,BASE,sha,jwrite,train_one,measured_latency,score,verify,analyze
from core import Model,make_data,P,POSITIONS,ARMS,TASKS,PROTOCOL,bytes_inference

def worker(args):
    task,seed,pos,arm,out=args
    torch.set_num_threads(1)
    return train_one(Path(out),seed,task,pos,arm)

def main(out):
    if out.exists():raise FileExistsError(out)
    torch.set_num_threads(1)
    out.mkdir(parents=True);(out/'checkpoints').mkdir()
    (out/'PROTOCOL.json').write_bytes((BASE/'PROTOCOL.json').read_bytes())
    jwrite(out/'ENVIRONMENT.json',{'id':'MA-1182','python':platform.python_version(),'torch':torch.__version__,
        'cpu':platform.processor(),'cpu_affinity':sorted(os.sched_getaffinity(0)),'torch_threads_per_worker':1,
        'workers':4,'scheduler':'ProcessPoolExecutor, independent jobs, same run.train_one','protocol_sha256':sha(BASE/'PROTOCOL.json')})
    jobs=[(task,seed,pos,arm,str(out)) for task in TASKS for seed in CONFIG['seeds'] for pos in POSITIONS for arm in ARMS]
    assert len(jobs)==CONFIG['total_models']==135
    started=time.perf_counter(); rows=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=get_context('spawn')) as pool:
        futures=[pool.submit(worker,job) for job in jobs]
        for f in as_completed(futures):
            row=f.result();rows.append(row)
            with (out/'TRAIN_ROWS.jsonl').open('a') as h:h.write(json.dumps(row,sort_keys=True)+'\n')
            print(f'TRAIN {len(rows):3d}/{len(jobs)} {row["key"]} DEV={row["dev_ce"]:.5f} train_s={row["train_wall_s"]:.2f}',flush=True)
    idx={(r['task'],r['seed'],r['pos'],r['arm']):r for r in rows}
    rows=[idx[t,s,p,a] for t,s,p,a,_ in jobs]
    assert len(rows)==135 and all(sha(out/'checkpoints'/r['checkpoint'])==r['checkpoint_sha256'] for r in rows)
    # HARD SEAL BEFORE fresh or OOD generation, exactly as original source/run.py main.
    jwrite(out/'SEALED_MANIFEST.json',{'id':'MA-1182','fresh_generated':False,'ood_generated':False,
        'protocol_sha256':sha(BASE/'PROTOCOL.json'),'rows':rows,'elapsed_preseal_s':time.perf_counter()-started,
        'execution_only_adapter':'source/parallel_driver.py'})
    jwrite(out/'RUN_STATE.json',{'sealed_sha256':sha(out/'SEALED_MANIFEST.json'),'postseal_started':True})
    print('SEALED_SHA256',sha(out/'SEALED_MANIFEST.json'),'CHECKPOINTS',len(rows),flush=True)
    scored=[]
    for i,r in enumerate(rows):
        ck=out/'checkpoints'/r['checkpoint'];assert sha(ck)==r['checkpoint_sha256']
        m=Model(r['arm'],r['pos'],r['seed']);m.load_state_dict(torch.load(ck,weights_only=True)['state']);m.eval()
        outrow=dict(r)
        for split,label,clength in [(3,'fresh',P['train_sequence_content_length']),(4,'ood',P['ood_content_length'])]:
            x,y=make_data(r['task'],r['seed'],CONFIG['n_'+label],split,content_length=clength)
            sc=score(m,x,y)
            outrow.update({f'{label}_ce':sc['ce'],f'{label}_acc':sc['acc']})
        outrow['latency']=measured_latency(m,r['seed'],r['task'],r['pos'])
        scored.append(outrow)
        if i%15==14:print(f'EVAL {i+1}/{len(rows)}',flush=True)
    jwrite(out/'FULL_RESULTS.json',scored)
    with (out/'RESULTS_CORE.csv').open('w',newline='') as h:
        keys=['key','task','seed','pos','arm','selected_step','dev_ce','dev_accuracy','fresh_ce','fresh_acc','ood_ce','ood_acc','params','bytes','train_wall_s','batch64_forward_us','checkpoint','checkpoint_sha256']
        w=csv.DictWriter(h,fieldnames=keys);w.writeheader()
        for r in scored:
            d={k:r.get(k) for k in keys};d['batch64_forward_us']=r['latency']['batch64_forward_median_us'];w.writerow(d)
    jwrite(out/'ANALYSIS.json',analyze(scored))
    jwrite(out/'VERIFICATION.json',{**verify(out),'source_protocol_sha256':sha(BASE/'PROTOCOL.json'),'results_csv_sha256':sha(out/'RESULTS_CORE.csv'),'sealed_manifest_sha256':sha(out/'SEALED_MANIFEST.json'),'total_models':len(scored),'total_wall_s':time.perf_counter()-started})
    print('FINISHED',round(time.perf_counter()-started,3),'verdict',json.loads((out/'ANALYSIS.json').read_text())['verdict'],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.out)
