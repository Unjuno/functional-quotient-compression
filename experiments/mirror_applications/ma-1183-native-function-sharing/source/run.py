"""Sequential preregistered execution. Run train, then score after seal, then verify."""
from __future__ import annotations
import argparse,os,sys,hashlib,json,time,platform,subprocess,statistics,csv
from pathlib import Path
import torch,numpy as np
from engine import ARMS,ACTS,GAMMAS,train_one,verify_and_evaluate,configure
ROOT=Path(__file__).resolve().parents[1]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,obj):Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,sort_keys=True)+'\n')
def getproto():return json.loads((ROOT/'PROTOCOL.json').read_text())
def source_hashes():return {str(f.relative_to(ROOT)):sha(f) for p in [ROOT/'source',ROOT/'tests'] for f in p.glob('*.py')}
def rows_csv(path,rows):
    ks=['key','arm','act','gamma','seed','params','packetbytes','selected_step','dev','init_dev','init_div','final_div','dev_swap','fresh','ood','fresh_swap','fresh_swap_delta','latency_batch128_us','elapsed_s','checkpoint','checkpoint_sha']
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=ks,extrasaction='ignore');w.writeheader();w.writerows(rows)

def run_train():
    configure()
    out=ROOT/'results';out.mkdir(exist_ok=True)
    if (out/'SEAL.json').exists():raise RuntimeError('Already sealed: will not overwrite')
    if (out/'DEV_ONLY.json').exists():raise RuntimeError('Partial run exists: do not overwrite')
    p=getproto();t0=time.perf_counter()
    seedlist=p['seeds'];results=[]
    for act in p['activations']:
        for gamma in GAMMAS:
            for seed in seedlist:
                for arm in p['arms']:
                    row=train_one(arm,act,gamma,seed,p,str(out))
                    results.append(row)
                print(f'TRAIN {len(results):03d}/210 act={act} gamma={gamma} seed={seed}  mirrorDEV={results[-5]["dev"]:.5f}',flush=True)
    assert len(results)==210
    dump(out/'DEV_ONLY.json',results)
    cfg={"torch":torch.__version__,"numpy":np.__version__,"python":sys.version,"machine":platform.platform(),"cpu":platform.processor(),"threads":torch.get_num_threads(),"affinity":sorted(os.sched_getaffinity(0)),"wall_train_s":time.perf_counter()-t0}
    dump(out/'ENVIRONMENT.json',cfg)
    # Check all files before seal, not just metadata.
    for row in results:
        if sha(row['checkpoint'])!=row['checkpoint_sha']:raise RuntimeError('Hash mismatch')
    manifest={'id':p['id'],'fresh_generated':False,'code_hashes':source_hashes(),'protocol_sha256':sha(ROOT/'PROTOCOL.json'),'rows':[{k:r[k] for k in ['key','checkpoint_sha','selected_step','dev']} for r in results]}
    dump(out/'SEAL.json',manifest)
    print('SEALED',len(results),'checkpoints','manifest sha',sha(out/'SEAL.json'),flush=True)

def run_score():
    configure();out=ROOT/'results';p=getproto()
    assert (out/'SEAL.json').exists() and (out/'DEV_ONLY.json').exists()
    if (out/'FULL_RESULTS.json').exists():raise RuntimeError('Already scored: refuse to overwrite')
    seal=json.loads((out/'SEAL.json').read_text());assert seal['fresh_generated'] is False
    assert seal['protocol_sha256']==sha(ROOT/'PROTOCOL.json')
    assert seal['code_hashes']==source_hashes(), 'code drift since seal'
    dev=json.loads((out/'DEV_ONLY.json').read_text())
    assert len(dev)==len(seal['rows'])==210
    for a,b in zip(dev,seal['rows']):
        assert a['key']==b['key'] and a['checkpoint_sha']==b['checkpoint_sha']
        if sha(a['checkpoint'])!=a['checkpoint_sha']:raise RuntimeError('checkpoint drift')
    result=[];t0=time.perf_counter()
    for row in dev:
        result.append(verify_and_evaluate(row,p))
        if len(result)%35==0:print('SCORE',len(result),'of',len(dev),flush=True)
    dump(out/'FULL_RESULTS.json',result);rows_csv(out/'RESULTS_CORE.csv',result)
    def group(act,gamma,arm):return sorted([r for r in result if r['act']==act and r['gamma']==gamma and r['arm']==arm],key=lambda r:r['seed'])
    tab=[]
    for act in ACTS:
      for gamma in GAMMAS:
        for arm in ARMS:
            rr=group(act,gamma,arm)
            tab.append({'act':act,'gamma':gamma,'arm':arm,'fresh_median':statistics.median(x['fresh'] for x in rr),'ood_median':statistics.median(x['ood'] for x in rr),'dev_median':statistics.median(x['dev'] for x in rr),'bytes_median':statistics.median(x['packetbytes'] for x in rr),'train_s_median':statistics.median(x['elapsed_s'] for x in rr),'fwd_us_median':statistics.median(x['latency_batch128_us'] for x in rr),'swap_delta_median':statistics.median(x['fresh_swap_delta'] for x in rr)})
    pairs=[]
    for seed in p['seeds']:
        z={arm:group('relu',.35,arm)[p['seeds'].index(seed)] for arm in ARMS}
        ratio={a:z['mirror']['fresh']/z[a]['fresh'] for a in ['film','coupled','ind_budget']}
        won=all(x<=p['primary_screen']['mirror_fresh_nmse_ratio_threshold'] for x in ratio.values())
        pairs.append({'seed':seed,'ratios':ratio,'passed':won})
    passes=sum(a['passed'] for a in pairs)
    analysis={'id':p['id'],'primary':{'ratio_threshold':p['primary_screen']['mirror_fresh_nmse_ratio_threshold'],'required':p['primary_screen']['paired_seed_successes_required'],'passes':passes,'passed':passes>=p['primary_screen']['paired_seed_successes_required'],'paired':pairs},'groups':tab,'fresh_ood_evaluation_wall_s':time.perf_counter()-t0,'model_count':len(result)}
    dump(out/'ANALYSIS.json',analysis)
    print('SCREEN',passes,'of',len(pairs),'PASS' if analysis['primary']['passed'] else 'FAIL',flush=True)
    for a in pairs:print(a,flush=True)

def run_verify():
    configure();out=ROOT/'results';p=getproto()
    seal=json.loads((out/'SEAL.json').read_text());assert seal['fresh_generated'] is False
    assert sha(ROOT/'PROTOCOL.json')==seal['protocol_sha256'] and source_hashes()==seal['code_hashes']
    allrows=json.loads((out/'FULL_RESULTS.json').read_text())
    sha_ok=0;error=0.0;t0=time.perf_counter()
    for row in allrows:
        sha_ok+=sha(row['checkpoint'])==row['checkpoint_sha']
        z=verify_and_evaluate(row,p)
        for k in ('fresh','ood','fresh_swap','dev'):
            error=max(error,abs(row[k]-z[k]))
        if len(allrows)>0 and sha_ok%35==0:print('VERIFY',sha_ok,'of',len(allrows),flush=True)
    assert sha_ok==len(allrows)==210
    assert error<=1e-12
    dump(out/'VERIFICATION.json',{'id':p['id'],'checkpoint_hash_ok':sha_ok,'recomputed_rows':len(allrows),'max_abs_score_error':error,'protocol_sha256':seal['protocol_sha256'],'seal_sha256':sha(out/'SEAL.json'),'wall_replay_s':time.perf_counter()-t0})
    print('VERIFIED',sha_ok,'max error',error,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['train','score','verify']);a=ap.parse_args()
    {'train':run_train,'score':run_score,'verify':run_verify}[a.action]()
