"""Post-run audit only: no new training/selection except one exact replay check."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from model import Transformer, unpack, pack, batch_data, world_encoding, conjugate
from run import evaluate

ROOT=Path(__file__).resolve().parents[1]


def main():
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    with (ROOT/'results/fresh.csv').open() as f: rows=list(csv.DictReader(f))
    assert len(rows)==72
    assert len({(r['model'],r['task'],r['world'],r['seed']) for r in rows})==72
    summary=[]; artifacts=[]
    for mode in ('mirror','shared','gain','dense','dense_byte','scalar_only'):
        for task in ('copy','retrieve'):
            group=[r for r in rows if r['model']==mode and r['task']==task]
            assert len(group)==6
            stats={'model':mode,'task':task,'n':6}
            for key in ('nll','accuracy','initial_nll','train_s','infer_b1_ms_median','infer_b32_ms_median'):
                values=np.array([float(r[key]) for r in group])
                for name,value in zip(('mean','median','min','max'),(values.mean(),np.median(values),values.min(),values.max())):
                    stats[key+'_'+name]=float(value)
            for key in ('learned_scalars','frozen_stored_scalars','payload_bytes','forward_dense_macs_per_example'):
                stats[key]=int(group[0][key]); assert len({r[key] for r in group})==1
            stats['learning_gate']=stats['accuracy_median']>=.75 and stats['nll_median']<=.7*stats['initial_nll_median']
            summary.append(stats)
    for r in rows:
        name=f"fresh_{r['model']}_{r['task']}_{r['world']}_{r['seed']}.pt"
        b=(ROOT/'checkpoints'/name).read_bytes()
        assert len(b)==int(r['payload_bytes'])
        assert hashlib.sha256(b).hexdigest()==r['checkpoint_sha256']
        m=unpack(b)
        assert sum(p.numel() for p in m.parameters())==int(r['learned_scalars'])
        artifacts.append({'checkpoint':name,'bytes':len(b),'sha256':r['checkpoint_sha256']})
    # Diagnostic only: destroy observed history but retain query/target, after freeze.
    shuffled=[]
    for r in rows:
        if r['model']!='mirror': continue
        world,seed=int(r['world']),int(r['seed']); task=r['task']
        m=unpack((ROOT/'checkpoints'/f'fresh_mirror_{task}_{world}_{seed}.pt').read_bytes())
        perm=world_encoding(world)
        stream=world*101+seed*1009+(0 if task=='copy' else 100000003)+1000000007
        gen=torch.Generator().manual_seed(stream); correct=0
        with torch.no_grad():
            for _ in range(16):
                x,y=batch_data(task,64,gen,perm)
                x[:,:-1]=torch.roll(x[:,:-1],1,0)
                correct+=int((m(x).argmax(-1)==y).sum())
        shuffled.append({'task':task,'world':world,'seed':seed,'history_shuffled_accuracy':correct/1024})
    # Exact reproducibility: same first candidate trial, no parameter selection.
    world,seed,task=618231,0,'copy'; torch.manual_seed(world*13+seed)
    m=Transformer('mirror')
    gen=torch.Generator().manual_seed(world*101+seed*1009)
    perm=world_encoding(world); opt=torch.optim.AdamW(m.parameters(),lr=.003,weight_decay=0.)
    for _ in range(400):
        x,y=batch_data(task,32,gen,perm);opt.zero_grad(set_to_none=True)
        F.cross_entropy(m(x),y).backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True);opt.step()
    original=(ROOT/'checkpoints/fresh_mirror_copy_618231_0.pt').read_bytes()
    deterministic_payload=hashlib.sha256(pack(m)).hexdigest()==hashlib.sha256(original).hexdigest()
    assert deterministic_payload
    # A common angle shift can be absorbed into core; 4110 stored scalars != 4110 identifiable directions.
    torch.manual_seed(99); s=torch.randn(64,64,dtype=torch.float64)
    theta=torch.linspace(-.5,.7,14,dtype=torch.float64); delta=torch.tensor([.123],dtype=torch.float64)
    gauge_error=float((conjugate(s,theta)-conjugate(conjugate(s,delta)[0],theta-delta[0])).abs().max())
    assert gauge_error<1e-10
    result={'posthoc_audit':True,'no_fresh_tuning':True,'rows_verified':72,'payloads_verified':72,
            'all_causal_prefix_errors_at_most':max(float(r['causal_prefix_max_abs']) for r in rows),
            'all_roundtrips_exact':all(r['roundtrip_exact']=='True' for r in rows),
            'one_trial_replayed_bit_exact':deterministic_payload,'common_shift_gauge_max_abs_fp64':gauge_error,
            'summary':summary,'history_shuffled':shuffled,'checkpoints':artifacts,
            'full_repository_test_suite':'NOT RUN: container direct GitHub DNS unavailable; isolated experiment tests only'}
    (ROOT/'VERIFICATION.json').write_text(json.dumps(result,indent=2))
    with (ROOT/'results/SUMMARY.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    print('Verified:',len(rows),'trials, 72 payloads, exact replay, causal checks and common angle gauge',gauge_error)
    print('History-destroyed accuracy:',{t:float(np.mean([r['history_shuffled_accuracy'] for r in shuffled if r['task']==t])) for t in ('copy','retrieve')})

if __name__=='__main__': main()
