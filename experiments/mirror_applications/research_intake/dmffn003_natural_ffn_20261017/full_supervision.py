"""Stage C: natural-image nine-task supervised FFN compression. No Mirror-generated teacher."""
from __future__ import annotations
import argparse,csv,hashlib,json,time
from pathlib import Path
import torch
from torch.nn import functional as F
from experiment import split_world
from model import METHODS,Student,payload,reload_payload
ROOT=Path(__file__).parent

def loss_all(logits,labels):return F.binary_cross_entropy_with_logits(logits,labels)

def compute_metrics(model,data):
    with torch.no_grad():
        logits=model(data[0]);labs=data[1]
        return {'all_BCE':float(F.binary_cross_entropy_with_logits(logits,labs)),
                'all_accuracy':float(((logits>=0)==(labs>=.5)).float().mean()),
                'worst_pair_BCE':float(max(F.binary_cross_entropy_with_logits(logits[:,i],labs[:,i]) for i in range(9)))}

def train(world,seed,method,data,lr,steps,checkpoints,save_dir=None):
    torch.manual_seed(1410000+world*17+seed)
    m=Student(method);opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=.0001)
    gen=torch.Generator().manual_seed(1411000+world*31+seed)
    x,y=data['train'];rows=[];start=time.perf_counter()
    for step in range(1,steps+1):
        ix=torch.randint(len(x),(64,),generator=gen)
        loss=loss_all(m(x[ix]),y[ix])
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        if step in checkpoints:
            val=compute_metrics(m,data['val']);test=compute_metrics(m,data['test'])
            rows.append({'world':world,'seed':seed,'method':method,'step':step,'lr':lr,
                         **{'val_'+k:v for k,v in val.items()},
                         **{'test_'+k:v for k,v in test.items()},'wall_s':time.perf_counter()-start})
    bs=payload(m,{'world':world,'seed':seed,'objective':'full_nine_label_BCE','step':steps})
    clone=reload_payload(bs)
    with torch.no_grad():assert torch.equal(clone(data['test'][0]),m(data['test'][0]))
    if save_dir is not None:
        save_dir.mkdir(exist_ok=True,parents=True)
        (save_dir/f'full_{world}_{seed}_{method}.pt').write_bytes(bs)
    for row in rows:row.update({'inference_bytes':len(bs),'model_sha256':hashlib.sha256(bs).hexdigest(),'parameters':sum(p.numel() for p in m.parameters())})
    return rows

def dev():
    data=split_world(79210);runs=[]
    for lr in (0.005,0.015):
        for method in METHODS:
            rec=train(79210,0,method,data,lr,400,(400,))[-1]
            runs.append({'lr':lr,'method':method,'val_all_BCE':rec['val_all_BCE']})
            print('FULL_DEV',lr,method,round(rec['val_all_BCE'],6),flush=True)
    best=min((.005,.015),key=lambda v:sum(r['val_all_BCE'] for r in runs if r['lr']==v))
    (ROOT/'results/FULL_DEV.json').write_text(json.dumps({'rows':runs,'selected_lr':best},indent=2)+'\n')

def fresh():
    prot=json.loads((ROOT/'FULL_PROTOCOL.json').read_text())
    lock=json.loads((ROOT/'FULL_LOCK.json').read_text())
    for fn,sha in lock['source_sha256'].items():assert hashlib.sha256((ROOT/fn).read_bytes()).hexdigest()==sha,fn
    lr=json.loads((ROOT/'results/FULL_DEV.json').read_text())['selected_lr']
    output=[]
    for world in prot['worlds']:
        data=split_world(world)
        for seed in prot['initializations']:
            for method in METHODS:
                r=train(world,seed,method,data,lr,prot['updates'],prot['checkpoints'],ROOT/'models')
                output.extend(r)
                print('FULL',world,seed,method,'BCE',round(r[-1]['test_all_BCE'],6),'bytes',r[-1]['inference_bytes'],flush=True)
    with (ROOT/'results/FULL_FRESH.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=output[0].keys());w.writeheader();w.writerows(output)
    print('FULL MODELS',len(output)//len(prot['checkpoints']))

if __name__=='__main__':
    torch.set_num_threads(1)
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=('dev','fresh'),required=True)
    a=p.parse_args();dev() if a.phase=='dev' else fresh()
