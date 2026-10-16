"""Follow-up on same real images, labels supervised directly; frozen as separate protocol."""
from __future__ import annotations
import argparse,csv,hashlib,json,time
from pathlib import Path
import torch
from torch.nn import functional as F
from model import METHODS,SEEN,HOLD,Student,payload,reload_payload
from experiment import split_world
ROOT=Path(__file__).parent

def loss_seen(logits,target):
    return F.binary_cross_entropy_with_logits(logits[:,SEEN],target[:,SEEN])

def compute_metrics(model,data):
    with torch.no_grad():
        x,y=data; logits=model(x)
        out={}
        for nm,ids in [('seen',SEEN),('unseen',HOLD),('all',tuple(range(9)))]:
            out[nm+'_BCE']=float(F.binary_cross_entropy_with_logits(logits[:,ids],y[:,ids]))
            out[nm+'_accuracy']=float(((logits[:,ids]>=0)==(y[:,ids]>=.5)).float().mean())
        out['worst_pair_BCE']=float(max(F.binary_cross_entropy_with_logits(logits[:,i],y[:,i]) for i in range(9)))
        return out

def train(world,initialization,method,data,lr,updates,checkpoints,out_dir=None):
    torch.manual_seed(1400000+world*31+initialization)
    model=Student(method)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.0001)
    rng=torch.Generator().manual_seed(510000+world*17+initialization)
    x,y=data['train'];records=[];t=time.perf_counter()
    for step in range(1,updates+1):
        indices=torch.randint(x.shape[0],(64,),generator=rng)
        p=model(x[indices]);loss=loss_seen(p,y[indices])
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        if step in checkpoints:
            records.append({'world':world,'seed':initialization,'method':method,'step':step,'lr':lr,
                            **{'val_'+k:v for k,v in compute_metrics(model,data['val']).items()},
                            **{'test_'+k:v for k,v in compute_metrics(model,data['test']).items()},
                            'wall_s':round(time.perf_counter()-t,5)})
    p=payload(model,{'world':world,'seed':initialization,'objective':'direct_binary_BCE','step':updates})
    if out_dir is not None:
        out_dir.mkdir(exist_ok=True,parents=True)
        (out_dir/f'direct_{world}_{initialization}_{method}.pt').write_bytes(p)
    clone=reload_payload(p)
    with torch.no_grad():assert torch.equal(clone(data['test'][0]),model(data['test'][0]))
    for record in records:
        record.update({'inference_bytes':len(p),'model_sha256':hashlib.sha256(p).hexdigest(),
                       'parameters':sum(p.numel() for p in model.parameters())})
    return records

def run_dev():
    data=split_world(79110);records=[]
    for lr in (0.005,0.015):
        for method in METHODS:
            result=train(79110,0,method,data,lr,400,(400,))[-1]
            records.append({'lr':lr,'method':method,'val_seen_BCE':result['val_seen_BCE']})
            print('DEVELOPMENT DIRECT',lr,method,round(result['val_seen_BCE'],5),flush=True)
    best=min([.005,.015],key=lambda lr:sum(r['val_seen_BCE'] for r in records if r['lr']==lr))
    (ROOT/'results/DIRECT_DEV.json').write_text(json.dumps({'world':79110,'rows':records,'selected_lr':best},indent=2)+'\n')

def run_fresh():
    protocol=json.loads((ROOT/'DIRECT_PROTOCOL.json').read_text())
    freeze=json.loads((ROOT/'DIRECT_LOCK.json').read_text())
    for file,hash_value in freeze['source_sha256'].items():
        assert hashlib.sha256((ROOT/file).read_bytes()).hexdigest()==hash_value,file
    dev=json.loads((ROOT/'results/DIRECT_DEV.json').read_text());lr=dev['selected_lr']
    records=[]
    for world in protocol['worlds']:
        data=split_world(world)
        for seed in protocol['initializations']:
            for method in METHODS:
                got=train(world,seed,method,data,lr,protocol['updates'],protocol['checkpoints'],ROOT/'models')
                records.extend(got);last=got[-1]
                print('DIRECT',world,seed,method,'test_seen_unseen_BCE',round(last['test_seen_BCE'],4),round(last['test_unseen_BCE'],4),flush=True)
    with (ROOT/'results/DIRECT_FRESH.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=records[0].keys());w.writeheader();w.writerows(records)
    print('Direct models',len(records)//len(protocol['checkpoints']))

if __name__=='__main__':
    torch.set_num_threads(1)
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['dev','fresh'],required=True)
    args=ap.parse_args()
    if args.phase=='dev':run_dev()
    else:run_fresh()
