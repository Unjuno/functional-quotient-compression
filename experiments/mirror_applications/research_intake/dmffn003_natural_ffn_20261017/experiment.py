"""CPU natural-digit teacher-to-student compression, without heldout-pair leakage."""
from __future__ import annotations
import argparse,csv,hashlib,io,json,os,platform,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from model import D,HOLD,SEEN,METHODS,Student,Teachers,labels_digits,payload,reload_payload

ROOT=Path(__file__).parent

def split_world(world):
    digits=load_digits()
    idx=np.arange(len(digits.target));y=digits.target
    tr,other=train_test_split(idx,train_size=0.60,random_state=world,stratify=y)
    val,te=train_test_split(other,train_size=0.50,random_state=world+20000,stratify=y[other])
    mu=digits.data[tr].mean(axis=0)/16
    sd=np.maximum(digits.data[tr].std(axis=0)/16,0.10)
    def make(indices):
        x=torch.tensor((digits.data[indices]/16-mu)/sd,dtype=torch.float32)
        lab=labels_digits(torch.as_tensor(y[indices],dtype=torch.long))
        return x,lab
    return {'train':make(tr),'val':make(val),'test':make(te),'split_index_sha256':hashlib.sha256(tr.tobytes()+val.tobytes()+te.tobytes()).hexdigest(),'samples':{'train':len(tr),'val':len(val),'test':len(te)}}

def soft_bce(logits,target_logits,ind):
    return F.binary_cross_entropy_with_logits(logits[:,ind],torch.sigmoid(target_logits[:,ind].detach()/2),reduction='mean')

def soft_kl(logits,target_logits,ind):
    target=torch.sigmoid(target_logits[:,ind]/2).clamp(1e-7,1-1e-7)
    return (F.binary_cross_entropy_with_logits(logits[:,ind],target,reduction='none')
           +(target*target.log()+(1-target)*torch.log1p(-target))).mean()

def evaluate(model,teacher,data):
    model.eval()
    with torch.no_grad():
        x,y=data
        logits=model(x)
        logits_teacher=teacher(x)
        results={}
        for label,ids in [('seen',SEEN),('unseen',HOLD),('all',tuple(range(9)))]:
            results[f'{label}_BCE']=float(F.binary_cross_entropy_with_logits(logits[:,ids],y[:,ids]))
            results[f'{label}_accuracy']=float(((logits[:,ids]>0)==(y[:,ids]>.5)).float().mean())
            results[f'{label}_teacher_KL']=float(soft_kl(logits,logits_teacher,ids))
        results['worst_pair_BCE']=float(max(F.binary_cross_entropy_with_logits(logits[:,j],y[:,j]) for j in range(9)))
        return results

def teacher_quality(model,dataset):
    with torch.no_grad():
        x,y=dataset
        logits=model(x)
        acc=((logits>0)==(y>.5)).float().mean().item()
        return {'accuracy':acc,'BCE':float(F.binary_cross_entropy_with_logits(logits,y))}

def fit_teacher(world,data,out,updates=800):
    torch.manual_seed(world+510000)
    model=Teachers();opt=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.0001)
    gen=torch.Generator().manual_seed(world+777)
    x,y=data['train']
    for step in range(updates):
        index=torch.randint(len(x),(64,),generator=gen)
        loss=F.binary_cross_entropy_with_logits(model(x[index]),y[index],reduction='sum')/64
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    model.eval()
    stats={s:teacher_quality(model,data[s]) for s in ['train','val','test']}
    if out is not None:
        out.mkdir(exist_ok=True,parents=True)
        bs=payload(model,{'world':world,'teacher_updates':updates})
        (out/f'teacher_world{world}.pt').write_bytes(bs)
    return model,stats

def fit_student(world,init_seed,method,teacher,data,lr,updates,checkpoints,save_dir=None):
    torch.manual_seed(100000+init_seed+world*23)
    model=Student(method)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=.0001)
    gen=torch.Generator().manual_seed(33000+init_seed+world*11)
    x,y=data['train']
    with torch.no_grad():teacher_logits=teacher(x)
    rows=[];start=time.perf_counter()
    for step in range(1,updates+1):
        index=torch.randint(len(x),(64,),generator=gen)
        pred=model(x[index]);loss=soft_bce(pred,teacher_logits[index],SEEN)
        opt.zero_grad(set_to_none=True);loss.backward();opt.step()
        if step in checkpoints:
            train=evaluate(model,teacher,data['train']);val=evaluate(model,teacher,data['val']);test=evaluate(model,teacher,data['test'])
            rows.append({'world':world,'init':init_seed,'method':method,'update':step,'lr':lr,'train_seen_BCE':train['seen_BCE'],**{'val_'+k:v for k,v in val.items()},**{'test_'+k:v for k,v in test.items()},'wall_s':time.perf_counter()-start})
    bs=payload(model,{'world':world,'init':init_seed,'updates':updates,'lr':lr})
    if save_dir is not None:
        save_dir.mkdir(exist_ok=True,parents=True)
        (save_dir/f'{world}_{init_seed}_{method}.pt').write_bytes(bs)
    restored=reload_payload(bs)
    with torch.no_grad():diff=(model(data['test'][0])-restored(data['test'][0])).abs().max().item()
    assert diff==0,diff
    sz=len(bs);h=hashlib.sha256(bs).hexdigest()
    for row in rows:row.update({'inference_bytes':sz,'model_sha256':h,'parameters':sum(p.numel() for p in model.parameters())})
    return rows

def run_dev(args):
    data=split_world(79010);t,st=fit_teacher(79010,data,ROOT/'models')
    print('Teacher dev',st,flush=True)
    rec=[]
    for lr in [0.005,0.015]:
        for method in METHODS:
            rows=fit_student(79010,0,method,t,data,lr,400,(400,))
            rec.append({'lr':lr,'method':method,'seen_val_teacher_KL':rows[-1]['val_seen_teacher_KL']})
            print('DEV',lr,method,round(rec[-1]['seen_val_teacher_KL'],6),flush=True)
    ROOT.joinpath('results/DEV_SELECTION.json').write_text(json.dumps({'world':79010,'teacher':st,'runs':rec,'selected_lr':min([.005,.015],key=lambda lr:sum(r['seen_val_teacher_KL'] for r in rec if r['lr']==lr)),'split_sha256':data['split_index_sha256']},indent=2)+'\n')

def run_fresh(args):
    protocol=json.loads((ROOT/'PROTOCOL.json').read_text())
    lock=json.loads((ROOT/'FRESH_LOCK.json').read_text())
    for filename,digest in lock['source_sha256'].items():
        assert hashlib.sha256((ROOT/filename).read_bytes()).hexdigest()==digest,filename
    assert protocol['worlds']==[79021,79022,79023]
    lr=json.loads((ROOT/'results/DEV_SELECTION.json').read_text())['selected_lr']
    all_rows=[];all_teachers={}
    for world in protocol['worlds']:
        data=split_world(world);teacher,qual=fit_teacher(world,data,ROOT/'models')
        all_teachers[world]={'quality':qual,'split_sha256':data['split_index_sha256'],'samples':data['samples']}
        for init in protocol['initializations']:
            for method in METHODS:
                rows=fit_student(world,init,method,teacher,data,lr,protocol['updates'],protocol['checkpoints'],ROOT/'models')
                all_rows.extend(rows)
                last=rows[-1]
                print('FRESH',world,init,method,'seen/unseen BCE',round(last['test_seen_BCE'],4),round(last['test_unseen_BCE'],4),'bytes',last['inference_bytes'],flush=True)
    with (ROOT/'results/FRESH_CORE.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=all_rows[0].keys());w.writeheader();w.writerows(all_rows)
    (ROOT/'results/TEACHERS.json').write_text(json.dumps(all_teachers,indent=2)+'\n')
    print('total student fits',len(all_rows)//len(protocol['checkpoints']))

def main():
    torch.set_num_threads(1)
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['dev','fresh'],required=True);args=ap.parse_args()
    if args.mode=='dev':run_dev(args)
    else:run_fresh(args)
if __name__=='__main__':main()
