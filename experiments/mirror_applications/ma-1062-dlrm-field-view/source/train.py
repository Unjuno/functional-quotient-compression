#!/usr/bin/env python3
"""Small DLRM screen for MA-1062; audit data is accessed only in --phase audit."""
import argparse, hashlib, json, math, os, statistics, time, zipfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]
RUNS=ROOT/'runs'; ARCHIVE=Path('/tmp/ma1062-ml-100k.zip')
P=json.loads((ROOT/'PROTOCOL.json').read_text())
HIDDEN=P['model']['embedding_dim']; CATS=(943,1682,12,7)
METHODS=['full','shared_hash_128','shared_hash_512','field_hash_128x4','mirror_128','diag8_128','diag8_512','diag16_512','mirror_512']
DEV_SEEDS=P['development']['training_seeds']; FRESH_SEEDS=P['fresh']['training_seeds']
EPOCH_CANDIDATES=P['model']['epoch_candidates']; BATCH=P['model']['batch_size']; LR=P['model']['learning_rate']

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def verify_archive():
    if not ARCHIVE.exists() or sha(ARCHIVE)!=P['dataset']['archive_sha256']: raise ValueError('MovieLens archive missing or hash mismatch')

def parse_rating_lines(raw):
    rows=[]
    for line in raw.decode('ascii').splitlines():
        u,i,r,t=map(int,line.split('\t'))
        dt=datetime.fromtimestamp(t,timezone.utc)
        rows.append((u,i,dt.month,dt.weekday(),t,float(r>=4)))
    return rows

def read_base():
    verify_archive()
    with zipfile.ZipFile(ARCHIVE) as z: return parse_rating_lines(z.read('ml-100k/ua.base'))

def read_audit_after_freeze():
    freeze=json.loads((ROOT/'FRESH_FREEZE.json').read_text())
    if freeze.get('audit_numeric_values_opened') is not False: raise ValueError('fresh split already opened or malformed freeze')
    if freeze.get('protocol_sha256')!=sha(ROOT/'PROTOCOL.json') or freeze.get('runner_sha256')!=sha(ROOT/'source/train.py'):
        raise ValueError('frozen protocol or runner hash mismatch')
    verify_archive()
    with zipfile.ZipFile(ARCHIVE) as z: return parse_rating_lines(z.read('ml-100k/ua.test'))

def split_train_dev(rows):
    train=[]; dev=[]; seen_train=set(); seen_dev=set()
    for row in rows:
        key=f'{row[0]}:{row[1]}'
        bucket=int(hashlib.sha256(key.encode()).hexdigest(),16)%10
        (dev if bucket==0 else train).append(row)
        (seen_dev if bucket==0 else seen_train).add(key)
    if seen_train & seen_dev: raise ValueError('user/movie pair crossed train/dev split')
    return train,dev

def arrays(rows,mean,std):
    a=np.asarray(rows,dtype=np.float32)
    cats=torch.from_numpy(a[:,:4].astype(np.int64))
    dense=torch.from_numpy(((a[:,4]-mean)/max(std,1.)).reshape(-1,1))
    labels=torch.from_numpy(a[:,5])
    return cats,dense,labels

def field_hash(ids,field,H):
    # Stable integer field-salted multiplicative hash; no Python hash randomization.
    prime=2654435761
    salt=(field+1)*2246822519
    return torch.remainder(ids*prime+salt,H)

class DLRM(nn.Module):
    def __init__(self,method,mean,std):
        super().__init__(); self.method=method; self.mean=float(mean); self.std=float(std); self.dim=HIDDEN
        if method=='full':
            self.tables=nn.ModuleList([nn.Embedding(c+1,HIDDEN) for c in CATS]); self.shared=None; self.field_tables=None; self.H=0
        elif method.startswith('field_hash'):
            self.tables=None; self.shared=None; self.field_tables=nn.ModuleList([nn.Embedding(128,HIDDEN) for _ in CATS]); self.H=128
        else:
            self.tables=None; self.field_tables=None; self.H=128 if method.endswith('128') else 512
            self.shared=nn.Embedding(self.H,HIDDEN)
        if method.startswith('mirror'):
            self.view=nn.Parameter(torch.zeros(4,8))
        elif method.startswith('diag8'):
            self.view=nn.Parameter(torch.zeros(4,8))
        elif method.startswith('diag16'):
            self.view=nn.Parameter(torch.zeros(4,16))
        else: self.register_parameter('view',None)
        self.bottom=nn.Sequential(nn.Linear(1,HIDDEN),nn.ReLU(),nn.Linear(HIDDEN,HIDDEN))
        self.top=nn.Sequential(nn.Linear(90,32),nn.ReLU(),nn.Linear(32,1))
        for t in (self.tables or []): nn.init.normal_(t.weight,mean=0.,std=.02)
        for t in (self.field_tables or []): nn.init.normal_(t.weight,mean=0.,std=.02)
        if self.shared is not None: nn.init.normal_(self.shared.weight,mean=0.,std=.02)

    def embed_fields(self,cats):
        em=[]
        for f in range(4):
            ids=cats[:,f]
            if self.tables is not None: x=self.tables[f](ids)
            elif self.field_tables is not None: x=self.field_tables[f](field_hash(ids,f,128))
            else: x=self.shared(field_hash(ids,f,self.H))
            if self.view is not None and self.method.startswith('mirror'):
                pairs=x.reshape(-1,8,2); a=self.view[f]; c=torch.cos(a); s=torch.sin(a)
                x=torch.stack((c*pairs[:,:,0]-s*pairs[:,:,1],s*pairs[:,:,0]+c*pairs[:,:,1]),dim=-1).reshape(-1,16)
            elif self.view is not None and self.method.startswith('diag8'):
                x=(x.reshape(-1,8,2)*(1.+self.view[f]).unsqueeze(-1)).reshape(-1,16)
            elif self.view is not None and self.method.startswith('diag16'):
                x=x*(1.+self.view[f])
            em.append(x)
        return torch.stack(em,dim=1)

    def forward(self,cats,dense):
        e=self.embed_fields(cats); d=self.bottom(dense).unsqueeze(1); v=torch.cat((d,e),dim=1)
        dots=[]
        for i in range(5):
            for j in range(i+1,5): dots.append((v[:,i]*v[:,j]).sum(1,keepdim=True))
        features=torch.cat((v.flatten(1),torch.cat(dots,dim=1)),dim=1)
        return self.top(features).squeeze(1)

def metric(logits,labels):
    x=np.asarray(logits,dtype=np.float64); y=np.asarray(labels,dtype=np.float64)
    ll=float(np.mean(np.logaddexp(0,x)-y*x)); probs=1/(1+np.exp(-np.clip(x,-80,80)))
    order=np.argsort(probs,kind='mergesort'); ranks=np.empty(len(order),dtype=np.float64); ranks[order]=np.arange(1,len(order)+1)
    # Average tied ranks for deterministic Mann-Whitney AUC.
    vals=probs[order]; i=0
    while i<len(vals):
        j=i+1
        while j<len(vals) and vals[j]==vals[i]: j+=1
        if j-i>1: ranks[order[i:j]]=float(i+1+j)/2
        i=j
    pos=y==1; npos=int(pos.sum()); nneg=len(y)-npos
    auc=float((ranks[pos].sum()-npos*(npos+1)/2)/(npos*nneg)) if npos and nneg else float('nan')
    return {'logloss':ll,'auc':auc}

@torch.no_grad()
def evaluate(model,rows):
    mean,std=model.mean,model.std; cats,dense,y=arrays(rows,mean,std); model.eval(); out=[]
    for s in range(0,len(y),4096): out.extend(model(cats[s:s+4096],dense[s:s+4096]).cpu().tolist())
    return metric(out,y.numpy()),np.asarray(out,dtype=np.float32)

def train_one(method,train_rows,eval_rows,seed,epochs,save_path=None):
    ts=np.asarray([r[4] for r in train_rows],dtype=np.float64); mean=float(ts.mean()); std=float(ts.std())
    tc,td,ty=arrays(train_rows,mean,std); ec,ed,ey=arrays(eval_rows,mean,std)
    torch.manual_seed(seed); np.random.seed(seed); model=DLRM(method,mean,std); opt=torch.optim.Adam(model.parameters(),lr=LR,weight_decay=1e-6)
    snapshots={}; started=time.perf_counter(); updates=0
    for epoch in range(1,epochs+1):
        model.train(); g=torch.Generator().manual_seed(seed+epoch*101); order=torch.randperm(len(ty),generator=g)
        for s in range(0,len(order),BATCH):
            idx=order[s:s+BATCH]; opt.zero_grad(set_to_none=True); logits=model(tc[idx],td[idx]); loss=F.binary_cross_entropy_with_logits(logits,ty[idx]); loss.backward(); opt.step(); updates+=1
        if eval_rows is not None and epoch in EPOCH_CANDIDATES:
            m,preds=evaluate(model,eval_rows); snapshots[epoch]={'metrics':m,'predictions':preds}
    elapsed=time.perf_counter()-started
    if save_path:
        payload={'format':'ma1062-dlrm-v1','method':method,'timestamp_mean':mean,'timestamp_std':std,'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'embedding_dim':HIDDEN,'categories':CATS}
        torch.save(payload,save_path)
    return model,snapshots,{'wall_seconds':elapsed,'optimizer_updates':updates,'training_examples':len(train_rows)*epochs}

def run_dev():
    rows=read_base(); train,dev=split_train_dev(rows); details={}; chosen={}
    for method in METHODS:
        details[method]={}
        for seed in DEV_SEEDS:
            _,snap,compute=train_one(method,train,dev,seed,max(EPOCH_CANDIDATES))
            details[method][str(seed)]={'epochs':{str(e):v['metrics'] for e,v in snap.items()},'compute':compute}
        meanloss={e:np.mean([details[method][str(seed)]['epochs'][str(e)]['logloss'] for seed in DEV_SEEDS]) for e in EPOCH_CANDIDATES}
        chosen[method]=min(meanloss,key=meanloss.get)
    out={'experiment_id':'MA-1062','dataset_archive_sha256':sha(ARCHIVE),'base_rows':len(rows),'train_rows':len(train),'dev_rows':len(dev),'train_dev_pair_overlap':False,'methods':details,'selected_epochs':chosen,'audit_opened':False}
    RUNS.mkdir(parents=True,exist_ok=True); (RUNS/'development.json').write_text(json.dumps(out,indent=2)+'\n')
    (RUNS/'DEVELOPMENT_SELECTION.json').write_text(json.dumps({'experiment_id':'MA-1062','selected_epochs':chosen,'selection':'per-method minimum mean development logloss across two seeds from epoch candidates 3,5,8','audit_opened':False},indent=2)+'\n')
    print(json.dumps({'status':'development_complete','selected_epochs':chosen,'train_rows':len(train),'dev_rows':len(dev)},indent=2))

def load_model_from_payload(path):
    obj=torch.load(path,map_location='cpu',weights_only=False); model=DLRM(obj['method'],obj['timestamp_mean'],obj['timestamp_std']); model.load_state_dict(obj['state_dict']); model.eval(); return model,obj

def batch_latency(model,rows,repeats=100):
    cats,dense,_=arrays(rows[:1024],model.mean,model.std)
    def sample(fn):
        for _ in range(10): fn()
        vals=[]
        with torch.no_grad():
            for _ in range(repeats):
                t=time.perf_counter(); fn(); vals.append(time.perf_counter()-t)
        return {'p50_ms':float(np.median(vals)*1000),'p95_ms':float(np.quantile(vals,.95)*1000)}
    return {'lookup':sample(lambda:model.embed_fields(cats)),'end_to_end':sample(lambda:model(cats,dense))}

def run_audit():
    if not (RUNS/'DEVELOPMENT_SELECTION.json').exists(): raise ValueError('development selection missing')
    selection=json.loads((RUNS/'DEVELOPMENT_SELECTION.json').read_text())
    if selection['audit_opened'] is not False: raise ValueError('development selection invalid')
    audit=read_audit_after_freeze(); train=read_base()
    train_pairs={(r[0],r[1]) for r in train}; audit_pairs={(r[0],r[1]) for r in audit}
    if train_pairs & audit_pairs: raise ValueError('official audit pairs overlap the training partition')
    out={'experiment_id':'MA-1062','audit_rows':len(audit),'seeds':FRESH_SEEDS,'selected_epochs':selection['selected_epochs'],'methods':{},'payloads':{},'compute':{}}
    adir=RUNS/'audit'; adir.mkdir(parents=True,exist_ok=True)
    for method in METHODS:
        out['methods'][method]={}; out['payloads'][method]={}; out['compute'][method]={}
        for seed in FRESH_SEEDS:
            payload=adir/f'{method}_seed{seed}.pt'; started=time.perf_counter()
            _,_,comp=train_one(method,train,None,seed,int(selection['selected_epochs'][method]),payload)
            # Reopen the actual serialized state and use it as the canonical evaluator.
            model,obj=load_model_from_payload(payload); metrics,preds=evaluate(model,audit)
            # Keep official audit labels and row identifiers out of persisted artifacts.
            np.savez_compressed(adir/f'{method}_seed{seed}.npz',logits=preds)
            latency=batch_latency(model,audit)
            metadata={'experiment_id':'MA-1062','method':method,'seed':seed,'model':'compact DLRM screening implementation','timestamp_mean':obj['timestamp_mean'],'timestamp_std':obj['timestamp_std'],'selected_epochs':selection['selected_epochs'][method],'archive_sha256':P['dataset']['archive_sha256']}
            meta=adir/f'{method}_seed{seed}.json'; meta.write_text(json.dumps(metadata,indent=2)+'\n')
            actual=payload.stat().st_size+meta.stat().st_size
            out['methods'][method][str(seed)]={**metrics,'prediction_sha256':sha(adir/f'{method}_seed{seed}.npz'),'payload_sha256':sha(payload),'latency_ms':latency}
            out['payloads'][method][str(seed)]={'payload_file_bytes':payload.stat().st_size,'metadata_file_bytes':meta.stat().st_size,'complete_payload_bytes':actual}
            out['compute'][method][str(seed)]={**comp,'end_to_end_wall_seconds':time.perf_counter()-started,'categorical_embedding_values_per_example':4*HIDDEN,'approx_lookup_bytes_per_1024_batch':1024*4*HIDDEN*4}
    (adir/'results.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({'status':'audit_complete','results':{m:{s:{'logloss':v['logloss'],'auc':v['auc']} for s,v in r.items()} for m,r in out['methods'].items()}},indent=2))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--phase',choices=['dev','audit'],required=True); a=ap.parse_args()
    torch.set_num_threads(1)
    if a.phase=='dev': run_dev()
    else: run_audit()
if __name__=='__main__': main()
