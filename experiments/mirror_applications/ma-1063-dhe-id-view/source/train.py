#!/usr/bin/env python3
"""Deterministic CPU screen for MA-1063. Audit is opened only after FRESH_FREEZE."""
import argparse, hashlib, json, math, time, zipfile
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1]; RUNS=ROOT/'runs'; ARCHIVE=Path('/tmp/ma1063-ml-1m.zip')
P=json.loads((ROOT/'PROTOCOL.json').read_text()); M=P['model']; K=M['dense_hash']['k']; DIM=M['embedding_dim']
METHODS=['full_tables','dhe_shared','dhe_diag','dhe_film','dhe_mirror','dhe_independent']

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify_archive():
    if sha(ARCHIVE)!=P['dataset']['archive_sha256'] or ARCHIVE.stat().st_size!=P['dataset']['archive_bytes']: raise ValueError('archive hash/size mismatch')
def raw_lines(member):
    verify_archive()
    with zipfile.ZipFile(ARCHIVE) as z:
        return z.read(member).decode('latin-1').splitlines()
def bucket(u,i): return int(hashlib.sha256(f'{u}:{i}'.encode()).hexdigest(),16)%10
def split(include_audit=False):
    train=[]; dev=[]; audit=[]
    for line in raw_lines('ml-1m/ratings.dat'):
        us,ins,rs=line.split('::')[:3]; u=int(us); i=int(ins); b=bucket(u,i)
        if b==9 and not include_audit: continue
        row=(u,i,int(rs))
        (train if b<8 else dev if b==8 else audit).append(row)
    train=np.asarray(train,dtype=np.int64); dev=np.asarray(dev,dtype=np.int64); audit=np.asarray(audit,dtype=np.int64)
    pairs=lambda x:set((int(u),int(i)) for u,i,_ in x)
    assert not(pairs(train)&pairs(dev) or pairs(train)&pairs(audit) or pairs(dev)&pairs(audit))
    return train,dev,audit

def dense_hash(ids,salt):
    # PA338 universal hashing, fixed seed 0; values are exact float32 in [-1,1].
    rng=np.random.default_rng(0+salt); p=int(M['dense_hash']['prime']); m=int(M['dense_hash']['m'])
    aa=rng.integers(1,p,size=K,dtype=np.int64); bb=rng.integers(0,p,size=K,dtype=np.int64)
    idx=((ids[:,None].astype(np.int64)*aa+bb)%p)%m
    vals=np.random.default_rng(0).uniform(-1,1,size=m).astype(np.float32)
    return vals[idx]
class Decoder(nn.Module):
    def __init__(self):
        super().__init__(); blocks=[]; d=K
        for _ in range(4): blocks += [nn.Linear(d,64),nn.BatchNorm1d(64),nn.Mish()]; d=64
        blocks += [nn.Linear(64,DIM)]; self.net=nn.Sequential(*blocks)
    def forward(self,x): return self.net(x)
class Model(nn.Module):
    def __init__(self,method,n_user,n_item):
        super().__init__(); self.method=method
        if method=='full_tables': self.u=nn.Embedding(n_user+1,DIM); self.i=nn.Embedding(n_item+1,DIM); self.du=self.di=None
        elif method=='dhe_independent': self.du=Decoder(); self.di=Decoder(); self.u=self.i=None
        else: self.du=Decoder(); self.di=None; self.u=self.i=None
        if method in ('dhe_mirror','dhe_diag'): self.view=nn.Parameter(torch.zeros(2,DIM//2))
        elif method=='dhe_film': self.view=nn.Parameter(torch.zeros(2,2*DIM))
        else: self.register_parameter('view',None)
        self.head=nn.Linear(DIM,1,bias=True)
        if self.u is not None: nn.init.normal_(self.u.weight,std=.02); nn.init.normal_(self.i.weight,std=.02)
    def embed(self,ids,role):
        if self.method=='full_tables': return (self.u if role==0 else self.i)(ids)
        enc=F.one_hot(ids*0,num_classes=1).float() if False else None
        # Dense hash arrays are registered at run time as paid deterministic-free metadata.
        x=self.hash_u[ids] if role==0 else self.hash_i[ids]
        y=(self.du if role==0 or self.method!='dhe_independent' else self.di)(x)
        if self.method=='dhe_mirror':
            p=y.reshape(-1,DIM//2,2); a=self.view[role]; c=torch.cos(a); s=torch.sin(a)
            y=torch.stack((c*p[:,:,0]-s*p[:,:,1],s*p[:,:,0]+c*p[:,:,1]),-1).reshape(-1,DIM)
        elif self.method=='dhe_diag': y=y*(1+self.view[role].repeat_interleave(2))
        elif self.method=='dhe_film': y=y*(1+self.view[role,:DIM])+self.view[role,DIM:]
        return y
    def forward(self,u,i):
        z=self.embed(u,0)*self.embed(i,1); return self.head(z).squeeze(-1)

def tensors(split_rows,maxu,maxi):
    u=split_rows[:,0].astype(np.int64); i=split_rows[:,1].astype(np.int64); y=(split_rows[:,2]>=4).astype(np.float32)
    hu=np.zeros((maxu+1,K),np.float32); hi=np.zeros((maxi+1,K),np.float32)
    uu=np.arange(1,maxu+1); ii=np.arange(1,maxi+1); hu[uu]=dense_hash(uu,int(M['dense_hash']['field_salts'][0])); hi[ii]=dense_hash(ii,int(M['dense_hash']['field_salts'][1]))
    return tuple(torch.from_numpy(x) for x in (u,i,y,hu,hi))
def metric(logits,y):
    from sklearn.metrics import roc_auc_score, log_loss
    p=1/(1+np.exp(-np.clip(logits,-60,60))); return {'logloss':float(log_loss(y,p,labels=[0,1])),'auc':float(roc_auc_score(y,p))}
def eval_model(model,rows,maxt):
    u,i,y,hu,hi=tensors(rows,maxt[0],maxt[1]); model.hash_u=hu; model.hash_i=hi; model.eval(); out=[]
    with torch.no_grad():
        for s in range(0,len(y),4096): out.extend(model(u[s:s+4096],i[s:s+4096]).numpy().tolist())
    return metric(np.asarray(out),y.numpy()),np.asarray(out,dtype=np.float32)
def latency(model, rows, maxt):
    u,i,_,hu,hi=tensors(rows[:4096],maxt[0],maxt[1]); model.hash_u=hu; model.hash_i=hi; model.eval(); batch=min(1024,len(u)); vals=[]
    with torch.no_grad():
        for _ in range(8): model(u[:batch],i[:batch])
        for _ in range(40):
            t=time.perf_counter(); model(u[:batch],i[:batch]); vals.append(time.perf_counter()-t)
    return {'batch_size':batch,'p50_ms':float(np.quantile(vals,.5)*1000),'p95_ms':float(np.quantile(vals,.95)*1000),'qps':float(batch/np.mean(vals))}
def train_one(method,tr,ev,seed,epochs):
    maxu=6040; maxi=3952; torch.manual_seed(seed); np.random.seed(seed)
    model=Model(method,maxu,maxi); tu,ti,ty,hu,hi=tensors(tr,maxu,maxi); model.hash_u=hu; model.hash_i=hi
    opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=1e-6); hist={}; start=time.perf_counter()
    for ep in range(1,epochs+1):
        model.train(); g=torch.Generator().manual_seed(seed+ep*313); order=torch.randperm(len(ty),generator=g)
        for s in range(0,len(ty),4096):
            ix=order[s:s+4096]; opt.zero_grad(set_to_none=True); pred=model(tu[ix],ti[ix]); loss=F.binary_cross_entropy_with_logits(pred,ty[ix]); loss.backward(); opt.step()
        if ev is not None:
            hist[ep]=eval_model(model,ev,(maxu,maxi))[0]
    return model,hist,{'wall_seconds':time.perf_counter()-start,'updates':math.ceil(len(tr)/4096)*epochs}
def run_dev():
    tr,dev,_=split(); seeds=P['development']['seeds']; out={}; chosen={}
    for method in METHODS:
        out[method]={}
        for seed in seeds:
            _,hist,comp=train_one(method,tr,dev,seed,max(P['model']['epoch_candidates']))
            out[method][str(seed)]={'metrics':hist,'compute':comp}
        chosen[method]=min(P['model']['epoch_candidates'],key=lambda e:np.mean([out[method][str(s)]['metrics'][e]['logloss'] for s in seeds]))
    RUNS.mkdir(exist_ok=True); (RUNS/'development.json').write_text(json.dumps({'train_rows':len(tr),'dev_rows':len(dev),'seeds':seeds,'methods':out,'selected_epochs':chosen,'audit_opened':False},indent=2)+'\n')
    (RUNS/'DEVELOPMENT_SELECTION.json').write_text(json.dumps({'selected_epochs':chosen,'audit_opened':False},indent=2)+'\n'); print(json.dumps(chosen))
def load_freeze():
    f=json.loads((ROOT/'FRESH_FREEZE.json').read_text())
    if f['protocol_sha256']!=sha(ROOT/'PROTOCOL.json') or f['runner_sha256']!=sha(ROOT/'source/train.py'): raise ValueError('freeze hash mismatch')
    return f
def run_audit():
    f=load_freeze(); sel=json.loads((RUNS/'DEVELOPMENT_SELECTION.json').read_text()); tr,_,audit=split(include_audit=True); maxu=6040; maxi=3952; out={}; d=RUNS/'audit'; d.mkdir(parents=True,exist_ok=True)
    freq=np.bincount(tr[:,1],minlength=maxi+1)
    def slices(pred):
        y=(audit[:,2]>=4).astype(np.float32); f=freq[audit[:,1]]; result={}
        for name,mask in [('item_freq_le5',f<=5),('item_freq_6_20',(f>=6)&(f<=20)),('item_freq_gt20',f>20)]:
            if int(mask.sum())>=100: result[name]={'n':int(mask.sum()),**metric(pred[mask],y[mask])}
        return result
    for method in METHODS:
        out[method]={}
        for seed in P['fresh']['seeds']:
            model,_,comp=train_one(method,tr,None,seed,int(sel['selected_epochs'][method]));
            # Record actual model state and compact reconstruction metadata; hash input is deterministic code, not an array payload.
            ck=d/f'{method}_{seed}.pt'; torch.save({'method':method,'state_dict':model.state_dict(),'maxu':maxu,'maxi':maxi},ck)
            meta={'experiment_id':'MA-1063','method':method,'seed':seed,'embedding_dim':DIM,'hash':M['dense_hash'],'decoder':M['decoder'],'head':M['recommendation_head'],'selected_epochs':sel['selected_epochs'][method]}
            mp=d/f'{method}_{seed}.json'; mp.write_text(json.dumps(meta,sort_keys=True,separators=(',',':'))+'\n')
            replay=Model(method,maxu,maxi); replay.load_state_dict(torch.load(ck,map_location='cpu',weights_only=False)['state_dict']); replay.hash_u=model.hash_u; replay.hash_i=model.hash_i
            m,pred=eval_model(replay,audit,(maxu,maxi)); np.savez_compressed(d/f'{method}_{seed}.npz',logits=pred)
            out[method][str(seed)]={'metrics':m,'train_only_item_frequency_slices':slices(pred),'checkpoint_bytes':ck.stat().st_size,'metadata_bytes':mp.stat().st_size,'complete_payload_bytes':ck.stat().st_size+mp.stat().st_size,'checkpoint_sha256':sha(ck),'metadata_sha256':sha(mp),'logits_sha256':sha(d/f'{method}_{seed}.npz'),'latency':latency(replay,audit,(maxu,maxi)),'compute':comp}
    (d/'results.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--phase',choices=['dev','audit'],required=True); p=ap.parse_args(); torch.set_num_threads(1)
    run_dev() if p.phase=='dev' else run_audit()
