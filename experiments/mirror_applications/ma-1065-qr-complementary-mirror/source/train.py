#!/usr/bin/env python3
"""MA-1065 QR complementary embedding + role Mirror CPU screen."""
import argparse, hashlib, json, math, time, zipfile
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, log_loss

ROOT=Path(__file__).resolve().parents[1]; RUNS=ROOT/'runs'; ARCHIVE=Path('/tmp/ma1063-ml-1m.zip')
P=json.loads((ROOT/'PROTOCOL.json').read_text()); METHODS=['full_tables','independent_qr','shared_qr','shared_qr_diag','shared_qr_film','shared_qr_mirror','shared_hash_200']
DEV=P['development']['seeds']; FRESH=P['fresh']['seeds']; EPOCHS=P['model']['epoch_candidates']; D=32; NU=6040; NI=3952; N=NU+NI
BASE1063=ROOT.parent/'ma-1063-dhe-id-view/source/train.py'; BASE1064=ROOT.parent/'ma-1064-dhe-rare-id-fallback/source/train.py'

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def h10(s): return int(hashlib.sha256(s.encode()).hexdigest(),16)%10
def exclude_pair(u,i): return h10(f'{u}:{i}')==9 or (h10(f'{u}:{i}')!=9 and h10(f'MA-1064:{u}:{i}')==9)
def new_bucket(u,i): return h10(f'MA-1065:{u}:{i}')
def read_rows(include_audit=False):
    if sha(ARCHIVE)!=P['dataset']['archive_sha256'] or ARCHIVE.stat().st_size!=P['dataset']['archive_bytes']: raise ValueError('MovieLens archive hash/size mismatch')
    tr=[]; dv=[]; au=[]; excluded=0
    with zipfile.ZipFile(ARCHIVE) as z:
        for line in z.read('ml-1m/ratings.dat').decode('latin-1').splitlines():
            us,is_,rs=line.split('::')[:3]; u=int(us); i=int(is_)
            if exclude_pair(u,i): excluded+=1; continue
            b=new_bucket(u,i)
            if b==9 and not include_audit: continue
            row=(u,i,int(rs)); (tr if b<8 else dv if b==8 else au).append(row)
    tr=np.asarray(tr,dtype=np.int64); dv=np.asarray(dv,dtype=np.int64); au=np.asarray(au,dtype=np.int64)
    pairs=lambda x:set((int(u),int(i)) for u,i,_ in x)
    assert not(pairs(tr)&pairs(dv) or pairs(tr)&pairs(au) or pairs(dv)&pairs(au))
    return tr,dv,au,excluded

def global_ids(u,i): return u-1, NU+(i-1)
class QRModel(nn.Module):
    def __init__(self,method):
        super().__init__(); self.method=method
        if method=='full_tables': self.user=nn.Embedding(NU,D); self.item=nn.Embedding(NI,D)
        elif method=='independent_qr':
            self.uq=nn.Embedding(77,D); self.ur=nn.Embedding(79,D); self.iq=nn.Embedding(63,D); self.ir=nn.Embedding(63,D)
        elif method=='shared_hash_200': self.hash=nn.Embedding(200,D)
        else:
            self.sq=nn.Embedding(99,D); self.sr=nn.Embedding(101,D)
            if method in ('shared_qr_diag','shared_qr_mirror'): self.view=nn.Parameter(torch.zeros(2,16))
            elif method=='shared_qr_film': self.view=nn.Parameter(torch.zeros(2,64))
            else: self.register_parameter('view',None)
        self.head=nn.Linear(D,1)
        for name in ('user','item','uq','ur','iq','ir','hash','sq','sr'):
            emb=getattr(self,name,None)
            if emb is not None: nn.init.normal_(emb.weight,std=.1 if name not in ('user','item') else .02)
    def embed_role(self,u,i,role):
        if self.method=='full_tables': return (self.user(u-1) if role==0 else self.item(i-1))
        if self.method=='independent_qr':
            ids=(u-1) if role==0 else (i-1); m=79 if role==0 else 63
            q=ids//m; r=ids%m
            return (self.uq(q)*self.ur(r)) if role==0 else (self.iq(q)*self.ir(r))
        gu,gi=global_ids(u,i); g=gu if role==0 else gi
        if self.method=='shared_hash_200':
            idx=torch.remainder(g*2654435761+2246822519,2**32)%200
            return self.hash(idx)
        x=self.sq(g//101)*self.sr(g%101)
        if self.method=='shared_qr_mirror':
            a=self.view[role]; c=torch.cos(a); s=torch.sin(a); p=x.reshape(-1,16,2)
            x=torch.stack((c*p[:,:,0]-s*p[:,:,1],s*p[:,:,0]+c*p[:,:,1]),dim=-1).reshape(-1,D)
        elif self.method=='shared_qr_diag': x=x*(1+self.view[role].repeat_interleave(2))
        elif self.method=='shared_qr_film': x=x*(1+self.view[role,:32])+self.view[role,32:]
        return x
    def forward(self,u,i): return self.head(self.embed_role(u,i,0)*self.embed_role(u,i,1)).squeeze(-1)

def metric(pred,y):
    p=1/(1+np.exp(-np.clip(np.asarray(pred,dtype=np.float64),-60,60)))
    return {'auc':float(roc_auc_score(y,p)),'logloss':float(log_loss(y,p,labels=[0,1]))}
def evaluate(model,rows):
    u=torch.from_numpy(rows[:,0].astype(np.int64)); i=torch.from_numpy(rows[:,1].astype(np.int64)); y=(rows[:,2]>=4).astype(np.float32); model.eval(); out=[]
    with torch.no_grad():
        for s in range(0,len(y),4096): out.extend(model(u[s:s+4096],i[s:s+4096]).cpu().numpy().tolist())
    pred=np.asarray(out,dtype=np.float32); return metric(pred,y),pred
def train_one(method,tr,dev,seed,epochs):
    torch.manual_seed(seed); np.random.seed(seed); model=QRModel(method); u=torch.from_numpy(tr[:,0].astype(np.int64)); i=torch.from_numpy(tr[:,1].astype(np.int64)); y=torch.from_numpy((tr[:,2]>=4).astype(np.float32)); opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=1e-6); hist={}; updates=0; start=time.perf_counter()
    for ep in range(1,epochs+1):
        model.train(); order=torch.randperm(len(y),generator=torch.Generator().manual_seed(seed+ep*313))
        for s in range(0,len(y),4096):
            ix=order[s:s+4096]; opt.zero_grad(set_to_none=True); logits=model(u[ix],i[ix]); loss=F.binary_cross_entropy_with_logits(logits,y[ix]); loss.backward(); opt.step(); updates+=1
        if dev is not None: hist[ep]=evaluate(model,dev)[0]
    return model,hist,{'wall_seconds':time.perf_counter()-start,'updates':updates,'training_examples':len(tr)*epochs}
def collisions():
    ids=np.arange(N,dtype=np.int64); h=((ids*2654435761+2246822519)%(2**32))%200
    return {'full_tables':0,'independent_qr':0,'shared_qr':0,'shared_qr_diag':0,'shared_qr_film':0,'shared_qr_mirror':0,'shared_hash_200':int(N-len(np.unique(h)))}
def latency(model,rows):
    u=torch.from_numpy(rows[:4096,0].astype(np.int64)); i=torch.from_numpy(rows[:4096,1].astype(np.int64)); n=min(1024,len(rows)); vals=[]; model.eval()
    with torch.no_grad():
        for _ in range(8): model(u[:n],i[:n])
        for _ in range(40):
            t=time.perf_counter(); model(u[:n],i[:n]); vals.append(time.perf_counter()-t)
    return {'batch_size':n,'p50_ms':float(np.quantile(vals,.5)*1000),'p95_ms':float(np.quantile(vals,.95)*1000),'qps':float(n/np.mean(vals))}
def run_dev():
    RUNS.mkdir(parents=True,exist_ok=True); tr,dv,_,excluded=read_rows(False); details={}; epochs={}
    for m in METHODS:
        details[m]={}
        for seed in DEV:
            _,hist,comp=train_one(m,tr,dv,seed,max(EPOCHS)); details[m][str(seed)]={'metrics':hist,'compute':comp}
        epochs[m]=min(EPOCHS,key=lambda e:np.mean([details[m][str(s)]['metrics'][e]['logloss'] for s in DEV]))
    out={'experiment_id':'MA-1065','train_rows':len(tr),'dev_rows':len(dv),'excluded_previous_audit_rows':excluded,'previous_audit_metrics_opened':False,'seeds':DEV,'method_details':details,'selected_epochs':epochs,'collisions':collisions()}
    (RUNS/'development.json').write_text(json.dumps(out,indent=2)+'\n')
    (RUNS/'DEVELOPMENT_SELECTION.json').write_text(json.dumps({'experiment_id':'MA-1065','selected_epochs':epochs,'collision_counts':collisions(),'audit_opened':False},indent=2)+'\n')
    print(json.dumps({'status':'development_complete','train_rows':len(tr),'dev_rows':len(dv),'excluded_previous_audits':excluded,'selected_epochs':epochs},indent=2))
def load_freeze():
    f=json.loads((ROOT/'FRESH_FREEZE.json').read_text())
    for k,path in [('protocol_sha256',ROOT/'PROTOCOL.json'),('runner_sha256',ROOT/'source/train.py'),('selection_sha256',RUNS/'DEVELOPMENT_SELECTION.json')]:
        if f[k]!=sha(path): raise ValueError(f'freeze mismatch: {k}')
    return f
def run_audit():
    load_freeze(); sel=json.loads((RUNS/'DEVELOPMENT_SELECTION.json').read_text()); tr,_,audit,excluded=read_rows(True); d=RUNS/'audit'; d.mkdir(parents=True,exist_ok=True); results={}
    for method in METHODS:
        results[method]={}
        for seed in FRESH:
            ep=int(sel['selected_epochs'][method]); model,_,comp=train_one(method,tr,None,seed,ep); ck=d/f'{method}_{seed}.pt'; torch.save({'method':method,'state_dict':model.state_dict()},ck)
            meta={'experiment_id':'MA-1065','method':method,'seed':seed,'embedding_dim':D,'shared_qr':P['model']['shared_qr'],'independent_qr':P['model']['independent_qr'],'hash_control':P['model']['hash_control'],'head':P['model']['head'],'selected_epoch':ep,'collision_count':collisions()[method]}
            mp=d/f'{method}_{seed}.json'; mp.write_text(json.dumps(meta,sort_keys=True,separators=(',',':'))+'\n')
            obj=torch.load(ck,map_location='cpu',weights_only=False); replay=QRModel(method); replay.load_state_dict(obj['state_dict']); metrics,pred=evaluate(replay,audit); lp=d/f'{method}_{seed}.npz'; np.savez_compressed(lp,logits=pred)
            results[method][str(seed)]={'metrics':metrics,'payload_bytes':ck.stat().st_size+mp.stat().st_size,'checkpoint_bytes':ck.stat().st_size,'metadata_bytes':mp.stat().st_size,'checkpoint_sha256':sha(ck),'metadata_sha256':sha(mp),'logits_sha256':sha(lp),'logits_bytes':lp.stat().st_size,'collision_count':collisions()[method],'latency':latency(replay,audit),'compute':comp,'selected_epoch':ep}
    out={'experiment_id':'MA-1065','audit_rows':len(audit),'excluded_previous_audit_rows':excluded,'prior_audit_pairs_in_current_split':False,'methods':results,'audit_labels_or_ids_persisted':False,'collisions':collisions()}
    (d/'results.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps({'status':'audit_complete','rows':len(audit),'metrics':{m:{s:v['metrics'] for s,v in z.items()} for m,z in results.items()}},indent=2))
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--phase',choices=['dev','audit'],required=True); a=ap.parse_args(); torch.set_num_threads(1)
    run_dev() if a.phase=='dev' else run_audit()
