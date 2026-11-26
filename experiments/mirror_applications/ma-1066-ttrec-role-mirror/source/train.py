#!/usr/bin/env python3
"""MA-1066 TT-Rec role-view experiment runner. Requires CUDA and vendored FBTT-Embedding."""
import argparse, csv, hashlib, json, math, os, time, urllib.request, zipfile
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, log_loss
from views import apply_givens

ROOT=Path(__file__).resolve().parents[1]
P=json.loads((ROOT/'PROTOCOL.json').read_text())
RUNS=ROOT/'runs'; ARCHIVE=Path(os.environ.get('MOVIELENS_ARCHIVE','/tmp/ma1066-ml-1m.zip'))
VENDOR=ROOT/'source'/'vendor'/'FBTT-Embedding'; import sys; sys.path.insert(0,str(VENDOR))

METHODS=['full_tables','independent_tt','shared_tt','shared_tt_diag','shared_tt_film','shared_tt_mirror']
NU=6040; NI=3952; N=NU+NI; D=32
DEV=P['development']['seeds']; FRESH=P['fresh']['seeds']; EPOCHS=P['model']['epoch_candidates']

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def h10(s): return int(hashlib.sha256(s.encode()).hexdigest(),16)%10
def excluded(u,i): return h10(f'{u}:{i}')==9 or (h10(f'{u}:{i}')!=9 and h10(f'MA-1064:{u}:{i}')==9)
def split_bucket(u,i): return h10(f'MA-1066:{u}:{i}')
def acquire_archive():
    if ARCHIVE.exists(): return
    ARCHIVE.parent.mkdir(parents=True,exist_ok=True)
    urllib.request.urlretrieve(P['dataset']['url'],ARCHIVE)
def read_rows(include_audit=False):
    acquire_archive()
    if sha(ARCHIVE)!=P['dataset']['archive_sha256'] or ARCHIVE.stat().st_size!=P['dataset']['archive_bytes']:
        raise ValueError('MovieLens archive SHA-256/size mismatch')
    tr=[]; dv=[]; au=[]; prior=0
    with zipfile.ZipFile(ARCHIVE) as z:
        for line in z.read('ml-1m/ratings.dat').decode('latin-1').splitlines():
            us,it,rt=line.split('::')[:3]; u=int(us); i=int(it)
            if excluded(u,i): prior+=1; continue
            b=split_bucket(u,i)
            if b==9 and not include_audit: continue
            (tr if b<8 else dv if b==8 else au).append((u,i,int(rt)))
    tr=np.asarray(tr,dtype=np.int64); dv=np.asarray(dv,dtype=np.int64); au=np.asarray(au,dtype=np.int64)
    pairs=lambda a:set((int(u),int(i)) for u,i,_ in a)
    assert not(pairs(tr)&pairs(dv) or pairs(tr)&pairs(au) or pairs(dv)&pairs(au))
    return tr,dv,au,prior

def make_tt(n,p_shapes,device):
    from tt_embeddings_ops import TTEmbeddingBag
    m=TTEmbeddingBag(num_embeddings=n,embedding_dim=D,tt_ranks=[16,16],tt_p_shapes=p_shapes,tt_q_shapes=[4,4,2],sparse=False,use_cache=False,weight_dist='approx-normal')
    return m.to(device)

class TTRecRoleModel(nn.Module):
    def __init__(self,method,device):
        super().__init__(); self.method=method
        if method=='full_tables': self.user=nn.Embedding(NU,D); self.item=nn.Embedding(NI,D)
        elif method=='independent_tt': self.user_tt=make_tt(NU,[19,18,18],device); self.item_tt=make_tt(NI,[16,16,16],device)
        else: self.shared_tt=make_tt(N,[22,22,22],device)
        if method in ('shared_tt_diag','shared_tt_mirror'): self.view=nn.Parameter(torch.zeros(2,16))
        elif method=='shared_tt_film': self.view=nn.Parameter(torch.zeros(2,64))
        else: self.register_parameter('view',None)
        self.head=nn.Linear(D,1)
    def lookup(self,table,ids):
        offsets=torch.arange(ids.numel(),device=ids.device,dtype=torch.long)
        return table(ids,offsets)
    def embed(self,u,i,role):
        ids=(u-1) if role==0 else (i-1)
        if self.method=='full_tables': return self.user(ids) if role==0 else self.item(ids)
        if self.method=='independent_tt': return self.lookup(self.user_tt if role==0 else self.item_tt,ids)
        gid=ids if role==0 else NU+ids
        x=self.lookup(self.shared_tt,gid)
        if self.method=='shared_tt_diag': x=x*(1+self.view[role].repeat_interleave(2))
        elif self.method=='shared_tt_film': x=x*(1+self.view[role,:32])+self.view[role,32:]
        elif self.method=='shared_tt_mirror': x=apply_givens(x,self.view[role])
        return x
    def forward(self,u,i): return self.head(self.embed(u,i,0)*self.embed(u,i,1)).squeeze(-1)

def metric(logits,y):
    prob=1/(1+np.exp(-np.clip(np.asarray(logits,dtype=np.float64),-60,60)))
    return {'auc':float(roc_auc_score(y,prob)),'logloss':float(log_loss(y,prob,labels=[0,1]))}
def tensors(rows,device):
    u=torch.as_tensor(rows[:,0].copy(),device=device,dtype=torch.long); i=torch.as_tensor(rows[:,1].copy(),device=device,dtype=torch.long); y=torch.as_tensor((rows[:,2]>=4).astype(np.float32),device=device)
    return u,i,y
def evaluate(model,rows,device):
    u,i,y=tensors(rows,device); model.eval(); out=[]
    with torch.no_grad():
        for s in range(0,len(y),4096): out.append(model(u[s:s+4096],i[s:s+4096]).float().cpu().numpy())
    logits=np.concatenate(out).astype(np.float32); return metric(logits,y.cpu().numpy()),logits
def train_one(method,train_rows,dev_rows,seed,epochs,device):
    torch.manual_seed(seed); np.random.seed(seed); model=TTRecRoleModel(method,device)
    u,i,y=tensors(train_rows,device); opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=1e-6); history={}; updates=0
    torch.cuda.synchronize(); start=time.perf_counter()
    for ep in range(1,epochs+1):
        model.train(); order=torch.randperm(len(y),device=device,generator=torch.Generator(device=device).manual_seed(seed+ep*313))
        for s in range(0,len(y),4096):
            ix=order[s:s+4096]; opt.zero_grad(set_to_none=True); logits=model(u[ix],i[ix]); loss=F.binary_cross_entropy_with_logits(logits,y[ix]); loss.backward(); opt.step(); updates+=1
        if dev_rows is not None: history[ep]=evaluate(model,dev_rows,device)[0]
    torch.cuda.synchronize(); elapsed=time.perf_counter()-start
    return model,history,{'wall_seconds':elapsed,'updates':updates,'training_examples':len(y)*epochs}
def collisions(): return {m:0 for m in METHODS}
def qps(model,rows,device):
    u,i,_=tensors(rows[:2048],device); n=min(1024,len(u)); vals=[]; model.eval()
    with torch.no_grad():
        for _ in range(15): model(u[:n],i[:n])
        torch.cuda.synchronize()
        for _ in range(60):
            torch.cuda.synchronize(); t=time.perf_counter(); model(u[:n],i[:n]); torch.cuda.synchronize(); vals.append(time.perf_counter()-t)
    return {'batch_size':n,'p50_ms':float(np.quantile(vals,.5)*1000),'p95_ms':float(np.quantile(vals,.95)*1000),'qps':float(n/np.mean(vals))}
def write_results_core(results):
    fields=['condition','world_or_seed','method','serialized_bytes','train_tokens_or_examples','optimizer_updates','active_compute_proxy','wall_time_s','primary_metric','primary_value','secondary_metric','secondary_value','status_note']
    rows=[]
    for method,seeds in results['methods'].items():
        for seed,v in seeds.items():
            rows.append({'condition':'fresh_gpu','world_or_seed':seed,'method':method,'serialized_bytes':v['payload_bytes'],'train_tokens_or_examples':v['compute']['training_examples'],'optimizer_updates':v['compute']['updates'],'wall_time_s':v['compute']['wall_seconds'],'primary_metric':'AUC','primary_value':v['metrics']['auc'],'secondary_metric':'logloss','secondary_value':v['metrics']['logloss'],'status_note':f"GPU={results['device']}; cache=disabled; QPS={v['throughput']['qps']}; p95_ms={v['throughput']['p95_ms']}; replay_max={v['max_replay_logit_diff']}"})
    with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n'); w.writeheader(); w.writerows(rows)
def run_dev():
    if not torch.cuda.is_available(): raise RuntimeError('MA-1066 requires a reviewed CUDA GPU profile; CPU is not an admissible QPS substitute')
    device=torch.device('cuda'); tr,dv,_,prior=read_rows(False); details={}; selected={}
    for method in METHODS:
        details[method]={}
        for seed in DEV:
            _,hist,comp=train_one(method,tr,dv,seed,max(EPOCHS),device); details[method][str(seed)]={'metrics':hist,'compute':comp}
        selected[method]=min(EPOCHS,key=lambda ep:np.mean([details[method][str(s)]['metrics'][ep]['logloss'] for s in DEV]))
    out={'experiment_id':'MA-1066','train_rows':len(tr),'dev_rows':len(dv),'excluded_previous_audit_rows':prior,'audit_metrics_opened':False,'device':torch.cuda.get_device_name(0),'seeds':DEV,'methods':details,'selected_epochs':selected,'collisions':collisions()}
    RUNS.mkdir(exist_ok=True); (RUNS/'development.json').write_text(json.dumps(out,indent=2)+'\n'); (RUNS/'DEVELOPMENT_SELECTION.json').write_text(json.dumps({'selected_epochs':selected,'audit_opened':False},indent=2)+'\n')
def load_freeze():
    f=json.loads((ROOT/'FRESH_FREEZE.json').read_text())
    for key,path in [('protocol_sha256',ROOT/'PROTOCOL.json'),('runner_sha256',ROOT/'source/train.py'),('selection_sha256',RUNS/'DEVELOPMENT_SELECTION.json'),('tt_ops_sha256',VENDOR/'tt_embeddings_ops.py'),('tt_cuda_sha256',VENDOR/'tt_embeddings_cuda.cu')]:
        if f[key]!=sha(path): raise ValueError(f'freeze mismatch: {key}')
    if f['audit_opened'] is not False: raise ValueError('freeze must precede audit access')
    if f['gpu_model']!=torch.cuda.get_device_name(0) or f['cuda_version']!=torch.version.cuda or f['torch_version']!=torch.__version__: raise ValueError('reviewed runtime mismatch')
    return f
def run_audit():
    if not torch.cuda.is_available(): raise RuntimeError('MA-1066 requires a reviewed CUDA GPU profile')
    freeze=load_freeze(); d=RUNS/'audit'; d.mkdir(parents=True,exist_ok=True); marker=d/'AUDIT_OPENED.json'
    if marker.exists() or (d/'results.json').exists(): raise RuntimeError('Audit access already started; do not rerun or overwrite')
    marker.write_text(json.dumps({'experiment_id':'MA-1066','audit_opened':True,'freeze_sha256':sha(ROOT/'FRESH_FREEZE.json')},sort_keys=True)+'\n')
    device=torch.device('cuda'); sel=json.loads((RUNS/'DEVELOPMENT_SELECTION.json').read_text()); tr,_,audit,prior=read_rows(True); out={}
    for method in METHODS:
        out[method]={}
        for seed in FRESH:
            model,_,comp=train_one(method,tr,None,seed,int(sel['selected_epochs'][method]),device)
            d=RUNS/'audit'; d.mkdir(parents=True,exist_ok=True); ck=d/f'{method}_{seed}.pt'; meta=d/f'{method}_{seed}.json'
            torch.save({'method':method,'state_dict':model.state_dict()},ck)
            meta.write_text(json.dumps({'experiment_id':'MA-1066','method':method,'seed':seed,'device':torch.cuda.get_device_name(0),'backend':'facebookresearch/FBTT-Embedding@b95947ccd32cf9ab851242cc606562f2a74d9efa','embedding_dim':D,'shared_tt':P['model']['tt_shared'],'independent_tt':P['model']['tt_independent'],'cache':False,'selected_epoch':int(sel['selected_epochs'][method])},sort_keys=True)+'\n')
            metrics,logits=evaluate(model,audit,device); replay=TTRecRoleModel(method,device); obj=torch.load(ck,map_location=device,weights_only=False); replay.load_state_dict(obj['state_dict']); _,replay_logits=evaluate(replay,audit,device)
            lp=d/f'{method}_{seed}.npz'; np.savez_compressed(lp,logits=logits)
            out[method][str(seed)]={'metrics':metrics,'payload_bytes':ck.stat().st_size+meta.stat().st_size,'checkpoint_bytes':ck.stat().st_size,'metadata_bytes':meta.stat().st_size,'checkpoint_sha256':sha(ck),'metadata_sha256':sha(meta),'logits_sha256':sha(lp),'logits_bytes':lp.stat().st_size,'max_replay_logit_diff':float(np.max(np.abs(logits-replay_logits))),'throughput':qps(replay,audit,device),'compute':comp,'selected_epoch':int(sel['selected_epochs'][method])}
    results={'experiment_id':'MA-1066','audit_rows':len(audit),'excluded_previous_audit_rows':prior,'audit_ids_or_labels_persisted':False,'device':torch.cuda.get_device_name(0),'methods':out}
    (RUNS/'audit').joinpath('results.json').write_text(json.dumps(results,indent=2)+'\n'); write_results_core(results)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--phase',choices=['dev','audit'],required=True); args=ap.parse_args()
    if args.phase=='dev': run_dev()
    else: run_audit()
