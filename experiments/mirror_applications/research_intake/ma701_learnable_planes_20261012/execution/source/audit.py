"""Post-training numerical replay and cache-aware inference benchmark."""
import argparse,copy,csv,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .experiment import load_model,make_world,metrics
from .planes import apply_planes

class Compiled(nn.Module):
    def __init__(self,trunk,up,head,feature_bias,bias):
        super().__init__();self.trunk=copy.deepcopy(trunk)
        for name,v in [('up',up),('head',head),('feature_bias',feature_bias),('bias',bias)]:self.register_buffer(name,v.detach().clone())
    def forward(self,x):
        h=self.trunk(x)
        z=F.gelu(torch.einsum('bi,kdi->bkd',h,self.up)+self.feature_bias)
        return torch.einsum('bkd,kod->bko',z,self.head)+self.bias

class CompiledShared(nn.Module):
    """Native common FFN executes once; only the cheap readouts differ."""
    def __init__(self,model):
        super().__init__();self.trunk=copy.deepcopy(model.trunk)
        self.register_buffer('up',model.up.detach().clone())
        h=model.task_head if model.method=='multihead' else model.head.unsqueeze(0)
        self.register_buffer('head',(h@model.down).detach().clone())
        self.register_buffer('bias',model.bias.detach().clone())
    def forward(self,x):
        z=F.gelu(self.trunk(x)@self.up.T)
        return torch.einsum('bd,kod->bko',z,self.head)+self.bias

@torch.no_grad()
def compile_model(model):
    if model.method in ('shared','multihead'):return CompiledShared(model).eval()
    k=model.k;up=model.up.unsqueeze(0).repeat(k,1,1)
    fb=torch.zeros(k,32)
    if model.method in ('learned','fixed'):
        eye=torch.eye(32).unsqueeze(1).expand(-1,k,-1)
        q=apply_planes(eye,model.a,model.b,model.code.unsqueeze(0)).permute(1,2,0)
        up=q@up
    elif model.method=='native4':up=up*(1+model.code@model.basis).unsqueeze(-1)
    elif model.method=='film':
        up=up*(1+model.affine[:,0]).unsqueeze(-1);fb=model.affine[:,1]
    elif model.method=='lora2':up=up+model.lo_b@model.lo_a
    elif model.method=='householder':
        for j in range(2):
            n=F.normalize(model.normals[:,j],dim=-1)
            up=up-2*n.unsqueeze(-1)*(n.unsqueeze(1)@up)
    elif model.method=='private_ffn':up=up+model.delta_up
    head=model.task_head if model.method in ('multihead','private_ffn') else model.head.unsqueeze(0).expand(k,-1,-1)
    return Compiled(model.trunk,up,head@model.down,fb,model.bias).eval()

@torch.no_grad()
def benchmark(model,x):
    t=time.perf_counter();compiled=compile_model(model);construction=time.perf_counter()-t
    out=[]
    err=float((model(x)-compiled(x)).abs().max())
    for batch in [1,64]:
        sample=x[:batch]
        reps={'eager':[],'folded':[]}
        for fn in [model,compiled]:
            for _ in range(15):fn(sample)
        for rep in range(7):
            order=[('eager',model),('folded',compiled)]
            if rep%2:order.reverse()
            for name,fn in order:
                t=time.perf_counter()
                for _ in range(10):fn(sample)
                reps[name].append((time.perf_counter()-t)*1000/10)
        for name,values in reps.items():
            out.append(dict(mode=name,batch=batch,median_ms=float(np.median(values)),min_ms=min(values),max_ms=max(values),repetitions_ms=values))
    return dict(timing=out,fold_error=err,construction_ms=construction*1000,
                raw_state_bytes=sum(v.numel()*v.element_size() for v in model.state_dict().values()),
                folded_state_bytes=sum(v.numel()*v.element_size() for v in compiled.state_dict().values()))

def read_hash_lock(path):
    lock=json.loads(Path(path).read_text())
    return lock.get('hashes',lock)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('runs'));ap.add_argument('--benchmark',action='store_true')
    args=ap.parse_args();torch.set_num_threads(1);torch.set_num_interop_threads(1)
    checks=[];timing=[];flat=[];maxerr=0.;scalars=0
    lock=read_hash_lock('PRE_RUN_LOCK.json')
    for name,h in lock.items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h,('source changed',name)
    for p in sorted(args.root.rglob('metric.json')):
        r=json.loads(p.read_text());artifact=p.parent/'model.npz'
        assert artifact.stat().st_size==r['payload']['bytes']
        assert hashlib.sha256(artifact.read_bytes()).hexdigest()==r['payload']['sha256']
        m=load_model(artifact);w=make_world(r['seed'],r['family'])
        for split in ['val','test']:
            if r[split] is None:continue
            measured=metrics(m,w,split)
            for key in ['loss','worst','accuracy']:
                if r[split][key] is not None:
                    delta=abs(measured[key]-r[split][key]);maxerr=max(maxerr,delta);scalars+=1
                    assert delta<=1e-6
            for a,b in zip(measured['per_task'],r[split]['per_task']):maxerr=max(maxerr,abs(a-b));scalars+=1;assert abs(a-b)<=1e-6
        checks.append(str(p))
        row=dict(stage=r['stage'],family=r['family'],seed=r['seed'],method=r['label'],steps=r['steps'],bytes=r['payload']['bytes'],learned_scalars=r['learned_scalars'],train_seconds=r['train_seconds'],lr=r['cfg']['lr'],axis_lr=r['cfg']['axis_lr'],warmup=r['cfg']['warmup'],sigma=r['cfg']['sigma'])
        for split in ['val','test']:
            for key in ['loss','worst','accuracy']:row[split+'_'+key]=None if r[split] is None else r[split][key]
        flat.append(row)
        if args.benchmark and r['stage']!='calibration' and r['steps']==800:
            bm=benchmark(m,w['x_test'][:64]);bm.update(family=r['family'],seed=r['seed'],method=r['label'],steps=r['steps']);timing.append(bm)
    out=args.root.parent
    with open(out/'RESULTS_CORE.csv','w') as f:
        writer=csv.DictWriter(f,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    report=dict(payloads=len(checks),replayed_scalars=scalars,max_abs_error=maxerr,source_lock_verified=True,fresh_seeds_opened=[],timing_cases=len(timing))
    (out/'VERIFICATION_NUMERIC.json').write_text(json.dumps(report,indent=2))
    (out/'TIMING.json').write_text(json.dumps(timing,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
