#!/usr/bin/env python3
"""MA-603 synthetic dynamic-weight FFN mechanism screen."""
import argparse, hashlib, io, json, math, platform, time
from pathlib import Path
import torch
from torch import nn

D, H, O, K = 12, 8, 8, 4
METHODS = ("static", "condconv", "mirror", "mlp_gate", "oracle")

class Teacher(nn.Module):
    def __init__(self, seed):
        super().__init__(); g=torch.Generator().manual_seed(seed)
        self.b1=nn.Parameter(torch.randn(K,H,D,generator=g)*.24,requires_grad=False)
        self.c1=nn.Parameter(torch.randn(K,H,generator=g)*.08,requires_grad=False)
        self.b2=nn.Parameter(torch.randn(K,O,H,generator=g)*.28,requires_grad=False)
        self.c2=nn.Parameter(torch.randn(K,O,generator=g)*.08,requires_grad=False)
        self.g1=nn.Parameter(torch.randn(2,D,generator=g)*.45,requires_grad=False)
        self.g2=nn.Parameter(torch.randn(K,2,generator=g)*.45,requires_grad=False)
    def coeff(self,x): return torch.softmax(torch.tanh(x@self.g1.T)@self.g2.T,dim=-1)
    def forward(self,x): return mix_ffn(x,self.coeff(x),self.b1,self.c1,self.b2,self.c2)

def mix_ffn(x,c,w1,b1,w2,b2):
    a=torch.einsum('nk,khd->nhd',c,w1)
    ab=torch.einsum('nk,kh->nh',c,b1)
    h=torch.tanh(torch.einsum('nd,nhd->nh',x,a)+ab)
    # output weights synthesized before evaluating second FFN layer
    zb=torch.einsum('nk,ko->no',c,b2)
    z=torch.einsum('nk,koh->noh',c,w2)
    return torch.einsum('nh,noh->no',h,z)+zb

class Student(nn.Module):
    def __init__(self, method, seed, teacher=None):
        super().__init__(); torch.manual_seed(seed); self.method=method
        g=torch.Generator().manual_seed(seed)
        if method=='oracle':
            self.teacher=teacher; return
        self.w1=nn.Parameter(torch.randn(K,H,D,generator=g)*.12); self.b1=nn.Parameter(torch.zeros(K,H))
        self.w2=nn.Parameter(torch.randn(K,O,H,generator=g)*.12); self.b2=nn.Parameter(torch.zeros(K,O))
        if method=='static':
            self.raw=nn.Parameter(torch.zeros(1,K))
        elif method in ('condconv','mirror'):
            self.router=nn.Linear(D,K)
            if method=='mirror': self.angle=nn.Linear(D,2)
        elif method=='mlp_gate':
            self.router=nn.Sequential(nn.Linear(D,2),nn.Tanh(),nn.Linear(2,K))
    def coeff(self,x):
        if self.method=='static': return self.raw.softmax(-1).expand(x.shape[0],-1)
        c=torch.softmax(self.router(x),dim=-1)
        if self.method=='mirror':
            ang=torch.tanh(self.angle(x))*0.9
            # fixed pair schedule: rotate coefficient pairs (0,1) and (2,3)
            out=c.clone()
            for theta,i,j in ((ang[:,0],0,1),(ang[:,1],2,3)):
                cs, sn=theta.cos(),theta.sin()
                out[:,i]=cs*c[:,i]-sn*c[:,j]; out[:,j]=sn*c[:,i]+cs*c[:,j]
            c=out
        return c
    def forward(self,x):
        if self.method=='oracle': return self.teacher(x)
        return mix_ffn(x,self.coeff(x),self.w1,self.b1,self.w2,self.b2)

def make_data(seed, teacher, n):
    g=torch.Generator().manual_seed(seed)
    x=torch.randn(n,D,generator=g)
    with torch.no_grad(): y=teacher(x)
    return x,y

def bytes_and_hash(model,method,seed):
    state={k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()}
    bio=io.BytesIO(); torch.save({'state':state,'method':method,'dims':[D,H,O,K],'seed':seed,'format':'MA603-v1'},bio)
    blob=bio.getvalue()
    return blob,hashlib.sha256(blob).hexdigest()

def flop_proxy(method):
    # multiply-add proxy/input. Counts router plus coefficient-weighted synthesis and two FFN layers.
    router={'static':0,'condconv':2*D*K,'mirror':2*D*K+2*D*2+12,'mlp_gate':2*D*2+2*2*K,'oracle':2*D*2+2*2*K}[method]
    synth=K*(H*D+H+O*H+O)
    ffn=2*(H*D+O*H)
    return router+synth+ffn

def fit(method, world, teacher, train, dev, steps):
    torch.manual_seed(world+1700); torch.set_num_threads(1)
    model=Student(method,world+4000,teacher if method=='oracle' else None)
    if method=='oracle':
        with torch.no_grad(): loss=((model(dev[0])-dev[1])**2).mean().item()
        return model,{'dev_mse':loss,'seconds':0.0,'updates':0,'train_mse':None}
    opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=0)
    x,y=train; start=time.perf_counter(); n=x.shape[0]; g=torch.Generator().manual_seed(world+8000)
    for step in range(steps):
        ix=torch.randint(n,(128,),generator=g); pred=model(x[ix]); loss=((pred-y[ix])**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    sec=time.perf_counter()-start
    with torch.no_grad():
        tr=((model(x)-y)**2).mean().item(); dv=((model(dev[0])-dev[1])**2).mean().item()
    return model,{'dev_mse':dv,'train_mse':tr,'seconds':sec,'updates':steps}

def run_world(world,steps):
    teacher=Teacher(world)
    tr=make_data(world+100,teacher,4096); dv=make_data(world+200,teacher,1024)
    metrics={}
    for method in METHODS:
        model,m=fit(method,world,teacher,tr,dv,steps)
        blob,sha=bytes_and_hash(model,method,world)
        run_dir=Path(__file__).resolve().parents[1]/'runs'/'dev_payloads'; run_dir.mkdir(parents=True,exist_ok=True)
        payload_path=run_dir/f'{world}_{method}.pt'; payload_path.write_bytes(blob)
        packed=torch.load(io.BytesIO(blob),map_location='cpu',weights_only=False)
        replay=Student(method,world+4000,Teacher(world) if method=='oracle' else None)
        replay.load_state_dict(packed['state'])
        with torch.no_grad(): replay_error=float((replay(dv[0])-model(dv[0])).abs().max().item())
        m.update({'bytes':len(blob),'payload_sha256':sha,'payload_file':str(payload_path.relative_to(Path(__file__).resolve().parents[1])),'replay_max_abs_error':replay_error,'params':sum(p.numel() for p in model.parameters()),'synthesis_flop_proxy_per_example':flop_proxy(method),'wall_clock_seconds':m['seconds'],'train_examples':4096,'dev_examples':1024})
        metrics[method]=m
    return metrics

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--worlds',nargs='+',type=int,default=[60301,60302]); ap.add_argument('--steps',type=int,default=1200); ap.add_argument('--out',required=True); a=ap.parse_args()
    torch.set_num_threads(1)
    res={'experiment_id':'MA-603','phase':'development','python':platform.python_version(),'torch':torch.__version__,'device':'cpu','steps':a.steps,'worlds':{str(w):run_world(w,a.steps) for w in a.worlds}}
    out=Path(a.out); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(res,indent=2)+'\n')
    print(json.dumps(res,indent=2))
if __name__=='__main__': main()
