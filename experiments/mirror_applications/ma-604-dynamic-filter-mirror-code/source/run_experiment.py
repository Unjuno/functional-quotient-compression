#!/usr/bin/env python3
"""MA-604 dynamic filter code versus generic filter generation."""
import argparse, hashlib, io, json, platform, time
from pathlib import Path
import torch
from torch import nn
D,O,K=12,8,3
METHODS=('static','mirror','basis','full_filter','oracle')

def givens_filter(w,angles):
    # Per-example left rotations on output pairs (0,1),(2,3), then a right rotation on input pair (0,1).
    out=w.unsqueeze(0).expand(angles.shape[0],-1,-1).clone()
    for idx,(i,j) in enumerate(((0,1),(2,3))):
        c,s=angles[:,idx].cos()[:,None],angles[:,idx].sin()[:,None]
        a,b=out[:,i,:].clone(),out[:,j,:].clone()
        out[:,i,:]=c*a-s*b; out[:,j,:]=s*a+c*b
    c,s=angles[:,2].cos()[:,None],angles[:,2].sin()[:,None]
    a,b=out[:,:,0].clone(),out[:,:,1].clone()
    out[:,:,0]=c*a-s*b; out[:,:,1]=s*a+c*b
    return out

class Teacher(nn.Module):
    def __init__(self,seed):
        super().__init__(); g=torch.Generator().manual_seed(seed)
        self.basis=nn.Parameter(torch.randn(4,O,D,generator=g)*.18,requires_grad=False)
        self.g1=nn.Parameter(torch.randn(4,D,generator=g)*.45,requires_grad=False)
        self.g2=nn.Parameter(torch.randn(4,4,generator=g)*.45,requires_grad=False)
    def weights(self,x):
        c=torch.softmax(torch.tanh(x@self.g1.T)@self.g2.T,dim=-1)
        return torch.einsum('nk,kod->nod',c,self.basis)
    def forward(self,x): return torch.einsum('nod,nd->no',self.weights(x),x)

class Student(nn.Module):
    def __init__(self,method,seed,teacher=None):
        super().__init__(); torch.manual_seed(seed); self.method=method
        if method=='oracle': self.teacher=teacher; return
        if method=='static': self.w=nn.Parameter(torch.randn(O,D)*.1)
        elif method=='mirror': self.w=nn.Parameter(torch.randn(O,D)*.1); self.code=nn.Linear(D,3)
        elif method=='basis': self.w=nn.Parameter(torch.randn(K,O,D)*.1); self.router=nn.Linear(D,K)
        elif method=='full_filter': self.generator=nn.Sequential(nn.Linear(D,32),nn.Tanh(),nn.Linear(32,O*D))
    def weights(self,x):
        if self.method=='oracle': return self.teacher.weights(x)
        if self.method=='static': return self.w.unsqueeze(0).expand(x.shape[0],-1,-1)
        if self.method=='mirror': return givens_filter(self.w,torch.tanh(self.code(x))*.8)
        if self.method=='basis': return torch.einsum('nk,kod->nod',torch.softmax(self.router(x),-1),self.w)
        return self.generator(x).reshape(-1,O,D)
    def forward(self,x): return torch.einsum('nod,nd->no',self.weights(x),x)

def data(seed,teacher,n):
    g=torch.Generator().manual_seed(seed); x=torch.randn(n,D,generator=g)
    with torch.no_grad(): y=teacher(x)
    return x,y

def flop_proxy(method):
    return {'static':D*O,'mirror':2*D*3+3*(D+O)+D*O,'basis':2*D*K+K*O*D+D*O,'full_filter':2*D*32+2*32*O*D+D*O,'oracle':2*D*4+2*4*4+4*O*D+D*O}[method]

def pack(model,method,seed):
    bio=io.BytesIO(); torch.save({'state':{k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},'method':method,'dims':[D,O,K],'seed':seed,'format':'MA604-v1'},bio)
    return bio.getvalue()

def fit(method,world,teacher,tr,dv,steps):
    torch.manual_seed(world+5000); model=Student(method,world+7000,teacher if method=='oracle' else None)
    if method=='oracle':
        with torch.no_grad(): loss=((model(dv[0])-dv[1])**2).mean()/dv[1].var()
        return model,{'dev_nrmse2':loss.item(),'train_nrmse2':None,'updates':0,'seconds':0.0}
    opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=0); x,y=tr; g=torch.Generator().manual_seed(world+9000); start=time.perf_counter()
    for _ in range(steps):
        ix=torch.randint(x.shape[0],(128,),generator=g); loss=((model(x[ix])-y[ix])**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    seconds=time.perf_counter()-start
    with torch.no_grad():
        trm=((model(x)-y)**2).mean()/y.var(); dvm=((model(dv[0])-dv[1])**2).mean()/dv[1].var()
    return model,{'dev_nrmse2':dvm.item(),'train_nrmse2':trm.item(),'updates':steps,'seconds':seconds}

def run_world(world,steps):
    teacher=Teacher(world); tr=data(world+100,teacher,4096); dv=data(world+200,teacher,1024); out={}
    root=Path(__file__).resolve().parents[1]/'runs/dev_payloads'; root.mkdir(parents=True,exist_ok=True)
    for method in METHODS:
        model,m=fit(method,world,teacher,tr,dv,steps); blob=pack(model,method,world); p=root/f'{world}_{method}.pt'; p.write_bytes(blob)
        payload=torch.load(io.BytesIO(blob),map_location='cpu',weights_only=False)
        replay=Student(method,world+7000,Teacher(world) if method=='oracle' else None); replay.load_state_dict(payload['state'])
        with torch.no_grad(): err=(replay(dv[0])-model(dv[0])).abs().max().item()
        m.update({'payload_bytes':len(blob),'payload_sha256':hashlib.sha256(blob).hexdigest(),'payload_file':str(p.relative_to(Path(__file__).resolve().parents[1])),'replay_max_abs_error':err,'params_diagnostic':sum(p.numel() for p in model.parameters()),'compute_proxy':flop_proxy(method),'wall_clock_seconds':m['seconds'],'train_examples':4096,'dev_examples':1024})
        out[method]=m
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--worlds',type=int,nargs='+',default=[60401,60402]); ap.add_argument('--steps',type=int,default=1200); ap.add_argument('--out',required=True); a=ap.parse_args(); torch.set_num_threads(1)
    result={'experiment_id':'MA-604','phase':'development','python':platform.python_version(),'torch':torch.__version__,'device':'cpu','steps':a.steps,'worlds':{str(w):run_world(w,a.steps) for w in a.worlds}}
    p=Path(a.out); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
