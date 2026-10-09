#!/usr/bin/env python3
"""MA-606 one dynamic Givens-view FFN versus soft-MoE."""
import argparse,hashlib,io,json,platform,time
from pathlib import Path
import torch
from torch import nn
D,H,O,K=16,12,10,4
METHODS=('static','mirror','film','soft_moe','oracle')

def rotate_hidden(v,angles):
    z=v.clone()
    for k,(i,j) in enumerate(((0,1),(2,3),(4,5))):
        c,s=angles[:,k].cos()[:,None],angles[:,k].sin()[:,None]
        a,b=z[:,i].clone(),z[:,j].clone()
        z[:,i]=c[:,0]*a-s[:,0]*b; z[:,j]=s[:,0]*a+c[:,0]*b
    return z

class FFN(nn.Module):
    def __init__(self,w1,b1,w2,b2):
        super().__init__(); self.w1=nn.Parameter(w1);self.b1=nn.Parameter(b1);self.w2=nn.Parameter(w2);self.b2=nn.Parameter(b2)
    def forward(self,x):return torch.tanh(x@self.w1.T+self.b1)@self.w2.T+self.b2

class Teacher(nn.Module):
    def __init__(self,seed):
        super().__init__();g=torch.Generator().manual_seed(seed)
        w1=torch.randn(H,D,generator=g)*.22;b1=torch.randn(H,generator=g)*.08
        w2=torch.randn(O,H,generator=g)*.18;b2=torch.randn(O,generator=g)*.05
        self.w1=nn.Parameter(w1[None]+torch.randn(K,H,D,generator=g)*.15,requires_grad=False)
        self.b1=nn.Parameter(b1[None]+torch.randn(K,H,generator=g)*.15,requires_grad=False)
        self.w2=nn.Parameter(w2[None]+torch.randn(K,O,H,generator=g)*.15,requires_grad=False)
        self.b2=nn.Parameter(b2[None]+torch.randn(K,O,generator=g)*.15,requires_grad=False)
        self.gate=nn.Linear(D,K)
        with torch.no_grad():self.gate.weight.copy_(torch.randn(K,D,generator=g)*.35);self.gate.bias.copy_(torch.randn(K,generator=g)*.1)
        for p in self.gate.parameters():p.requires_grad_(False)
    def forward(self,x):
        c=torch.softmax(self.gate(x),-1); y=[]
        for k in range(K):y.append(torch.tanh(x@self.w1[k].T+self.b1[k])@self.w2[k].T+self.b2[k])
        return torch.einsum('nk,nko->no',c,torch.stack(y,1))

class Student(nn.Module):
    def __init__(self,method,seed,teacher=None):
        super().__init__();torch.manual_seed(seed);self.method=method
        if method=='oracle':self.teacher=teacher;return
        if method=='soft_moe':
            self.w1=nn.Parameter(torch.randn(K,H,D)*.12);self.b1=nn.Parameter(torch.zeros(K,H));self.w2=nn.Parameter(torch.randn(K,O,H)*.12);self.b2=nn.Parameter(torch.zeros(K,O));self.router=nn.Linear(D,K)
        else:
            self.w1=nn.Parameter(torch.randn(H,D)*.12);self.b1=nn.Parameter(torch.zeros(H));self.w2=nn.Parameter(torch.randn(O,H)*.12);self.b2=nn.Parameter(torch.zeros(O))
            if method=='mirror':self.code=nn.Linear(D,3)
            if method=='film':self.mod=nn.Linear(D,2*H)
    def forward(self,x):
        if self.method=='oracle':return self.teacher(x)
        if self.method=='soft_moe':
            c=torch.softmax(self.router(x),-1); ys=[]
            for k in range(K):ys.append(torch.tanh(x@self.w1[k].T+self.b1[k])@self.w2[k].T+self.b2[k])
            return torch.einsum('nk,nko->no',c,torch.stack(ys,1))
        pre=x@self.w1.T+self.b1
        if self.method=='mirror':pre=rotate_hidden(pre,torch.tanh(self.code(x))*.8)
        h=torch.tanh(pre)
        if self.method=='film':
            scale,bias=self.mod(x).chunk(2,-1);h=h*(1+scale)+bias
        return h@self.w2.T+self.b2

def data(seed,t,n):
    g=torch.Generator().manual_seed(seed);x=torch.randn(n,D,generator=g)
    with torch.no_grad():y=t(x)
    return x,y

def proxy(m):
    base=2*(D*H+H*O)
    return {'static':base,'mirror':base+2*D*3+2*H*3,'film':base+2*D*(2*H)+2*H,'soft_moe':2*D*K+K*base,'oracle':2*D*K+K*base}[m]

def fit(method,world,t,tr,dv,steps):
    torch.manual_seed(world+1900);model=Student(method,world+5000,t if method=='oracle' else None)
    if method=='oracle':
        with torch.no_grad():return model,{'dev_nrmse2':(((model(dv[0])-dv[1])**2).mean()/dv[1].var()).item(),'train_nrmse2':None,'updates':0,'seconds':0.}
    opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=0);x,y=tr;g=torch.Generator().manual_seed(world+7000);start=time.perf_counter()
    for _ in range(steps):
        ix=torch.randint(x.shape[0],(128,),generator=g);loss=((model(x[ix])-y[ix])**2).mean();opt.zero_grad();loss.backward();opt.step()
    sec=time.perf_counter()-start
    with torch.no_grad():trm=((model(x)-y)**2).mean()/y.var();dvm=((model(dv[0])-dv[1])**2).mean()/dv[1].var()
    return model,{'dev_nrmse2':dvm.item(),'train_nrmse2':trm.item(),'updates':steps,'seconds':sec}

def pack(model,m,seed):
    bio=io.BytesIO();torch.save({'state':{k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},'method':m,'dims':[D,H,O,K],'seed':seed,'format':'MA606-v1'},bio);return bio.getvalue()

def run_world(world,steps):
    t=Teacher(world);tr=data(world+100,t,4096);dv=data(world+200,t,1024);out={};root=Path(__file__).resolve().parents[1]/'runs/dev_payloads';root.mkdir(parents=True,exist_ok=True)
    for m in METHODS:
        model,r=fit(m,world,t,tr,dv,steps);blob=pack(model,m,world);p=root/f'{world}_{m}.pt';p.write_bytes(blob)
        dat=torch.load(io.BytesIO(blob),map_location='cpu',weights_only=False);re=Student(m,world+5000,Teacher(world) if m=='oracle' else None);re.load_state_dict(dat['state'])
        with torch.no_grad():err=(re(dv[0])-model(dv[0])).abs().max().item()
        r.update({'payload_bytes':len(blob),'payload_sha256':hashlib.sha256(blob).hexdigest(),'payload_file':str(p.relative_to(Path(__file__).resolve().parents[1])),'replay_max_abs_error':err,'params_diagnostic':sum(p.numel() for p in model.parameters()),'compute_proxy':proxy(m),'wall_clock_seconds':r['seconds'],'train_examples':4096,'dev_examples':1024});out[m]=r
    return out

def main():
    a=argparse.ArgumentParser();a.add_argument('--worlds',type=int,nargs='+',default=[60601,60602]);a.add_argument('--steps',type=int,default=1200);a.add_argument('--out',required=True);q=a.parse_args();torch.set_num_threads(1)
    z={'experiment_id':'MA-606','phase':'development','python':platform.python_version(),'torch':torch.__version__,'device':'cpu','steps':q.steps,'worlds':{str(w):run_world(w,q.steps) for w in q.worlds}}
    p=Path(q.out);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(z,indent=2)+'\n');print(json.dumps(z,indent=2))
if __name__=='__main__':main()
