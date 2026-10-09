#!/usr/bin/env python3
"""MA-607 AdapterFusion with Givens-compressed source adapters."""
import argparse,csv,hashlib,io,json,platform,time
from pathlib import Path
import torch
from torch import nn
D,O,R,K=32,32,6,4
PAIRS=[(i,j) for i in range(R) for j in range(i+1,R)]
MODES=('full','mirror','shared','oracle')

def rotate(v,angles):
    z=v.clone()
    for n,(i,j) in enumerate(PAIRS):
        c,s=angles[n].cos(),angles[n].sin();a,b=z[...,i].clone(),z[...,j].clone();z[...,i]=c*a-s*b;z[...,j]=s*a+c*b
    return z

def random_rot(g):
    return torch.randn(len(PAIRS),generator=g)*.35

class Teacher:
    def __init__(self,seed):
        g=torch.Generator().manual_seed(seed);self.a=torch.randn(R,D,generator=g)*.12;self.b=torch.randn(O,R,generator=g)*.12
        self.angles=torch.stack([random_rot(g) for _ in range(K)]);self.gate=nn.Linear(D,K)
        with torch.no_grad():self.gate.weight.copy_(torch.randn(K,D,generator=g)*.3);self.gate.bias.copy_(torch.randn(K,generator=g)*.1)
    def adapter(self,t,x):return rotate(x@self.a.T,self.angles[t])@self.b.T
    def outputs(self,x):return torch.stack([self.adapter(t,x) for t in range(K)],1)
    def __call__(self,x):return torch.einsum('nk,nko->no',torch.softmax(self.gate(x),-1),self.outputs(x))
    def forward(self,x):return torch.einsum('nk,nko->no',torch.softmax(self.gate(x),-1),self.outputs(x))

class Bank(nn.Module):
    def __init__(self,mode,seed,teacher=None):
        super().__init__();torch.manual_seed(seed);self.mode=mode
        if mode=='oracle':
            self.a=nn.Parameter(teacher.a.clone(),requires_grad=False);self.b=nn.Parameter(teacher.b.clone(),requires_grad=False);self.angles=nn.Parameter(teacher.angles.clone(),requires_grad=False);return
        if mode=='full':
            self.a=nn.Parameter(torch.randn(K,R,D)*.04);self.b=nn.Parameter(torch.randn(K,O,R)*.04)
        else:
            self.a=nn.Parameter(torch.randn(R,D)*.04);self.b=nn.Parameter(torch.randn(O,R)*.04)
            if mode=='mirror':self.angles=nn.Parameter(torch.zeros(K,len(PAIRS)))
            if mode=='shared':self.coeff=nn.Parameter(torch.eye(R).unsqueeze(0).repeat(K,1,1))
    def adapter(self,t,x):
        if self.mode=='full':return (x@self.a[t].T)@self.b[t].T
        v=x@self.a.T
        if self.mode in ('mirror','oracle'):v=rotate(v,self.angles[t])
        else:v=v@self.coeff[t].T
        return v@self.b.T
    def outputs(self,x):return torch.stack([self.adapter(t,x) for t in range(K)],1)

def gen(seed,teacher,n):
    g=torch.Generator().manual_seed(seed);x=torch.randn(n,D,generator=g);return x,teacher(x)

def fit_bank(mode,world,teacher,sets,steps):
    bank=Bank(mode,world+3000,teacher if mode=='oracle' else None)
    if mode=='oracle':return bank,0.
    opt=torch.optim.AdamW(bank.parameters(),lr=.003,weight_decay=0);start=time.perf_counter()
    for step in range(steps):
        losses=[]
        for t,(x,y) in enumerate(sets):
            ix=torch.randint(x.shape[0],(128,),generator=torch.Generator().manual_seed(world+step*11+t));losses.append(((bank.adapter(t,x[ix])-y[ix])**2).mean())
        loss=torch.stack(losses).mean();opt.zero_grad();loss.backward();opt.step()
    return bank,time.perf_counter()-start

def fit_fusion(bank,world,teacher,steps):
    gate=nn.Linear(D,K);torch.manual_seed(world+9000);gate.reset_parameters();opt=torch.optim.AdamW(gate.parameters(),lr=.003,weight_decay=0)
    g=torch.Generator().manual_seed(world+9100);x=torch.randn(4096,D,generator=g)
    with torch.no_grad():target=teacher(x);ys=bank.outputs(x)
    start=time.perf_counter()
    for step in range(steps):
        ix=torch.randint(x.shape[0],(128,),generator=torch.Generator().manual_seed(world+step+10000));c=torch.softmax(gate(x[ix]),-1);pred=torch.einsum('nk,nko->no',c,ys[ix]);loss=((pred-target[ix])**2).mean();opt.zero_grad();loss.backward();opt.step()
    return gate,time.perf_counter()-start

def eval_model(bank,gate,x,y):
    with torch.no_grad():
        ys=bank.outputs(x);pred=torch.einsum('nk,nko->no',torch.softmax(gate(x),-1),ys)
        return ((pred-y)**2).mean().item()/y.var().item()

def proxy(mode):
    proj=K*(D*R+R*O)
    return {'full':proj,'shared':proj+K*R*R,'mirror':proj+K*len(PAIRS)*R,'oracle':proj+K*len(PAIRS)*R}[mode]

def pack(bank,gate,mode,seed):
    bio=io.BytesIO();state={'bank':{k:v.detach().cpu().contiguous() for k,v in bank.state_dict().items()},'fusion_gate':{k:v.detach().cpu().contiguous() for k,v in gate.state_dict().items()}}
    torch.save({'state':state,'mode':mode,'dims':[D,O,R,K],'seed':seed,'format':'MA607-v1'},bio);return bio.getvalue()

def run_world(world,steps,fusion_steps):
    t=Teacher(world);sets=[]
    for i in range(K):
        g=torch.Generator().manual_seed(world+100+i);x=torch.randn(2048,D,generator=g);y=t.adapter(i,x);sets.append((x,y))
    gx,gy=gen(world+200,t,1024);result={};root=Path(__file__).resolve().parents[1]/'runs/dev_payloads';root.mkdir(parents=True,exist_ok=True)
    for mode in MODES:
        bank,bt=fit_bank(mode,world,t,sets,steps);gate,ft=fit_fusion(bank,world,t,fusion_steps);blob=pack(bank,gate,mode,world);p=root/f'{world}_{mode}.pt';p.write_bytes(blob)
        dat=torch.load(io.BytesIO(blob),weights_only=False)
        rb=Bank(mode,world+3000,t if mode=='oracle' else None);rb.load_state_dict(dat['state']['bank']);rg=nn.Linear(D,K);rg.load_state_dict(dat['state']['fusion_gate'])
        with torch.no_grad():err=(torch.einsum('nk,nko->no',torch.softmax(gate(gx),-1),bank.outputs(gx))-torch.einsum('nk,nko->no',torch.softmax(rg(gx),-1),rb.outputs(gx))).abs().max().item()
        r={'fusion_nrmse2':eval_model(bank,gate,gx,gy),'source_training_seconds':bt,'fusion_training_seconds':ft,'updates_source':steps if mode!='oracle' else 0,'updates_fusion':fusion_steps,'source_examples_per_task':2048 if mode!='oracle' else 0,'fusion_examples':4096,'dev_examples':1024,'payload_bytes':len(blob),'payload_sha256':hashlib.sha256(blob).hexdigest(),'payload_file':str(p.relative_to(Path(__file__).resolve().parents[1])),'replay_max_abs_error':err,'params_diagnostic':sum(p.numel() for p in bank.parameters())+sum(p.numel() for p in gate.parameters()),'compute_proxy':proxy(mode),'wall_clock_seconds':bt+ft}
        result[mode]=r
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--worlds',type=int,nargs='+',default=[60701,60702]);p.add_argument('--steps',type=int,default=1200);p.add_argument('--fusion-steps',type=int,default=800);p.add_argument('--out',required=True);a=p.parse_args();torch.set_num_threads(1)
    z={'experiment_id':'MA-607','phase':'development','python':platform.python_version(),'torch':torch.__version__,'device':'cpu','source_steps':a.steps,'fusion_steps':a.fusion_steps,'worlds':{str(w):run_world(w,a.steps,a.fusion_steps) for w in a.worlds}}
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(z,indent=2)+'\n');print(json.dumps(z,indent=2))
if __name__=='__main__':main()
