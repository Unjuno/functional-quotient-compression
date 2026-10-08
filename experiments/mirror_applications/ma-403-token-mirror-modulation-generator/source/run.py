from __future__ import annotations
import argparse,hashlib,io,json,math,time
from pathlib import Path
import numpy as np
import torch
D=16; METHODS=('global_film','token_film','token_rank1','mirror_generator','full_affine_generator')

def rot(angles):
    # angles shape [...,8]; output [...,16,16]
    out=torch.eye(D).expand(*angles.shape[:-1],D,D).clone()
    for j in range(8):
        a=angles[...,j];c=torch.cos(a);s=torch.sin(a);i=2*j
        out[...,i,i]=c;out[...,i,i+1]=-s;out[...,i+1,i]=s;out[...,i+1,i+1]=c
    return out

def world(seed,n=4096):
    g=torch.Generator().manual_seed(seed)
    w1=torch.randn(D,D,generator=g)/math.sqrt(D);w2=torch.randn(D,D,generator=g)/math.sqrt(D)
    x=torch.randn(n,D,generator=g);h=torch.relu(x@w1)@w2
    intercept=torch.randn(8,generator=g)*0.35;slope=torch.randn(8,generator=g)*1.2
    ptrain=torch.rand(n,generator=g)*0.8;ptest=.8+torch.rand(n,generator=g)*.2
    def targets(p):
        angles=intercept[None,:]+p[:,None]*slope[None,:]
        return torch.einsum('nij,nj->ni',rot(angles),h[:len(p)])
    # Independent inputs/features for held-out examples, same paid feature block and teacher.
    xt=torch.randn(n,D,generator=g);ht=torch.relu(xt@w1)@w2
    ytrain=targets(ptrain)
    angles=intercept[None,:]+ptest[:,None]*slope[None,:]
    ytest=torch.einsum('nij,nj->ni',rot(angles),ht)
    return x,h,ptrain,ytrain,xt,ht,ptest,ytest,w1,w2

class Fit(torch.nn.Module):
    def __init__(self,method):
        super().__init__();self.method=method
        if method=='global_film':self.scale=torch.nn.Parameter(torch.ones(D));self.bias=torch.nn.Parameter(torch.zeros(D))
        elif method=='token_film':self.sf=torch.nn.Parameter(torch.ones(D));self.ss=torch.nn.Parameter(torch.zeros(D));self.bf=torch.nn.Parameter(torch.zeros(D));self.bs=torch.nn.Parameter(torch.zeros(D))
        elif method=='token_rank1':self.u0=torch.nn.Parameter(torch.zeros(D));self.u1=torch.nn.Parameter(torch.zeros(D));self.v0=torch.nn.Parameter(torch.randn(D)*.01);self.v1=torch.nn.Parameter(torch.zeros(D));self.bf=torch.nn.Parameter(torch.zeros(D));self.bs=torch.nn.Parameter(torch.zeros(D))
        elif method=='mirror_generator':self.ang0=torch.nn.Parameter(torch.zeros(8));self.ang1=torch.nn.Parameter(torch.zeros(8))
        elif method=='full_affine_generator':self.m0=torch.nn.Parameter(torch.eye(D));self.m1=torch.nn.Parameter(torch.zeros(D,D));self.b0=torch.nn.Parameter(torch.zeros(D));self.b1=torch.nn.Parameter(torch.zeros(D))
    def forward(self,h,p):
        if self.method=='global_film':return h*self.scale+self.bias
        if self.method=='token_film':return h*(self.sf+p[:,None]*self.ss)+(self.bf+p[:,None]*self.bs)
        if self.method=='token_rank1':
            u=self.u0+p[:,None]*self.u1;v=self.v0+p[:,None]*self.v1
            return h+(h*v).sum(-1,keepdim=True)*u+self.bf+p[:,None]*self.bs
        if self.method=='mirror_generator':
            angles=self.ang0+p[:,None]*self.ang1
            return torch.einsum('nij,nj->ni',rot(angles),h)
        m=self.m0[None]+p[:,None,None]*self.m1[None]
        return torch.einsum('nij,nj->ni',m,h)+self.b0+p[:,None]*self.b1

def fit(method,h,p,y,seed,updates=400):
    torch.manual_seed(seed);m=Fit(method);opt=torch.optim.Adam(m.parameters(),lr=.04);start=time.perf_counter()
    for _ in range(updates):
        ix=torch.randint(int(.8*len(h)),(1024,));pred=m(h[ix],p[ix]);loss=((pred-y[ix])**2).mean();opt.zero_grad();loss.backward();opt.step()
    return m,time.perf_counter()-start

def pack(model,w1,w2,method):
    arrays={'shared_w1':w1.numpy().astype(np.float16),'shared_w2':w2.numpy().astype(np.float16)}
    for k,v in model.state_dict().items():arrays[k]=v.detach().numpy().astype(np.float16)
    arrays['metadata_utf8']=np.frombuffer(json.dumps({'method':method,'position_code':'linear','dtype':'float16'},sort_keys=True,separators=(',',':')).encode(),dtype=np.uint8)
    b=io.BytesIO();np.savez_compressed(b,**arrays);return b.getvalue()

def replay(raw,h,p,method):
    with np.load(io.BytesIO(raw),allow_pickle=False) as a:
        w1=torch.tensor(a['shared_w1'].astype(np.float32));w2=torch.tensor(a['shared_w2'].astype(np.float32));m=Fit(method)
        m.load_state_dict({k:torch.tensor(a[k].astype(np.float32)) for k in m.state_dict()})
    return m(torch.relu(h@w1)@w2,p)

def nrmse(a,b):return float(torch.sqrt(torch.mean((a-b)**2))/torch.sqrt(torch.mean((b-b.mean())**2)))
def benchmark(model,h,p):
    h=h[:1024];p=p[:1024]
    with torch.no_grad():
        for _ in range(10):model(h,p)
        start=time.perf_counter()
        for _ in range(25):model(h,p)
    return 25*len(h)/(time.perf_counter()-start)
def run(out,seeds=(40301,40302),updates=400,n=4096):
    out.mkdir(parents=True,exist_ok=True);rows=[];screens=[];torch.set_num_threads(1)
    for seed in seeds:
        x,h,p,y,xt,ht,pt,yt,w1,w2=world(seed,n)
        for method in METHODS:
            m,sec=fit(method,h,p,y,seed+len(method),updates);raw=pack(m,w1,w2,method);(out/f'dev{seed}_{method}.npz').write_bytes(raw)
            pred=replay(raw,xt,pt,method);score=nrmse(pred,yt);rate=benchmark(m,ht,pt);digest=hashlib.sha256(raw).hexdigest()
            ops={'global_film':32,'token_film':64,'token_rank1':128,'mirror_generator':64,'full_affine_generator':544}[method]
            rows.append({'condition':'heldout_position_[0.8,1]','world_or_seed':seed,'method':method,'serialized_bytes':len(raw),'train_tokens_or_examples':updates*1024,'optimizer_updates':updates,'active_compute_proxy':ops,'wall_time_s':round(sec,6),'primary_metric':'NRMSE','primary_value':score,'secondary_metric':'examples_per_second','secondary_value':rate,'status_note':digest})
        by={r['method']:r for r in rows if r['world_or_seed']==seed};mi=by['mirror_generator'];tf=by['token_film'];fa=by['full_affine_generator']
        screens.append({'seed':seed,'pass':mi['primary_value']<=.02 and mi['primary_value']<=.1*tf['primary_value'] and mi['serialized_bytes']<=.75*tf['serialized_bytes'] and mi['serialized_bytes']<=.1*fa['serialized_bytes'] and mi['secondary_value']>=.8*tf['secondary_value'],'mirror_nrmse':mi['primary_value'],'token_film_nrmse':tf['primary_value'],'mirror_bytes':mi['serialized_bytes'],'token_film_bytes':tf['serialized_bytes'],'full_affine_bytes':fa['serialized_bytes'],'mirror_rate':mi['secondary_value'],'token_film_rate':tf['secondary_value']})
    import csv
    with (out/'RESULTS_CORE.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    result={'experiment_id':'MA-403','protocol_frozen':True,'fresh_accessed':False,'development_gate_passed':all(r['pass'] for r in screens),'seed_results':screens}
    (out/'screen.json').write_text(json.dumps(result,indent=2)+'\n');return result
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[1]/'runs');z=a.parse_args();print(json.dumps(run(z.out),indent=2))
