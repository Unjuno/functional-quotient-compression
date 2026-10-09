from __future__ import annotations
import argparse,hashlib,io,json,math,time
from pathlib import Path
import numpy as np
import torch
D=32;CTX=8;ALPHAS=(0.,.25,.5,.75,1.)
METHODS=('shared','style_moddemod','mirror_rotation','rank1_residual','independent_full')

def rotation(a):
    r=torch.eye(D).expand(*a.shape[:-1],D,D).clone()
    for j in range(D//2):
        c=torch.cos(a[...,j]);s=torch.sin(a[...,j]);i=2*j
        r[...,i,i]=c;r[...,i,i+1]=-s;r[...,i+1,i]=s;r[...,i+1,i+1]=c
    return r

def style_matrix(w,log_s):
    mod=w[None,:,:]*torch.exp(log_s.clamp(-3,3))[:,None,:]
    return mod*torch.rsqrt((mod*mod).sum(-1)+1e-8)[:,:,None]

def world(seed,n=4096):
    g=torch.Generator().manual_seed(seed)
    w1=torch.randn(D,D,generator=g)/math.sqrt(D);w2=torch.randn(D,D,generator=g)/math.sqrt(D);w=torch.randn(D,D,generator=g)/math.sqrt(D)
    x=torch.randn(n,D,generator=g);h=torch.relu(x@w1)@w2
    style=torch.randn(CTX,D,generator=g)*.45;angle=torch.randn(CTX,D//2,generator=g)*.6
    sm=style_matrix(w,style);rm=w[None,:,:]@rotation(angle)
    tm=torch.stack([alpha*sm[c]+(1-alpha)*rm[c] for alpha in ALPHAS for c in range(CTX)])
    y=torch.einsum('ni,toi->nto',h,tm)
    xt=torch.randn(n,D,generator=g);ht=torch.relu(xt@w1)@w2
    ytest=torch.einsum('ni,toi->nto',ht,tm)
    return x,h,y,xt,ytest,w1,w2,w,tm

class Fit(torch.nn.Module):
    def __init__(self,method,w,tasks=CTX*len(ALPHAS)):
        super().__init__();self.method=method
        if method=='style_moddemod':self.log_s=torch.nn.Parameter(torch.zeros(tasks,D))
        elif method=='mirror_rotation':self.angles=torch.nn.Parameter(torch.zeros(tasks,D//2))
        elif method=='rank1_residual':self.u=torch.nn.Parameter(torch.zeros(tasks,D));self.v=torch.nn.Parameter(torch.randn(tasks,D)*.01)
        elif method=='independent_full':self.matrix=torch.nn.Parameter(w[None,:,:].repeat(tasks,1,1))
    def matrices(self,w):
        if self.method=='shared':return w[None,:,:].expand(CTX*len(ALPHAS),-1,-1)
        if self.method=='style_moddemod':return style_matrix(w,self.log_s)
        if self.method=='mirror_rotation':return w[None,:,:]@rotation(self.angles)
        if self.method=='rank1_residual':return w[None,:,:]+self.u[:,:,None]*self.v[:,None,:]
        return self.matrix
    def forward(self,h,w):return torch.einsum('ni,toi->nto',h,self.matrices(w))

def fit(method,h,y,w,seed,updates=400):
    torch.manual_seed(seed);m=Fit(method,w);params=list(m.parameters());start=time.perf_counter()
    if params:
        opt=torch.optim.Adam(params,lr=.04);train_n=int(.8*len(h))
        for _ in range(updates):
            ix=torch.randint(train_n,(1024,));pred=m(h[ix],w);loss=((pred-y[ix])**2).mean();opt.zero_grad();loss.backward();opt.step()
    return m,time.perf_counter()-start

def pack(model,w1,w2,w,method):
    arrays={'shared_w1':w1.numpy().astype(np.float16),'shared_w2':w2.numpy().astype(np.float16),'shared_ffn_w':w.numpy().astype(np.float16)}
    for k,v in model.state_dict().items():arrays[k]=v.detach().numpy().astype(np.float16)
    arrays['metadata_utf8']=np.frombuffer(json.dumps({'method':method,'contexts':CTX,'alphas':ALPHAS,'width':D,'dtype':'float16'},sort_keys=True,separators=(',',':')).encode(),dtype=np.uint8)
    b=io.BytesIO();np.savez_compressed(b,**arrays);return b.getvalue()

def replay(raw,x,method):
    with np.load(io.BytesIO(raw),allow_pickle=False) as a:
        w1=torch.tensor(a['shared_w1'].astype(np.float32));w2=torch.tensor(a['shared_w2'].astype(np.float32));w=torch.tensor(a['shared_ffn_w'].astype(np.float32));m=Fit(method,w)
        m.load_state_dict({k:torch.tensor(a[k].astype(np.float32)) for k in m.state_dict()})
    h=torch.relu(x@w1)@w2
    with torch.no_grad():return m(h,w)

def metric(pred,y):
    den=torch.sqrt(((y-y.mean(dim=0,keepdim=True))**2).mean(dim=(0,2))).clamp_min(1e-8)
    return (((pred-y)**2).mean(dim=(0,2)).sqrt()/den).reshape(len(ALPHAS),CTX).mean(dim=1).tolist()

def benchmark(model,h,w):
    h=h[:512]
    with torch.no_grad():
        for _ in range(10):model(h,w)
        start=time.perf_counter()
        for _ in range(25):model(h,w)
    return 25*len(h)*CTX*len(ALPHAS)/(time.perf_counter()-start)

def run(out,seeds=(40501,40502),updates=400,n=4096):
    out.mkdir(parents=True,exist_ok=True);rows=[];screens=[];torch.set_num_threads(1)
    ops={'shared':0,'style_moddemod':5152,'mirror_rotation':2176,'rank1_residual':2100,'independent_full':2016}
    for seed in seeds:
        x,h,y,xt,ytest,w1,w2,w,tm=world(seed,n)
        for method in METHODS:
            m,sec=fit(method,h,y,w,seed+len(method),updates);raw=pack(m,w1,w2,w,method);(out/f'dev{seed}_{method}.npz').write_bytes(raw)
            scores=metric(replay(raw,xt,method),ytest);rate=benchmark(m,h,w);digest=hashlib.sha256(raw).hexdigest()
            for ai,alpha in enumerate(ALPHAS):rows.append({'condition':f'alpha={alpha}','world_or_seed':seed,'method':method,'serialized_bytes':len(raw),'train_tokens_or_examples':updates*1024,'optimizer_updates':updates,'active_compute_proxy':ops[method],'wall_time_s':round(sec,6),'primary_metric':'NRMSE','primary_value':scores[ai],'secondary_metric':'logical_examples_per_second','secondary_value':rate,'status_note':digest})
        look={(r['method'],r['condition']):r for r in rows if r['world_or_seed']==seed};mi=look[('mirror_rotation','alpha=0.0')];st=look[('style_moddemod','alpha=0.0')];ind=look[('independent_full','alpha=0.0')]
        screens.append({'seed':seed,'pass':mi['primary_value']<=.02 and mi['primary_value']<=.1*st['primary_value'] and mi['serialized_bytes']<=.95*st['serialized_bytes'] and mi['serialized_bytes']<=.1*ind['serialized_bytes'] and mi['secondary_value']>=.8*st['secondary_value'],'mirror_nrmse':mi['primary_value'],'style_nrmse':st['primary_value'],'mirror_bytes':mi['serialized_bytes'],'style_bytes':st['serialized_bytes'],'independent_bytes':ind['serialized_bytes'],'mirror_rate':mi['secondary_value'],'style_rate':st['secondary_value']})
    import csv
    with (out/'RESULTS_CORE.csv').open('w',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');wr.writeheader();wr.writerows(rows)
    result={'experiment_id':'MA-405','protocol_frozen':True,'fresh_accessed':False,'development_gate_passed':all(r['pass'] for r in screens),'seed_results':screens}
    (out/'screen.json').write_text(json.dumps(result,indent=2)+'\n');return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[1]/'runs');a=p.parse_args();print(json.dumps(run(a.out),indent=2))
