from __future__ import annotations
import argparse, hashlib, io, json, math, time
from pathlib import Path
import numpy as np
import torch

METHODS = ('shared_identity','film','rank1','mirror','mirror_rank1','independent')
ALPHAS = (0.0,0.25,0.5,0.75,1.0)
CONTEXTS = 8
D = 16


def make_world(seed: int, n: int = 4096):
    g = torch.Generator().manual_seed(seed)
    w1 = torch.randn(D,D,generator=g)/math.sqrt(D)
    w2 = torch.randn(D,D,generator=g)/math.sqrt(D)
    x = torch.randn(n,D,generator=g)
    h = torch.relu(x @ w1) @ w2
    angles = torch.randn(CONTEXTS,8,generator=g)*0.8
    dense = torch.randn(CONTEXTS,D,D,generator=g)/math.sqrt(D)
    dense = dense + torch.eye(D)*0.25
    transforms=[]
    for alpha in ALPHAS:
        row=[]
        for c in range(CONTEXTS):
            r=rotation(angles[c])
            row.append(alpha*r+(1-alpha)*dense[c])
        transforms += row
    mats=torch.stack(transforms)
    y=torch.einsum('nd,tdk->ntk',h,mats)
    return x,h,y,w1,w2,mats


def rotation(angles: torch.Tensor):
    # Product of eight independent rotations over fixed adjacent feature pairs.
    out=torch.eye(D,dtype=angles.dtype,device=angles.device)
    for j in range(8):
        a=angles[j]; c=torch.cos(a); s=torch.sin(a); i=2*j
        out[i,i]=c; out[i,i+1]=-s; out[i+1,i]=s; out[i+1,i+1]=c
    return out


class ConditionalFit(torch.nn.Module):
    def __init__(self, method: str, tasks: int):
        super().__init__(); self.method=method
        if method=='film': self.scale=torch.nn.Parameter(torch.ones(tasks,D)); self.bias=torch.nn.Parameter(torch.zeros(tasks,D))
        elif method=='rank1': self.u=torch.nn.Parameter(torch.zeros(tasks,D)); self.v=torch.nn.Parameter(torch.randn(tasks,D)*0.01); self.bias=torch.nn.Parameter(torch.zeros(tasks,D))
        elif method=='mirror': self.angle=torch.nn.Parameter(torch.zeros(tasks,8))
        elif method=='mirror_rank1': self.angle=torch.nn.Parameter(torch.zeros(tasks,8)); self.u=torch.nn.Parameter(torch.zeros(tasks,D)); self.v=torch.nn.Parameter(torch.randn(tasks,D)*0.01); self.bias=torch.nn.Parameter(torch.zeros(tasks,D))
        elif method=='independent': self.matrix=torch.nn.Parameter(torch.eye(D).repeat(tasks,1,1)); self.bias=torch.nn.Parameter(torch.zeros(tasks,D))
        elif method=='shared_identity': pass
        else: raise ValueError(method)

    def forward(self,h,task_ids):
        z=h[:,None,:]
        if self.method=='shared_identity': return z.expand(-1,task_ids.numel(),-1)
        if self.method=='film': return z*self.scale[task_ids][None,:,:]+self.bias[task_ids][None,:,:]
        if self.method in ('rank1','mirror_rank1'):
            if self.method=='mirror_rank1': z=apply_rotation(z,self.angle[task_ids][None,:,:])
            u=self.u[task_ids]; v=self.v[task_ids]
            return z + (z*v[None,:,:]).sum(-1,keepdim=True)*u[None,:,:]+self.bias[task_ids][None,:,:]
        if self.method=='mirror': return apply_rotation(z,self.angle[task_ids][None,:,:])
        return torch.einsum('nd,tdk->ntk',h,self.matrix[task_ids])+self.bias[task_ids][None,:,:]


def apply_rotation(z, angles):
    # z shape [batch, tasks, features], angles [1, tasks, 8]
    blocks=z.reshape(*z.shape[:-1],8,2)
    c=torch.cos(angles); s=torch.sin(angles)
    a=blocks[...,0]; b=blocks[...,1]
    return torch.stack((a*c-b*s,a*s+b*c),dim=-1).reshape(z.shape[0],angles.shape[1],D)


def fit(method,h,y,updates=400,seed=0):
    torch.manual_seed(seed)
    tasks=y.shape[1]
    model=ConditionalFit(method,tasks)
    train_n=max(1,int(h.shape[0]*0.8)); parameters=list(model.parameters())
    if not parameters: return model,0.0
    opt=torch.optim.Adam(parameters,lr=0.04)
    start=time.perf_counter()
    for step in range(updates):
        ix=torch.randint(train_n,(1024,))
        task_ids=torch.arange(tasks)
        pred=model(h[ix],task_ids)
        loss=((pred-y[ix])**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    elapsed=time.perf_counter()-start
    return model,elapsed


def payload(model,w1,w2,method):
    arrays={'shared_w1':w1.detach().cpu().numpy().astype(np.float16),'shared_w2':w2.detach().cpu().numpy().astype(np.float16)}
    for name,value in model.state_dict().items(): arrays[name]=value.detach().cpu().numpy().astype(np.float16)
    meta=json.dumps({'method':method,'contexts':CONTEXTS,'alphas':ALPHAS,'dimension':D,'dtype':'float16'},sort_keys=True,separators=(',',':')).encode()
    arrays['metadata_utf8']=np.frombuffer(meta,dtype=np.uint8)
    buff=io.BytesIO(); np.savez_compressed(buff,**arrays); raw=buff.getvalue()
    return raw,arrays


def predict_from_payload(raw,h,method):
    with np.load(io.BytesIO(raw),allow_pickle=False) as z: a={k:z[k].copy() for k in z.files}
    # Inputs are common X; reconstruct paid shared block from serialized values.
    h=torch.relu(torch.as_tensor(h,dtype=torch.float32) @ torch.as_tensor(a['shared_w1'].astype(np.float32))) @ torch.as_tensor(a['shared_w2'].astype(np.float32))
    m=ConditionalFit(method,40)
    state={k:torch.as_tensor(v.astype(np.float32)) for k,v in a.items() if k in m.state_dict()}
    m.load_state_dict(state)
    with torch.no_grad(): return m(h,torch.arange(40)).numpy()


def evaluate(model,h,y):
    with torch.no_grad(): pred=model(h,torch.arange(y.shape[1]))
    den=((y-y.mean(dim=0,keepdim=True))**2).mean(dim=(0,2)).sqrt().clamp_min(1e-8)
    err=((pred-y)**2).mean(dim=(0,2)).sqrt()/den
    return err.reshape(5,8).mean(dim=1).numpy().tolist()


def run(out: Path,seeds=(40101,40102),updates=400,n=4096):
    out.mkdir(parents=True,exist_ok=True); rows=[]; artifacts={}
    torch.set_num_threads(1)
    for seed in seeds:
        x,h,y,w1,w2,mats=make_world(seed,n)
        for method in METHODS:
            model,seconds=fit(method,h,y,updates,seed+len(method))
            raw,_=payload(model,w1,w2,method)
            path=out/f'dev{seed}_{method}.npz'; path.write_bytes(raw); digest=hashlib.sha256(raw).hexdigest(); artifacts[(seed,method)]=(raw,digest)
            # roundtrip uses loaded coefficients; compare saved metric to original metric
            replay=predict_from_payload(raw,x.numpy(),method)
            replay_t=torch.as_tensor(replay)
            hh=torch.relu(x @ torch.as_tensor(np.asarray(np.load(io.BytesIO(raw))['shared_w1'],dtype=np.float32))) @ torch.as_tensor(np.asarray(np.load(io.BytesIO(raw))['shared_w2'],dtype=np.float32))
            yy=y
            den=((yy-yy.mean(dim=0,keepdim=True))**2).mean(dim=(0,2)).sqrt().clamp_min(1e-8)
            err=((replay_t-yy)**2).mean(dim=(0,2)).sqrt()/den
            nrmse=err.reshape(5,8).mean(dim=1).numpy().tolist()
            with torch.no_grad():
                batch=h[:1024]; task_ids=torch.arange(40)
                for _ in range(10): model(batch,task_ids)
                start=time.perf_counter()
                for _ in range(25): model(batch,task_ids)
                rate=25*batch.shape[0]/max(time.perf_counter()-start,1e-9)
            op={'shared_identity':0,'film':32,'rank1':64,'mirror':48,'mirror_rank1':112,'independent':512}[method]
            for ai,alpha in enumerate(ALPHAS):
                rows.append({'seed':seed,'split':'development','alpha':alpha,'method':method,'nrmse':nrmse[ai],'payload_bytes':len(raw),'train_examples':updates*1024,'updates':updates,'train_wall_seconds':seconds,'inference_examples_per_second':rate,'active_ops_per_example':op,'disposition':'SCREENING','artifact_sha256':digest})
    import csv
    with (out/'RESULTS_CORE.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    # Fixed screen gate checks alpha=1 in both seeds.
    by={(r['seed'],r['method'],r['alpha']):r for r in rows}
    passed=True; why=[]
    for seed in seeds:
        mir=by[(seed,'mirror',1.0)]; film=by[(seed,'film',1.0)]; rank=by[(seed,'rank1',1.0)]; ind=by[(seed,'independent',1.0)]
        ok=(mir['nrmse']<=0.02 and mir['nrmse']<=0.1*film['nrmse'] and mir['nrmse']<=0.1*rank['nrmse'] and mir['payload_bytes']<=.75*film['payload_bytes'] and mir['payload_bytes']<=.1*ind['payload_bytes'] and mir['inference_examples_per_second']>=.8*film['inference_examples_per_second'])
        passed &= ok; why.append({'seed':seed,'passed':ok,'mirror_nrmse':mir['nrmse'],'film_nrmse':film['nrmse'],'rank1_nrmse':rank['nrmse'],'mirror_bytes':mir['payload_bytes'],'film_bytes':film['payload_bytes'],'independent_bytes':ind['payload_bytes'],'mirror_rate':mir['inference_examples_per_second'],'film_rate':film['inference_examples_per_second']})
    result={'experiment_id':'MA-401','protocol_frozen':True,'fresh_accessed':False,'development_gate_passed':bool(passed),'seed_results':why,'note':'Development screen only; no fresh data accessed.'}
    (out/'screen.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[1]/'runs');p.add_argument('--updates',type=int,default=400);p.add_argument('--n',type=int,default=4096)
    a=p.parse_args();print(json.dumps(run(a.out,updates=a.updates,n=a.n),indent=2))
