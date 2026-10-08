"""MA-840 stratified tangent-code and private nonlinear residual screen."""
from __future__ import annotations
import csv,hashlib,json,time
from pathlib import Path
import numpy as np,torch
from torch import nn
from safetensors.torch import save_file
ROOT=Path(__file__).resolve().parents[1];torch.set_num_threads(1)
D=16;O=8;R=2

def world(seed):
    rng=np.random.default_rng(seed);torch.manual_seed(seed)
    W0=rng.normal(0,.18,(O,D)).astype('float32')
    q,_=np.linalg.qr(rng.normal(size=(O*D,R)));B=q[:,:R].T.astype('float32')
    def task(hard,idx):
        c=rng.normal(0,.08,(R,)).astype('float32');delta=(c@B).reshape(O,D)
        if hard:
            u=rng.normal(size=O);u=u/(np.linalg.norm(u)+1e-9);v=rng.normal(size=D);v=v/(np.linalg.norm(v)+1e-9);delta+=.45*np.outer(u,v).astype('float32')
        xs=rng.normal(0,1,(32,D)).astype('float32');xq=rng.normal(0,1,(256,D)).astype('float32')
        return {'hard':hard,'code':c,'delta':delta,'support':xs,'query':xq,'id':idx}
    dev=[task(False,f'dev{i}') for i in range(4)]
    # Estimate the shared tangent basis only from development task deltas.
    mat=np.stack([t['delta'].reshape(-1) for t in dev]);_,_,vt=np.linalg.svd(mat,full_matrices=False);basis=vt[:R].astype('float32')
    fresh=[task(False,f'easy{i}') for i in range(4)]+[task(True,f'hard{i}') for i in range(4)]
    return W0,basis,dev,fresh

def infer_base(x,W):return torch.tanh(x@W.T)
def tangent(x,W,B,c):
    z=x@W.T;y0=torch.tanh(z);jac=1-y0*y0
    bb=torch.tensor(B,dtype=x.dtype);cc=c
    d=torch.einsum('r,rod,bd->bo',cc,bb.reshape(R,O,D),x)
    return y0+jac*d

def hybrid(x,W,B,c,u,v):
    y0=torch.tanh(x@W.T);j=1-y0*y0
    shared=torch.einsum('r,rod,bd->bo',c,torch.tensor(B).reshape(R,O,D),x)
    priv=(x@v[:,None])@u[None,:]
    exact_priv=torch.tanh(x@W.T+priv)-y0-j*priv
    return y0+j*shared+exact_priv

def target(x,W,delta):return torch.tanh(x@(W+delta).T)

def fit_code(x,y,W,B,steps=300):
    c=nn.Parameter(torch.zeros(R));opt=torch.optim.Adam([c],lr=.04);t0=time.perf_counter()
    for _ in range(steps):loss=((tangent(x,W,B,c)-y)**2).mean();opt.zero_grad();loss.backward();opt.step()
    return c.detach(),steps,time.perf_counter()-t0

def fit_hybrid(x,y,W,B,steps=800):
    c=nn.Parameter(torch.zeros(R));u=nn.Parameter(torch.randn(O)*.01);v=nn.Parameter(torch.randn(D)*.01);opt=torch.optim.Adam([c,u,v],lr=.025);t0=time.perf_counter()
    for _ in range(steps):
        loss=((hybrid(x,W,B,c,u,v)-y)**2).mean()+.0005*(u.square().sum()+v.square().sum());opt.zero_grad();loss.backward();opt.step()
    return c.detach(),u.detach(),v.detach(),steps,time.perf_counter()-t0

def fit_full(x,y,steps=1000):
    delta=nn.Parameter(torch.zeros(O,D));opt=torch.optim.Adam([delta],lr=.02);t0=time.perf_counter()
    for _ in range(steps):loss=((torch.tanh(x@(torch.tensor(W0_global)+delta).T)-y)**2).mean();opt.zero_grad();loss.backward();opt.step()
    return delta.detach(),steps,time.perf_counter()-t0

def ser(method,shared,states,path):
    data={}
    data['W0']=torch.tensor(shared[0])
    if method!='full':data['basis']=torch.tensor(shared[1])
    for i,s in enumerate(states):
      for k,v in s.items():data[f't{i}.{k}']=v.detach().cpu().contiguous()
    save_file(data,str(path));raw=path.read_bytes();return len(raw),hashlib.sha256(raw).hexdigest()

def run(seed):
    global W0_global
    W0,B,dev,tasks=world(seed);W0_global=torch.tensor(W0);Wt=torch.tensor(W0);results=[];states={'tangent':[],'hybrid':[],'full':[]};times={'tangent':[],'hybrid':[],'full':[]}
    for t in tasks:
      xs=torch.tensor(t['support']);xq=torch.tensor(t['query']);y=target(xs,W0,t['delta']);yq=target(xq,W0,t['delta'])
      c,n,tc=fit_code(xs,y,Wt,B);pt=tangent(xq,Wt,B,c);err=float(((pt-yq)**2).mean());states['tangent'].append({'code':c});times['tangent'].append(.0)
      results.append({'world':seed,'task':t['id'],'stratum':'high' if t['hard'] else 'low','method':'tangent','query_mse':err,'support_examples':len(xs)*n,'updates':n,'active_macs_per_example':2*R*O+2*O,'train_seconds':tc,'infer_us_per_example':0.0,'private_active_fraction':0.0})
      c2,u,v,nh,th=fit_hybrid(xs,y,Wt,B);ph=hybrid(xq,Wt,B,c2,u,v);eh=float(((ph-yq)**2).mean());states['hybrid'].append({'code':c2,'u':u,'v':v})
      results.append({'world':seed,'task':t['id'],'stratum':'high' if t['hard'] else 'low','method':'hybrid','query_mse':eh,'support_examples':len(xs)*nh,'updates':nh,'active_macs_per_example':2*R*O+2*O+O*D+O*D,'train_seconds':th,'infer_us_per_example':0.0,'private_active_fraction':1.0})
      delta,nu,tf=fit_full(xs,y);pf=torch.tanh(xq@(Wt+delta).T);ef=float(((pf-yq)**2).mean());states['full'].append({'delta':delta})
      results.append({'world':seed,'task':t['id'],'stratum':'high' if t['hard'] else 'low','method':'full','query_mse':ef,'support_examples':len(xs)*nu,'updates':nu,'active_macs_per_example':2*O*D,'train_seconds':tf,'infer_us_per_example':0.0,'private_active_fraction':1.0})
    # Report eager CPU inference per input example on each fitted task.
    for task in tasks:
      xq=torch.tensor(task['query']);i=int(task['id'][-1])+(0 if task['id'].startswith('easy') else 4)
      funcs={'tangent':lambda:tangent(xq,Wt,B,states['tangent'][i]['code']), 'hybrid':lambda:hybrid(xq,Wt,B,states['hybrid'][i]['code'],states['hybrid'][i]['u'],states['hybrid'][i]['v']), 'full':lambda:torch.tanh(xq@(Wt+states['full'][i]['delta']).T)}
      for m,fn in funcs.items():
        with torch.no_grad():
          for _ in range(4):fn()
          tic=time.perf_counter()
          for _ in range(50):fn()
        us=(time.perf_counter()-tic)/50/len(xq)*1e6
        for row in results:
          if row['task']==task['id'] and row['method']==m:row['infer_us_per_example']=us
    sizes={}
    for m in states:
      p=ROOT/'source'/f'.{m}-{seed}.safetensors';size,sha=ser(m,(W0,B),states[m],p);p.unlink();sizes[m]=(size,sha)
      for r in results:
       if r['method']==m:r['serialized_bytes']=size;r['payload_sha256']=sha
    return results,(W0,B,dev)

def main():
    cfg=json.loads((ROOT/'PROTOCOL.json').read_text());allrows=[];devmeta=[]
    for seed in cfg['fresh']['worlds_or_seeds']:
      rows,(W0,B,dev)=run(seed);allrows+=rows;devmeta.append({'seed':seed,'dev_task_ids':[t['id'] for t in dev],'basis_shape':list(B.shape),'basis_rank':int(np.linalg.matrix_rank(B))})
      for st in ['low','high']:
       print(seed,st,flush=True)
       for m in ['tangent','hybrid','full']:
        q=[r for r in rows if r['stratum']==st and r['method']==m];print(m,'mse',np.mean([r['query_mse'] for r in q]),'bytes',q[0]['serialized_bytes'],flush=True)
    (ROOT/'source'/'audit_results.json').write_text(json.dumps(allrows,indent=2)+'\n');(ROOT/'source'/'development_ids.json').write_text(json.dumps(devmeta,indent=2)+'\n')
    with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
if __name__=='__main__':main()
