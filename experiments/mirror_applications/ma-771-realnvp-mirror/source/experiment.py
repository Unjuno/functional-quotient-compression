"""MA-771: shared invertible chart plus per-task latent activation views."""
from __future__ import annotations
import csv, hashlib, json, os, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from safetensors.torch import save_file
ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(1)

def chart(x,a,b):
    s=a*torch.sin(x[...,0]);t=b*torch.tanh(x[...,0])
    return torch.stack([x[...,0],x[...,1]*torch.exp(s)+t],-1)
def chart_inv(y,a,b):
    s=a*torch.sin(y[...,0]);t=b*torch.tanh(y[...,0])
    return torch.stack([y[...,0],(y[...,1]-t)*torch.exp(-s)],-1)
def chart_logdet(x,a):return a*torch.sin(x[...,0])
def view(x,a,b,m):return chart_inv(chart(x,a,b)+m,a,b)
def view_inv(y,a,b,m):return chart_inv(chart(y,a,b)-m,a,b)
def view_logdet(x,a,b,m):
    y=view(x,a,b,m)
    return chart_logdet(x,a)-chart_logdet(y,a)

def gen(seed,ntasks=8,nsupport=8,nquery=256):
    rng=np.random.default_rng(seed);tasks=[]
    for _ in range(ntasks):
        m=rng.uniform(-.8,.8,size=(2,)).astype('float32')
        xs=rng.uniform(-2,2,size=(nsupport,2)).astype('float32');xq=rng.uniform(-2,2,size=(nquery,2)).astype('float32')
        tasks.append((m,xs,xq))
    return tasks

def fit_chart(seed,steps=500):
    torch.manual_seed(seed);np.random.seed(seed);dev=gen(seed,ntasks=8,nsupport=64,nquery=32)
    a=nn.Parameter(torch.tensor(.2));b=nn.Parameter(torch.tensor(.1));codes=nn.Parameter(torch.zeros(8,2))
    opt=torch.optim.Adam([a,b,codes],lr=.025);start=time.perf_counter();updates=0;examples=0
    targets=[]
    for i,(m,xs,xq) in enumerate(dev):
        x=torch.tensor(xs);y=view(x,torch.tensor(.7),torch.tensor(.4),torch.tensor(m));targets.append((x,y))
    for _ in range(steps):
        ids=np.random.randint(0,len(dev),size=32);ls=[]
        for i in ids:
            x,y=targets[i];ix=torch.randint(len(x),(4,));pred=view(x[ix],a,b,codes[i]);ls.append(((pred-y[ix])**2).mean())
        loss=torch.stack(ls).mean();opt.zero_grad();loss.backward();opt.step();updates+=1;examples+=128
    elapsed=time.perf_counter()-start
    return float(a.detach()),float(b.detach()),codes.detach().numpy(),updates,examples,elapsed

class Affine(nn.Module):
    def __init__(self):super().__init__();self.W=nn.Parameter(torch.eye(2));self.bias=nn.Parameter(torch.zeros(2))
    def forward(self,x):return x@self.W.T+self.bias
class Shift(nn.Module):
    def __init__(self):super().__init__();self.bias=nn.Parameter(torch.zeros(2))
    def forward(self,x):return x+self.bias
class ResidualMLP(nn.Module):
    def __init__(self):super().__init__();self.net=nn.Sequential(nn.Linear(2,16),nn.Tanh(),nn.Linear(16,2))
    def forward(self,x):return x+self.net(x)

def fit_model(model,x,y,steps,lr=.025):
    opt=torch.optim.Adam(model.parameters(),lr=lr);start=time.perf_counter();n=0
    for _ in range(steps):
        ix=torch.randint(len(x),(min(8,len(x)),));loss=((model(x[ix])-y[ix])**2).mean();opt.zero_grad();loss.backward();opt.step();n+=1
    return n,time.perf_counter()-start

def payload(method,shared,states,path):
    data={}
    if method=='mirror':data.update({'chart_alpha':torch.tensor([shared[0]]),'chart_beta':torch.tensor([shared[1]])})
    for i,state in enumerate(states):
        for k,v in state.items():data[f'task{i}.{k}']=v.detach().cpu().contiguous()
    save_file(data,str(path));raw=path.read_bytes();return len(raw),hashlib.sha256(raw).hexdigest()

def evaluate(seed,shared,mode='audit'):
    torch.manual_seed(seed);np.random.seed(seed);tasks=gen(seed+9000,ntasks=6,nsupport=8,nquery=256);methods=['mirror','shift','affine','mlp','independent']
    outputs=[];models={m:[] for m in methods};t0=time.perf_counter()
    for ti,(m,xs,xq) in enumerate(tasks):
        xt=torch.tensor(xs);xq=torch.tensor(xq);mt=torch.tensor(m);ys=view(xt,torch.tensor(.7),torch.tensor(.4),mt);yq=view(xq,torch.tensor(.7),torch.tensor(.4),mt)
        # Mirror: fixed shared chart, optimize only the task address from support.
        code=nn.Parameter(torch.zeros(2));opt=torch.optim.Adam([code],lr=.035);start=time.perf_counter()
        for _ in range(250):
            ix=torch.randint(len(xt),(8,));loss=((view(xt[ix],torch.tensor(shared[0]),torch.tensor(shared[1]),code)-ys[ix])**2).mean();opt.zero_grad();loss.backward();opt.step()
        mirror_s=time.perf_counter()-start;pm=view(xq,torch.tensor(shared[0]),torch.tensor(shared[1]),code.detach());mr=float(((pm-yq)**2).mean());cycle=float((view_inv(pm,torch.tensor(shared[0]),torch.tensor(shared[1]),code.detach())-xq).abs().max());lderr=float((view_logdet(xq,torch.tensor(shared[0]),torch.tensor(shared[1]),code.detach())+view_logdet(pm,torch.tensor(shared[0]),torch.tensor(shared[1]),-code.detach())).abs().max())
        models['mirror'].append({'code':code.detach()});outputs.append({'seed':seed,'task':ti,'method':'mirror','query_mse':mr,'inverse_cycle_max':cycle,'logdet_inverse_error':lderr,'support_examples':2000,'updates':250,'train_seconds':mirror_s,'infer_us_per_example':0.0,'active_macs_per_example':44})
        for name,cls,steps in [('shift',Shift,250),('affine',Affine,400),('mlp',ResidualMLP,600)]:
            model=cls();u,sec=fit_model(model,xt,ys,steps);pred=model(xq);err=float(((pred-yq)**2).mean());models[name].append(model.state_dict());outputs.append({'seed':seed,'task':ti,'method':name,'query_mse':err,'inverse_cycle_max':None,'logdet_inverse_error':None,'support_examples':u*8,'updates':u,'train_seconds':sec,'infer_us_per_example':0.0,'active_macs_per_example':({'shift':2,'affine':8,'mlp':130}[name])})
        # Independent full flow: each task pays its own chart and code.
        aa=nn.Parameter(torch.tensor(.2));bb=nn.Parameter(torch.tensor(.1));cc=nn.Parameter(torch.zeros(2));opt=torch.optim.Adam([aa,bb,cc],lr=.025);start=time.perf_counter()
        for _ in range(500):
            ix=torch.randint(len(xt),(8,));loss=((view(xt[ix],aa,bb,cc)-ys[ix])**2).mean();opt.zero_grad();loss.backward();opt.step()
        sec=time.perf_counter()-start;pred=view(xq,aa.detach(),bb.detach(),cc.detach());err=float(((pred-yq)**2).mean());cy=float((view_inv(pred,aa.detach(),bb.detach(),cc.detach())-xq).abs().max());lde=float((view_logdet(xq,aa.detach(),bb.detach(),cc.detach())+view_logdet(pred,aa.detach(),bb.detach(),-cc.detach())).abs().max());models['independent'].append({'alpha':aa.detach(),'beta':bb.detach(),'code':cc.detach()});outputs.append({'seed':seed,'task':ti,'method':'independent','query_mse':err,'inverse_cycle_max':cy,'logdet_inverse_error':lde,'support_examples':4000,'updates':500,'train_seconds':sec,'infer_us_per_example':0.0,'active_macs_per_example':44})
    # Measure batched eager query runtime; warm up then average 100 forwards.
    for name in methods:
        selected=[x for x in outputs if x['method']==name]
        for i,row in enumerate(selected):
            x=torch.tensor(tasks[i][2])
            if name=='mirror': fn=lambda: view(x,torch.tensor(shared[0]),torch.tensor(shared[1]),models[name][i]['code'])
            elif name=='independent': fn=lambda: view(x,models[name][i]['alpha'],models[name][i]['beta'],models[name][i]['code'])
            else:
                model={'shift':Shift,'affine':Affine,'mlp':ResidualMLP}[name]();model.load_state_dict(models[name][i]);model.eval();fn=lambda: model(x)
            with torch.no_grad():
                for _ in range(5):fn()
                start=time.perf_counter()
                for _ in range(100):fn()
            row['infer_us_per_example']=(time.perf_counter()-start)/100/len(x)*1e6
    # Serialize task payloads, charging shared chart once for Mirror.
    all_sizes={}
    for name in methods:
        path=ROOT/'source'/f'.{name}-{seed}.safetensors';shared_pair=shared if name=='mirror' else (0.,0.)
        size,digest=payload(name,shared_pair,models[name],path);path.unlink();all_sizes[name]=(size,digest)
        for row in outputs:
            if row['method']==name:row['serialized_bytes']=size;row['payload_sha256']=digest
    return outputs,all_sizes

def main():
    cfg=json.loads((ROOT/'PROTOCOL.json').read_text());dev_seed=cfg['development']['worlds_or_seeds'][0]
    shared=fit_chart(dev_seed)
    (ROOT/'source'/'development_fit.json').write_text(json.dumps({'seed':dev_seed,'chart_alpha':shared[0],'chart_beta':shared[1],'task_codes':shared[2].tolist(),'updates':shared[3],'examples':shared[4],'train_seconds':shared[5]},indent=2)+'\n')
    seeds=cfg['fresh']['worlds_or_seeds'];rows=[]
    for seed in seeds:
        out,_=evaluate(seed,shared[:2]);rows.extend(out)
        print(seed,'shared',shared[:2],flush=True)
        for m in ['mirror','shift','affine','mlp','independent']:
            q=[r for r in out if r['method']==m];print(m,'MSE',float(np.mean([r['query_mse'] for r in q])),'bytes',q[0]['serialized_bytes'],'cycle',max([r['inverse_cycle_max'] or 0 for r in q]),'s',round(float(np.mean([r['train_seconds'] for r in q])),3),flush=True)
    (ROOT/'source'/'audit_results.json').write_text(json.dumps(rows,indent=2)+'\n')
    with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print('development_shared',shared[:2],'updates',shared[3],'examples',shared[4],'seconds',shared[5])
if __name__=='__main__':
    import numpy as np
    main()
