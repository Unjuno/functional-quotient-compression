#!/usr/bin/env python3
"""Pinned DLinear plus compact periodic coordinate experiment."""
import argparse, csv, gzip, hashlib, json, math, random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from types import SimpleNamespace

HERE=Path(__file__).resolve().parents[1]; RAW=HERE/'runs/raw/exchange_rate.txt.gz'; OUT=HERE/'runs'
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from DLinear_upstream import Model as DLinear, series_decomp
torch.set_num_threads(1)
NAMES=('Australia','Britain','Canada','Switzerland','China','Japan','NewZealand','Singapore')
PERIOD,H=252,96
METHODS=('dlinear','fixed','gain','mirror','free_fourier','independent')

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load_data(include_audit=False):
    n=7588; ntr,ndev=int(n*.6),int(n*.8)
    with gzip.open(RAW,'rt') as f:
        lines=[]
        for i,line in enumerate(f):
            if not include_audit and i>=ndev: break
            lines.append(line)
    vals=np.loadtxt(lines,delimiter=',',dtype=np.float32)
    nloaded,c=vals.shape; assert c==8 and nloaded==(n if include_audit else ndev)
    mu=vals[:ntr].mean(0); sd=vals[:ntr].std(0)+1e-8; z=(vals-mu)/sd
    scale=np.mean(np.abs(z[PERIOD:ntr]-z[:ntr-PERIOD]),axis=0)
    data={'z':z,'ntrain':ntr,'ndev':ndev,'scale':scale}
    ranges=[('train',0,ntr),('dev',ntr,ndev)]
    if include_audit: ranges.append(('audit',ndev,n))
    for split,lo,hi in ranges:
        starts=np.arange(max(PERIOD,lo),hi-H+1,1 if split=='train' else H,dtype=np.int64)
        data[split]={'starts':starts,'lo':lo,'hi':hi}
    return data,{'rows':n,'channels':c,'train_end':ntr,'dev_end':ndev,'audit_end':n,
        'sha256_gzip':sha(RAW),'normalization_mean':mu.tolist(),'normalization_std':sd.tolist(),
        'mase_seasonal_scale':scale.tolist()}

def make_dlinear(ctx,individual=False,channels=8):
    cfg=SimpleNamespace(seq_len=ctx,pred_len=H,individual=individual,enc_in=channels)
    return DLinear(cfg)

class ViewModel(nn.Module):
    def __init__(self,base,mode):
        super().__init__(); self.base=base; self.mode=mode; self.decomp=series_decomp(25)
        if mode in ('gain','mirror'): self.code=nn.Parameter(torch.zeros(8))
        elif mode=='free_fourier': self.code=nn.Parameter(torch.zeros(8,2))
    def forward(self,x):
        y=self.base(x)
        if self.mode in ('dlinear','independent'): return y
        resid,_=self.decomp(x); b,l,c=resid.shape
        t=torch.arange(-l,0,device=x.device,dtype=x.dtype)
        omega=2*math.pi/PERIOD
        co=torch.cos(omega*t); si=torch.sin(omega*t)
        a=2.0/l*torch.einsum('blc,l->bc',resid,co)
        bb=2.0/l*torch.einsum('blc,l->bc',resid,si)
        future=torch.arange(1,H+1,device=x.device,dtype=x.dtype)*omega
        c0=torch.cos(future)[None,:,None]; s0=torch.sin(future)[None,:,None]
        if self.mode=='fixed': phase=torch.zeros(c,device=x.device,dtype=x.dtype)
        elif self.mode=='mirror': phase=math.pi*torch.tanh(self.code)
        elif self.mode=='gain':
            seasonal=a[:,None,:]*c0+bb[:,None,:]*s0
            return y+seasonal*(1.0+0.5*torch.tanh(self.code)[None,None,:])
        elif self.mode=='free_fourier':
            return y+self.code[:,0][None,None,:]*c0+self.code[:,1][None,None,:]*s0
        angle=future[None,:,None]+phase[None,None,:]
        return y+a[:,None,:]*torch.cos(angle)+bb[:,None,:]*torch.sin(angle)

def tensors(data,split,ctx):
    z=data['z']; starts=data[split]['starts']; starts=starts[starts>=ctx]
    x=np.stack([z[s-ctx:s] for s in starts]); y=np.stack([z[s:s+H] for s in starts])
    assert np.all(starts>=data[split]['lo']) if split!='train' else np.all(starts>=ctx)
    assert np.all(starts+H<=data[split]['hi'])
    return torch.tensor(x),torch.tensor(y),starts

def fit(method,seed,ctx,lr,data,updates=300):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    tr_x,tr_y,_=tensors(data,'train',ctx); gen=torch.Generator().manual_seed(seed)
    if method=='independent':
        models=[]; wall=time.perf_counter()
        for ch in range(8):
            m=ViewModel(make_dlinear(ctx,True,1),'independent'); opt=torch.optim.Adam(m.parameters(),lr=lr)
            for _ in range(updates):
                ix=torch.randint(len(tr_x),(32,),generator=gen); opt.zero_grad(set_to_none=True)
                loss=((m(tr_x[ix,:,ch:ch+1])-tr_y[ix,:,ch:ch+1])**2).mean(); loss.backward(); opt.step()
            models.append(m)
        return models,time.perf_counter()-wall,updates*8,updates*32*8
    model=ViewModel(make_dlinear(ctx),method); opt=torch.optim.Adam(model.parameters(),lr=lr); wall=time.perf_counter()
    for _ in range(updates):
        ix=torch.randint(len(tr_x),(32,),generator=gen); opt.zero_grad(set_to_none=True)
        loss=((model(tr_x[ix])-tr_y[ix])**2).mean(); loss.backward(); opt.step()
    return [model],time.perf_counter()-wall,updates,updates*32

def metric(pred,y,scale):
    return np.mean(np.mean(np.abs(pred-y),axis=(0,1))/scale)

def evaluate(models,method,seed,ctx,split,lr,data,protocol_hash):
    x,y,starts=tensors(data,split,ctx); results=[]
    reps=10; start=time.perf_counter()
    with torch.no_grad():
        for _ in range(reps):
            if method=='independent': pred=torch.cat([m(x[:,:,i:i+1]) for i,m in enumerate(models)],dim=-1)
            else: pred=models[0](x)
    latency=(time.perf_counter()-start)*1000/(reps*len(x))
    pred=pred.numpy(); yy=y.numpy(); m=metric(pred,yy,data['scale'])
    import io
    payload={'states':[m.state_dict() for m in models],'method':method,'seed':seed,'context':ctx,'horizon':H,
        'annual_period':PERIOD,'learning_rate':lr,'protocol_sha256':protocol_hash,'data_sha256':sha(RAW),
        'series_order':NAMES,'decoder':'pinned DLinear plus source/run_screen.py periodic operator'}
    buf=io.BytesIO(); torch.save(payload,buf); path=OUT/split/'checkpoints'/f'{method}_seed{seed}.pt'; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(buf.getvalue())
    return {'method':method,'seed':seed,'split':split,'macro_mase':float(m),'per_series_mase':(np.mean(np.abs(pred-yy),axis=(0,1))/data['scale']).tolist(),
        'target_windows':len(x),'target_start_min':int(starts.min()),'target_end_max_exclusive':int(starts.max()+H),
        'payload_bytes':path.stat().st_size,'payload_path':str(path.relative_to(HERE)),'payload_sha256':sha(path),
        'updates':(300*8 if method=='independent' else 300),'examples':(32*300*8 if method=='independent' else 32*300),
        'active_compute_proxy':int((32*300*8 if method=='independent' else 32*300)*(2*ctx*H*8 + (ctx*8 if method in ('fixed','gain','mirror','free_fourier') else 0))),
        'train_wall_seconds':None,'inference_ms_per_window':float(latency)}

def tune_dev(data):
    records=[]
    for ctx in (252,504):
        for lr in (.001,.005):
            models,wall,updates,examples=fit('dlinear',105601,ctx,lr,data,updates=100)
            # Development metrics, and only development metrics, select the DLinear settings.
            x,y,_=tensors(data,'dev',ctx)
            with torch.no_grad(): pred=models[0](x).numpy()
            score=metric(pred,y.numpy(),data['scale'])
            records.append({'context':ctx,'learning_rate':lr,'dev_macro_mase':float(score),'screen_wall_seconds':wall})
    best=min(records,key=lambda q:q['dev_macro_mase'])
    # Tune the last-value/annual seasonal naive blend on same-context development windows.
    ctx=best['context']; x,y,_=tensors(data,'dev',ctx); xx=x.numpy(); yy=y.numpy()
    naive=np.repeat(xx[:,-1:,:],H,axis=1); annual=np.stack([xx[:,(-PERIOD+j)%ctx,:] for j in range(H)],axis=1)
    blend=[]
    for alpha in np.linspace(0,1,21): blend.append((metric(alpha*annual+(1-alpha)*naive,yy,data['scale']),float(alpha)))
    best_alpha=min(blend)[1]
    return {'dlinear_grid':records,'selected_context':ctx,'selected_learning_rate':best['learning_rate'],
        'selected_dev_mase':best['dev_macro_mase'],'naive_blend_alpha_annual':best_alpha,
        'naive_dev_mase':min(blend)[0],'seed':105601,'locked':True}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--split',choices=('dev','audit'),default='dev')
    p.add_argument('--tune',action='store_true'); p.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS))
    p.add_argument('--seeds',nargs='+',type=int); p.add_argument('--context',type=int); p.add_argument('--lr',type=float)
    a=p.parse_args(); data,meta=load_data(include_audit=(a.split=='audit')); OUT.mkdir(exist_ok=True)
    if a.tune:
        assert a.split=='dev'; select=tune_dev(data); (OUT/'dev').mkdir(exist_ok=True)
        (OUT/'dev/tuning.json').write_text(json.dumps({'metadata':meta,'selection':select},indent=2)+'\n'); print(json.dumps(select,indent=2)); return
    if a.split=='audit': select=json.loads((OUT/'dev/tuning.json').read_text())['selection']; ctx=select['selected_context']; lr=select['selected_learning_rate']
    else: ctx=a.context or 252; lr=a.lr or .001
    if a.context: ctx=a.context
    if a.lr: lr=a.lr
    seeds=a.seeds or ([105601] if a.split=='dev' else [105602,105603])
    assert ctx in (252,504) and lr in (.001,.005)
    results=[]; protocol_hash=sha(HERE/'PROTOCOL.json')
    for seed in seeds:
        for method in a.methods:
            models,wall,updates,examples=fit(method,seed,ctx,lr,data)
            row=evaluate(models,method,seed,ctx,a.split,lr,data,protocol_hash); row['train_wall_seconds']=wall; row['updates']=updates; row['examples']=examples
            results.append(row)
        # Seasonal naive controls are deterministic and receive equal context and target origins.
        x,y,starts=tensors(data,a.split,ctx); xx=x.numpy(); yy=y.numpy()
        naive=np.repeat(xx[:,-1:,:],H,axis=1); annual=np.stack([xx[:,(-PERIOD+j)%ctx,:] for j in range(H)],axis=1)
        alpha=float(json.loads((OUT/'dev/tuning.json').read_text())['selection']['naive_blend_alpha_annual']) if a.split=='audit' else 0.0
        for name,pred in [('last_value_naive',naive),('annual_seasonal_naive',annual),('tuned_naive_blend',alpha*annual+(1-alpha)*naive)]:
            results.append({'method':name,'seed':seed,'split':a.split,'macro_mase':float(metric(pred,yy,data['scale'])),
                'per_series_mase':(np.mean(np.abs(pred-yy),axis=(0,1))/data['scale']).tolist(),'target_windows':len(x),
                'target_start_min':int(starts.min()),'target_end_max_exclusive':int(starts.max()+H),'payload_bytes':0,
                'updates':0,'examples':0,'active_compute_proxy':int(len(x)*H*8),'train_wall_seconds':0.0,
                'inference_ms_per_window':0.0})
    folder=OUT/a.split; folder.mkdir(exist_ok=True)
    out={'metadata':meta,'split':a.split,'context':ctx,'learning_rate':lr,'results':results}
    (folder/'results.json').write_text(json.dumps(out,indent=2)+'\n'); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
