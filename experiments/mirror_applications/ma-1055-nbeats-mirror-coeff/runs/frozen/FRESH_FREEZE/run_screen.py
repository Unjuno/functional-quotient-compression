#!/usr/bin/env python3
"""Frozen small CPU screen for MA-1055; standard library + numpy + torch."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn

HERE=Path(__file__).resolve().parents[1]; RAW=HERE/'runs/raw/ETTm1.csv'; RAW_AUDIT=HERE/'runs/raw/ETTm2.csv'; OUT=HERE/'runs'
sys.path.insert(0,str(Path(__file__).resolve().parent))
from nbeats_upstream import NBeats, NBeatsBlock, SeasonalityBasis, TrendBasis
torch.set_num_threads(1)
CTX,H,WIDTH=512,96,128
CHANNELS=('HUFL','HULL','MUFL','MULL')
METHODS=('native','shared','gain','common_shift','factor_phase','flat_phase')

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def load_data(include_audit=False):
    with RAW.open(newline='') as f: rows=list(csv.DictReader(f))
    dates=np.array([np.datetime64(z['date']) for z in rows]); n=len(rows); a,b=int(n*.6),int(n*.8)
    out={}
    for ci,col in enumerate(CHANNELS):
        vals=np.array([float(z[col]) for z in rows],np.float32); mu=float(vals[:a].mean()); sd=float(vals[:a].std()); z=(vals-mu)/(sd+1e-8)
        for name,(lo,hi) in {'train':(0,a),'dev':(a,b)}.items():
            # Fully disjoint target and context intervals; conservative same-split context requirement.
            starts=np.arange(lo+CTX,hi-H+1,H,dtype=np.int64)
            if name=='train':
                rng=np.random.RandomState(105501+ci); starts=np.sort(rng.choice(starts,min(96,len(starts)),replace=False))
            x=np.stack([z[s-CTX:s] for s in starts]); y=np.stack([z[s:s+H] for s in starts])
            weekday=np.array([int(np.datetime64(dates[s],'D').astype(object).weekday()>=5) for s in starts],np.int64)
            out[(ci,name)]={'x':x,'y':y,'starts':starts,'regime':weekday,'lo':lo,'hi':hi,
                'mase_scale':float(np.mean(np.abs(z[96:a]-z[:a-96])))}
    if include_audit:
        with RAW_AUDIT.open(newline='') as f: audit_rows=list(csv.DictReader(f))
        audit_dates=np.array([np.datetime64(q['date']) for q in audit_rows]); na=len(audit_rows); alo=int(na*.8)
        for ci,col in enumerate(CHANNELS):
            tr=np.array([float(q[col]) for q in rows],np.float32); mu=float(tr[:a].mean()); sd=float(tr[:a].std())
            vals=np.array([float(q[col]) for q in audit_rows],np.float32); z=(vals-mu)/(sd+1e-8)
            starts=np.arange(alo,na-H+1,H,dtype=np.int64)
            x=np.stack([z[s-CTX:s] for s in starts]); y=np.stack([z[s:s+H] for s in starts])
            weekday=np.array([int(np.datetime64(audit_dates[s],'D').astype(object).weekday()>=5) for s in starts],np.int64)
            trainz=(tr-mu)/(sd+1e-8)
            out[(ci,'audit')]={'x':x,'y':y,'starts':starts,'regime':weekday,'lo':alo,'hi':na,
                'mase_scale':float(np.mean(np.abs(trainz[96:a]-trainz[:a-96])))}
    return out,{'rows':n,'train_end':a,'dev_end':b,'channels':CHANNELS,'sha256':sha(RAW),
        'audit_dataset':str(RAW_AUDIT.name) if include_audit else None,
        'audit_sha256':sha(RAW_AUDIT) if include_audit else None}

def make_model(seed):
    torch.manual_seed(seed); blocks=[]
    for i in range(6):
        if i<3: basis,size=TrendBasis(2,CTX,H),6
        else:
            basis=SeasonalityBasis(3,CTX,H); size=4*basis.forecast_cos_template.shape[0]
        blocks.append(NBeatsBlock(CTX,size,basis,layers=2,layer_size=WIDTH))
    return NBeats(nn.ModuleList(blocks))

class RoleModel(nn.Module):
    def __init__(self,base,method):
        super().__init__(); self.base=base; self.method=method
        # Shared coefficients get only two coarse non-DC bands; role-indexed flat baseline stores
        # independent band phases. gain/common shift have matched compact state.
        if method in ('gain','factor_phase'):
            self.series_code=nn.Parameter(torch.zeros(4,2)); self.regime_code=nn.Parameter(torch.zeros(2,2))
        elif method=='common_shift': self.code=nn.Parameter(torch.zeros(2))
        elif method=='flat_phase': self.code=nn.Parameter(torch.zeros(8,2))
        self.freq_band=None

    def forward(self,x,channel,regime):
        residual=x.flip((1,)); mask=torch.ones_like(residual); forecast=x[:,-1:]
        for block in self.base.blocks:
            h=residual
            for layer in block.layers: h=torch.relu(layer(h))
            theta=block.basis_parameters(h)
            if isinstance(block.basis_function,SeasonalityBasis):
                basis=block.basis_function; k=basis.forecast_cos_template.shape[0]
                cf=theta[:,:k]; sf=theta[:,k:2*k]
                # The first harmonic is DC. Rotate paired coefficients exactly in the native
                # Fourier coordinate system; preserve all backcast parameters.
                freq=torch.arange(k,device=x.device); band=(freq >= max(1,k//2)).long()
                if self.method=='gain':
                    rolecode=self.series_code[channel]+self.regime_code[regime]
                    gain=.05*torch.tanh(rolecode[:,band])
                    cf=cf*(1+gain); sf=sf*(1+gain)
                elif self.method in ('common_shift','factor_phase','flat_phase'):
                    if self.method=='common_shift': delta=.25*torch.tanh(self.code).unsqueeze(0).expand(len(x),-1)
                    elif self.method=='factor_phase': delta=.25*torch.tanh(self.series_code[channel])+.25*torch.tanh(self.regime_code[regime])
                    else: delta=.25*torch.tanh(self.code[channel*2+regime])
                    angle=delta[:,band]*2*math.pi
                    cf,sf=cf*torch.cos(angle)-sf*torch.sin(angle),cf*torch.sin(angle)+sf*torch.cos(angle)
                theta=torch.cat((cf,sf,theta[:,2*k:]),dim=-1)
            backcast,block_forecast=block.basis_function(theta)
            residual=(residual-backcast)*mask; forecast=forecast+block_forecast
        return forecast

def payload_bytes(model,method,seed):
    import io
    b=io.BytesIO(); torch.save({'state_dict':model.state_dict(),'method':method,'seed':seed,
        'data_sha256':sha(RAW),'protocol_sha256':sha(HERE/'PROTOCOL.json')},b)
    return len(b.getvalue())

def run_one(method,seed,data,eval_split):
    start=time.perf_counter(); models=[]
    if method=='native':
        # Independent official models per physical series; each fit uses only that series' train windows.
        for ci in range(4):
            model=RoleModel(make_model(seed+ci),method); d=data[(ci,'train')]
            x=torch.tensor(d['x'],dtype=torch.float32); y=torch.tensor(d['y'],dtype=torch.float32)
            c=torch.full((len(x),),ci,dtype=torch.long); r=torch.tensor(d['regime'],dtype=torch.long)
            opt=torch.optim.Adam(model.parameters(),lr=1e-3); gen=torch.Generator().manual_seed(seed+ci)
            for step in range(300):
                idx=torch.randint(len(x),(32,),generator=gen); opt.zero_grad(set_to_none=True)
                loss=((model(x[idx],c[idx],r[idx])-y[idx])**2).mean(); loss.backward(); opt.step()
            models.append(model)
    else:
        model=RoleModel(make_model(seed),method)
        xs=[]; ys=[]; cs=[]; rs=[]
        for ci in range(4):
            d=data[(ci,'train')]; xs.append(d['x']); ys.append(d['y']); cs.extend([ci]*len(d['x'])); rs.extend(d['regime'].tolist())
        x=torch.tensor(np.concatenate(xs),dtype=torch.float32); y=torch.tensor(np.concatenate(ys),dtype=torch.float32)
        c=torch.tensor(cs,dtype=torch.long); r=torch.tensor(rs,dtype=torch.long)
        opt=torch.optim.Adam(model.parameters(),lr=1e-3); gen=torch.Generator().manual_seed(seed)
        for step in range(300):
            idx=torch.randint(len(x),(32,),generator=gen); opt.zero_grad(set_to_none=True)
            loss=((model(x[idx],c[idx],r[idx])-y[idx])**2).mean(); loss.backward(); opt.step()
        models=[model]
    train_seconds=time.perf_counter()-start
    results=[]
    for ci in range(4):
        for split in (eval_split,):
            d=data[(ci,split)]; xx=torch.tensor(d['x']); rr=torch.tensor(d['regime']); cc=torch.full((len(xx),),ci,dtype=torch.long)
            with torch.no_grad(): pred=models[ci if method=='native' else 0](xx,cc,rr).numpy()
            mase=float(np.mean(np.abs(pred-d['y']))/d['mase_scale'])
            results.append({'channel':CHANNELS[ci],'split':split,'mase':mase,'n':len(xx)})
    import io
    payload={'models':[m.state_dict() for m in models],'method':method,'seed':seed,
        'data_sha256':sha(RAW),'audit_data_sha256':sha(RAW_AUDIT) if eval_split=='audit' else None,
        'protocol_sha256':sha(HERE/'PROTOCOL.json'),'channel_order':CHANNELS,
        'coordinate_shape':list(models[0].state_dict().get('series_code',torch.empty(0)).shape)}
    buf=io.BytesIO(); torch.save(payload,buf); payload_path=OUT/eval_split/'checkpoints'/f'{method}_seed{seed}.pt'
    payload_path.parent.mkdir(parents=True,exist_ok=True); payload_path.write_bytes(buf.getvalue())
    payload_bytes=payload_path.stat().st_size
    loaded=torch.load(payload_path,map_location='cpu',weights_only=False)
    assert loaded['method']==method and loaded['seed']==seed and len(loaded['models'])==len(models)
    for model,state in zip(models,loaded['models']): model.load_state_dict(state)
    return {'method':method,'seed':seed,'metrics':results,'payload_bytes':payload_bytes,
        'payload_path':str(payload_path.relative_to(HERE)),'payload_sha256':sha(payload_path),
        'updates':300*len(models),'examples':32*300*len(models),'wall_seconds':train_seconds,
        'active_compute_proxy':32*300*len(models)*6*2*WIDTH*CTX}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--split',choices=['dev','audit'],default='dev')
    p.add_argument('--seeds',nargs='+',type=int,default=[105501]); p.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS))
    a=p.parse_args(); data,meta=load_data(include_audit=a.split=='audit')
    for (_,split),d in data.items():
        assert np.all(d['starts']>=d['lo']+CTX) if split!='audit' else np.all(d['starts']>=d['lo'])
        assert np.all(d['starts']+H<=d['hi'])
    out=[]
    for seed in a.seeds:
        for method in a.methods: out.append(run_one(method,seed,data,a.split))
    folder=OUT/a.split; folder.mkdir(parents=True,exist_ok=True)
    f=folder/'results.json'; f.write_text(json.dumps({'metadata':meta,'split':a.split,'results':out},indent=2)+'\n')
    print(json.dumps({'output':str(f),'results':out},indent=2))
if __name__=='__main__': main()
