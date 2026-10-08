"""MA-921 synthetic Flow Map Matching endpoint-View mechanism screen."""
from __future__ import annotations
import argparse, hashlib, io, json, math, time
from pathlib import Path
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'source'
GRID_N=21
GRID=torch.linspace(0.0,1.0,GRID_N)
K_BASIS=4
DIM=2
HOLDOUT=[(0,20),(2,14),(4,16),(6,18),(1,10),(3,12),(5,15),(7,17),(9,20),(0,11),(8,19),(10,20)]
COMPOSITIONS=[(0,8,20),(2,8,14),(4,10,16),(6,12,18),(1,5,10),(3,8,12),(5,10,15),(7,12,17),(9,14,20),(0,6,11),(8,13,19),(10,15,20)]
ALL_PAIRS=[(i,j) for i in range(GRID_N) for j in range(i+1,GRID_N)]
HELD_SET=set(HOLDOUT)
TRAIN_PAIRS=[p for p in ALL_PAIRS if p not in HELD_SET]
PAIR_TO_ID={p:i for i,p in enumerate(ALL_PAIRS)}


def generator(seed:int):
    g=torch.Generator().manual_seed(seed)
    a0=torch.randn(DIM,DIM,generator=g,dtype=torch.float64)*0.22
    a1=torch.randn(DIM,DIM,generator=g,dtype=torch.float64)*0.22
    return a0,a1


def ode_matrix(a0,a1,t):
    return a0+math.sin(2*math.pi*t)*a1


def transition(a0,a1,s_idx,t_idx,step=0.002):
    """RK4 transition matrix for dP/dt=A(t)P, endpoints lie on step-aligned grid."""
    s=float(GRID[s_idx]);t=float(GRID[t_idx]);n=max(1,round((t-s)/step));h=(t-s)/n
    def f(time,p): return ode_matrix(a0,a1,time)@p
    p=torch.eye(DIM,dtype=torch.float64);now=s
    for _ in range(n):
        k1=f(now,p);k2=f(now+h/2,p+h*k1/2);k3=f(now+h/2,p+h*k2/2);k4=f(now+h,p+h*k3)
        p=p+h*(k1+2*k2+2*k3+k4)/6;now+=h
    return p.to(torch.float32)


_WORLD_CACHE={}
def make_world(seed:int):
    if seed not in _WORLD_CACHE:
        a0,a1=generator(seed)
        _WORLD_CACHE[seed]={pair:transition(a0,a1,*pair) for pair in ALL_PAIRS}
    return _WORLD_CACHE[seed]


def sample_pairs(seed:int,pairs,n_per_pair:int,maps):
    g=torch.Generator().manual_seed(seed)
    xs=[];ys=[];sids=[];tids=[];pids=[]
    for i,j in pairs:
        x=torch.randn(n_per_pair,DIM,generator=g)
        y=x@maps[(i,j)].T
        xs.append(x);ys.append(y);sids.extend([i]*n_per_pair);tids.extend([j]*n_per_pair);pids.extend([PAIR_TO_ID[(i,j)]]*n_per_pair)
    return {'x':torch.cat(xs),'y':torch.cat(ys),'s':torch.tensor(sids),'t':torch.tensor(tids),'pid':torch.tensor(pids)}


def time_features(sids,tids):
    s=GRID[sids];t=GRID[tids]
    return torch.stack([s,t,torch.sin(math.pi*s),torch.cos(math.pi*s),torch.sin(math.pi*t),torch.cos(math.pi*t)],dim=-1)

class NativeFMM(nn.Module):
    """Ordinary state- and two-time-conditioned Flow Map MLP."""
    def __init__(self):
        super().__init__();self.net=nn.Sequential(nn.Linear(DIM+6,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh(),nn.Linear(32,DIM))
    def forward(self,x,s,t,pid):
        return x+self.net(torch.cat([x,time_features(s,t)],dim=-1))

class SharedView(nn.Module):
    def __init__(self,method,rank,seed):
        super().__init__();torch.manual_seed(seed);self.method=method;self.rank=rank
        if method=='independent':
            self.interval_delta=nn.Parameter(torch.zeros(len(ALL_PAIRS),DIM,DIM))
        else:
            self.basis=nn.Parameter(torch.randn(K_BASIS,DIM,DIM)*0.04)
            if method=='native_basis':
                self.cond=nn.Sequential(nn.Linear(6,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh(),nn.Linear(32,K_BASIS))
            else:
                self.start=nn.Parameter(torch.randn(GRID_N,rank)*0.06)
                self.end=nn.Parameter(torch.randn(GRID_N,rank)*0.06)
                self.decode=nn.Parameter(torch.randn(rank,K_BASIS)*0.06)
    def forward(self,x,s,t,pid):
        if self.method=='independent':
            delta=self.interval_delta[pid]
        else:
            if self.method=='native_basis':
                coeff=self.cond(time_features(s,t))
            else:
                left=self.start[s];right=self.end[t]
                code=left+right if self.method=='additive' else left*right
                coeff=code@self.decode
            delta=torch.einsum('bk,kij->bij',coeff,self.basis)
        return x+torch.bmm(delta,x.unsqueeze(-1)).squeeze(-1)


def build_model(method,rank,seed):
    if method=='fmm':
        torch.manual_seed(seed);return NativeFMM()
    return SharedView(method,rank,seed)


def train_model(method,rank,seed,updates,train_seed):
    maps=make_world(seed);data=sample_pairs(train_seed,TRAIN_PAIRS,32,maps)
    model=build_model(method,rank,seed+train_seed+17000);opt=torch.optim.Adam(model.parameters(),lr=0.012)
    started=time.perf_counter()
    for _ in range(updates):
        opt.zero_grad(set_to_none=True);loss=(model(data['x'],data['s'],data['t'],data['pid'])-data['y']).square().mean();loss.backward();opt.step()
    return model,time.perf_counter()-started,len(data['x'])*updates,float(loss.detach())


def serialize(model,method,rank):
    payload={'metadata':{'format':'MA921-torch-v1','method':method,'rank':rank,'grid':GRID.tolist(),'heldout_pairs':HOLDOUT,'precision':'float32'},'state_dict':model.state_dict()}
    b=io.BytesIO();torch.save(payload,b);blob=b.getvalue()
    return blob,hashlib.sha256(blob).hexdigest()


def deserialize(blob):
    obj=torch.load(io.BytesIO(blob),map_location='cpu',weights_only=True)
    meta=obj['metadata'];model=build_model(meta['method'],meta['rank'],12345);model.load_state_dict(obj['state_dict']);model.eval()
    return model,meta


def macs_per_sample(method,rank):
    if method in ('fmm','native_basis'):
        output=DIM if method=='fmm' else K_BASIS
        return (DIM+6)*32+32*32+32*output + (0 if method=='fmm' else K_BASIS*DIM*DIM+K_BASIS*DIM)
    if method=='independent': return DIM*DIM
    coeff=rank*K_BASIS+(rank if method=='mirror' else 0)
    return K_BASIS*DIM*DIM+K_BASIS*DIM+coeff


def evaluate(model,method,world_seed,eval_seed):
    maps=make_world(world_seed)
    all_data=sample_pairs(eval_seed,ALL_PAIRS,64,maps)
    model.eval()
    with torch.inference_mode():
        started=time.perf_counter();pred=model(all_data['x'],all_data['s'],all_data['t'],all_data['pid']);infer_wall=time.perf_counter()-started
    err=(pred-all_data['y']).square().mean(dim=1)
    held_ids=torch.tensor([PAIR_TO_ID[p] for p in HOLDOUT]);held=torch.isin(all_data['pid'],held_ids)
    train_mask=~held
    held_mse=float(err[held].mean()) if method!='independent' else None
    seen_mse=float(err[train_mask].mean())
    # Semigroup and composition scores use only the predeclared triples whose component pairs were in training.
    comp_true=[];semigroup=[]
    with torch.inference_mode():
        for i,k,j in COMPOSITIONS:
            g=torch.Generator().manual_seed(eval_seed+1000+i*31+k*7+j)
            x=torch.randn(64,DIM,generator=g);target=x@maps[(i,j)].T
            ss=torch.full((64,),i,dtype=torch.long);kk=torch.full((64,),k,dtype=torch.long);tt=torch.full((64,),j,dtype=torch.long)
            pid_ik=torch.full((64,),PAIR_TO_ID[(i,k)]);pid_kj=torch.full((64,),PAIR_TO_ID[(k,j)]);pid_ij=torch.full((64,),PAIR_TO_ID[(i,j)])
            one=model(x,ss,kk,pid_ik);composed=model(one,kk,tt,pid_kj);direct=model(x,ss,tt,pid_ij)
            denom=float(target.square().mean())+1e-12
            comp_true.append(float((composed-target).square().mean())/denom)
            semigroup.append(float((composed-direct).square().mean())/denom)
    return {'heldout_map_mse':held_mse,'seen_map_mse':seen_mse,'composition_vs_true_normalized_mse':sum(comp_true)/len(comp_true),
            'semigroup_discrepancy_normalized_mse':sum(semigroup)/len(semigroup),'inference_wall_s':infer_wall,
            'eval_examples':len(all_data['x']),'nfe_direct':1,'nfe_composition':2}


def run(stage,updates,seeds,ranks):
    methods=['fmm','independent','native_basis']
    for rank in ranks:methods += [('additive',rank),('mirror',rank)]
    rows=[]
    for seed in seeds:
        for item in methods:
            method,rank=(item,4) if isinstance(item,str) else item
            model,wall,exposures,train_loss=train_model(method,rank,seed,updates,seed+1)
            metrics=evaluate(model,method,seed,seed+2)
            blob,digest=serialize(model,method,rank)
            restored,meta=deserialize(blob)
            if meta['method']!=method or not all(torch.equal(a,b) for a,b in zip(model.state_dict().values(),restored.state_dict().values())):
                raise RuntimeError('serialized flow model did not reconstruct exactly')
            if stage=='fresh':
                payload_dir=OUT/'payloads';payload_dir.mkdir(exist_ok=True)
                (payload_dir/f'fresh_{seed}_{method}_r{rank}.pt').write_bytes(blob)
            rows.append({'world_seed':seed,'method':method,'rank':rank,'updates':updates,'serialized_bytes':len(blob),'payload_sha256':digest,
                         'train_examples':exposures,'optimizer_updates':updates,
                         'train_macs_proxy':exposures*macs_per_sample(method,rank)*3,
                         'train_wall_s':wall,'inference_macs_proxy':metrics['eval_examples']*macs_per_sample(method,rank),
                         'train_loss':train_loss,**metrics})
    return rows


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['development','fresh'],required=True);ap.add_argument('--updates',type=int,default=200);args=ap.parse_args()
    if args.stage=='development':seeds=[9211,9212];ranks=[2,4,8];updates=args.updates
    else:
        frozen=json.loads((OUT/'frozen_config.json').read_text());seeds=frozen['fresh_seeds'];ranks=[frozen['selected_rank']];updates=frozen['updates']
    rows=run(args.stage,updates,seeds,ranks);out=OUT/f'{args.stage}_{updates}_updates.json'
    out.write_text(json.dumps({'stage':args.stage,'seeds':seeds,'rows':rows},indent=2)+'\n')
    print(json.dumps({'stage':args.stage,'updates':updates,'rows':len(rows),'output':str(out)},indent=2))
if __name__=='__main__':main()
