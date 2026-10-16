"""GIC002: independent batched replicas of a two-expert nonlinear Mirror bank.
All training happens on CPU. No hidden regime labels enter non-oracle training.
"""
from __future__ import annotations
import argparse, hashlib, json, math, struct, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.special import ndtr

ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/'PROTOCOL.json').read_text())
D,H=5,16
HEADER=768

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def rotate(x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Adjacent-plane rotations; a omits sample axis (axis1)."""
    q=x.shape[-1]//2
    a=F.pad(a,(0,q-a.shape[-1]))
    c,s=a.cos().unsqueeze(1),a.sin().unsqueeze(1)
    z=x.reshape(*x.shape[:-1],q,2);u,v=z[...,0],z[...,1]
    return torch.stack((c*u-s*v,s*u+c*v),dim=-1).reshape_as(x)

class Bank(nn.Module):
    def __init__(self,kind: str,q: int,seeds: list[int],zero_route: bool=False):
        super().__init__()
        if kind not in ('mirror','gain','independent') or not 1<=q<=8 or not seeds:
            raise ValueError('invalid bank configuration')
        self.kind,self.q,self.seeds=kind,q,list(seeds)
        arr={k:[] for k in ['w1','b1','w2','b2','router','rb']}
        if kind!='independent': arr['codes']=[]
        for seed in seeds:
            base=np.random.default_rng(100000+seed)
            code=np.random.default_rng(200000+seed)
            route=np.random.default_rng(300000+seed)
            w1=base.normal(size=(D,H))/math.sqrt(D)
            b1=base.normal(size=H)*.1
            w2=base.normal(size=H)/math.sqrt(H)
            b2=np.zeros(())
            if kind=='independent':
                w1=np.repeat(w1[None],2,0)+code.normal(size=(2,D,H))*.01
                b1=np.repeat(b1[None],2,0)+code.normal(size=(2,H))*.01
                w2=np.repeat(w2[None],2,0)+code.normal(size=(2,H))*.01
                b2=np.zeros(2)
            else: arr['codes'].append(code.normal(size=(2,8))[:,:q]*.1)
            for k,v in [('w1',w1),('b1',b1),('w2',w2),('b2',b2)]: arr[k].append(v)
            arr['router'].append(route.normal(size=6)*(.0 if zero_route else .3))
            arr['rb'].append(np.array(route.normal()*(.0 if zero_route else .3)))
        for k,values in arr.items(): setattr(self,k,nn.Parameter(torch.tensor(np.stack(values),dtype=torch.float32)))

    def forward(self,x: torch.Tensor,context: torch.Tensor,uniform: bool=False,oracle: torch.Tensor|None=None):
        s=len(self.seeds)
        if self.kind=='independent':
            z=torch.einsum('nd,sedh->sneh',x,self.w1)+self.b1[:,None]
            experts=(F.gelu(z)*self.w2[:,None]).sum(-1)+self.b2[:,None]
        else:
            z=torch.einsum('nd,sdh->snh',x,self.w1)+self.b1[:,None]
            both=z[:,:,None,:].expand(-1,-1,2,-1)
            if self.kind=='mirror':
                hidden=rotate(F.gelu(rotate(both,self.codes)),-self.codes)
            else:
                gains=F.pad(self.codes,(0,8-self.q)).exp().repeat_interleave(2,-1)
                hidden=F.gelu(both)*gains[:,None]
            experts=(hidden*self.w2[:,None,None,:]).sum(-1)+self.b2[:,None,None]
        if oracle is not None:
            p=oracle.to(x.dtype).unsqueeze(0).expand(s,-1)
        elif uniform:
            p=torch.full((s,len(x)),.5,dtype=x.dtype,device=x.device)
        else:
            p=torch.sigmoid(torch.einsum('nd,sd->sn',context,self.router)+self.rb[:,None])
        return experts[:,:,0]*(1-p)+experts[:,:,1]*p,p,experts

def world(seed: int,family: str) -> dict[str,np.ndarray]:
    if family not in ('aligned','unrelated'): raise ValueError(family)
    rng=np.random.default_rng(seed)
    w=rng.normal(size=(2,D,H))*1.3/math.sqrt(D);b=rng.normal(size=(2,H))*.2
    v=rng.normal(size=(2,H))/math.sqrt(H);bias=rng.normal(size=2)*.1
    angles=rng.uniform(-2.5,2.5,size=(2,8))
    if family=='aligned': w[1]=w[0];b[1]=b[0];v[1]=v[0];bias[1]=bias[0]
    def teacher(x):
        z=np.einsum('nd,edh->neh',x,w)+b
        if family=='aligned':
            zz=z.reshape(-1,2,8,2);c=np.cos(angles)[None];s=np.sin(angles)[None]
            u,t=zz[...,0],zz[...,1]
            z=np.stack((c*u-s*t,s*u+c*t),-1).reshape(-1,2,H)
            z=z*ndtr(z);zz=z.reshape(-1,2,8,2);u,t=zz[...,0],zz[...,1]
            z=np.stack((c*u+s*t,-s*u+c*t),-1).reshape(-1,2,H)
        else: z=z*ndtr(z)
        return (z*v).sum(-1)+bias
    data={}
    for split,n in [('train',512),('val',256),('test',1024)]:
        x=rng.normal(size=(n,D));reg=rng.integers(0,2,size=n)
        c=np.column_stack((2*reg-1+rng.normal(size=n)*.25,x))
        target=teacher(x)[np.arange(n),reg]
        noisy=target+rng.normal(size=n)*CFG['noise_sd']
        for suffix,a in [('x',x),('c',c),('y',noisy),('clean',target),('reg',reg)]:
            data[split+'_'+suffix]=np.asarray(a,dtype=np.int64 if suffix=='reg' else np.float32)
    return data

def save(m: Bank,index: int,path: Path) -> str:
    meta={'version':1,'kind':m.kind,'q':m.q,'dtype':'<f4','D':D,'H':H,'tensors':[]}
    raw=[]
    for k,p in m.named_parameters():
        a=np.array(p.detach().cpu().numpy()[index],dtype='<f4',copy=True,order='C')
        meta['tensors'].append([k,list(a.shape),a.nbytes]);raw.append(a.tobytes())
    j=json.dumps(meta,sort_keys=True,separators=(',',':')).encode()
    if len(j)>HEADER-12: raise ValueError('header overflow')
    payload=struct.pack('<8sI',b'GIC002v1',len(j))+j+b'\0'*(HEADER-12-len(j))+b''.join(raw)
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()

def load(path: Path) -> Bank:
    payload=path.read_bytes()
    if len(payload)<HEADER or payload[:8]!=b'GIC002v1':raise ValueError('invalid trusted-fixture payload')
    n=struct.unpack('<I',payload[8:12])[0]
    if n>HEADER-12:raise ValueError('invalid header size')
    meta=json.loads(payload[12:12+n]);m=Bank(meta['kind'],meta['q'],[0]);offset=HEADER
    expected=dict(m.named_parameters())
    if set(expected)!=set(t[0] for t in meta['tensors']):raise ValueError('tensor names')
    with torch.no_grad():
        for k,shape,count in meta['tensors']:
            if list(expected[k].shape[1:])!=shape or int(np.prod(shape))*4!=count:raise ValueError('tensor shape')
            a=np.frombuffer(payload[offset:offset+count],dtype='<f4').reshape(shape).copy();offset+=count
            expected[k][0].copy_(torch.from_numpy(a))
    if offset!=len(payload):raise ValueError('payload length')
    return m

def parsed(method: str):
    prefix,policy=method.split('_')
    if prefix=='independent':return 'independent',8,policy
    kind='mirror' if prefix.startswith('mirror') else 'gain'
    return kind,int(prefix[len(kind):]),policy

def tensors(data: dict) -> dict:
    return {k:torch.from_numpy(v) for k,v in data.items()}

def evaluate(m: Bank,data: dict,ntrain: int,oracle: bool=False,uniform: bool=False) -> dict:
    out={}
    with torch.no_grad():
        for split in ('train','val','test'):
            n=ntrain if split=='train' else len(data[split+'_x'])
            reg=data[split+'_reg'][:n]
            y,p,experts=m(data[split+'_x'][:n],data[split+'_c'][:n],uniform=uniform,oracle=reg if oracle else None)
            clean=(y-data[split+'_clean'][:n]).square()
            out[split+'_mse']=clean.mean(1).numpy().tolist()
            out[split+'_noisy_mse']=(y-data[split+'_y'][:n]).square().mean(1).numpy().tolist()
            if split=='test':
                regime=torch.stack([clean[:,reg==i].mean(1) for i in (0,1)],1)
                out['worst_mse']=regime.max(1).values.numpy().tolist()
                out['usage']=p.mean(1).numpy().tolist()
                a=((p>.5)==reg.bool()).float().mean(1);a=torch.maximum(a,1-a)
                out['route_match']=a.numpy().tolist()
                out['expert_difference']=(experts[:,:,0]-experts[:,:,1]).square().mean(1).numpy().tolist()
    return out

def gradient_diagnostic(m: Bank,x: torch.Tensor,c: torch.Tensor,y: torch.Tensor,uniform=False,oracle=None) -> dict:
    if m.kind=='independent':return {}
    pred,p,e=m(x,c,uniform=uniform,oracle=oracle)
    upstream=(2*(pred-y)/len(x)).detach()
    params=[m.w1,m.b1,m.w2,m.b2]
    gs=[]
    for i,weight in [(0,1-p),(1,p)]:
        surrogate=(upstream*weight.detach()*e[:,:,i]).sum()
        grads=torch.autograd.grad(surrogate,params,retain_graph=True)
        gs.append(torch.cat([g.reshape(len(m.seeds),-1) for g in grads],1))
    a,b=gs;na=a.norm(dim=1);nb=b.norm(dim=1)
    return {'g0':na.tolist(),'g1':nb.tolist(),'cos':((a*b).sum(1)/(na*nb).clamp_min(1e-30)).tolist(),
            'cancel':((a+b).norm(dim=1)/(na+nb).clamp_min(1e-30)).tolist()}

def fit(seed: int,family: str,ntrain: int,method: str,inits: list[int],lr: float,out: Path,steps: int|None=None):
    torch.set_num_threads(1);steps=CFG['steps'] if steps is None else steps
    torch.use_deterministic_algorithms(True)
    kind,q,policy=parsed(method);m=Bank(kind,q,inits,zero_route=policy in ('warm','equal'))
    data=tensors(world(seed,family));stream=np.random.default_rng(seed+400000).integers(0,ntrain,size=(CFG['steps'],CFG['batch_size']))
    opt=torch.optim.Adam(m.parameters(),lr=lr,foreach=False)
    history=[];best=np.full(len(inits),np.inf);best_steps=[0]*len(inits);best_state={k:p.detach().clone() for k,p in m.named_parameters()}
    start=time.perf_counter()
    diagnostics={}
    for step in range(steps+1):
        uniform=policy=='warm' and step<CFG['warmup_steps']
        if step in (0,steps) and kind!='independent':
            diagnostics[str(step)]=gradient_diagnostic(m,data['train_x'][:ntrain],data['train_c'][:ntrain],data['train_y'][:ntrain],uniform=uniform,oracle=data['train_reg'][:ntrain] if policy=='oracle' else None)
        if step in CFG['checkpoint_steps'] or step==steps:
            ev=evaluate(m,data,ntrain,oracle=policy=='oracle',uniform=uniform)
            history.append({'step':step,**ev})
            for j,loss in enumerate(ev['val_noisy_mse']):
                if step>0 and not uniform and loss<best[j]:
                    best[j]=loss;best_steps[j]=step
                    for k,p in m.named_parameters():best_state[k][j]=p.detach()[j]
        if step==steps:break
        idx=stream[step];x=data['train_x'][idx];c=data['train_c'][idx];y=data['train_y'][idx]
        opt.zero_grad(set_to_none=True)
        pred=m(x,c,uniform=uniform,oracle=data['train_reg'][idx] if policy=='oracle' else None)[0]
        loss=(pred-y).square().mean(1).sum() # sum across independent replicas, not average
        loss.backward()
        if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
        if step==375:
            for g in opt.param_groups:g['lr']=lr*.1
        opt.step()
    elapsed=time.perf_counter()-start
    out.mkdir(parents=True,exist_ok=True)
    final=evaluate(m,data,ntrain,oracle=policy=='oracle')
    records=[]
    for j,init in enumerate(inits):
        f=out/f'{method}_s{init}_final.bin';h=save(m,j,f)
        rec={'world':seed,'family':family,'ntrain':ntrain,'method':method,'init':init,'lr':lr,'steps':steps,
             'params':sum(p[j].numel() for p in m.parameters()),'bytes':f.stat().st_size,'sha256':h,'file':str(f.relative_to(ROOT)),
             'replica_group_seconds':elapsed,'replicas_in_group':len(inits),'best_step':best_steps[j]}
        rec.update({k:v[j] for k,v in final.items()});records.append(rec)
    with torch.no_grad():
        for k,p in m.named_parameters():p.copy_(best_state[k])
    selected=evaluate(m,data,ntrain,oracle=policy=='oracle')
    for j,init in enumerate(inits):
        f=out/f'{method}_s{init}_best.bin';h=save(m,j,f)
        records[j]['best_file']=str(f.relative_to(ROOT));records[j]['best_sha256']=h
        records[j].update({'best_'+k:v[j] for k,v in selected.items()})
    result={'records':records,'history':history,'gradients':diagnostics}
    (out/f'{method}.json').write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False))
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--world',type=int,required=True);ap.add_argument('--family',required=True);ap.add_argument('--ntrain',type=int,required=True)
    ap.add_argument('--method',required=True);ap.add_argument('--inits',default='0,1');ap.add_argument('--lr',type=float,default=.01);ap.add_argument('--out',required=True)
    args=ap.parse_args();r=fit(args.world,args.family,args.ntrain,args.method,list(map(int,args.inits.split(','))),args.lr,ROOT/args.out)
    print(json.dumps([{k:x[k] for k in ('method','init','test_mse','worst_mse','bytes','replica_group_seconds')} for x in r['records']]))
