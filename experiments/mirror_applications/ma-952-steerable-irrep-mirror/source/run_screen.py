"""MA-952 CPU screen: task codes on C4 steerable irrep readouts."""
import csv, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from e2cnn import gspaces, nn as enn

ROOT=Path(__file__).resolve().parents[1]
SEEDS=[95221,95223,95227]; UPDATES=500; BATCH=64; THREADS=1
torch.set_num_threads(THREADS)

def make_data(seed,n,train=False):
    rng=np.random.default_rng(seed); xs=[]; ys=[]
    for _ in range(n):
        cls=int(rng.integers(0,2)); ori=int(rng.integers(0,3 if train else 4))
        img=rng.normal(0,.015,(16,16)).astype('float32')
        img[6:10,6:10]+=0.25+0.45*cls
        img[1:4,6:10]+=0.9
        img=np.rot90(img,ori).copy();xs.append(img[None]);ys.append([float(ori==0),float(ori==1),float(cls)])
    return torch.from_numpy(np.stack(xs)),torch.tensor(ys,dtype=torch.float32)

class SteerableTasks(nn.Module):
    def __init__(self,condition):
        super().__init__();self.condition=condition;self.gspace=gspaces.Rot2dOnR2(N=4)
        g=self.gspace; self.input_type=enn.FieldType(g,[g.trivial_repr])
        hidden=enn.FieldType(g,[g.regular_repr]*4)
        out=enn.FieldType(g,[g.fibergroup.irrep(0)]*4+[g.fibergroup.irrep(1)]*2+[g.fibergroup.irrep(2)]*4)
        self.c1=enn.R2Conv(self.input_type,hidden,3,padding=1)
        self.act=enn.NormNonLinearity(hidden)
        self.c2=enn.R2Conv(hidden,out,3,padding=1)
        self.out_type=out
        # The three contiguous feature blocks are the three real C4 irrep families.
        self.blocks=((0,4),(4,8),(8,12))
        if condition=='native_heads': self.head=nn.Parameter(torch.empty(3,12)); self.bias=nn.Parameter(torch.zeros(3))
        elif condition=='mirror_irrep':
            self.shared=nn.Parameter(torch.empty(12));self.code=nn.Parameter(torch.ones(3,3));self.bias=nn.Parameter(torch.zeros(3))
        elif condition=='ordinary_gate':
            self.shared=nn.Parameter(torch.empty(12));self.gate=nn.Parameter(torch.zeros(3,12));self.bias=nn.Parameter(torch.zeros(3))
        elif condition=='mirror_private':
            self.shared=nn.Parameter(torch.empty(12));self.code=nn.Parameter(torch.ones(3,3));self.private=nn.Parameter(torch.zeros(12));self.bias=nn.Parameter(torch.zeros(3))
        else: raise ValueError(condition)
        for name in ['head','shared']:
            p=getattr(self,name,None)
            if isinstance(p,nn.Parameter):
                if p.ndim>=2:nn.init.xavier_uniform_(p)
                else:nn.init.normal_(p,std=.25)
        if condition=='mirror_private': nn.init.normal_(self.private,std=.25)
    def feature_map(self,x):
        z=enn.GeometricTensor(x,self.input_type);z=self.c2(self.act(self.c1(z)))
        return z.tensor
    def features(self,x):return self.feature_map(x).mean((-2,-1))
    def forward(self,x):
        z=self.features(x)
        if self.condition=='native_heads':return z@self.head.T+self.bias
        if self.condition=='ordinary_gate':return (z[:,None]*self.shared[None,None]*torch.sigmoid(self.gate)[None]).sum(-1)+self.bias
        outs=[]
        for t in range(3):
            w=torch.cat([self.shared[a:b]*self.code[t,j] for j,(a,b) in enumerate(self.blocks)])
            if self.condition=='mirror_private' and t==2:w=w+self.private
            outs.append((z*w).sum(-1)+self.bias[t])
        return torch.stack(outs,1)

def equivariance_error(m,x):
    errs=[];g=m.gspace.fibergroup
    with torch.no_grad():
        z=m.features(x[:16])
        for k in range(1,4):
            a=m.features(torch.rot90(x[:16],k,(-2,-1)))
            mats=[torch.as_tensor(rep(k),dtype=z.dtype) for rep in m.out_type.representations]
            rho=torch.block_diag(*mats)
            b=z@rho.T
            errs.append((a-b).abs().max().item())
    return max(errs)

def fit_eval(condition,seed,xtr,ytr,xte,yte):
    torch.manual_seed(seed);m=SteerableTasks(condition);opt=torch.optim.Adam(m.parameters(),lr=.003)
    rng=torch.Generator().manual_seed(seed+37);t0=time.perf_counter()
    for _ in range(UPDATES):
        ix=torch.randint(len(xtr),(BATCH,),generator=rng);loss=F.binary_cross_entropy_with_logits(m(xtr[ix]),ytr[ix])
        opt.zero_grad();loss.backward();opt.step()
    train_s=time.perf_counter()-t0;m.eval();path=ROOT/'source'/f'{condition}_{seed}.pt'
    torch.save({'state_dict':m.state_dict(),'condition':condition,'metadata':{'group':'C4','tasks':['orientation0','orientation1','object_class'],'e2cnn':'0.2.3'}},path)
    nbytes=path.stat().st_size;loaded=torch.load(path,map_location='cpu',weights_only=True);clone=SteerableTasks(condition);clone.load_state_dict(loaded['state_dict']);clone.eval()
    with torch.no_grad():
        logits=clone(xte);acc=((logits>=0)==yte).float().mean(0).tolist();replay=float((logits-m(xte)).abs().max())
    eq=equivariance_error(clone,xte);t0=time.perf_counter()
    with torch.no_grad():
        for i in range(0,len(xte),BATCH):clone(xte[i:i+BATCH])
    infer_s=time.perf_counter()-t0
    # Dense spatial-convolution proxy; e2cnn basis expansion / sampling overhead is reflected in wall time.
    mac=16*16*9*(1*16+16*16)+12*3
    if condition=='ordinary_gate':mac+=36
    if condition=='mirror_private':mac+=12
    row={'seed':seed,'condition':condition,'acc_task0_orbit':acc[0],'acc_task1_orbit':acc[1],'acc_task2_nonorbit':acc[2],
         'equivariance_max_abs':eq,'payload_bytes':nbytes,'train_examples':len(xtr),'test_examples':len(xte),'updates':UPDATES,
         'train_seconds':train_s,'inference_examples_per_second':len(xte)/infer_s,'dense_mac_proxy_per_example':mac,
         'serialization_replay_max_abs':replay}
    path.unlink();return row

def main():
    rows=[]
    for seed in SEEDS:
        xtr,ytr=make_data(seed,512,True);xte,yte=make_data(seed+1000,256,False)
        for c in ['native_heads','mirror_irrep','mirror_private','ordinary_gate']:
            r=fit_eval(c,seed,xtr,ytr,xte,yte);rows.append(r);print(json.dumps(r),flush=True)
    with (ROOT/'RESULTS_CORE.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (ROOT/'source/results.json').write_text(json.dumps({'results':rows,'seeds':SEEDS,'train_orientations':[0,1,2],
        'heldout_orientation':3,'device':'cpu','torch':torch.__version__,'e2cnn':'0.2.3',
        'task_relation':'task0/task1 are exact C4 orbit copies; task2 is a non-orbit class label'},indent=2)+'\n')
    print('max equivariance',max(r['equivariance_max_abs'] for r in rows),'max replay',max(r['serialization_replay_max_abs'] for r in rows))
if __name__=='__main__':main()
