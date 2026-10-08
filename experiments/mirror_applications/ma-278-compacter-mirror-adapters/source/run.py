import argparse, json, time, io
from pathlib import Path
import torch

torch.set_num_threads(1)
ROOT=Path(__file__).resolve().parents[1]
N,T,D,R=6,16,16,2
METHODS=['tied','compacter','scalar','mirror','independent']

def orth(seed, shape):
    g=torch.Generator().manual_seed(seed); a=torch.randn(*shape,generator=g); q,_=torch.linalg.qr(a); return q

def world_data(seed):
    g=torch.Generator().manual_seed(seed)
    # teacher shared Kronecker slow atoms times task-specific rank-one factors
    slow=torch.randn(R,4,4,generator=g)*.25
    left=torch.randn(N,R,4,1,generator=g)*.45
    right=torch.randn(N,R,1,4,generator=g)*.45
    targets=[]
    for t in range(N):
        delta=torch.zeros(D,D)
        for r in range(R): delta += torch.kron(slow[r], left[t,r]@right[t,r])
        targets.append(delta)
    W=torch.stack(targets)
    X=torch.randn(8192,D,generator=g); xv=torch.randn(2048,D,generator=g)
    noise=torch.randn(N,8192,D,generator=g)*.005
    y=torch.einsum('bd,tdh->tbh',X,W)+noise
    yv=torch.einsum('bd,tdh->tbh',xv,W)
    return X,xv,y,yv

class Model(torch.nn.Module):
    def __init__(self,method,seed):
        super().__init__(); self.method=method
        g=torch.Generator().manual_seed(seed)
        if method=='tied': self.delta=torch.nn.Parameter(torch.zeros(D,D))
        elif method=='compacter':
            self.slow=torch.nn.Parameter(torch.randn(R,4,4,generator=g)*.03); self.l=torch.nn.Parameter(torch.randn(N,R,4,1,generator=g)*.03); self.r=torch.nn.Parameter(torch.randn(N,R,1,4,generator=g)*.03)
        elif method=='scalar':
            self.slow=torch.nn.Parameter(torch.randn(R,4,4,generator=g)*.03); self.l=torch.nn.Parameter(torch.randn(R,4,1,generator=g)*.03); self.r=torch.nn.Parameter(torch.randn(R,1,4,generator=g)*.03); self.c=torch.nn.Parameter(torch.randn(N,R,generator=g)*.03)
        elif method=='mirror':
            self.slow=torch.nn.Parameter(torch.randn(R,4,4,generator=g)*.03); self.l=torch.nn.Parameter(torch.randn(R,4,1,generator=g)*.03); self.r=torch.nn.Parameter(torch.randn(R,1,4,generator=g)*.03); self.angles=torch.nn.Parameter(torch.randn(N,R,generator=g)*.03)
        else: self.delta=torch.nn.Parameter(torch.randn(N,D,D,generator=g)*.003)
    def matrices(self):
        if self.method=='tied': return self.delta.expand(N,-1,-1)
        if self.method=='independent': return self.delta
        out=[]
        for t in range(N):
            d=torch.zeros(D,D)
            for k in range(R):
                if self.method=='compacter': a,b=self.l[t,k],self.r[t,k]
                elif self.method=='scalar': a,b=self.l[k],self.r[k]; a=a*self.c[t,k]
                else:
                    # Givens coordinate rotates paired shared fast factors; smooth and differentiable.
                    theta=self.angles[t,k]; c,s=torch.cos(theta),torch.sin(theta)
                    a,b=self.l[k]*c+self.r[k].T*s, self.r[k]*c-self.l[k].T*s
                d=d+torch.kron(self.slow[k],a@b)
            out.append(d)
        return torch.stack(out)
    def forward(self,x): return self.matrices()

def payload(model):
    b=io.BytesIO(); torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'config':{'method':model.method,'N':N,'D':D,'R':R}},b); return len(b.getvalue())

def run(seed,lr,updates,split):
    X,xv,y,yv=world_data(seed); rows=[]
    for mi,m in enumerate(METHODS):
        model=Model(m,seed*100+mi); opt=torch.optim.Adam(model.parameters(),lr=lr); st=time.time(); seen=0
        for step in range(updates):
            ix=(torch.arange(128)+step*128)%len(X); xb=X[ix]; yb=y[:,ix]
            pred=torch.einsum('bd,tdh->tbh',xb,model.matrices()); loss=(pred-yb).square().mean(); opt.zero_grad(); loss.backward(); opt.step(); seen+=len(ix)*N
        train_s=time.time()-st
        with torch.no_grad():
            mats=model.matrices(); pred=torch.einsum('bd,tdh->tbh',xv,mats); mse=(pred-yv).square().mean().item(); per=(pred-yv).square().mean((1,2)).tolist()
            t0=time.time()
            for _ in range(10): torch.einsum('bd,tdh->tbh',xv,mats)
            throughput=(10*len(xv)*N)/(time.time()-t0)
        # linear map MAC proxy: forward plus backward ~= 3*T*D^2 per example/update batch
        mac=updates*128*N*D*D*3
        rows.append({'condition':split,'world_or_seed':seed,'method':m,'serialized_bytes':payload(model),'train_tokens_or_examples':seen,'optimizer_updates':updates,'active_compute_proxy':mac,'wall_time_s':train_s,'primary_metric':'heldout_mse','primary_value':mse,'secondary_metric':'mean_task_mse','secondary_value':sum(per)/N,'per_task':per,'status_note':f'lr={lr}; inference_examples_per_s={throughput:.3f}'})
    return rows

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--world',type=int,required=True);p.add_argument('--lr',type=float,required=True);p.add_argument('--updates',type=int,default=500);p.add_argument('--split',default='development');p.add_argument('--out',required=True);a=p.parse_args()
    rows=run(a.world,a.lr,a.updates,a.split); Path(a.out).write_text(json.dumps(rows,indent=2))
    for r in rows: print(r['world_or_seed'],r['method'],f"mse={r['primary_value']:.6g}",f"bytes={r['serialized_bytes']}",f"sec={r['wall_time_s']:.1f}")
