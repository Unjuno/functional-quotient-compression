import numpy as np, torch
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn

class DigitsMLP(nn.Module):
    def __init__(self):
        super().__init__(); self.fc1=nn.Linear(64,128); self.fc2=nn.Linear(128,64); self.fc3=nn.Linear(64,10)
    def forward(self,x):
        return self.fc3(torch.relu(self.fc2(torch.relu(self.fc1(x)))))

def data_splits():
    d=load_digits(); x=torch.tensor(d.images.reshape(-1,64).astype(np.float32)/16.0); y=torch.tensor(d.target.astype(np.int64))
    ix=np.arange(len(y)); trainval,audit=train_test_split(ix,test_size=.2,random_state=7210,stratify=y.numpy())
    train,dev=train_test_split(trainval,test_size=.2,random_state=7211,stratify=y.numpy()[trainval])
    return x[train],y[train],x[dev],y[dev],x[audit],y[audit]

def fit(seed, x, y, epochs=80, capture=False):
    torch.manual_seed(seed); m=DigitsMLP();opt=torch.optim.SGD(m.parameters(),lr=.05,momentum=.9)
    snapshots=[]; n=len(y);gen=torch.Generator().manual_seed(seed+313)
    updates=0
    for epoch in range(1,epochs+1):
        lr=.05 if epoch<=40 else (.01 if epoch<=60 else .003)
        for group in opt.param_groups:group['lr']=lr
        perm=torch.randperm(n,generator=gen)
        for start in range(0,n,128):
            ids=perm[start:start+128];opt.zero_grad(set_to_none=True);loss=nn.functional.cross_entropy(m(x[ids]),y[ids]);loss.backward();opt.step();updates+=1
        if capture and epoch>=61:snapshots.append(flatten(m).detach().clone())
    return m,snapshots,updates

def flatten(m):
    return torch.cat([p.detach().reshape(-1) for p in m.parameters()])

def load_flat(m,flat):
    off=0
    with torch.no_grad():
        for p in m.parameters():
            n=p.numel();p.copy_(flat[off:off+n].reshape_as(p));off+=n
    if off!=flat.numel():raise ValueError('flat weight count mismatch')
    return m

def metrics(logits,y):
    probs=logits.softmax(-1);nll=float(nn.functional.cross_entropy(logits,y));pred=probs.argmax(-1);acc=float((pred==y).float().mean());one=torch.nn.functional.one_hot(y,10).float();brier=float((probs-one).square().sum(-1).mean())
    conf,cls=probs.max(-1);ece=0.0
    for lo in torch.linspace(0,0.9,10):
        hi=lo+0.1;mask=(conf>=lo)&(conf<(hi if hi<1 else hi+1e-6))
        if mask.any():ece+=float(mask.float().mean())*abs(float((pred[mask]==y[mask]).float().mean())-float(conf[mask].mean()))
    return {'nll':nll,'accuracy':acc,'ece10':ece,'brier':brier}

def posterior_logits(mean,basis,scales,diag_std,x,seed,k=8,rademacher=False):
    g=torch.Generator().manual_seed(seed);out=[];p=mean.numel();diag_std=diag_std if diag_std is not None else torch.zeros_like(mean)
    for _ in range(k):
        z=(torch.randint(0,2,(len(scales),),generator=g).float()*2-1) if rademacher else torch.randn(len(scales),generator=g)
        eps=torch.randn(p,generator=g)
        theta=mean+(z*scales)@basis+eps*diag_std
        m=load_flat(DigitsMLP(),theta);out.append(m(x).softmax(-1))
    probs=torch.stack(out).mean(0)
    return probs.clamp_min(1e-12).log(),torch.stack(out)

def rank1_logits(mean,x,seed,sigma,k=8):
    g=torch.Generator().manual_seed(seed);pred=[]
    shapes=[(128,64),(64,128),(10,64)];offset=0
    # Parameters order is weight,bias per layer; sample independent row/column factors for each matrix.
    state=mean.clone();off=0
    layers=[]
    for outd,ind in shapes:
        wn=outd*ind;w=mean[off:off+wn].reshape(outd,ind);off+=wn
        bn=outd;b=mean[off:off+bn];off+=bn
        layers.append((w,b))
    for _ in range(k):
        hs=x
        for li,(w,b) in enumerate(layers):
            u=1+sigma*torch.randn(w.shape[0],generator=g);v=1+sigma*torch.randn(w.shape[1],generator=g)
            hs=nn.functional.linear(hs,w*(u[:,None]*v[None,:]),b)
            if li<2:hs=torch.relu(hs)
        pred.append(hs.softmax(-1))
    ps=torch.stack(pred)
    return ps.mean(0).clamp_min(1e-12).log(),ps

def disagreement(member_probs):
    k=len(member_probs);d=[]
    for i in range(k):
        for j in range(i+1,k):d.append((member_probs[i].argmax(-1)!=member_probs[j].argmax(-1)).float().mean())
    return float(torch.stack(d).mean()) if d else 0.0
