from __future__ import annotations
from dataclasses import dataclass
import math, time
import torch
from torch.nn import functional as F

@dataclass(frozen=True)
class Shapes:
    inp:int=8
    hidden:int=32
    out:int=2
    @property
    def nparam(self):
        h=self.hidden
        return h*self.inp+h+h*h+h+self.out*h+self.out

def pack(vals):
    return torch.cat([v.reshape(-1) for v in vals])

def unpack(th:torch.Tensor, sh:Shapes):
    h,i,o=sh.hidden,sh.inp,sh.out
    p=0; out=[]
    for shape in [(h,i),(h,),(h,h),(h,),(o,h),(o,)]:
        n=math.prod(shape); out.append(th[p:p+n].reshape(shape)); p+=n
    if p != len(th): raise ValueError('theta size mismatch')
    return out

def init_theta(sh:Shapes, seed:int, dtype=None)->torch.Tensor:
    if dtype is None: dtype=torch.get_default_dtype()
    g=torch.Generator().manual_seed(seed)
    def xavier(o,i):
        bound=math.sqrt(6/(i+o))
        return (2*torch.rand(o,i,generator=g,dtype=dtype)-1)*bound
    return pack([
        xavier(sh.hidden,sh.inp), torch.zeros(sh.hidden,dtype=dtype),
        xavier(sh.hidden,sh.hidden), torch.zeros(sh.hidden,dtype=dtype),
        xavier(sh.out,sh.hidden), torch.zeros(sh.out,dtype=dtype),
    ])

def model_forward(th,x,sh):
    w1,b1,w2,b2,w3,b3=unpack(th,sh)
    h=F.gelu(F.linear(x,w1,b1), approximate='tanh')
    h=F.gelu(F.linear(h,w2,b2), approximate='tanh')
    return F.linear(h,w3,b3)

def population_loss(pop:torch.Tensor,x:torch.Tensor,y:torch.Tensor,sh:Shapes)->torch.Tensor:
    P=pop.shape[0]; h,i,o=sh.hidden,sh.inp,sh.out; p=0
    def take(shape):
        nonlocal p
        n=math.prod(shape); z=pop[:,p:p+n].reshape(P,*shape); p+=n; return z
    w1=take((h,i)); b1=take((h,)); w2=take((h,h)); b2=take((h,)); w3=take((o,h)); b3=take((o,))
    z=torch.einsum('bi,phi->pbh',x,w1)+b1[:,None,:]
    z=F.gelu(z,approximate='tanh')
    z=torch.einsum('pbi,phi->pbh',z,w2)+b2[:,None,:]
    z=F.gelu(z,approximate='tanh')
    pred=torch.einsum('pbi,poi->pbo',z,w3)+b3[:,None,:]
    return (pred-y[None,:,:]).square().mean((1,2))

def law(z,changed=False):
    p0,p1,v0,v1=z.unbind(-1)
    a0=-torch.sin(p0)-.15*v0+.10*p1*v1+.20*torch.sin(p0+p1)
    a1=-.7*torch.sin(p1)-.15*v1+.10*p0*v0-.15*torch.sin(p0-p1)
    if changed:
        a0=a0+.35*torch.sin(2*p0)+.20*v1
        a1=a1-.25*p0+.25*v0
    return torch.stack((a0,a1),-1)

@dataclass
class World:
    train_x:torch.Tensor
    train_y:torch.Tensor
    test_x:torch.Tensor
    test_y:torch.Tensor
    train_y1:torch.Tensor
    test_y1:torch.Tensor

def make_world(seed:int,ntrain:int=1536,ntest:int=4096)->World:
    g=torch.Generator().manual_seed(seed)
    H=[];bias=[]
    for s in range(3):
        if s==0:
            q=torch.eye(4); sc=torch.ones(4); r=torch.eye(4); b=torch.zeros(4)
        else:
            q,_=torch.linalg.qr(torch.randn(4,4,generator=g)); r,_=torch.linalg.qr(torch.randn(4,4,generator=g))
            sc=torch.exp(.6*(torch.rand(4,generator=g)-.5)); b=.3*(torch.rand(4,generator=g)-.5)
        H.append(q@torch.diag(sc)@r.T); bias.append(b)
    H=torch.stack(H); bias=torch.stack(bias); inv=torch.linalg.inv(H)
    def sample(n,off):
        gg=torch.Generator().manual_seed(seed*31+off)
        z=2*torch.rand(n//3,4,generator=gg)-1
        xs=[];ys=[];ys1=[]
        for s in range(3):
            raw=z@H[s].T+bias[s]+.01*torch.randn(z.shape,generator=gg)
            can=(raw-bias[s])@inv[s].T
            sid=F.one_hot(torch.full((len(z),),s),3).float()
            x=torch.cat((can,sid,torch.ones(len(z),1)),1)
            xs.append(x); ys.append(law(z)); ys1.append(law(z,True))
        return torch.cat(xs),torch.cat(ys),torch.cat(ys1)
    tx,ty,ty1=sample(ntrain,101); vx,vy,vy1=sample((ntest//3)*3,102)
    return World(tx,ty,vx,vy,ty1,vy1)

def pretrain_parent(world,sh,init_seed,steps=800,batch=192,lr=.003):
    th=init_theta(sh,init_seed).detach().clone().requires_grad_(True)
    opt=torch.optim.AdamW([th],lr=lr,weight_decay=1e-4)
    g=torch.Generator().manual_seed(init_seed+90001)
    for _ in range(steps):
        idx=torch.randint(len(world.train_x),(batch,),generator=g)
        loss=(model_forward(th,world.train_x[idx],sh)-world.train_y[idx]).square().mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_([th],1.0); opt.step()
    return th.detach()

@dataclass(frozen=True)
class MirrorView:
    a1:torch.Tensor
    a2:torch.Tensor
    rho:float

def _nilpotent_rank1(hidden:int,g:torch.Generator,dtype):
    u=torch.randn(hidden,generator=g,dtype=dtype); u=u/u.norm()
    v=torch.randn(hidden,generator=g,dtype=dtype); v=v-u*(u@v); v=v/v.norm()
    return torch.outer(u,v)

def make_mirror_bank(hidden:int,k:int,seed:int,rho:float=0.15,dtype=torch.float32):
    g=torch.Generator().manual_seed(seed); out=[]
    for _ in range(k):
        out.append(MirrorView(_nilpotent_rank1(hidden,g,dtype), _nilpotent_rank1(hidden,g,dtype), float(rho)))
    return out

def mirror_view_theta(th:torch.Tensor, sh:Shapes, view:MirrorView)->torch.Tensor:
    if view.rho == 0.0: return th.clone()
    w1,b1,w2,b2,w3,b3=unpack(th,sh)
    I=torch.eye(sh.hidden,dtype=th.dtype,device=th.device)
    q1=I+view.rho*view.a1; q2=I+view.rho*view.a2
    r1=I-view.rho*view.a1; r2=I-view.rho*view.a2
    return pack([q1@w1,q1@b1,q2@w2@r1,q2@b2,w3@r2,b3])

def packed_multiview_losses(pop,x,y,sh,bank):
    expanded=[]
    for th in pop:
        for view in bank:
            expanded.append(mirror_view_theta(th,sh,view))
    losses=population_loss(torch.stack(expanded),x,y,sh)
    return losses.reshape(len(pop),len(bank))

def packed_aligned_view_losses(pop,x,y,sh,views):
    if len(pop) != len(views): raise ValueError('one view required per population member')
    effective=torch.stack([mirror_view_theta(th,sh,v) for th,v in zip(pop,views)])
    return population_loss(effective,x,y,sh)

def aggregate_view_losses(lv:torch.Tensor, mode='mean', lam:float=0.5)->torch.Tensor:
    if mode=='mean': return lv.mean(1)
    if mode=='worst': return lv.max(1).values
    if mode=='mean_var': return lv.mean(1)+lam*lv.var(1,unbiased=False)
    raise ValueError(mode)

def count_branch_evals(k_mirrors:int,pairs:int)->int:
    return 2*int(k_mirrors)*int(pairs)

def es_estimate(th,x,y,sh,bank,pairs:int,sigma:float,seed:int,mode='shared_eps',objective='mean',lam:float=0.5):
    if mode not in ('shared_eps','independent_eps'): raise ValueError(mode)
    K=len(bank); N=len(th); g=torch.Generator().manual_seed(seed)
    if mode=='shared_eps':
        eps=torch.randn(pairs,N,generator=g,dtype=th.dtype)
        cands=torch.cat([th[None,:]+sigma*eps, th[None,:]-sigma*eps],0)
        lv=packed_multiview_losses(cands,x,y,sh,bank)
        lp=aggregate_view_losses(lv[:pairs],objective,lam); lm=aggregate_view_losses(lv[pairs:],objective,lam)
        grad=eps.T@(lp-lm)/(2*pairs*sigma)
    else:
        eps=torch.randn(K,pairs,N,generator=g,dtype=th.dtype)
        flat=eps.reshape(K*pairs,N)
        plus=th[None,:]+sigma*flat; minus=th[None,:]-sigma*flat
        aligned=[]
        for view in bank: aligned.extend([view]*pairs)
        lv_plus=packed_aligned_view_losses(plus,x,y,sh,aligned).reshape(K,pairs)
        lv_minus=packed_aligned_view_losses(minus,x,y,sh,aligned).reshape(K,pairs)
        delta=lv_plus-lv_minus
        grads=torch.einsum('kpn,kp->kn',eps,delta)/(2*pairs*sigma)
        grad=grads.mean(0)
    return grad, {'branch_evals':count_branch_evals(K,pairs),'mode':mode,'pairs':pairs,'mirrors':K}

def exact_mean_gradient(th,x,y,sh,bank):
    q=th.detach().clone().requires_grad_(True)
    losses=[]
    for view in bank:
        losses.append((model_forward(mirror_view_theta(q,sh,view),x,sh)-y).square().mean())
    loss=torch.stack(losses).mean(); loss.backward()
    return float(loss.detach()), q.grad.detach()

def nmse(pred,y):
    return float((pred-y).square().sum()/(y-y.mean(0)).square().sum())

def evaluate_views(th,x,y,sh,bank):
    vals=[]
    with torch.no_grad():
        for view in bank:
            vals.append(nmse(model_forward(mirror_view_theta(th,sh,view),x,sh),y))
    return {'mean_nmse':float(sum(vals)/len(vals)),'worst_nmse':float(max(vals)),'min_nmse':float(min(vals)),'per_view':vals}
