"""Synthetic four-member ensemble comparison: independent, tied, BatchEnsemble, Mirror."""
from __future__ import annotations
import argparse,hashlib,io,json,time,zipfile
from pathlib import Path
import numpy as np
import torch
from torch import nn

E=4;C=4;D=16

def rotate(w,theta):
    if w.ndim>1 and theta.ndim==1: theta=theta.reshape(theta.shape[0],*([1]*(w.ndim-1)))
    q=w.reshape(*w.shape[:-1],D//2,2);c=theta.cos();s=theta.sin();a=q[...,0];b=q[...,1]
    return torch.stack((c*a-s*b,s*a+c*b),-1).reshape_as(w)

def make_world(seed,regime):
    g=torch.Generator().manual_seed(seed)
    if regime=='aligned':
        base=torch.randn(C,D,generator=g)*.65;angles=torch.rand(E,generator=g)*1.2-.6;teachers=rotate(base.expand(E,-1,-1),angles)
    else:
        teachers=torch.randn(E,C,D,generator=g)*.65;angles=None
    gx=torch.Generator().manual_seed(seed+11);xtrain=torch.randn(2048,D,generator=gx);xtest=torch.randn(4096,D,generator=torch.Generator().manual_seed(seed+12))
    trp=torch.softmax(torch.einsum('nd,ecd->enc',xtrain,teachers),-1);tep=torch.softmax(torch.einsum('nd,ecd->enc',xtest,teachers),-1)
    ytrain=torch.stack([torch.multinomial(trp[e],1,generator=torch.Generator().manual_seed(seed+100+e)).squeeze(-1) for e in range(E)])
    ytest=torch.stack([torch.multinomial(tep[e],1,generator=torch.Generator().manual_seed(seed+200+e)).squeeze(-1) for e in range(E)])
    return teachers,angles,(xtrain,ytrain,trp),(xtest,ytest,tep)

def fit_independent(seed,x,y):
    ws=[];bs=[];wall=0
    for e in range(E):
        torch.manual_seed(seed+e);m=nn.Linear(D,C);o=torch.optim.Adam(m.parameters(),lr=.01);g=torch.Generator().manual_seed(seed+300+e);st=time.perf_counter()
        for _ in range(1000):
            ix=torch.randint(x.shape[0],(128,),generator=g);loss=nn.functional.cross_entropy(m(x[ix]),y[e,ix]);o.zero_grad();loss.backward();o.step()
        wall+=time.perf_counter()-st;ws.append(m.weight.detach().clone());bs.append(m.bias.detach().clone())
    return torch.stack(ws),torch.stack(bs),wall

def fit_tied(seed,x,y):
    torch.manual_seed(seed+400);w=nn.Parameter(torch.randn(C,D)*.01);b=nn.Parameter(torch.zeros(C));o=torch.optim.Adam([w,b],lr=.01);g=torch.Generator().manual_seed(seed+401);st=time.perf_counter()
    for k in range(1000):
        e=k%E;ix=torch.randint(x.shape[0],(128,),generator=g);loss=nn.functional.cross_entropy(x[ix]@w.T+b,y[e,ix]);o.zero_grad();loss.backward();o.step()
    return w.detach(),b.detach(),time.perf_counter()-st

def fit_be(seed,x,y):
    torch.manual_seed(seed+500);w=nn.Parameter(torch.randn(C,D)*.01);b=nn.Parameter(torch.zeros(C));r=nn.Parameter(torch.ones(E,C));s=nn.Parameter(torch.ones(E,D));o=torch.optim.Adam([w,b,r,s],lr=.01);g=torch.Generator().manual_seed(seed+501);st=time.perf_counter()
    for k in range(1000):
        e=k%E;ix=torch.randint(x.shape[0],(128,),generator=g);wm=w*r[e,:,None]*s[e,None,:];loss=nn.functional.cross_entropy(x[ix]@wm.T+b,y[e,ix]);o.zero_grad();loss.backward();o.step()
    return {'weight':w.detach(),'bias':b.detach(),'r':r.detach(),'s':s.detach()},time.perf_counter()-st

def fit_mirror(seed,x,y):
    torch.manual_seed(seed+600);w=nn.Parameter(torch.randn(C,D)*.01);b=nn.Parameter(torch.zeros(C));a=nn.Parameter(torch.zeros(E));o=torch.optim.Adam([w,b,a],lr=.01);g=torch.Generator().manual_seed(seed+601);st=time.perf_counter()
    for k in range(1000):
        e=k%E;ix=torch.randint(x.shape[0],(128,),generator=g);wm=rotate(w,a[e]);loss=nn.functional.cross_entropy(x[ix]@wm.T+b,y[e,ix]);o.zero_grad();loss.backward();o.step()
    return {'weight':w.detach(),'bias':b.detach(),'angles':a.detach()},time.perf_counter()-st

def archive(path,method,state,seed,regime):
    arrays={k:v.detach().cpu().numpy().astype(np.float16) for k,v in state.items()};meta={'experiment_id':'MA-260','method':method,'seed':seed,'regime':regime,'members':E,'classes':C,'input_dim':D,'dtype':'FP16','member_id':'externally supplied'}
    path.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_STORED) as z:
        z.writestr('metadata.json',json.dumps(meta,sort_keys=True).encode())
        for k,a in arrays.items():q=io.BytesIO();np.save(q,a,allow_pickle=False);z.writestr(f'arrays/{k}.npy',q.getvalue())
    raw=path.read_bytes();return len(raw),hashlib.sha256(raw).hexdigest()

def decode(path):
    with zipfile.ZipFile(path) as z:meta=json.loads(z.read('metadata.json'));a={Path(n).stem:torch.from_numpy(np.load(io.BytesIO(z.read(n)),allow_pickle=False).astype(np.float32)) for n in z.namelist() if n.endswith('.npy')}
    m=meta['method']
    if m=='independent':w,b=a['weight'],a['bias']
    elif m=='tied':w=a['weight'].expand(E,-1,-1);b=a['bias'].expand(E,-1)
    elif m=='batchensemble':w=a['weight'][None,:,:]*a['r'][:,:,None]*a['s'][:,None,:];b=a['bias'].expand(E,-1)
    else:w=torch.stack([rotate(a['weight'],v) for v in a['angles']]);b=a['bias'].expand(E,-1)
    return meta,a,w,b

def ece(probs,labels,bins=10):
    conf,pred=probs.max(-1);correct=(pred==labels).float();edges=torch.linspace(0,1,bins+1);score=0.
    for i in range(bins):
        mask=(conf>=edges[i])&(conf<(edges[i+1] if i<bins-1 else edges[i+1]+1e-7))
        if mask.any():score+=float(mask.float().mean())*abs(float(correct[mask].mean()-conf[mask].mean()))
    return score

def metrics(w,b,x,y,teacher_probs):
    with torch.no_grad():p=torch.softmax(torch.einsum('nd,ecd->enc',x,w)+b[:,None,:],-1)
    nll=[];acc=[];cal=[]
    for e in range(E):nll.append(float(-p[e].clamp_min(1e-12).log().gather(1,y[e,:,None]).mean()));acc.append(float((p[e].argmax(-1)==y[e]).float().mean()));cal.append(ece(p[e],y[e]))
    mix=p.mean(0);target=teacher_probs.mean(0);mixture_nll=float(-(target*mix.clamp_min(1e-12).log()).sum(-1).mean());mixture_acc=float((mix.argmax(-1)==target.argmax(-1)).float().mean())
    js=[]
    for i in range(E):
      for j in range(i+1,E):
        mid=(p[i]+p[j])/2;kl1=(p[i]*(p[i].clamp_min(1e-12).log()-mid.clamp_min(1e-12).log())).sum(-1);kl2=(p[j]*(p[j].clamp_min(1e-12).log()-mid.clamp_min(1e-12).log())).sum(-1);js.append(float(((kl1+kl2)/2).mean()))
    return {'mean_member_nll':float(np.mean(nll)),'per_member_nll':nll,'mean_member_accuracy':float(np.mean(acc)),'per_member_accuracy':acc,'mean_member_ece':float(np.mean(cal)),'per_member_ece':cal,'mixture_nll':mixture_nll,'mixture_accuracy':mixture_acc,'mean_pairwise_JS_divergence':float(np.mean(js)),'per_pair_JS':js}

def run(seed,regime,split,out):
    teachers,teacher_angles,train,test=make_world(seed,regime);x,y,_=train;xt,yt,tp=test
    wi,bi,tind=fit_independent(seed,x,y);wt,bt,ttied=fit_tied(seed,x,y);be,tbe=fit_be(seed,x,y);mir,tm=fit_mirror(seed,x,y)
    states={'independent':{'weight':wi,'bias':bi},'tied':{'weight':wt,'bias':bt},'batchensemble':be,'mirror':mir};wall={'independent':tind,'tied':ttied,'batchensemble':tbe,'mirror':tm};methods={}
    for method,state in states.items():
        path=out/f'{regime}_{seed}_{method}.zip';size,digest=archive(path,method,state,seed,regime);_,_,w,b=decode(path);t0=time.perf_counter();q=metrics(w,b,xt,yt,tp);infer=time.perf_counter()-t0
        methods[method]={'metrics':q,'actual_payload_bytes':size,'payload_sha256':digest,'inference_seconds':infer,'test_examples_per_second':E*len(xt)/max(infer,1e-9),'weight_elements':int(w.numel())}
    result={'experiment_id':'MA-260','seed':seed,'split':split,'regime':regime,'teacher_angles':teacher_angles.tolist() if teacher_angles is not None else None,'train_examples':E*x.shape[0],'optimizer_updates':{'independent':4000,'tied':1000,'batchensemble':1000,'mirror':1000},'examples_seen':{'independent':E*1000*128,'tied':1000*128,'batchensemble':1000*128,'mirror':1000*128},'training_wall_seconds':wall,'methods':methods,'scope_note':'synthetic linear ensemble with externally supplied member IDs'}
    out.mkdir(parents=True,exist_ok=True);(out/f'{regime}_{seed}_result.json').write_text(json.dumps(result,indent=2)+'\n');return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--regime',choices=('aligned','unrelated'),required=True);ap.add_argument('--split',choices=('development','fresh'),required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();print(json.dumps(run(a.seed,a.regime,a.split,a.out),indent=2))
if __name__=='__main__':main()
