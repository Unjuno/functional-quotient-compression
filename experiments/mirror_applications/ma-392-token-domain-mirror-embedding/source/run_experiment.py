"""Held-out token-domain generalization with shared and Mirror domain views."""
from __future__ import annotations
import argparse,hashlib,io,json,time,zipfile
from pathlib import Path
import numpy as np
import torch
from torch import nn

V=128;DOMAINS=8;C=8


def labels(token,domain):return (token%C+domain)%C


def make_world(seed):
    rng=np.random.default_rng(seed+811)
    heldout=torch.from_numpy((rng.random((DOMAINS,V))<.25))
    # Guarantee training coverage by construction: each item/domain is withheld at most once per token.
    for t in range(V):
        if heldout[:,t].all():heldout[rng.integers(DOMAINS),t]=False
    for d in range(DOMAINS):
        if heldout[d].all():heldout[d,rng.integers(V)]=False
    seen=~heldout
    pairs=torch.nonzero(seen,as_tuple=False);g=torch.Generator().manual_seed(seed)
    train_ix=torch.randint(len(pairs),(32768,),generator=g);train=pairs[train_ix]
    def eval_split(repeats,offset):
        allpairs=torch.cartesian_prod(torch.arange(DOMAINS),torch.arange(V))
        ids=allpairs.repeat(repeats,1)
        gen=torch.Generator().manual_seed(seed+offset);ids=ids[torch.randperm(len(ids),generator=gen)]
        d,t=ids[:,0],ids[:,1]
        return t,d,labels(t,d),heldout[d,t]
    train_t,train_d=train[:,1],train[:,0]
    train_y=labels(train_t,train_d)
    return {'train':(train_t,train_d,train_y),'validation':eval_split(8,1000),'test':eval_split(8,2000)},heldout


class DomainModel(nn.Module):
    def __init__(self,method):
        super().__init__();self.method=method
        if method=='independent':self.full=nn.Parameter(torch.randn(DOMAINS,V,C)*.02)
        else:self.token=nn.Parameter(torch.randn(V,C)*.02)
        if method=='bias':self.bias=nn.Parameter(torch.zeros(DOMAINS,C))
        elif method=='rank4':
            self.a=nn.Parameter(torch.randn(DOMAINS,C,4)*.01);self.b=nn.Parameter(torch.randn(DOMAINS,4,C)*.01);self.bias=nn.Parameter(torch.zeros(DOMAINS,C))
        elif method=='domain_map':self.matrix=nn.Parameter(torch.eye(C).repeat(DOMAINS,1,1));self.bias=nn.Parameter(torch.zeros(DOMAINS,C))
        elif method=='mirror':self.phase=nn.Parameter(torch.zeros(DOMAINS))

    def forward(self,token,domain):
        if self.method=='independent':return self.full[domain,token]
        x=self.token[token]
        if self.method=='bias':return x+self.bias[domain]
        if self.method=='rank4':
            residual=torch.bmm(torch.bmm(x.unsqueeze(1),self.a[domain]),self.b[domain]).squeeze(1)
            return x+residual+self.bias[domain]
        if self.method=='domain_map':return torch.bmm(x.unsqueeze(1),self.matrix[domain]).squeeze(1)+self.bias[domain]
        # Real Fourier/Givens representation of a cyclic shift. A single phase
        # generates all paired rotations at integer frequencies.
        z=torch.fft.rfft(x,n=C,dim=-1)
        k=torch.arange(z.shape[-1],device=x.device,dtype=x.dtype)
        angle=self.phase[domain,None]*k[None,:]
        ph=torch.polar(torch.ones_like(angle),-angle)
        return torch.fft.irfft(z*ph,n=C,dim=-1)


def archive(path,model):
    arrays={k:v.detach().cpu().numpy().astype(np.float16) for k,v in model.state_dict().items()}
    path.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_STORED) as z:
        z.writestr('metadata.json',json.dumps({'vocabulary':V,'domains':DOMAINS,'classes':C,'target':'(token mod 8 + domain) mod 8'},sort_keys=True).encode())
        for k,a in sorted(arrays.items()):
            b=io.BytesIO();np.save(b,a,allow_pickle=False);z.writestr(f'arrays/{k}.npy',b.getvalue())
    blob=path.read_bytes();return len(blob),hashlib.sha256(blob).hexdigest()


def reload(path,model):
    with zipfile.ZipFile(path) as z:a={Path(n).stem:np.load(io.BytesIO(z.read(n)),allow_pickle=False) for n in z.namelist() if n.endswith('.npy')}
    model.load_state_dict({k:torch.from_numpy(a[k].astype(np.float32)) for k in model.state_dict()})


@torch.no_grad()
def score(model,split):
    t,d,y,is_heldout=split;logits=model(t,d);nll=nn.functional.cross_entropy(logits,y,reduction='none');correct=logits.argmax(-1)==y
    def summary(mask):
        if not mask.any():return {'examples':0,'accuracy':None,'nll':None}
        return {'examples':int(mask.sum()),'accuracy':float(correct[mask].float().mean()),'nll':float(nll[mask].mean())}
    return {'all':summary(torch.ones_like(is_heldout,dtype=torch.bool)),'seen':summary(~is_heldout),'heldout':summary(is_heldout)}


def fit(seed,method,updates,out):
    splits,mask=make_world(seed);torch.manual_seed(seed+sum(map(ord,method)));model=DomainModel(method)
    if method!='independent':
        gen=torch.Generator().manual_seed(seed+521);nn.init.normal_(model.token,mean=0,std=.03)
    opt=torch.optim.Adam(model.parameters(),lr=.01);t,d,y=splits['train'];start=time.perf_counter()
    for _ in range(updates):
        ix=torch.randint(len(t),(512,));loss=nn.functional.cross_entropy(model(t[ix],d[ix]),y[ix])
        opt.zero_grad();loss.backward();opt.step()
    train_wall=time.perf_counter()-start
    path=out/f'{seed}_{method}.zip';size,digest=archive(path,model);reload(path,model)
    val=score(model,splits['validation']);t0=time.perf_counter();test=score(model,splits['test']);infer_wall=time.perf_counter()-t0
    mac={'independent':0,'bias':8,'rank4':64,'domain_map':64,'mirror':24}[method]
    return {'validation':val,'test':test,'actual_payload_bytes':size,'payload_sha256':digest,'updates':updates,
            'train_examples':len(t),'examples_seen':updates*512,'train_wall_seconds':train_wall,
            'operator_macs_per_query':mac,'domains_per_query':1,'test_inference_examples_per_second':len(splits['test'][0])/max(infer_wall,1e-9),
            'heldout_pairs':int(mask.sum()),'seen_pairs':int((~mask).sum())}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--updates',type=int,default=1600);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    result={'experiment_id':'MA-392','seed':a.seed,'methods':{}}
    for method in ('independent','bias','rank4','domain_map','mirror'):result['methods'][method]=fit(a.seed,method,a.updates,a.out)
    a.out.mkdir(parents=True,exist_ok=True);(a.out/f'{a.seed}_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
