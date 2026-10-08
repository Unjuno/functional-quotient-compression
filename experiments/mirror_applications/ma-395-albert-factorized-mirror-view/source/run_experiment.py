"""Domain transfer with ALBERT-style factorized embeddings and an inner Mirror view."""
from __future__ import annotations
import argparse,hashlib,io,json,time,zipfile
from pathlib import Path
import numpy as np
import torch
from torch import nn

V=256;DOMAINS=8;C=8;ED=8;MD=16


def labels(token,domain):return (token%C+domain)%C


def make_world(seed):
    rng=np.random.default_rng(seed+811);held=torch.from_numpy(rng.random((DOMAINS,V))<.25)
    for t in range(V):
        if held[:,t].all():held[rng.integers(DOMAINS),t]=False
    for d in range(DOMAINS):
        if held[d].all():held[d,rng.integers(V)]=False
    seen=~held;pairs=torch.nonzero(seen,as_tuple=False);g=torch.Generator().manual_seed(seed)
    train=pairs[torch.randint(len(pairs),(32768,),generator=g)];dt,t=train[:,0],train[:,1]
    def ev(repeat,offset):
        p=torch.cartesian_prod(torch.arange(DOMAINS),torch.arange(V)).repeat(repeat,1)
        gen=torch.Generator().manual_seed(seed+offset);p=p[torch.randperm(len(p),generator=gen)]
        d,x=p[:,0],p[:,1];return x,d,labels(x,d),held[d,x]
    return {'train':(t,dt,labels(t,dt)),'validation':ev(8,1000),'test':ev(8,2000)},held


def rotate(x,theta):
    z=torch.fft.rfft(x,n=ED,dim=-1)
    k=torch.arange(z.shape[-1],device=x.device,dtype=x.dtype)
    angle=theta[...,None]*k
    phase=torch.polar(torch.ones_like(angle),-angle)
    return torch.fft.irfft(z*phase,n=ED,dim=-1)


class FactorizedDomainModel(nn.Module):
    def __init__(self,method):
        super().__init__();self.method=method
        if method=='independent':self.emb=nn.Parameter(torch.randn(DOMAINS,V,ED)*.02)
        else:self.emb=nn.Parameter(torch.randn(V,ED)*.02)
        self.proj=nn.Parameter(torch.randn(ED,MD)*.04)
        self.classifier=nn.Linear(MD,C)
        if method=='post_rank4':
            self.a=nn.Parameter(torch.randn(DOMAINS,MD,4)*.01);self.b=nn.Parameter(torch.randn(DOMAINS,4,MD)*.01);self.bias=nn.Parameter(torch.zeros(DOMAINS,MD))
        elif method=='post_full':self.matrix=nn.Parameter(torch.eye(MD).repeat(DOMAINS,1,1));self.bias=nn.Parameter(torch.zeros(DOMAINS,MD))
        elif method=='mirror':self.phase=nn.Parameter(torch.zeros(DOMAINS))

    def forward(self,token,domain):
        z=self.emb[domain,token] if self.method=='independent' else self.emb[token]
        if self.method=='mirror':z=rotate(z,self.phase[domain])
        h=z@self.proj
        if self.method=='post_rank4':h=h+torch.bmm(torch.bmm(h.unsqueeze(1),self.a[domain]),self.b[domain]).squeeze(1)+self.bias[domain]
        elif self.method=='post_full':h=torch.bmm(h.unsqueeze(1),self.matrix[domain]).squeeze(1)+self.bias[domain]
        return self.classifier(h)


def archive(path,model):
    arrays={k:v.detach().cpu().numpy().astype(np.float16) for k,v in model.state_dict().items()};path.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_STORED) as z:
        z.writestr('metadata.json',json.dumps({'vocabulary':V,'domains':DOMAINS,'embedding_dim':ED,'model_dim':MD,'classes':C},sort_keys=True).encode())
        for k,a in sorted(arrays.items()):
            b=io.BytesIO();np.save(b,a,allow_pickle=False);z.writestr(f'arrays/{k}.npy',b.getvalue())
    blob=path.read_bytes();return len(blob),hashlib.sha256(blob).hexdigest()


def reload(path,model):
    with zipfile.ZipFile(path) as z:a={Path(n).stem:np.load(io.BytesIO(z.read(n)),allow_pickle=False) for n in z.namelist() if n.endswith('.npy')}
    model.load_state_dict({k:torch.from_numpy(a[k].astype(np.float32)) for k in model.state_dict()})


@torch.no_grad()
def score(model,split):
    t,d,y,h=split;logits=model(t,d);nll=nn.functional.cross_entropy(logits,y,reduction='none');ok=logits.argmax(-1)==y
    def stats(m):return {'examples':int(m.sum()),'accuracy':float(ok[m].float().mean()) if m.any() else None,'nll':float(nll[m].mean()) if m.any() else None}
    return {'all':stats(torch.ones_like(h,dtype=torch.bool)),'seen':stats(~h),'heldout':stats(h)}


def fit(seed,method,updates,out):
    data,mask=make_world(seed);torch.manual_seed(seed+sum(map(ord,method)));model=FactorizedDomainModel(method)
    rng=torch.Generator().manual_seed(seed+456);nn.init.normal_(model.proj,mean=0,std=.04);nn.init.xavier_uniform_(model.classifier.weight,generator=rng);nn.init.zeros_(model.classifier.bias)
    opt=torch.optim.Adam(model.parameters(),lr=.01);t,d,y=data['train'];start=time.perf_counter()
    for _ in range(updates):
        ix=torch.randint(len(t),(512,));loss=nn.functional.cross_entropy(model(t[ix],d[ix]),y[ix]);opt.zero_grad();loss.backward();opt.step()
    wall=time.perf_counter()-start;path=out/f'{seed}_{method}.zip';size,digest=archive(path,model);reload(path,model)
    val=score(model,data['validation']);t0=time.perf_counter();test=score(model,data['test']);infer=time.perf_counter()-t0
    mac={'independent':0,'shared':0,'post_rank4':128,'post_full':256,'mirror':32}[method]
    return {'validation':val,'test':test,'actual_payload_bytes':size,'payload_sha256':digest,'updates':updates,'train_examples':len(t),'examples_seen':updates*512,'train_wall_seconds':wall,'operator_macs_per_query':mac,'test_inference_examples_per_second':len(data['test'][0])/max(infer,1e-9),'heldout_pairs':int(mask.sum()),'seen_pairs':int((~mask).sum())}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--updates',type=int,default=1600);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    res={'experiment_id':'MA-395','seed':a.seed,'methods':{}}
    for m in ('independent','shared','post_rank4','post_full','mirror'):res['methods'][m]=fit(a.seed,m,a.updates,a.out)
    a.out.mkdir(parents=True,exist_ok=True);(a.out/f'{a.seed}_result.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res,indent=2))

if __name__=='__main__':main()
