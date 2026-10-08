"""Quotient/remainder embedding compositions with a per-ID Mirror coordinate."""
from __future__ import annotations
import argparse,hashlib,io,json,time,zipfile
from pathlib import Path
import numpy as np
import torch
from torch import nn

V=1024;D=16;QROWS=32;RROWS=32;CLASSES=8


def target(ids,family):
    q=ids//RROWS;r=ids%RROWS
    if family=='separable':return (3*q+5*r)%CLASSES
    if family=='xor':return (q ^ r)%CLASSES
    raise ValueError(family)


def make_world(seed,family):
    data={}
    for split,n,offset in [('train',65536,0),('validation',16384,1000),('test',32768,2000)]:
        g=torch.Generator().manual_seed(seed+offset)
        ids=torch.randint(V,(n,),generator=g)
        data[split]=(ids,target(ids,family).long())
    return data


class Composition(nn.Module):
    def __init__(self,method):
        super().__init__();self.method=method
        if method=='full':self.embedding=nn.Parameter(torch.randn(V,D)*.05)
        else:
            self.q=nn.Parameter(torch.randn(QROWS,D)*.08)
            self.r=nn.Parameter(torch.randn(RROWS,D)*.08)
            if method=='mirror':self.angle=nn.Parameter(torch.zeros(V))
            elif method=='concat':self.projection=nn.Parameter(torch.randn(2*D,D)*.05)
        self.classifier=nn.Linear(D,CLASSES)

    def embedding_rows(self):
        ids=torch.arange(V,device=next(self.parameters()).device);q=ids//RROWS;r=ids%RROWS
        if self.method=='full':return self.embedding[ids]
        if self.method=='add':return self.q[q]+self.r[r]
        if self.method=='multiply':return self.q[q]*self.r[r]
        if self.method=='concat':return torch.cat([self.q[q],self.r[r]],-1)@self.projection
        theta=self.angle[ids]
        return torch.cos(theta)[:,None]*self.q[q]+torch.sin(theta)[:,None]*self.r[r]

    def forward(self,ids):
        return self.classifier(self.embedding_rows()[ids])


def archive(path,model):
    state={k:v.detach().cpu().numpy().astype(np.float16) for k,v in model.state_dict().items()}
    path.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_STORED) as z:
        z.writestr('metadata.json',json.dumps({'vocabulary':V,'dim':D,'q_rows':QROWS,'r_rows':RROWS,'id_rule':'q=id//32,r=id%32'},sort_keys=True).encode())
        for k,a in sorted(state.items()):
            b=io.BytesIO();np.save(b,a,allow_pickle=False);z.writestr(f'arrays/{k}.npy',b.getvalue())
    blob=path.read_bytes();return len(blob),hashlib.sha256(blob).hexdigest()


def reload(path,model):
    with zipfile.ZipFile(path) as z:arrays={Path(n).stem:np.load(io.BytesIO(z.read(n)),allow_pickle=False) for n in z.namelist() if n.endswith('.npy')}
    model.load_state_dict({k:torch.from_numpy(arrays[k].astype(np.float32)) for k in model.state_dict()})


@torch.no_grad()
def evaluate(model,ids,y):
    model.eval();logits=model(ids);nll=nn.functional.cross_entropy(logits,y)
    rows=model.embedding_rows().half()
    decoded_unique=int(torch.unique(rows,dim=0).shape[0])
    return {'accuracy':float((logits.argmax(-1)==y).float().mean()),'nll':float(nll),
            'decoded_unique_embeddings':decoded_unique,'decoded_unique_fraction':decoded_unique/V,
            'address_unique_pairs':V}


def fit(seed,family,method,updates,out):
    data=make_world(seed,family);torch.manual_seed(seed+sum(map(ord,method)))
    model=Composition(method)
    head_rng=torch.Generator().manual_seed(seed+878);nn.init.xavier_uniform_(model.classifier.weight,generator=head_rng);nn.init.zeros_(model.classifier.bias)
    opt=torch.optim.Adam(model.parameters(),lr=.003);ids,y=data['train'];start=time.perf_counter()
    for _ in range(updates):
        ix=torch.randint(len(ids),(512,));loss=nn.functional.cross_entropy(model(ids[ix]),y[ix])
        opt.zero_grad();loss.backward();opt.step()
    wall=time.perf_counter()-start
    path=out/f'{seed}_{family}_{method}.zip';size,digest=archive(path,model);reload(path,model)
    validation=evaluate(model,*data['validation'])
    t0=time.perf_counter();test=evaluate(model,*data['test']);infer_wall=time.perf_counter()-t0
    mac={'full':0,'add':16,'multiply':16,'concat':32,'mirror':32}[method]
    return {'validation':validation,'test':test,'actual_payload_bytes':size,'payload_sha256':digest,
            'updates':updates,'train_examples':len(ids),'examples_seen':updates*512,'train_wall_seconds':wall,
            'operator_macs_per_query':mac,'quotient_remainder_lookups_per_query':2 if method!='full' else 0,
            'test_inference_examples_per_second':len(data['test'][0])/max(infer_wall,1e-9),
            'trainable_parameters':sum(p.numel() for p in model.parameters())}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--family',choices=['separable','xor'],required=True);ap.add_argument('--updates',type=int,default=1200);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    res={'experiment_id':'MA-391','seed':a.seed,'family':a.family,'methods':{}}
    for m in ('full','add','multiply','concat','mirror'):res['methods'][m]=fit(a.seed,a.family,m,a.updates,a.out)
    a.out.mkdir(parents=True,exist_ok=True);(a.out/f'{a.seed}_{a.family}_result.json').write_text(json.dumps(res,indent=2)+'\n');print(json.dumps(res,indent=2))

if __name__=='__main__':main()
