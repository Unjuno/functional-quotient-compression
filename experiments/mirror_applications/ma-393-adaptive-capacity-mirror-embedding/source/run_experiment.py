"""Zipf-frequency token classification with adaptive and rare-token Mirror tables."""
from __future__ import annotations
import argparse,hashlib,io,json,time,zipfile
from pathlib import Path
import numpy as np
import torch
from torch import nn

V=1024;D=32;C=8;HEAD=64;MID=192;TAIL=V-HEAD-MID


def make_world(seed):
    ranks=torch.arange(1,V+1,dtype=torch.float64)
    probs=(1/ranks**1.1);probs=probs/probs.sum()
    data={}
    for split,n,off in [('train',65536,0),('validation',16384,1000),('test',32768,2000)]:
        g=torch.Generator().manual_seed(seed+off)
        ids=torch.multinomial(probs,n,replacement=True,generator=g)
        data[split]=(ids,(ids%C).long())
    return data,probs


def rotate(x,theta):
    a,b=x[...,0::2],x[...,1::2];c=torch.cos(theta)[...,None];s=torch.sin(theta)[...,None]
    out=torch.empty_like(x);out[...,0::2]=c*a-s*b;out[...,1::2]=s*a+c*b;return out


class AdaptiveTokenModel(nn.Module):
    def __init__(self,method):
        super().__init__();self.method=method
        if method=='full':self.embedding=nn.Parameter(torch.randn(V,D)*.04)
        elif method=='adaptive':
            self.head=nn.Parameter(torch.randn(HEAD,D)*.04);self.middle=nn.Parameter(torch.randn(MID,16)*.04);self.tail=nn.Parameter(torch.randn(TAIL,8)*.04)
        elif method=='uniform12':self.embedding=nn.Parameter(torch.randn(V,12)*.04)
        else:
            self.head=nn.Parameter(torch.randn(HEAD,D)*.04);self.middle=nn.Parameter(torch.randn(MID,16)*.04)
            self.tail_code=nn.Parameter(torch.randn(TAIL,4)*.04);self.basis=nn.Parameter(torch.randn(4,D)*.04)
            if method=='tail_scalar':self.gain=nn.Parameter(torch.ones(TAIL))
            if method=='mirror':self.angle=nn.Parameter(torch.zeros(TAIL))
        self.classifier=nn.Linear(D,C)

    def token_embeddings(self,ids):
        if self.method=='full':return self.embedding[ids]
        if self.method=='uniform12':return torch.nn.functional.pad(self.embedding[ids],(0,20))
        out=torch.zeros((len(ids),D),device=ids.device,dtype=self.classifier.weight.dtype)
        hm=ids<HEAD;mm=(ids>=HEAD)&(ids<HEAD+MID);tm=ids>=HEAD+MID
        if hm.any():out[hm]=self.head[ids[hm]]
        if mm.any():out[mm,:16]=self.middle[ids[mm]-HEAD]
        if tm.any():
            ti=ids[tm]-HEAD-MID;z=self.tail[ti] if self.method=='adaptive' else self.tail_code[ti]@self.basis
            if self.method=='tail_scalar':z=self.gain[ti,None]*z
            if self.method=='mirror':z=rotate(z,self.angle[ti])
            out[tm]=z if self.method!='adaptive' else torch.nn.functional.pad(z,(0,24))
        return out

    def forward(self,ids):return self.classifier(self.token_embeddings(ids))


def archive(path,model):
    arrays={k:v.detach().cpu().numpy().astype(np.float16) for k,v in model.state_dict().items()}
    path.parent.mkdir(parents=True,exist_ok=True)
    meta=json.dumps({'vocabulary':V,'embedding_dim':D,'classes':C,'bands':[HEAD,MID,TAIL],'adaptive_dims':[32,16,8],'mirror_tail_code_dim':4},sort_keys=True).encode()
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_STORED) as z:
        z.writestr('metadata.json',meta)
        for k,a in sorted(arrays.items()):
            b=io.BytesIO();np.save(b,a,allow_pickle=False);z.writestr(f'arrays/{k}.npy',b.getvalue())
    blob=path.read_bytes();return len(blob),hashlib.sha256(blob).hexdigest()


def reload(path,model):
    with zipfile.ZipFile(path) as z:a={Path(n).stem:np.load(io.BytesIO(z.read(n)),allow_pickle=False) for n in z.namelist() if n.endswith('.npy')}
    model.load_state_dict({k:torch.from_numpy(a[k].astype(np.float32)) for k in model.state_dict()})


@torch.no_grad()
def evaluate(model,ids,y):
    logits=model(ids);loss=nn.functional.cross_entropy(logits,y,reduction='none');correct=(logits.argmax(-1)==y)
    bands={'head':ids<HEAD,'middle':(ids>=HEAD)&(ids<HEAD+MID),'tail':ids>=HEAD+MID};out={}
    for name,mask in bands.items():out[name]={'examples':int(mask.sum()),'accuracy':float(correct[mask].float().mean()),'nll':float(loss[mask].mean())}
    out['overall']={'examples':len(ids),'accuracy':float(correct.float().mean()),'nll':float(loss.mean())}
    return out


def fit(seed,method,updates,out):
    data,probs=make_world(seed);torch.manual_seed(seed+sum(map(ord,method)));model=AdaptiveTokenModel(method)
    gen=torch.Generator().manual_seed(seed+777);nn.init.xavier_uniform_(model.classifier.weight,generator=gen);nn.init.zeros_(model.classifier.bias)
    opt=torch.optim.Adam(model.parameters(),lr=.003);ids,y=data['train'];start=time.perf_counter()
    for _ in range(updates):
        ix=torch.randint(len(ids),(512,));loss=nn.functional.cross_entropy(model(ids[ix]),y[ix]);opt.zero_grad();loss.backward();opt.step()
    wall=time.perf_counter()-start;path=out/f'{seed}_{method}.zip';size,digest=archive(path,model);reload(path,model)
    val=evaluate(model,*data['validation']);t0=time.perf_counter();test=evaluate(model,*data['test']);infer_wall=time.perf_counter()-t0
    coverage={band:int(torch.unique(ids[(ids<HEAD) if band=='head' else ((ids>=HEAD)&(ids<HEAD+MID)) if band=='middle' else (ids>=HEAD+MID)]).numel()) for band in ('head','middle','tail')}
    mac={'full':0,'adaptive':0,'uniform12':0,'tail_basis':128,'tail_scalar':144,'mirror':192}[method]
    return {'validation':val,'test':test,'actual_payload_bytes':size,'payload_sha256':digest,'updates':updates,
            'train_examples':len(ids),'examples_seen':updates*512,'train_wall_seconds':wall,'unique_train_tokens_by_band':coverage,
            'operator_macs_per_tail_query':mac,'test_inference_examples_per_second':len(data['test'][0])/max(infer_wall,1e-9),
            'zipf_exponent':1.1,'band_sizes':{'head':HEAD,'middle':MID,'tail':TAIL}}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--updates',type=int,default=1200);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    result={'experiment_id':'MA-393','seed':a.seed,'methods':{}}
    for m in ('full','adaptive','uniform12','tail_basis','tail_scalar','mirror'):result['methods'][m]=fit(a.seed,m,a.updates,a.out)
    a.out.mkdir(parents=True,exist_ok=True);(a.out/f'{a.seed}_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
