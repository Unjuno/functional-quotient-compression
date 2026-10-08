"""Product-key token collisions resolved by scalar, native hash or Mirror codes."""
from __future__ import annotations
import argparse,hashlib,io,json,time,zipfile
from pathlib import Path
import numpy as np
import torch
from torch import nn

V=4096;D=16;C=4;N1=32;N2=32


def address(ids):return ids%N1,(ids//N1)%N2
def labels(ids):return ids//(N1*N2)


def make_world(seed):
    data={}
    for split,n,off in [('train',65536,0),('validation',16384,1000),('test',32768,2000)]:
        g=torch.Generator().manual_seed(seed+off);ids=torch.randint(V,(n,),generator=g);data[split]=(ids,labels(ids))
    return data


class ProductTokenModel(nn.Module):
    def __init__(self,method):
        super().__init__();self.method=method
        if method=='full':self.embedding=nn.Parameter(torch.randn(V,D)*.04)
        else:
            self.left=nn.Parameter(torch.randn(N1,D)*.08);self.right=nn.Parameter(torch.randn(N2,D)*.08)
            if method=='scalar':self.gain=nn.Parameter(torch.ones(V))
            elif method=='hash':self.importance=nn.Parameter(torch.ones(V,2))
            elif method=='mirror':self.angle=nn.Parameter(torch.zeros(V))
        self.classifier=nn.Linear(D,C)

    def token_embeddings(self,ids):
        if self.method=='full':return self.embedding[ids]
        q,r=address(ids);a=self.left[q];b=self.right[r]
        if self.method=='product':return a+b
        if self.method=='scalar':return self.gain[ids,None]*(a+b)
        if self.method=='hash':
            w=self.importance[ids];return w[:,0,None]*a+w[:,1,None]*b
        theta=self.angle[ids]
        return torch.cos(theta)[:,None]*a+torch.sin(theta)[:,None]*b

    def forward(self,ids):return self.classifier(self.token_embeddings(ids))


def archive(path,model):
    arrays={k:v.detach().cpu().numpy().astype(np.float16) for k,v in model.state_dict().items()}
    path.parent.mkdir(parents=True,exist_ok=True)
    meta=json.dumps({'vocabulary':V,'dim':D,'classes':C,'left_subkeys':N1,'right_subkeys':N2,'address_rule':'q=id%32,r=(id//32)%32','tokens_per_address':4},sort_keys=True).encode()
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
    logits=model(ids);loss=nn.functional.cross_entropy(logits,y,reduction='none');ok=logits.argmax(-1)==y
    by={}
    for c in range(C):
        mask=y==c;by[str(c)]={'examples':int(mask.sum()),'accuracy':float(ok[mask].float().mean()),'nll':float(loss[mask].mean())}
    all_ids=torch.arange(V);rows=model.token_embeddings(all_ids).half();unique=int(torch.unique(rows,dim=0).shape[0])
    return {'accuracy':float(ok.float().mean()),'nll':float(loss.mean()),'collision_group_metrics':by,'decoded_unique_embeddings':unique,'decoded_unique_fraction':unique/V}


def fit(seed,method,updates,out):
    data=make_world(seed);torch.manual_seed(seed+sum(map(ord,method)));model=ProductTokenModel(method)
    g=torch.Generator().manual_seed(seed+999);nn.init.xavier_uniform_(model.classifier.weight,generator=g);nn.init.zeros_(model.classifier.bias)
    opt=torch.optim.Adam(model.parameters(),lr=.003);ids,y=data['train'];start=time.perf_counter()
    for _ in range(updates):
        ix=torch.randint(len(ids),(512,));loss=nn.functional.cross_entropy(model(ids[ix]),y[ix]);opt.zero_grad();loss.backward();opt.step()
    wall=time.perf_counter()-start;path=out/f'{seed}_{method}.zip';size,digest=archive(path,model);reload(path,model)
    val=evaluate(model,*data['validation']);t0=time.perf_counter();test=evaluate(model,*data['test']);infer=time.perf_counter()-t0
    mac={'full':0,'product':0,'scalar':16,'hash':32,'mirror':32}[method]
    return {'validation':val,'test':test,'actual_payload_bytes':size,'payload_sha256':digest,'updates':updates,'train_examples':len(ids),'examples_seen':updates*512,'train_wall_seconds':wall,'subkey_lookups_per_query':0 if method=='full' else 2,'importance_macs_per_query':mac,'test_inference_examples_per_second':len(data['test'][0])/max(infer,1e-9)}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--updates',type=int,default=1200);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    result={'experiment_id':'MA-397','seed':a.seed,'methods':{}}
    for m in ('full','product','scalar','hash','mirror'):result['methods'][m]=fit(a.seed,m,a.updates,a.out)
    a.out.mkdir(parents=True,exist_ok=True);(a.out/f'{a.seed}_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
