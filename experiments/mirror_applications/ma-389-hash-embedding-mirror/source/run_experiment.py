"""Synthetic lexical-label screen for Hash Embeddings and Mirror importance codes."""
from __future__ import annotations
import argparse,hashlib,io,json,math,time,zipfile
from pathlib import Path
import numpy as np
import torch
from torch import nn

VOCAB=1024; DIM=16; CLASSES=8; POOL=64


def make_world(seed):
    # A balanced but token-specific lexical target; hash assignment is independent.
    rg=np.random.default_rng(seed+117)
    labels=np.arange(VOCAB,dtype=np.int64)%CLASSES
    rg.shuffle(labels)
    hg=np.random.default_rng(seed+991)
    indices=hg.integers(0,POOL,size=(VOCAB,2),dtype=np.uint8)
    # Uniform token queries make every vocabulary item equally important.
    def split(n,offset):
        g=torch.Generator().manual_seed(seed+offset)
        ids=torch.randint(VOCAB,(n,),generator=g)
        return ids,torch.from_numpy(labels[ids.numpy()].copy()).long()
    data={"train":split(65536,0),"validation":split(16384,1000),"test":split(32768,2000)}
    return data,torch.from_numpy(indices.copy()).long(),torch.from_numpy(labels).long()


class TokenModel(nn.Module):
    def __init__(self,method,hash_indices):
        super().__init__();self.method=method
        if method!='full':self.register_buffer('hash_indices',hash_indices.to(torch.uint8))
        if method=='full': self.embedding=nn.Parameter(torch.randn(VOCAB,DIM)*.04)
        else:
            self.components=nn.Parameter(torch.randn(POOL,DIM)*.08)
            if method=='hash': self.importance=nn.Parameter(torch.ones(VOCAB,2))
            elif method=='scalar': self.gain=nn.Parameter(torch.ones(VOCAB))
            elif method=='mirror': self.angle=nn.Parameter(torch.zeros(VOCAB))
        self.classifier=nn.Linear(DIM,CLASSES)

    def token_embeddings(self,ids):
        if self.method=='full': return self.embedding[ids]
        pair=self.hash_indices[ids.long()].long()
        a=self.components[pair[...,0]];b=self.components[pair[...,1]]
        if self.method=='hash':
            w=self.importance[ids.long()];return w[...,0,None]*a+w[...,1,None]*b
        if self.method=='unweighted': return a+b
        if self.method=='scalar': return self.gain[ids.long(),None]*(a+b)
        theta=self.angle[ids.long()]
        return torch.cos(theta)[...,None]*a+torch.sin(theta)[...,None]*b

    def forward(self,ids): return self.classifier(self.token_embeddings(ids))


def archive(path,model):
    state={k:v.detach().cpu().numpy() for k,v in model.state_dict().items()}
    # Quantize learned floating inference state to FP16; fixed component indices remain uint8.
    arrays={k:(v.astype(np.uint8) if k=='hash_indices' else v.astype(np.float16)) for k,v in state.items()}
    meta=json.dumps({'vocabulary':VOCAB,'dim':DIM,'classes':CLASSES,'hash_functions':2,'pool_size':POOL},sort_keys=True).encode()
    path.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_STORED) as z:
        z.writestr('metadata.json',meta)
        for name,array in sorted(arrays.items()):
            buf=io.BytesIO();np.save(buf,array,allow_pickle=False);z.writestr(f'arrays/{name}.npy',buf.getvalue())
    blob=path.read_bytes();return len(blob),hashlib.sha256(blob).hexdigest()


def load_archive(path,model):
    with zipfile.ZipFile(path) as z:
        arrays={Path(n).stem:np.load(io.BytesIO(z.read(n)),allow_pickle=False) for n in z.namelist() if n.endswith('.npy')}
    model.load_state_dict({k:torch.from_numpy(arrays[k].astype(np.uint8 if k=='hash_indices' else np.float32)) for k in model.state_dict()})
    return arrays


def collision_bins(indices):
    tuples=[tuple(map(int,x)) for x in indices.tolist()]
    from collections import Counter
    counts=Counter(tuples)
    return torch.tensor([counts[x] for x in tuples],dtype=torch.long)


@torch.no_grad()
def evaluate(model,ids,labels,degree_by_token,token_labels):
    model.eval();logits=model(ids);loss=nn.functional.cross_entropy(logits,labels,reduction='none')
    pred=logits.argmax(-1);ok=(pred==labels);sample_degrees=degree_by_token[ids]
    by={}
    for degree in torch.unique(sample_degrees).tolist():
        mask=sample_degrees==degree;by[str(degree)]={'examples':int(mask.sum()),'accuracy':float(ok[mask].float().mean()),'nll':float(loss[mask].mean())}
    # Per-token test correctness gives collision sensitivity without frequency weighting artifacts.
    ids_all=torch.arange(VOCAB); y_all=token_labels
    logits_all=model(ids_all);per_ok=(logits_all.argmax(-1)==y_all)
    per_nll=nn.functional.cross_entropy(logits_all,y_all,reduction='none')
    per_bins={}
    for degree in torch.unique(degree_by_token).tolist():
        mask=degree_by_token==degree;per_bins[str(degree)]={'tokens':int(mask.sum()),'accuracy':float(per_ok[mask].float().mean()),'nll':float(per_nll[mask].mean())}
    return {'accuracy':float(ok.float().mean()),'nll':float(loss.mean()),'collision_bins':by,'per_token_collision_bins':per_bins}


def fit(seed,method,updates,out):
    data,hashes,token_labels=make_world(seed)
    torch.manual_seed(seed+sum(map(ord,method)))
    model=TokenModel(method,hashes)
    # Use the same output head initialization across methods in each world.
    head_rng=torch.Generator().manual_seed(seed+878)
    nn.init.xavier_uniform_(model.classifier.weight,generator=head_rng)
    nn.init.zeros_(model.classifier.bias)
    opt=torch.optim.Adam(model.parameters(),lr=.003)
    ids,y=data['train'];start=time.perf_counter()
    for _ in range(updates):
        ix=torch.randint(len(ids),(512,));logits=model(ids[ix]);loss=nn.functional.cross_entropy(logits,y[ix])
        opt.zero_grad();loss.backward();opt.step()
    wall=time.perf_counter()-start
    dest=out/f'{seed}_{method}.zip';size,digest=archive(dest,model)
    arrays=load_archive(dest,model)
    degrees=collision_bins(hashes)
    validation=evaluate(model,*data['validation'],degrees,token_labels)
    infer_start=time.perf_counter();test=evaluate(model,*data['test'],degrees,token_labels);infer_wall=time.perf_counter()-infer_start
    # Mean forward multiply-accumulate proxy per query; table lookups are listed separately.
    gen={'full':0,'hash':32,'unweighted':0,'scalar':16,'mirror':32}[method]
    return {'validation':validation,'test':test,'actual_payload_bytes':size,'payload_sha256':digest,
            'updates':updates,'examples_seen':updates*512,'train_examples':len(ids),'train_wall_seconds':wall,
            'component_lookups_per_query':0 if method=='full' else 2,
            'importance_generation_macs_per_query':gen,
            'test_inference_examples_per_second':len(data['test'][0])/max(infer_wall,1e-9),
            'theoretical_trainable_parameters':sum(p.numel() for p in model.parameters())}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--updates',type=int,default=1200);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    result={'experiment_id':'MA-389','seed':a.seed,'methods':{}}
    for m in ('full','hash','unweighted','scalar','mirror'):
        result['methods'][m]=fit(a.seed,m,a.updates,a.out)
    a.out.mkdir(parents=True,exist_ok=True);(a.out/f'{a.seed}_result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
