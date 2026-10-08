import argparse, hashlib, io, json, time, zipfile
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

Q,R,V,DIM,HALF,DOMAINS,CLASSES=8,8,64,16,8,4,8
TRAIN_N,VAL_N,TEST_N,BATCH,UPDATES=65536,32768,32768,128,512
LR=3e-3
METHODS=('independent','tied_concat','domain_film','domain_add','mirror_domain')
torch.set_num_threads(1)

def held_split(seed):
    offset=int(np.random.default_rng(seed+9001).integers(DOMAINS))
    token=np.arange(V)[:,None]
    domain=np.arange(DOMAINS)[None,:]
    return ((domain-token%DOMAINS-offset)%DOMAINS==0).reshape(-1)

def make_data(seed):
    held=held_split(seed)
    rng=np.random.default_rng(seed+31000)
    a=rng.normal(0,.5,(Q,HALF)).astype(np.float32)
    b=rng.normal(0,.5,(R,HALF)).astype(np.float32)
    angles=rng.uniform(-1.1,1.1,DOMAINS).astype(np.float32)
    head=rng.normal(0,.7,(CLASSES,DIM)).astype(np.float32)
    emb=np.zeros((V*DOMAINS,DIM),np.float32)
    for token in range(V):
        q,r=divmod(token,R)
        for d in range(DOMAINS):
            c,s=np.cos(angles[d]),np.sin(angles[d])
            emb[token*DOMAINS+d]=np.r_[c*a[q]-s*b[r],s*a[q]+c*b[r]]
    labels=(emb@head.T).argmax(1).astype(np.int64)
    train_pairs=np.flatnonzero(~held)
    result={'held':held,'angles':angles}
    for name,n,salt,allowed in [('train',TRAIN_N,17,train_pairs),('validation',VAL_N,37,np.arange(V*DOMAINS)),('test',TEST_N,59,np.arange(V*DOMAINS))]:
        srng=np.random.default_rng(seed+salt)
        ids=srng.choice(allowed,n,replace=True).astype(np.int64)
        result[name]={'token':torch.tensor(ids//DOMAINS),'domain':torch.tensor(ids%DOMAINS),'labels':torch.tensor(labels[ids],dtype=torch.long),'held':torch.tensor(held[ids],dtype=torch.bool)}
    return result

class Model(torch.nn.Module):
    def __init__(self,method,seed):
        super().__init__(); self.method=method
        torch.manual_seed(seed+METHODS.index(method)*101)
        if method=='independent': self.embedding=torch.nn.Parameter(torch.randn(V*DOMAINS,DIM)*.04)
        else:
            self.q_table=torch.nn.Parameter(torch.randn(Q,HALF)*.04)
            self.r_table=torch.nn.Parameter(torch.randn(R,HALF)*.04)
            if method=='domain_film':
                self.scale=torch.nn.Parameter(torch.ones(DOMAINS,DIM)); self.shift=torch.nn.Parameter(torch.zeros(DOMAINS,DIM))
            elif method=='domain_add': self.domain_bias=torch.nn.Parameter(torch.zeros(DOMAINS,DIM))
            elif method=='mirror_domain': self.angle=torch.nn.Parameter(torch.zeros(DOMAINS))
        self.head=torch.nn.Parameter(torch.randn(CLASSES,DIM)*.04); self.bias=torch.nn.Parameter(torch.zeros(CLASSES))
    def embed(self,token,domain):
        if self.method=='independent': return self.embedding[token*DOMAINS+domain]
        a,b=self.q_table[token//R],self.r_table[token%R]
        x=torch.cat([a,b],dim=-1)
        if self.method=='tied_concat': return x
        if self.method=='domain_film': return x*self.scale[domain]+self.shift[domain]
        if self.method=='domain_add': return x+self.domain_bias[domain]
        angle=self.angle[domain]; c,s=torch.cos(angle)[:,None],torch.sin(angle)[:,None]
        return torch.cat([c*a-s*b,s*a+c*b],dim=-1)
    def forward(self,token,domain): return self.embed(token,domain)@self.head.T+self.bias

def evaluate(model,split,data):
    with torch.no_grad():
        logits=model(split['token'],split['domain']); correct=logits.argmax(1).eq(split['labels']); seen=~split['held']; held=split['held']
        role=torch.arange(V*DOMAINS); emb=model.embed(role//DOMAINS,role%DOMAINS).cpu().numpy().astype('<f2')
        return {'accuracy':float(correct.float().mean()),'nll':float(F.cross_entropy(logits,split['labels'])),'seen_accuracy':float(correct[seen].float().mean()),'heldout_accuracy':float(correct[held].float().mean()),'seen_nll':float(F.cross_entropy(logits[seen],split['labels'][seen])),'heldout_nll':float(F.cross_entropy(logits[held],split['labels'][held])),'unique_embeddings_fp16':int(len(np.unique(emb,axis=0))),'possible_roles':V*DOMAINS}

def arrays_for(model):
    a={'meta':np.asarray([Q,R,V,DOMAINS,DIM,CLASSES,METHODS.index(model.method)],dtype=np.uint16)}
    for k,v in model.state_dict().items(): a[k]=v.detach().cpu().numpy().astype('<f2')
    return a

def pack(arrays,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_STORED) as z:
        for name in sorted(arrays):
            buf=io.BytesIO();np.lib.format.write_array(buf,np.ascontiguousarray(arrays[name]),allow_pickle=False)
            info=zipfile.ZipInfo(name+'.npy',date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_STORED;info.external_attr=0o600<<16;z.writestr(info,buf.getvalue())
    raw=path.read_bytes();return len(raw),hashlib.sha256(raw).hexdigest()

def load_model(method,seed,arrays):
    model=Model(method,seed);model.load_state_dict({k:torch.tensor(arrays[k],dtype=v.dtype) for k,v in model.state_dict().items()});model.eval();return model

def train(method,seed,data,outdir):
    model=Model(method,seed);opt=torch.optim.Adam(model.parameters(),lr=LR);tr=data['train'];rng=np.random.default_rng(seed*47+METHODS.index(method));perm=rng.permutation(len(tr['token']));start=time.perf_counter()
    for step in range(UPDATES):
        ix=torch.from_numpy(perm[step*BATCH:(step+1)*BATCH]);logits=model(tr['token'][ix],tr['domain'][ix]);loss=F.cross_entropy(logits,tr['labels'][ix]);opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    wall=time.perf_counter()-start;arr=arrays_for(model);path=Path(outdir)/'payloads'/f'development_{seed}_{method}.npz';size,digest=pack(arr,path);loaded=load_model(method,seed,arr)
    return {'method':method,'serialized_bytes':size,'payload_sha256':digest,'optimizer_updates':UPDATES,'training_examples_seen':UPDATES*BATCH,'train_wall_s':wall,'validation':evaluate(loaded,data['validation'],data),'test':evaluate(loaded,data['test'],data),'macs_per_lookup_proxy':{'embedding_and_classifier':CLASSES*DIM,'view_coordinate_ops':2 if method=='mirror_domain' else (2*DIM if method=='domain_film' else 0)}}

def run(seed,condition,outdir,dest):
    data=make_data(seed);rows=[train(m,seed,data,outdir) for m in METHODS]
    result={'condition':condition,'seed':seed,'heldout_token_domain_pairs':np.flatnonzero(data['held']).tolist(),'heldout_pair_fraction':float(data['held'].mean()),'teacher_angles':data['angles'].tolist(),'summaries':rows}
    Path(dest).parent.mkdir(parents=True,exist_ok=True);Path(dest).write_text(json.dumps(result,indent=2)+'\n');return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--condition',choices=['development','fresh'],required=True);p.add_argument('--outdir',required=True);p.add_argument('--json',required=True);a=p.parse_args();run(a.seed,a.condition,a.outdir,a.json)
