import argparse
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import torch


torch.set_num_threads(1)
INPUT,HIDDEN,OUTPUT=8,12,3
UPDATES=300

def forward(x,w1,b1,w2,b2):
    return torch.relu(x@w1.T+b1)@w2.T+b2

def pack(arrays,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path,'w',compression=zipfile.ZIP_STORED) as z:
        for name in sorted(arrays):
            buf=io.BytesIO();np.lib.format.write_array(buf,np.ascontiguousarray(arrays[name]),allow_pickle=False)
            info=zipfile.ZipInfo(name+'.npy',date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_STORED;info.external_attr=0o600<<16;z.writestr(info,buf.getvalue())
    b=path.read_bytes();return len(b),hashlib.sha256(b).hexdigest()

def load(path):
    out={}
    with zipfile.ZipFile(path) as z:
        for f in z.namelist():out[f[:-4]]=np.load(io.BytesIO(z.read(f)),allow_pickle=False)
    return out

def world(seed):
    torch.manual_seed(seed);g=torch.Generator().manual_seed(seed+2)
    x=torch.randn(1024,INPUT,generator=g);a=torch.randn(INPUT,OUTPUT,generator=g)
    y=torch.sin(x@a)
    w1=torch.randn(HIDDEN,INPUT)*0.15;b1=torch.zeros(HIDDEN);w2=torch.randn(OUTPUT,HIDDEN)*0.15;b2=torch.zeros(OUTPUT)
    params=[torch.nn.Parameter(t.clone()) for t in (w1,b1,w2,b2)]
    opt=torch.optim.Adam(params,lr=0.01);start=time.perf_counter()
    for _ in range(UPDATES):
        pred=forward(x,*params);loss=(pred-y).square().mean();opt.zero_grad(set_to_none=True);loss.backward();opt.step()
    trainwall=time.perf_counter()-start
    testx=torch.randn(512,INPUT,generator=torch.Generator().manual_seed(seed+88));return [p.detach() for p in params],testx,trainwall

def arrays_for(model):
    return {f'w{i}':t.detach().cpu().numpy().astype('<f2') for i,t in enumerate(model)}

def transformed(model,perm):
    w1,b1,w2,b2=model;ix=torch.as_tensor(perm,dtype=torch.long)
    return [w1[ix],b1[ix],w2[:,ix],b2]

def eval_model(arr,x):
    p=[torch.from_numpy(np.array(arr[f'w{i}'],copy=True)).float() for i in range(4)]
    with torch.no_grad():return forward(x,*p)

def run(seed,split,outdir,jsonpath):
    model,x,trainwall=world(seed);rng=np.random.default_rng(seed+700);perms=[rng.permutation(HIDDEN) for _ in range(8)]
    reference=eval_model(arrays_for(model),x)
    rows=[];out=Path(outdir)
    # 8 separate checkpoint archives, each in the permuted but function-identical coordinate system.
    independent_paths=[]
    for i,p in enumerate(perms):
        q=transformed(model,p);path=out/f'{split}_{seed}_independent_{i}.zip';n,h=pack(arrays_for(q),path);independent_paths.append((path,n,h))
    independent_bytes=sum(n for _,n,_ in independent_paths)
    errors=[]
    for i,(path,_,_) in enumerate(independent_paths):
        pred=eval_model(load(path),x);errors.append(float(((pred-reference)**2).mean()))
    rows.append({'condition':split,'seed':seed,'method':'independent_8_checkpoints','serialized_bytes':independent_bytes,'payload_sha256':[h for _,_,h in independent_paths],
                 'updates':UPDATES,'train_examples':len(x),'train_wall_s':trainwall,'max_abs_output_diff':float(max(errors)**0.5),'rms_output_diff':float(np.mean(errors)**0.5),'variants':8})
    shared={**arrays_for(model),'permutations':np.stack(perms).astype(np.uint8),'meta':np.asarray([INPUT,HIDDEN,OUTPUT,8],dtype=np.uint16)}
    path=out/f'{split}_{seed}_shared_plus_permutations.zip';n,h=pack(shared,path);loaded=load(path)
    errors=[]
    for p in loaded['permutations']:
        q=transformed([torch.from_numpy(np.array(loaded[f'w{i}'],copy=True)).float() for i in range(4)],p)
        with torch.no_grad():pred=forward(x,*q)
        errors.append(float(((pred-reference)**2).mean()))
    rows.append({'condition':split,'seed':seed,'method':'shared_plus_paid_permutations','serialized_bytes':n,'payload_sha256':h,
                 'updates':UPDATES,'train_examples':len(x),'train_wall_s':trainwall,'max_abs_output_diff':float(max(errors)**0.5),'rms_output_diff':float(np.mean(errors)**0.5),'variants':8})
    # Negative control: permute incoming hidden weights but leave outgoing weights fixed.
    bad=transformed(model,perms[0]);bad[2]=model[2]
    with torch.no_grad():badpred=forward(x,*bad)
    rows.append({'condition':split,'seed':seed,'method':'incoming_only_negative_control','serialized_bytes':None,'payload_sha256':None,
                 'updates':UPDATES,'train_examples':len(x),'train_wall_s':trainwall,'max_abs_output_diff':float((badpred-reference).abs().max()),'rms_output_diff':float(((badpred-reference)**2).mean().sqrt()),'variants':1})
    result={'condition':split,'seed':seed,'summaries':rows};Path(jsonpath).parent.mkdir(parents=True,exist_ok=True);Path(jsonpath).write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True);ap.add_argument('--split',choices=['development','fresh'],required=True);ap.add_argument('--outdir',required=True);ap.add_argument('--json',required=True);a=ap.parse_args();run(a.seed,a.split,a.outdir,a.json)
