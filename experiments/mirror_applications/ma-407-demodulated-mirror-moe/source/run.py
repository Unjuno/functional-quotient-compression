#!/usr/bin/env python3
"""MA-407 synthetic demodulated Mirror expert screen."""
import argparse, hashlib, io, json, math, random, time
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'artifacts'
METHODS=['shared','mirror_raw','mirror_demod','ia3','rank1','independent']
DEV=[40700,40701]; FRESH=[40710,40711,40712]; SEEDS=[0,1,2]; LRS=[.003,.01]
UPDATES,BATCH=200,128; DIN,HID,CLS,CTX=32,48,10,4

def fixseed(s): random.seed(s);torch.manual_seed(s);torch.set_num_threads(1)

def rotate(w,angles):
    # Rotate four disjoint pairs of input rows.
    v=w
    for k in range(4):
        i,j=2*k,2*k+1; c=torch.cos(angles[k]);s=torch.sin(angles[k]); a,b=v[i].clone(),v[j].clone();v=v.clone();v[i]=c*a-s*b;v[j]=s*a+c*b
    return v

def world_data(world):
    g=torch.Generator().manual_seed(world)
    x=torch.randn(4096,DIN,generator=g); base1=torch.randn(DIN,HID,generator=g)/math.sqrt(DIN); base2=torch.randn(HID,CLS,generator=g)/math.sqrt(HID)
    teacher=[]
    for c in range(CTX):
        ang=torch.randn(4,generator=g)*.85; w=rotate(base1,ang)
        target_norm=base1.norm(dim=0).clamp_min(1e-8); effective=w*(target_norm/w.norm(dim=0).clamp_min(1e-8))[None,:]
        teacher.append((effective,base2))
    cid=torch.arange(4096)%CTX; y=[]
    for i in range(4096):
        w1,w2=teacher[int(cid[i])]; y.append(int((F.relu(x[i]@w1)@w2).argmax()))
    return x,cid,torch.tensor(y)

class Model(nn.Module):
    def __init__(self,m):
        super().__init__();self.method=m
        if m=='independent': self.w1=nn.Parameter(torch.randn(CTX,DIN,HID)*.12);self.w2=nn.Parameter(torch.randn(CTX,HID,CLS)*.12)
        else:self.w1=nn.Parameter(torch.randn(DIN,HID)*.12);self.w2=nn.Parameter(torch.randn(HID,CLS)*.12)
        if m.startswith('mirror'): self.angles=nn.Parameter(torch.zeros(CTX,4))
        if m=='ia3': self.scale=nn.Parameter(torch.ones(CTX,DIN))
        if m=='rank1': self.a=nn.Parameter(torch.randn(CTX,DIN,1)*.01);self.b=nn.Parameter(torch.randn(CTX,1,HID)*.01)

    def forward(self,x,c,return_hidden=False):
        if self.method=='independent': w=self.w1[c]
        elif self.method.startswith('mirror'):
            w=self.w1[None].expand(x.size(0),-1,-1).clone();cs=self.angles[c].cos();sn=self.angles[c].sin()
            for k in range(4):
                i,j=2*k,2*k+1;u,v=w[:,i].clone(),w[:,j].clone();w[:,i]=cs[:,k,None]*u-sn[:,k,None]*v;w[:,j]=sn[:,k,None]*u+cs[:,k,None]*v
            if self.method=='mirror_demod':
                norms=w.norm(dim=1).clamp_min(1e-8); target=self.w1.norm(dim=0).clamp_min(1e-8);w=w*(target[None,None,:]/norms[:,None,:])
        else:w=self.w1
        if self.method=='ia3': x=x*self.scale[c]
        elif self.method=='rank1': w=w+self.a[c]@self.b[c]
        h=F.relu(torch.bmm(x[:,None,:],w).squeeze(1) if w.ndim==3 else x@w)
        logits=torch.bmm(h[:,None,:],self.w2[c]).squeeze(1) if self.method=='independent' else h@self.w2
        return (logits,h) if return_hidden else logits

def serialize(model):
    b=io.BytesIO();torch.save({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},b,_use_new_zipfile_serialization=True);return b.getvalue()

def train_one(m,world,seed,lr,split):
    fixseed(world*100+seed);x,c,y=world_data(world);model=Model(m);opt=torch.optim.AdamW(model.parameters(),lr=lr);st=time.perf_counter()
    for _ in range(UPDATES):
        ix=torch.randint(0,3072,(BATCH,));loss=F.cross_entropy(model(x[ix],c[ix]),y[ix]);opt.zero_grad();loss.backward();opt.step()
    wall=time.perf_counter()-st
    with torch.no_grad():
        logits,h=model(x[3072:],c[3072:],True);acc=(logits.argmax(-1)==y[3072:]).float().mean().item();ce=F.cross_entropy(logits,y[3072:]).item()
        per=[h[c[3072:]==k].square().mean().sqrt().item() for k in range(CTX)];scale_cv=(torch.tensor(per).std(unbiased=False)/(torch.tensor(per).mean()+1e-12)).item()
        per_acc=[(logits[c[3072:]==k].argmax(-1)==y[3072:][c[3072:]==k]).float().mean().item() for k in range(CTX)]
    raw=serialize(model);name=f'{split}_{world}_{seed}_{m}_lr{lr:g}.pt';(OUT/'payloads'/name).write_bytes(raw)
    loaded=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True);clone=Model(m);clone.load_state_dict(loaded);roundtrip=serialize(clone);assert raw==roundtrip
    mac=2*DIN*HID+2*HID*CLS
    if m.startswith('mirror') or m in ('ia3','rank1'):mac+=DIN*HID
    return {'split':split,'world':world,'seed':seed,'method':m,'lr':lr,'payload_file':name,'serialized_bytes':len(raw),'payload_sha256':hashlib.sha256(raw).hexdigest(),'roundtrip_sha256':hashlib.sha256(roundtrip).hexdigest(),'examples':UPDATES*BATCH,'updates':UPDATES,'active_mac_proxy_per_example':mac,'wall_time_s':round(wall,6),'accuracy':acc,'cross_entropy':ce,'context_activation_rms':per,'activation_rms_cv':scale_cv,'per_context_accuracy':per_acc}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['dev','fresh'],required=True);args=ap.parse_args();OUT.mkdir(exist_ok=True);(OUT/'payloads').mkdir(exist_ok=True);rows=[];selection={}
    if args.phase=='dev':
        for m in METHODS:
            vals={lr:[] for lr in LRS}
            for lr in LRS:
                for w in DEV:
                    for s in SEEDS:
                        r=train_one(m,w,s,lr,'development');rows.append(r);vals[lr].append(r['accuracy'])
            selection[m]=max(LRS,key=lambda lr:sum(vals[lr])/len(vals[lr]))
        (OUT/'development_selection.json').write_text(json.dumps(selection,indent=2)+'\n');mode='w'
    else:
        selection=json.loads((OUT/'development_selection.json').read_text())
        for m in METHODS:
            for w in FRESH:
                for s in SEEDS:rows.append(train_one(m,w,s,selection[m],'fresh'))
        mode='a'
    with (OUT/'runs.jsonl').open(mode) as f:
        for r in rows:f.write(json.dumps(r,sort_keys=True)+'\n')
    print(json.dumps({'phase':args.phase,'rows':len(rows),'selection':selection},indent=2))
if __name__=='__main__':main()
