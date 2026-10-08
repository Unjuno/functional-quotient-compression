#!/usr/bin/env python3
"""Frozen small mechanism screen for MA-405."""
import argparse, csv, io, json, math, os, random, time
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts"
METHODS = ["shared", "style_mod_demod", "mirror_givens", "film", "rank1", "independent"]
DEV, FRESH = [40500,40501], [40510,40511,40512]
LRS=[.003,.01]
SEEDS=[0,1,2]
UPDATES,BATCH=200,128
DIN,HID,CLS,CTX=32,48,10,4

def fixseed(s):
    random.seed(s); torch.manual_seed(s)
    torch.set_num_threads(1)

def data(world):
    g=torch.Generator().manual_seed(world)
    x=torch.randn(4096,DIN,generator=g)
    base1=torch.randn(DIN,HID,generator=g)/math.sqrt(DIN)
    base2=torch.randn(HID,CLS,generator=g)/math.sqrt(HID)
    labels=[]
    perms=[]
    for c in range(CTX):
        p=torch.randperm(DIN,generator=g); perms.append(p)
    cids=torch.arange(4096)%CTX
    # Generate labels with one shared teacher under context-specific input permutations.
    for i in range(4096):
        c=int(cids[i]); z=x[i,perms[c]]
        logits=F.relu(z@base1)@base2
        labels.append(int(logits.argmax()))
    return x,cids,torch.tensor(labels),perms

class Net(nn.Module):
    def __init__(self,method):
        super().__init__(); self.method=method
        if method=="independent":
            self.w1=nn.Parameter(torch.randn(CTX,DIN,HID)*.12); self.w2=nn.Parameter(torch.randn(CTX,HID,CLS)*.12)
        else:
            self.w1=nn.Parameter(torch.randn(DIN,HID)*.12); self.w2=nn.Parameter(torch.randn(HID,CLS)*.12)
        if method=="style_mod_demod":
            self.code=nn.Parameter(torch.ones(CTX,DIN)); self.outscale=nn.Parameter(torch.ones(CTX,CLS))
        elif method=="mirror_givens":
            self.angles=nn.Parameter(torch.zeros(CTX,4))
        elif method=="film":
            self.scale=nn.Parameter(torch.ones(CTX,HID)); self.shift=nn.Parameter(torch.zeros(CTX,HID))
        elif method=="rank1":
            self.a=nn.Parameter(torch.randn(CTX,DIN,1)*.01); self.b=nn.Parameter(torch.randn(CTX,1,HID)*.01)

    def effective(self,c):
        if self.method=="independent": return self.w1[c],self.w2[c]
        w=self.w1
        if self.method=="style_mod_demod":
            # StyleGAN-like demodulation for each output channel; per-context input scaling.
            wm=w[None,:,:]*self.code[c,:,None]
            d=torch.rsqrt((wm.square().sum(1)+1e-8))
            wm=wm*d[:,None,:]
            return wm,self.w2
        if self.method=="mirror_givens":
            # Four disjoint input-plane rotations, right-multiplied into the shared matrix.
            ci=int(c.item()) if torch.is_tensor(c) else int(c)
            cs=torch.cos(self.angles[ci]); sn=torch.sin(self.angles[ci]); v=w
            for k in range(4):
                i,j=2*k,2*k+1
                vi,vj=v[i].clone(),v[j].clone()
                v=v.clone(); v[i]=cs[k]*vi-sn[k]*vj; v[j]=sn[k]*vi+cs[k]*vj
            return v,self.w2
        if self.method=="rank1": return w+self.a[c]@self.b[c],self.w2
        return w,self.w2

    def forward(self,x,c):
        if self.method=="mirror_givens":
            w1=self.w1[None].expand(x.shape[0],-1,-1).clone(); w2=self.w2
            cs=torch.cos(self.angles[c]); sn=torch.sin(self.angles[c])
            for k in range(4):
                i,j=2*k,2*k+1; vi,vj=w1[:,i].clone(),w1[:,j].clone()
                w1[:,i]=cs[:,k,None]*vi-sn[:,k,None]*vj
                w1[:,j]=sn[:,k,None]*vi+cs[:,k,None]*vj
        else: w1,w2=self.effective(c)
        if w1.ndim==2: h=x@w1
        else: h=torch.bmm(x[:,None,:],w1).squeeze(1)
        h=F.relu(h)
        if self.method=="film": h=h*self.scale[c]+self.shift[c]
        if self.method=="style_mod_demod":
            # Context-dependent output-channel scaling after shared second projection.
            pass
        if w2.ndim==2: y=h@w2
        else: y=torch.bmm(h[:,None,:],w2).squeeze(1)
        if self.method=="style_mod_demod": y=y*self.outscale[c]
        return y

def payload(model):
    # Deterministic inference state: compact ZIP of raw state tensors + fixed JSON metadata.
    b=io.BytesIO(); torch.save({k:v.detach().cpu().contiguous() for k,v in model.state_dict().items()},b,_use_new_zipfile_serialization=True)
    return b.getvalue()

def diversity(model):
    mats=[]
    for c in range(CTX):
        a,b=model.effective(torch.tensor([c])); mats.append(torch.cat([a.flatten(),b.flatten()]))
    d=[torch.linalg.vector_norm(mats[i]-mats[j]).item() for i in range(CTX) for j in range(i+1,CTX)]
    return sum(d)/len(d)

def run_one(method,world,seed,lr,split):
    fixseed(world*100+seed); x,c,y,_=data(world)
    model=Net(method); opt=torch.optim.AdamW(model.parameters(),lr=lr)
    st=time.perf_counter()
    for step in range(UPDATES):
        ix=torch.randint(0,len(x),(BATCH,)); logits=model(x[ix],c[ix]); loss=F.cross_entropy(logits,y[ix])
        opt.zero_grad(); loss.backward(); opt.step()
    train_s=time.perf_counter()-st
    with torch.no_grad():
        pred=model(x[3072:],c[3072:]); acc=(pred.argmax(-1)==y[3072:]).float().mean().item(); ce=F.cross_entropy(pred,y[3072:]).item()
    raw=payload(model)
    payload_dir=OUT/'payloads'; payload_dir.mkdir(exist_ok=True)
    payload_name=f"{split}_{world}_{seed}_{method}_lr{lr:g}.pt"
    (payload_dir/payload_name).write_bytes(raw)
    # MAC proxy: two dense projections per example; materialized modulation adds one weight-scale per context/example.
    mac=2*DIN*HID+2*HID*CLS
    if method in ("style_mod_demod","mirror_givens","rank1"): mac+=DIN*HID
    if method=="film": mac+=2*HID
    # Verify exact load/save roundtrip of the serialized inference state.
    loaded=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
    clone=Net(method); clone.load_state_dict(loaded); clone_raw=payload(clone)
    assert raw==clone_raw
    return {"split":split,"world":world,"seed":seed,"method":method,"lr":lr,"payload_file":payload_name,"serialized_bytes":len(raw),"examples":UPDATES*BATCH,"updates":UPDATES,"active_mac_proxy_per_example":mac,"wall_time_s":round(train_s,6),"accuracy":acc,"cross_entropy":ce,"effective_weight_pairwise_l2":diversity(model),"payload_sha256":__import__('hashlib').sha256(raw).hexdigest(),"roundtrip_sha256":__import__('hashlib').sha256(clone_raw).hexdigest()}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--phase',choices=['dev','fresh','all'],default='all'); a=ap.parse_args()
    OUT.mkdir(exist_ok=True)
    rows=[]; selection={}
    if a.phase in ('dev','all'):
        for m in METHODS:
            vals={lr:[] for lr in LRS}
            for lr in LRS:
                for w in DEV:
                    for s in SEEDS:
                        r=run_one(m,w,s,lr,'development'); rows.append(r); vals[lr].append(r['accuracy'])
            selection[m]=max(LRS,key=lambda lr:sum(vals[lr])/len(vals[lr]))
        (OUT/'development_selection.json').write_text(json.dumps(selection,indent=2)+'\n')
    else:
        selection=json.loads((OUT/'development_selection.json').read_text())
    if a.phase in ('fresh','all'):
        for m in METHODS:
            for w in FRESH:
                for s in SEEDS:
                    rows.append(run_one(m,w,s,selection[m],'fresh'))
    mode='w' if a.phase in ('dev','all') else 'a'
    with (OUT/'runs.jsonl').open(mode) as f:
        for r in rows:f.write(json.dumps(r,sort_keys=True)+'\n')
    print(json.dumps({'rows':len(rows),'selection':selection,'phase':a.phase},indent=2))
if __name__=='__main__':main()
