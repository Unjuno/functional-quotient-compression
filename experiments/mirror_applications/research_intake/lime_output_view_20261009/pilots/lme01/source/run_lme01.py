#!/usr/bin/env python3
"""LME01: trained five-output, one-trunk/one-PEFT CPU mechanism experiment.

Frozen protocol GitHub: research/mirror-lime-output-view-20261009
experiments/mirror_applications/research_intake/lime_output_view_20261009/LME01_FROZEN_PROTOCOL.json

This is a LiME-*style* output-modulation ablation with known task roles,
NOT a reproduction of native LiME's router or MMT-47 results.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

METHODS=("shared_broadcast","native_onepass_linearheads","lime_full_diagonal",
         "lime_linear4","lime_learned_linear4","mirror_rot4","mirror_shear4")
SEEDS={"dev":(71,72,73),"fresh":(701,702,703,704,705)}
K=5
D=32
RANK=8
CODE=4
STEPS=350
BATCH=128
TASKS=("parity","digit_ge5","left_right_balance","top_bottom_balance","center_intensity")


def patterns():
    g1=torch.Generator().manual_seed(71293)
    g2=torch.Generator().manual_seed(71294)
    a=torch.randn(D,CODE,generator=g1)
    b=torch.randn(D//2,CODE,generator=g2)
    q1,_=torch.linalg.qr(a,mode="reduced")
    q2,_=torch.linalg.qr(b,mode="reduced")
    return 3.0*q1,2.0*q2


def data_world(seed:int):
    digits=load_digits()
    raw=np.asarray(digits.data,dtype=np.float32)/16.0
    values=np.asarray(digits.target,dtype=np.int64)
    tr,te=train_test_split(np.arange(len(values)),test_size=.25,random_state=int(seed),stratify=values)
    tr=np.asarray(tr,dtype=np.int64)
    te=np.asarray(te,dtype=np.int64)
    pix=raw.reshape(-1,8,8)
    left_right=pix[:,:,:4].sum((1,2))-pix[:,:,4:].sum((1,2))
    top_bottom=pix[:,:4,:].sum((1,2))-pix[:,4:,:].sum((1,2))
    center=pix[:,2:6,2:6].sum((1,2))
    threshold=np.asarray([np.median(signal[tr]) for signal in (left_right,top_bottom,center)],dtype=np.float32)
    targets=np.stack((values%2,values>=5,left_right>threshold[0],top_bottom>threshold[1],center>threshold[2]),axis=-1).astype(np.int64)
    mean=raw[tr].mean(axis=0)
    sd=raw[tr].std(axis=0)+.15
    x=(raw-mean)/sd
    assert len(set(tr.tolist())&set(te.tolist()))==0
    return (torch.from_numpy(x[tr].copy()),torch.from_numpy(targets[tr].copy()),
            torch.from_numpy(x[te].copy()),torch.from_numpy(targets[te].copy()),
            {"thresholds":threshold.tolist(),"pixel_mean":mean,"pixel_std":sd,
             "train_ids_sha256":hashlib.sha256(tr.tobytes()).hexdigest(),
             "audit_ids_sha256":hashlib.sha256(te.tobytes()).hexdigest(),
             "dataset_sha256":hashlib.sha256(raw.tobytes()+values.tobytes()).hexdigest(),
             "train_count":len(tr),"test_count":len(te),
             "train_ids":tr,"audit_ids":te})


class SharedAdapterMultiOutput(nn.Module):
    """Single heavy trunk + one shared rank-8 output adapter, K useful role outputs."""
    def __init__(self, method:str, seed:int):
        super().__init__()
        if method not in METHODS:raise ValueError(method)
        self.method=method
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed*1439+137)
            self.trunk=nn.Sequential(nn.Linear(64,128),nn.GELU(),nn.Linear(128,D),nn.GELU())
            self.adapter=nn.Sequential(nn.Linear(D,RANK,bias=False),nn.GELU(),nn.Linear(RANK,D,bias=False))
            self.common=nn.Linear(D,2)
            if method=="native_onepass_linearheads":
                self.head_weight=nn.Parameter(self.common.weight.detach().clone().unsqueeze(0).expand(K,-1,-1).clone())
                self.head_bias=nn.Parameter(self.common.bias.detach().clone().unsqueeze(0).expand(K,-1).clone())
            elif method!="shared_broadcast":
                self.task_bias=nn.Parameter(torch.zeros(K,2))
                if method=="lime_full_diagonal":
                    self.p=nn.Parameter(torch.randn(K,D)*.025)
                else:
                    self.m=nn.Parameter(torch.randn(K,CODE)*.025)
                    if method=="lime_learned_linear4":
                        pp,_=patterns()
                        self.dictionary=nn.Parameter(pp.clone())
        pd,pr=patterns()
        self.register_buffer("fixed_scale",pd,persistent=False)
        self.register_buffer("fixed_planes",pr,persistent=False)

    def _member_readout(self,d:torch.Tensor)->torch.Tensor:
        if self.method=="lime_full_diagonal":
            return d.unsqueeze(1)*torch.exp(self.p).unsqueeze(0)
        if self.method in ("lime_linear4","lime_learned_linear4"):
            P=self.fixed_scale if self.method=="lime_linear4" else self.dictionary
            scale=1.0+self.m@P.T
            return d.unsqueeze(1)*scale.unsqueeze(0)
        if self.method in ("mirror_rot4","mirror_shear4"):
            angles=self.m@self.fixed_planes.T # [K,16]
            x=d.reshape(d.shape[0],D//2,2)
            a=x[...,0].unsqueeze(1)
            b=x[...,1].unsqueeze(1)
            if self.method=="mirror_rot4":
                ca=angles.cos().unsqueeze(0)
                sa=angles.sin().unsqueeze(0)
                p=ca*a-sa*b
                q=sa*a+ca*b
            else:
                p=a+angles.unsqueeze(0)*b
                q=b.expand(-1,K,-1)
            return torch.stack((p,q),dim=-1).reshape(d.shape[0],K,D)
        raise ValueError(self.method)

    def forward(self,x:torch.Tensor)->torch.Tensor:
        h=self.trunk(x)  # ONE trunk forward independent of K
        delta=self.adapter(h)  # ONE adapter forward independent of K
        if self.method=="shared_broadcast":
            return self.common(h+delta).unsqueeze(1).expand(-1,K,-1)
        if self.method=="native_onepass_linearheads":
            return torch.einsum("bd,kcd->bkc",h+delta,self.head_weight)+self.head_bias.unsqueeze(0)
        xd=self._member_readout(delta)
        return F.linear(h.unsqueeze(1)+xd,self.common.weight,self.common.bias)+self.task_bias.unsqueeze(0)

    def stored(self, meta:dict)->bytes:
        state={k:v.detach().cpu().contiguous().numpy().astype(np.float32) for k,v in self.state_dict().items()}
        if self.method=="native_onepass_linearheads":
            state.pop("common.weight",None)
            state.pop("common.bias",None)
        m={"method":self.method,"K":K,"D":D,"CODE":CODE,"adapter_rank":RANK,
           "pattern_seeds":[71293,71294],"pattern_source":"torch.randn QR fixed seed float32 torch2.10",
           "thresholds_train_only":meta["thresholds"],"tasks":TASKS}
        state["__meta__"]=np.frombuffer(json.dumps(m,sort_keys=True,separators=(",",":")).encode(),dtype=np.uint8)
        state["__preprocess_mean__"]=meta["pixel_mean"].astype(np.float32)
        state["__preprocess_std__"]=meta["pixel_std"].astype(np.float32)
        buf=io.BytesIO()
        np.savez(buf,**state)
        return buf.getvalue()

    def per_member_scalars(self)->int:
        if self.method=="native_onepass_linearheads":return K*(D*2+2)
        if self.method=="shared_broadcast":return 0
        if self.method=="lime_full_diagonal":return K*(D+2)
        if self.method=="lime_learned_linear4":return K*(CODE+2)+D*CODE
        return K*(CODE+2)


def score(model,x,y):
    model.eval()
    with torch.no_grad():
        logits=model(x)
        assert logits.shape==(len(x),K,2)
        ce=F.cross_entropy(logits.reshape(-1,2),y.reshape(-1),reduction="none").reshape(-1,K)
        per=ce.mean(0)
        accur=(logits.argmax(-1)==y).float().mean(0)
        diversity=torch.softmax(logits,dim=-1)[...,1].std(1).mean()
        return {"nll":float(per.mean()),"worst_nll":float(per.max()),
                "accuracy":float(accur.mean()),"prob_std":float(diversity),
                "task_nlls":";".join(f"{v:.9g}" for v in per.tolist()),
                "task_acc":";".join(f"{v:.9g}" for v in accur.tolist())}


def benchmark(model,x):
    model.eval()
    batch=x[:128]
    cnt={"trunk":0,"adapter":0}
    def trunk_hook(m,x,out):cnt["trunk"]+=1
    def adapter_hook(m,x,out):cnt["adapter"]+=1
    h1=model.trunk.register_forward_hook(trunk_hook)
    h2=model.adapter.register_forward_hook(adapter_hook)
    with torch.inference_mode():
        model(batch)
        h1.remove();h2.remove()
        assert cnt=={"trunk":1,"adapter":1},cnt
        for _ in range(20):model(batch)
        timings=[]
        for _ in range(80):
            t=time.perf_counter_ns()
            model(batch)
            timings.append((time.perf_counter_ns()-t)/1e6)
    return float(np.median(timings)),float(np.quantile(timings,.95))


def train_one(seed:int,method:str,timing:bool=True):
    torch.set_num_threads(1)
    trainx,trainy,auditx,audity,meta=data_world(seed)
    model=SharedAdapterMultiOutput(method,seed)
    opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.0001)
    g=torch.Generator().manual_seed(seed*7919+71)
    for step in range(STEPS):
        i=torch.randint(len(trainx),(BATCH,),generator=g)
        pred=model(trainx[i])
        loss=F.cross_entropy(pred.reshape(-1,2),trainy[i].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    sr=score(model,trainx,trainy)
    au=score(model,auditx,audity)
    sz=len(model.stored(meta))
    p50,p95=benchmark(model,auditx) if timing else (float("nan"),float("nan"))
    return {"phase":"","seed":seed,"method":method,"K":K,"train_nll":sr["nll"],
            "test_nll":au["nll"],"test_worst_task_nll":au["worst_nll"],
            "test_accuracy":au["accuracy"],"output_prob_std":au["prob_std"],
            "per_task_nll":au["task_nlls"],"per_task_accuracy":au["task_acc"],
            "serializer_bytes":sz,"role_trainable_scalars":model.per_member_scalars(),
            "cpu_p50_ms":p50,"cpu_p95_ms":p95,
            "one_trunk_forward":1,"one_adapter_forward":1,
            "original_train_id_sha256":meta["train_ids_sha256"],"audit_id_sha256":meta["audit_ids_sha256"],
            "digits_data_sha256":meta["dataset_sha256"],
            "training_steps":STEPS,"batch_size":BATCH,"source_dtype":"FP32","task_label_count":K}


def write_rows(out,phase):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for seed in SEEDS[phase]:
        for method in METHODS:
            v=train_one(seed,method)
            v["phase"]=phase
            rows.append(v)
        summary={r["method"]:round(r["test_nll"],4) for r in rows if r["seed"]==seed}
        print(json.dumps({"phase":phase,"seed":seed,"mean_nll":summary}),flush=True)
    with (out/f"{phase}_raw.csv").open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader();w.writerows(rows)
    return rows


def main():
    a=argparse.ArgumentParser()
    a.add_argument("--phase",choices=("dev","fresh","all"),default="dev")
    a.add_argument("--out",default="results")
    args=a.parse_args()
    torch.set_num_threads(1)
    if args.phase in ("dev","all"):write_rows(args.out,"dev")
    if args.phase in ("fresh","all"):write_rows(args.out,"fresh")

if __name__=="__main__":main()
