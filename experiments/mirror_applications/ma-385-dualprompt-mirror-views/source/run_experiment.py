"""Sequential synthetic DualPrompt experiment with shared and Mirror expert views."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch import nn


def rotate(v: torch.Tensor, theta: torch.Tensor):
    c, s = torch.cos(theta), torch.sin(theta)
    a, b = v[..., 0::2], v[..., 1::2]
    out = torch.empty_like(v)
    out[..., 0::2] = c[..., None] * a - s[..., None] * b
    out[..., 1::2] = s[..., None] * a + c[..., None] * b
    return out


def make_world(seed: int, family: str):
    slots, dim, classes = 8, 32, 4
    g = torch.Generator().manual_seed(seed + 400)
    centers = torch.zeros(slots, dim)
    centers[torch.arange(slots), torch.arange(slots)] = 3.0
    wg = torch.Generator().manual_seed(seed + 991)
    classifier = torch.randn(dim, classes, generator=wg) / math.sqrt(dim)
    pg = torch.Generator().manual_seed(seed + 1771)
    general = torch.randn(dim, generator=pg) * .25
    basis = torch.randn(dim, generator=pg) * .55
    if family == "aligned":
        angles = torch.linspace(-.9, .9, slots)
        deltas = torch.stack([rotate(basis, a) for a in angles])
    else:
        deltas = torch.randn(slots, dim, generator=pg) * .55
    datasets = {}
    for split, n, offset in (("train", 1024, 0), ("validation", 256, 1000), ("test", 512, 2000)):
        xs, ys = [], []
        for t in range(slots):
            tg = torch.Generator().manual_seed(seed + offset + t * 19)
            x = centers[t] + .50 * torch.randn(n, dim, generator=tg)
            y = ((x + general + deltas[t]) @ classifier).argmax(-1)
            xs.append(x); ys.append(y)
        datasets[split] = (torch.stack(xs), torch.stack(ys))
    # Retrieval keys are learned only from the training input centroids and
    # are serialized in every inference payload.
    keys = datasets["train"][0].mean(1)
    return datasets, keys, classifier, general, deltas


class ExpertPrompts(nn.Module):
    def __init__(self, method: str):
        super().__init__(); self.method = method
        self.general = nn.Parameter(torch.zeros(32))
        if method == "explicit": self.experts = nn.Parameter(torch.zeros(8, 32))
        elif method == "tied": self.expert = nn.Parameter(torch.zeros(32))
        elif method in ("scalar", "mirror"):
            self.basis = nn.Parameter(torch.zeros(32)); self.code = nn.Parameter(torch.zeros(8))
        elif method == "private_rank4":
            self.private_basis = nn.Parameter(torch.zeros(4, 32))
            self.private_code = nn.Parameter(torch.zeros(8, 4))
        elif method == "hyper":
            self.slot = nn.Embedding(8, 8)
            self.generator = nn.Sequential(nn.Linear(8, 32), nn.Tanh(), nn.Linear(32, 32))
        else: raise ValueError(method)

    def expert_table(self):
        if self.method == "explicit": return self.experts
        if self.method == "tied": return self.expert.expand(8, -1)
        if self.method == "scalar": return self.code[:, None] * self.basis[None, :]
        if self.method == "mirror": return rotate(self.basis[None, :].expand(8, -1), self.code)
        if self.method == "private_rank4": return self.private_code @ self.private_basis
        ids = torch.arange(8, device=self.slot.weight.device)
        return self.generator(self.slot(ids))

    def forward(self, x, slot):
        return (x + self.general + self.expert_table()[slot])


def nearest(x, keys): return torch.cdist(x.float(), keys.float()).argmin(-1)


@torch.no_grad()
def evaluate(model, dataset, keys, classifier, tasks):
    model.eval(); accs=[]; nlls=[]; retrieval=[]
    for t in tasks:
        x, y = dataset[0][t], dataset[1][t]
        slot = nearest(x, keys)
        logits = model(x, slot) @ classifier
        accs.append(float((logits.argmax(-1) == y).float().mean()))
        nlls.append(float(nn.functional.cross_entropy(logits, y)))
        retrieval.append(float((slot == t).float().mean()))
    return {"mean_accuracy":float(np.mean(accs)),"per_task_accuracy":accs,
            "mean_nll":float(np.mean(nlls)),"retrieval_accuracy":float(np.mean(retrieval))}


def pack(path: Path, model, keys, classifier):
    arrays={k:v.detach().cpu().numpy().astype(np.float16) for k,v in model.state_dict().items()}
    arrays.update({"keys":keys.cpu().numpy().astype(np.float16),"classifier":classifier.cpu().numpy().astype(np.float16)})
    path.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path,"w",compression=zipfile.ZIP_STORED) as z:
        z.writestr("metadata.json",json.dumps({"slots":8,"input_dim":32,"classes":4},sort_keys=True).encode())
        for name,a in sorted(arrays.items()):
            b=io.BytesIO();np.save(b,a,allow_pickle=False);z.writestr(f"arrays/{name.replace('.','_')}.npy",b.getvalue())
    payload=path.read_bytes()
    return len(payload),hashlib.sha256(payload).hexdigest()


def reload_payload(path: Path, model):
    with zipfile.ZipFile(path) as z:
        arrays={Path(name).stem:np.load(io.BytesIO(z.read(name)),allow_pickle=False)
                for name in z.namelist() if name.endswith('.npy')}
    model.load_state_dict({k:torch.from_numpy(arrays[k.replace('.','_')].astype('float32'))
                           for k in model.state_dict()})
    keys=torch.from_numpy(arrays['keys'].astype('float32'))
    classifier=torch.from_numpy(arrays['classifier'].astype('float32'))
    return keys,classifier


def train_method(seed,family,method,updates,device,out):
    data, keys, classifier, _, _ = make_world(seed,family)
    train_x,train_y=data["train"]
    model=ExpertPrompts(method).to(device); keys=keys.to(device); classifier=classifier.to(device)
    opt=torch.optim.Adam(model.parameters(),lr=.001)
    per_task=updates//8; task_history=[]; start=time.perf_counter()
    for t in range(8):
        x=train_x[t].to(device); y=train_y[t].to(device)
        for _ in range(per_task):
            ids=torch.randint(len(x),(128,),device=device)
            slot=nearest(x[ids],keys)
            logits=model(x[ids],slot) @ classifier
            loss=nn.functional.cross_entropy(logits,y[ids])
            opt.zero_grad();loss.backward();opt.step()
        task_history.append(evaluate(model,data["test"],keys,classifier,list(range(t+1))))
    wall=time.perf_counter()-start
    path=out/f"{family}_{seed}_{method}.zip"; size,digest=pack(path,model,keys,classifier)
    # The serialized FP16 inference state defines the reported test quality.
    stored_keys,stored_classifier=reload_payload(path,model)
    test=evaluate(model,data["test"],stored_keys,stored_classifier,list(range(8)))
    val=evaluate(model,data["validation"],stored_keys,stored_classifier,list(range(8)))
    generation={"explicit":0,"tied":0,"scalar":32,"mirror":64,"private_rank4":128,"hyper":1280}[method]
    return {"validation":val,"test":test,"seen_task_checkpoints":task_history,
            "updates":updates,"examples_seen":updates*128,"train_examples_available":8*1024,
            "train_wall_seconds":wall,"retrieval_distance_ops_per_query":8*32,
            "prompt_generation_macs_per_query":generation,"actual_payload_bytes":size,
            "payload_sha256":digest}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,required=True)
    ap.add_argument('--family',choices=['aligned','unrelated'],default='aligned')
    ap.add_argument('--updates',type=int,default=1600);ap.add_argument('--device',default='cpu')
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    res={"experiment_id":"MA-385","seed":a.seed,"family":a.family,"methods":{}}
    for method in ('explicit','tied','scalar','mirror','private_rank4','hyper'):
        res['methods'][method]=train_method(a.seed,a.family,method,a.updates,a.device,a.out)
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/f"{a.family}_{a.seed}_result.json").write_text(json.dumps(res,indent=2)+'\n')
    print(json.dumps(res,indent=2))

if __name__=='__main__':main()
