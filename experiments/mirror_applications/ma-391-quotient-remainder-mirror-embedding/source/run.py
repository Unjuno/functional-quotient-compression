"""Aligned quotient/remainder embedding screen for MA-391."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import CLASSES, DIM, METHODS, Q_ROWS, R_ROWS, VOCAB, QRBank, compute_proxy, from_payload, qr_indices

torch.set_num_threads(1)
UPDATES = 2000
BATCH = 128
LR = 0.02


def make_world(seed: int) -> dict[str, object]:
    g = torch.Generator().manual_seed(seed)
    q = 0.35 * torch.randn(Q_ROWS, DIM, generator=g)
    r = 0.35 * torch.randn(R_ROWS, DIM, generator=g)
    projection_seed = seed + 170003
    pg = torch.Generator().manual_seed(projection_seed)
    projection = torch.randn(2 * DIM, DIM, generator=pg) / (2 * DIM) ** 0.5
    teacher_angle = 2 * torch.pi * torch.rand(VOCAB, generator=g)
    tokens = torch.arange(VOCAB)
    qi, ri = qr_indices(tokens)
    teacher = torch.cos(teacher_angle)[:, None] * q[qi] + torch.sin(teacher_angle)[:, None] * r[ri]
    decoder = 0.5 * torch.randn(DIM, CLASSES, generator=g) / DIM ** 0.5
    labels = torch.argmax(teacher @ decoder, dim=1)
    pairs = torch.stack((qi, ri), dim=1)
    unique_pairs = int(torch.unique(pairs, dim=0).shape[0])
    return {"seed":seed,"q_table":q,"r_table":r,"decoder":decoder,"projection":projection,
            "projection_seed":projection_seed,"teacher":teacher,"labels":labels,
            "unique_pairs":unique_pairs,"teacher_angle":teacher_angle}


def nrmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((pred-target)**2) / torch.mean(target**2)))


def score(bank: QRBank, world: dict[str, object]) -> dict[str, float]:
    ids = torch.arange(VOCAB)
    with torch.no_grad():
        emb = bank(ids)
        logits = emb @ bank.decoder
        return {"embedding_nrmse":nrmse(emb,world["teacher"]),
                "decoder_nll":float(F.cross_entropy(logits,world["labels"])),
                "decoder_top1_accuracy":float((logits.argmax(1)==world["labels"]).float().mean())}


def fit(method: str, seed: int, world: dict[str, object]) -> tuple[QRBank, dict[str,float], float, int]:
    bank = QRBank(method, seed+1000, world["q_table"],world["r_table"],world["decoder"],
                  world["projection"],world["projection_seed"])
    params=list(bank.parameters())
    start=time.perf_counter()
    updates=UPDATES if params else 0
    if params:
        opt=torch.optim.Adam(params,lr=LR)
        gen=torch.Generator().manual_seed(seed+2000)
        target=world["teacher"]
        for _ in range(UPDATES):
            ids=torch.randint(VOCAB,(BATCH,),generator=gen)
            pred=bank(ids)
            loss=torch.mean((pred-target[ids])**2)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    elapsed=time.perf_counter()-start
    bank.eval()
    return bank,score(bank,world),elapsed,updates


def save_payload(path: Path, bank: QRBank) -> tuple[int,str]:
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**bank.payload_arrays())
    raw=path.read_bytes()
    return len(raw),hashlib.sha256(raw).hexdigest()


def load_payload(path: Path) -> QRBank:
    with np.load(path,allow_pickle=False) as z:
        payload={key:z[key] for key in z.files}
    return from_payload(payload)


def run(seed: int,condition: str,outdir: Path) -> dict[str,object]:
    world=make_world(seed)
    rows=[]
    for method in METHODS:
        t0=time.perf_counter()
        bank,metrics,train_s,updates=fit(method,seed,world)
        path=outdir/f"{condition}_{seed}_{method}.npz"
        size,digest=save_payload(path,bank)
        bank=load_payload(path)
        metrics=score(bank,world)
        ids=torch.arange(VOCAB)
        t1=time.perf_counter()
        with torch.no_grad(): _=bank(ids) @ bank.decoder
        inference_s=time.perf_counter()-t1
        rows.append({"method":method,"payload_path":path.name,"serialized_bytes":size,"payload_sha256":digest,
                     **metrics,**compute_proxy(method),"optimizer_updates":updates,
                     "training_tokens_seen":updates*BATCH,"training_wall_s":train_s,
                     "inference_tokens_per_s":VOCAB/max(inference_s,1e-9),"total_wall_s":time.perf_counter()-t0})
    return {"experiment_id":"MA-391","condition":condition,"seed":seed,"vocabulary_size":VOCAB,
            "embedding_dim":DIM,"quotient_rows":Q_ROWS,"remainder_rows":R_ROWS,"unique_address_pairs":world["unique_pairs"],
            "updates_per_trainable_method":UPDATES,"batch_size":BATCH,"learning_rate":LR,"results":rows}


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--seed',type=int,required=True)
    p.add_argument('--condition',choices=('development','fresh'),required=True)
    p.add_argument('--outdir',type=Path,required=True); p.add_argument('--json',type=Path,required=True)
    a=p.parse_args(); result=run(a.seed,a.condition,a.outdir)
    a.json.parent.mkdir(parents=True,exist_ok=True); a.json.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'seed':a.seed,'condition':a.condition,'unique_pairs':result['unique_address_pairs'],
      'results':[{k:r[k] for k in ('method','serialized_bytes','embedding_nrmse','decoder_nll','decoder_top1_accuracy','optimizer_updates')} for r in result['results']]},indent=2))


if __name__=='__main__': main()
