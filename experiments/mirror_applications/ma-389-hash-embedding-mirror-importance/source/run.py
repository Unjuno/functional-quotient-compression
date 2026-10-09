"""Aligned synthetic Hash Embedding importance-code screen."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import (BUCKETS, CLASSES, DIM, VOCAB, HashEmbeddingBank,
                   bank_from_payload, compute_proxy, hash_indices)

torch.set_num_threads(1)
UPDATES = 1500
BATCH = 64
LR = 0.01


def make_world(seed: int) -> dict[str, object]:
    g = torch.Generator().manual_seed(seed)
    hash_seed0, hash_seed1 = seed + 70001, seed + 90017
    indices = hash_indices(hash_seed0, hash_seed1)
    table0 = 0.4 * torch.randn(BUCKETS, DIM, generator=g)
    table1 = 0.4 * torch.randn(BUCKETS, DIM, generator=g)
    angle = 2 * torch.pi * torch.rand(VOCAB, generator=g)
    alpha = torch.stack((torch.cos(angle), torch.sin(angle)), dim=1)
    teacher = alpha[:, 0:1] * table0[indices[:, 0]] + alpha[:, 1:2] * table1[indices[:, 1]]
    decoder = 0.5 * torch.randn(DIM, CLASSES, generator=g) / DIM**0.5
    labels = torch.argmax(teacher @ decoder, dim=1)
    pair_keys = indices[:, 0] * BUCKETS + indices[:, 1]
    unique, counts = torch.unique(pair_keys, return_counts=True)
    count_by_key = torch.zeros(BUCKETS * BUCKETS, dtype=torch.long)
    count_by_key[unique] = counts
    collision = count_by_key[pair_keys] > 1
    return {"seed": seed, "hash_seed0": hash_seed0, "hash_seed1": hash_seed1,
            "indices": indices, "teacher_embedding": teacher, "decoder": decoder,
            "labels": labels, "collision_mask": collision,
            "collision_pair_count": int((counts > 1).sum()),
            "collided_token_fraction": float(collision.float().mean())}


def nrmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((pred-target)**2) / torch.mean(target**2)))


def score(bank: HashEmbeddingBank, world: dict[str, object]) -> dict[str, object]:
    ids = torch.arange(VOCAB)
    collision = world["collision_mask"]
    with torch.no_grad():
        emb = bank(ids)
        logits = emb @ bank.decoder
        nll = float(F.cross_entropy(logits, world["labels"]))
        acc = float((torch.argmax(logits, dim=1) == world["labels"]).float().mean())
        err_all = nrmse(emb, world["teacher_embedding"])
        err_col = nrmse(emb[collision], world["teacher_embedding"][collision]) if bool(collision.any()) else 0.0
        err_unique = nrmse(emb[~collision], world["teacher_embedding"][~collision]) if bool((~collision).any()) else 0.0
        acc_col = float((torch.argmax(logits[collision], dim=1) == world["labels"][collision]).float().mean()) if bool(collision.any()) else 0.0
        acc_unique = float((torch.argmax(logits[~collision], dim=1) == world["labels"][~collision]).float().mean()) if bool((~collision).any()) else 0.0
    return {"embedding_nrmse": err_all, "collision_embedding_nrmse": err_col,
            "unique_pair_embedding_nrmse": err_unique, "decoder_nll": nll,
            "decoder_top1_accuracy": acc, "collision_decoder_accuracy": acc_col,
            "unique_pair_decoder_accuracy": acc_unique}


def fit(method: str, seed: int, world: dict[str, object]):
    bank = HashEmbeddingBank(method, seed + 1000, world["hash_seed0"], world["hash_seed1"], world["decoder"])
    opt = torch.optim.Adam(bank.parameters(), lr=LR)
    gen = torch.Generator().manual_seed(seed + 2000)
    target = world["teacher_embedding"]
    start = time.perf_counter()
    for _ in range(UPDATES):
        ids = torch.randint(VOCAB, (BATCH,), generator=gen)
        pred = bank(ids)
        loss = torch.mean((pred-target[ids])**2)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    elapsed = time.perf_counter() - start
    bank.eval()
    metrics = score(bank, world)
    return bank, metrics, elapsed


def save_payload(path: Path, bank: HashEmbeddingBank) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **bank.payload_arrays())
    raw = path.read_bytes()
    return len(raw), hashlib.sha256(raw).hexdigest()


def load_payload(path: Path) -> HashEmbeddingBank:
    with np.load(path, allow_pickle=False) as archive:
        payload = {key: archive[key] for key in archive.files}
    return bank_from_payload(payload)


def run(seed: int, condition: str, outdir: Path) -> dict[str, object]:
    world = make_world(seed)
    rows = []
    for method in HashEmbeddingBank.METHODS:
        start = time.perf_counter()
        bank, metrics, train_s = fit(method, seed, world)
        path = outdir / f"{condition}_{seed}_{method}.npz"
        size, digest = save_payload(path, bank)
        # Quality is measured from the exact serialized inference state.
        bank = load_payload(path)
        metrics = score(bank, world)
        with torch.no_grad():
            ids = torch.arange(VOCAB)
            t0 = time.perf_counter()
            emb = bank(ids)
            _ = emb @ bank.decoder
            infer_s = time.perf_counter() - t0
        rows.append({"method": method, "payload_path": path.name,
                     "serialized_bytes": size, "payload_sha256": digest,
                     **metrics, **compute_proxy(method),
                     "optimizer_updates": UPDATES,
                     "training_tokens_seen": UPDATES * BATCH,
                     "training_wall_s": train_s,
                     "inference_tokens_per_s": VOCAB / max(infer_s, 1e-9),
                     "total_wall_s": time.perf_counter()-start})
    return {"experiment_id":"MA-389", "condition":condition, "seed":seed,
            "vocabulary_size":VOCAB, "embedding_dim":DIM, "buckets_per_table":BUCKETS,
            "collision_pair_count":world["collision_pair_count"],
            "collided_token_fraction":world["collided_token_fraction"],
            "updates":UPDATES,"batch_size":BATCH,"learning_rate":LR,"results":rows}


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument('--seed',type=int,required=True)
    p.add_argument('--condition',choices=('development','fresh'),required=True)
    p.add_argument('--outdir',type=Path,required=True)
    p.add_argument('--json',type=Path,required=True)
    a=p.parse_args()
    result=run(a.seed,a.condition,a.outdir)
    a.json.parent.mkdir(parents=True,exist_ok=True)
    a.json.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'seed':a.seed,'condition':a.condition,'collision_fraction':result['collided_token_fraction'],
                      'results':[{k:r[k] for k in ('method','serialized_bytes','embedding_nrmse','decoder_nll','decoder_top1_accuracy')} for r in result['results']]},indent=2))


if __name__=='__main__': main()
