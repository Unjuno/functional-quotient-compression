#!/usr/bin/env python3
"""CPU MovieLens-100K screen for MA-1064. Audit is evaluated only in stage=eval."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
DATA_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
DATA_SHA256 = "50d2a982c66986937beb9ffb3aa76efe955bf3d5c6b761f4e3a7cd717c6a3229"
N_USERS, N_ITEMS, DIM = 943, 1682, 32
SEEDS = (31, 47, 59)
METHODS = ("full_table", "dhe", "dhe_mirror", "dhe_rank2", "dhe_mirror_rank1", "dhe_rare_full")
HASH_SEEDS = (b"ma1064-h0", b"ma1064-h1", b"ma1064-h2", b"ma1064-h3")
BUCKETS, HASH_DIM = 128, 16


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download_and_read(archive_path: Path):
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    if not archive_path.exists():
        req = urllib.request.Request(DATA_URL, headers={"User-Agent": "MA-1064-research/1.0"})
        with urllib.request.urlopen(req, timeout=90) as response:
            archive_path.write_bytes(response.read())
    raw = archive_path.read_bytes()
    if sha256_bytes(raw) != DATA_SHA256:
        raise ValueError("MovieLens-100K archive SHA-256 does not match frozen protocol")
    with zipfile.ZipFile(archive_path) as zf:
        content = zf.read("ml-100k/u.data")
        readme = zf.read("ml-100k/README")
    arr = np.loadtxt(content.splitlines(), dtype=np.int64)
    if arr.shape != (100000, 4):
        raise ValueError(f"expected 100000 MovieLens ratings, got {arr.shape}")
    # Stable sorting preserves file order when timestamps tie.
    arr = arr[np.argsort(arr[:, 3], kind="stable")]
    users = torch.from_numpy((arr[:, 0] - 1).astype(np.int64))
    items = torch.from_numpy((arr[:, 1] - 1).astype(np.int64))
    labels = torch.from_numpy((arr[:, 2] >= 4).astype(np.float32))
    n = len(arr)
    tr_end, dev_end = int(n * 0.70), int(n * 0.80)
    split = {
        "train": (users[:tr_end], items[:tr_end], labels[:tr_end]),
        "dev": (users[tr_end:dev_end], items[tr_end:dev_end], labels[tr_end:dev_end]),
        "audit": (users[dev_end:], items[dev_end:], labels[dev_end:]),
    }
    train_counts = np.bincount(items[:tr_end].numpy(), minlength=N_ITEMS)
    rare_idx = np.flatnonzero((train_counts >= 1) & (train_counts <= 5)).astype(np.int64)
    data_info = {
        "url": DATA_URL,
        "archive_bytes": len(raw),
        "archive_sha256": sha256_bytes(raw),
        "ratings_sha256": sha256_bytes(content),
        "readme_sha256": sha256_bytes(readme),
        "events": n,
        "users": N_USERS,
        "items": N_ITEMS,
        "split_events": {k: len(v[0]) for k, v in split.items()},
        "train_rare_items_1_to_5": int(len(rare_idx)),
        "train_cold_items_0": int(np.sum(train_counts == 0)),
        "train_common_items_ge_6": int(np.sum(train_counts >= 6)),
        "rare_audit_events": int(np.isin(items[dev_end:].numpy(), rare_idx).sum()),
        "cold_audit_events": int((train_counts[items[dev_end:].numpy()] == 0).sum()),
        "common_audit_events": int((train_counts[items[dev_end:].numpy()] >= 6).sum()),
        "readme_excerpt": readme.decode("latin-1")[:1200],
    }
    return split, train_counts, rare_idx, data_info


def hash_bucket(item_id: int, seed: bytes) -> int:
    h = hashlib.blake2b(str(item_id).encode(), digest_size=8, key=seed)
    return int.from_bytes(h.digest(), "little") % BUCKETS


def make_hash_indices():
    return torch.tensor([[hash_bucket(i + 1, seed) for seed in HASH_SEEDS]
                         for i in range(N_ITEMS)], dtype=torch.long)


class MovieRecommender(nn.Module):
    def __init__(self, method: str, rare_item_ids: np.ndarray, init_seed: int):
        super().__init__()
        self.method = method
        self.dim = DIM
        self.user_embeddings = nn.Embedding(N_USERS, DIM)
        self.user_bias = nn.Embedding(N_USERS, 1)
        self.global_bias = nn.Parameter(torch.zeros(()))
        self.register_buffer("hash_indices", make_hash_indices(), persistent=False)
        rare_lookup = torch.full((N_ITEMS,), -1, dtype=torch.long)
        if len(rare_item_ids):
            rare_lookup[torch.from_numpy(rare_item_ids)] = torch.arange(len(rare_item_ids))
        self.register_buffer("rare_lookup", rare_lookup, persistent=False)
        self.register_buffer("rare_item_ids", torch.from_numpy(rare_item_ids.astype(np.int64)), persistent=False)

        self.item_table = None
        self.hash_tables = None
        self.decoder1 = None
        self.decoder2 = None
        self.mirror_plane = None
        self.mirror_angles = None
        self.residual_basis = None
        self.residual_codes = None
        self.full_residuals = None
        self._cached_plane = None

        if method == "full_table":
            self.item_table = nn.Embedding(N_ITEMS, DIM)
        else:
            self.hash_tables = nn.Parameter(torch.empty(4, BUCKETS, HASH_DIM))
            nn.init.normal_(self.hash_tables, mean=0.0, std=0.03)
            self.decoder1 = nn.Linear(4 * HASH_DIM, 64)
            self.decoder2 = nn.Linear(64, DIM)
        if method in ("dhe_mirror", "dhe_mirror_rank1"):
            self.mirror_plane = nn.Parameter(torch.empty(DIM, 2))
            nn.init.normal_(self.mirror_plane, mean=0.0, std=0.15)
            self.mirror_angles = nn.Parameter(torch.zeros(len(rare_item_ids), 1))
        if method in ("dhe_rank2",):
            self.residual_basis = nn.Parameter(torch.empty(DIM, 2))
            nn.init.normal_(self.residual_basis, mean=0.0, std=0.02)
            self.residual_codes = nn.Parameter(torch.zeros(len(rare_item_ids), 2))
        if method == "dhe_mirror_rank1":
            self.residual_basis = nn.Parameter(torch.empty(DIM, 1))
            nn.init.normal_(self.residual_basis, mean=0.0, std=0.02)
            self.residual_codes = nn.Parameter(torch.zeros(len(rare_item_ids), 1))
        if method == "dhe_rare_full":
            self.full_residuals = nn.Parameter(torch.zeros(len(rare_item_ids), DIM))

    def train(self, mode: bool = True):
        self._cached_plane = None
        return super().train(mode)

    def _base_item_vector(self, item_ids: torch.Tensor) -> torch.Tensor:
        if self.method == "full_table":
            return self.item_table(item_ids)
        buckets = self.hash_indices[item_ids]
        table_ids = torch.arange(4, device=item_ids.device).view(1, 4).expand(len(item_ids), 4)
        features = self.hash_tables[table_ids, buckets].reshape(len(item_ids), -1)
        return self.decoder2(F.relu(self.decoder1(features)))

    def _orthogonal_plane(self):
        if not self.training and self._cached_plane is not None:
            return self._cached_plane
        q, _ = torch.linalg.qr(self.mirror_plane, mode="reduced")
        if not self.training:
            self._cached_plane = q.detach()
        return q

    def item_vector(self, item_ids: torch.Tensor) -> torch.Tensor:
        vec = self._base_item_vector(item_ids)
        if self.method in ("dhe_mirror", "dhe_mirror_rank1"):
            slots = self.rare_lookup[item_ids]
            mask = (slots >= 0).to(vec.dtype).unsqueeze(1)
            safe_slots = slots.clamp_min(0)
            theta = self.mirror_angles[safe_slots].squeeze(1) * mask.squeeze(1)
            plane = self._orthogonal_plane()
            z = vec @ plane
            c, s = torch.cos(theta), torch.sin(theta)
            rotated = torch.stack((c * z[:, 0] - s * z[:, 1],
                                   s * z[:, 0] + c * z[:, 1]), dim=1)
            vec = vec + ((rotated - z) @ plane.T) * mask
        if self.method == "dhe_rank2":
            slots = self.rare_lookup[item_ids]
            mask = (slots >= 0).to(vec.dtype).unsqueeze(1)
            coeff = self.residual_codes[slots.clamp_min(0)] * mask
            vec = vec + coeff @ self.residual_basis.T
        elif self.method == "dhe_mirror_rank1":
            slots = self.rare_lookup[item_ids]
            mask = (slots >= 0).to(vec.dtype).unsqueeze(1)
            coeff = self.residual_codes[slots.clamp_min(0)] * mask
            vec = vec + coeff @ self.residual_basis.T
        elif self.method == "dhe_rare_full":
            slots = self.rare_lookup[item_ids]
            mask = (slots >= 0).to(vec.dtype).unsqueeze(1)
            vec = vec + self.full_residuals[slots.clamp_min(0)] * mask
        return vec

    def forward(self, user_ids, item_ids):
        ue = self.user_embeddings(user_ids)
        ie = self.item_vector(item_ids)
        return (ue * ie).sum(dim=1) / math.sqrt(DIM) + self.user_bias(user_ids).squeeze(1) + self.global_bias

    def inference_state(self):
        state = {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()}
        if self.method in ("dhe_mirror", "dhe_mirror_rank1"):
            state.pop("mirror_plane")
            state["mirror_plane_orthonormal"] = self._orthogonal_plane().detach().cpu().contiguous()
        if len(self.rare_item_ids) and self.method in (
                "dhe_mirror", "dhe_rank2", "dhe_mirror_rank1", "dhe_rare_full"):
            state["rare_item_ids_one_based"] = (self.rare_item_ids + 1).to(torch.int32).cpu()
        return state


def build(method, rare_item_ids, seed):
    torch.manual_seed(seed)
    return MovieRecommender(method, rare_item_ids, seed)


def eval_split(model, split, batch_size=2048):
    users, items, labels = split
    model.eval()
    logits = []
    with torch.no_grad():
        for start in range(0, len(users), batch_size):
            logits.append(model(users[start:start+batch_size], items[start:start+batch_size]).cpu())
    pred = torch.cat(logits).numpy()
    target = labels.numpy()
    logloss = float(np.logaddexp(0.0, pred).mean() - (target * pred).mean())
    prob = 1.0 / (1.0 + np.exp(-np.clip(pred, -40, 40)))
    auc = float(roc_auc_score(target, prob)) if len(np.unique(target)) == 2 else float("nan")
    return logloss, auc, pred


def model_meta(method, seed, rare_ids, best_epoch, obj):
    result = {
        "experiment": "MA-1064",
        "method": method,
        "dataset": "MovieLens-100K",
        "data_sha256": obj["archive_sha256"],
        "model_seed": str(seed),
        "user_id_range": "1-943",
        "item_id_range": "1-1682",
        "embedding_dim": str(DIM),
        "best_epoch": str(best_epoch),
        "rare_definition": "train interactions 1-5",
        "rare_count": str(len(rare_ids)),
        "audit": "chronological final 20%; not used to select checkpoint",
    }
    if method == "full_table":
        result["item_representation"] = "direct learned embedding table; row i is MovieLens item ID i+1"
    else:
        result["hash"] = "BLAKE2b-64; seeds ma1064-h0..h3; 128 buckets each"
        result["decoder"] = "4x16 hash vectors -> Linear(64,64), ReLU, Linear(64,32)"
    if method in ("dhe_mirror", "dhe_mirror_rank1"):
        result["mirror"] = "orthogonal 2D plane Givens angle per train-rare ID"
    else:
        result["mirror"] = "none"
    if method == "dhe_rank2":
        result["private_residual"] = "train-rare ID scalar codes over shared rank-2 basis"
    elif method == "dhe_mirror_rank1":
        result["private_residual"] = "train-rare ID scalar codes over shared rank-1 basis"
    elif method == "dhe_rare_full":
        result["private_residual"] = "train-rare ID full 32D residual vectors"
    else:
        result["private_residual"] = "none"
    return result


def save_inference(model, path, method, seed, rare_ids, best_epoch, data_info):
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = model_meta(method, seed, rare_ids, best_epoch, data_info)
    save_file(model.inference_state(), str(path), metadata=metadata)
    return path.stat().st_size, sha256_bytes(path.read_bytes())


def train_stage(split, train_counts, rare_ids, outdir: Path, data_info):
    train, dev = split["train"], split["dev"]
    x_users, x_items, x_y = train
    n = len(x_users)
    records = []
    checkpoint_dir = outdir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        for method in METHODS:
            generator = torch.Generator().manual_seed(seed)
            model = build(method, rare_ids, seed)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0001)
            best_dev = float("inf")
            best_epoch = 0
            best_state = None
            stale = 0
            updates = 0
            events_seen = 0
            started = time.perf_counter()
            for epoch in range(1, 21):
                model.train()
                order = torch.randperm(n, generator=generator)
                for start in range(0, n, 512):
                    ix = order[start:start+512]
                    logits = model(x_users[ix], x_items[ix])
                    loss = F.binary_cross_entropy_with_logits(logits, x_y[ix])
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                    updates += 1
                    events_seen += len(ix)
                dev_loss, _, _ = eval_split(model, dev)
                if dev_loss < best_dev - 1e-6:
                    best_dev, best_epoch, stale = dev_loss, epoch, 0
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                else:
                    stale += 1
                    if stale >= 3:
                        break
            train_wall = time.perf_counter() - started
            if best_state is None:
                raise RuntimeError("no development checkpoint selected")
            model.load_state_dict(best_state)
            model.eval()
            ckpt = checkpoint_dir / f"{method}_seed{seed}.pt"
            torch.save({"state": best_state, "method": method, "seed": seed,
                        "best_epoch": best_epoch, "updates": updates,
                        "train_wall_seconds": train_wall,
                        "dev_logloss": best_dev}, ckpt)
            records.append({"seed": seed, "method": method, "best_epoch": best_epoch,
                            "updates": updates, "events_seen": events_seen,
                            "train_wall_seconds": train_wall,
                            "dev_logloss": best_dev, "checkpoint": ckpt.name})
            print(json.dumps(records[-1]), flush=True)
    (outdir / "training_selection.json").write_text(json.dumps({"models": records,
        "data": data_info, "audit_used_for_selection": False}, indent=2) + "\n")
    return records


def inference_timing(model, audit, batch_size=256):
    users, items, _ = audit
    sample_u, sample_i = users[:batch_size], items[:batch_size]
    model.eval()
    with torch.no_grad():
        for _ in range(3):
            model(sample_u[:min(32, len(sample_u))], sample_i[:min(32, len(sample_i))])
        t0 = time.perf_counter()
        seen = 0
        for start in range(0, len(users), 1024):
            model(users[start:start+1024], items[start:start+1024])
            seen += len(users[start:start+1024])
        batch_wall = time.perf_counter() - t0
        latencies = []
        for i in range(min(batch_size, len(users))):
            t1 = time.perf_counter()
            model(users[i:i+1], items[i:i+1])
            latencies.append((time.perf_counter() - t1) * 1000.0)
    return seen / max(batch_wall, 1e-12), float(np.percentile(latencies, 95))


def audit_stage(split, train_counts, rare_ids, outdir: Path, data_info):
    audit = split["audit"]
    selected = json.loads((outdir / "training_selection.json").read_text())["models"]
    selection_index = {(row["method"], row["seed"]): row for row in selected}
    u, items, labels = audit
    counts_for_audit = train_counts[items.numpy()]
    strata = {
        "rare": (counts_for_audit >= 1) & (counts_for_audit <= 5),
        "cold": counts_for_audit == 0,
        "common": counts_for_audit >= 6,
    }
    rows = []
    serialized_dir = outdir / "inference_payloads"
    serialized_dir.mkdir(parents=True, exist_ok=True)
    for seed in SEEDS:
        for method in METHODS:
            ckpt_path = outdir / "checkpoints" / f"{method}_seed{seed}.pt"
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model = build(method, rare_ids, seed)
            model.load_state_dict(ckpt["state"])
            selection = selection_index[(method, seed)]
            payload_path = serialized_dir / f"{method}_seed{seed}.safetensors"
            payload_bytes, payload_sha = save_inference(model, payload_path, method, seed,
                                                        rare_ids, ckpt["best_epoch"], data_info)
            ll, auc, logits = eval_split(model, audit)
            prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
            row = {
                "seed": seed, "method": method, "updates": ckpt["updates"],
                "events_seen": selection["events_seen"],
                "train_wall_seconds": ckpt["train_wall_seconds"],
                "dev_logloss": ckpt["dev_logloss"], "audit_logloss": ll,
                "audit_auc": auc, "payload_bytes": payload_bytes,
                "payload_sha256": payload_sha,
            }
            for name, mask in strata.items():
                ids = np.flatnonzero(mask)
                y = labels.numpy()[ids]
                p = prob[ids]
                row[f"{name}_count"] = len(ids)
                row[f"{name}_logloss"] = float(np.logaddexp(0, logits[ids]).mean() - (y * logits[ids]).mean()) if len(ids) else float("nan")
                row[f"{name}_auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")
            row["cpu_events_per_second"], row["batch1_p95_ms"] = inference_timing(model, audit)
            row["active_macs_per_event"] = active_macs(method)
            rows.append(row)
            print(json.dumps(row), flush=True)
    fieldnames = ["seed", "method", "updates", "events_seen", "train_wall_seconds", "dev_logloss",
                  "audit_logloss", "audit_auc", "rare_count", "rare_logloss", "rare_auc",
                  "cold_count", "cold_logloss", "cold_auc", "common_count", "common_logloss",
                  "common_auc", "payload_bytes", "active_macs_per_event", "cpu_events_per_second",
                  "batch1_p95_ms", "status"]
    with (outdir / "RESULTS_CORE.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            row["status"] = "FRESH_AUDIT"
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    details = {"data": data_info, "models": rows,
               "audit_evaluated_after_checkpoint_selection": True,
               "claims": "one small MovieLens-100K experiment only; no production DHE/TT-Rec claim"}
    (outdir / "results.json").write_text(json.dumps(details, indent=2, sort_keys=True, allow_nan=True) + "\n")


def active_macs(method):
    # One interaction's score includes one user and one item vector dot product.
    if method == "full_table":
        return DIM
    if method == "dhe":
        # Hash-vector reads are not MACs; count decoder and user-item score.
        return 64 * 64 + 64 * DIM + DIM
    dhe = 64 * 64 + 64 * DIM + DIM
    if method == "dhe_mirror":
        return dhe + 2 * DIM + 2 * DIM
    if method == "dhe_rank2":
        return dhe + 2 * DIM
    if method == "dhe_mirror_rank1":
        return dhe + 2 * DIM + 2 * DIM + DIM
    if method == "dhe_rare_full":
        return dhe
    return dhe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("train", "eval"), required=True)
    ap.add_argument("--archive", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    split, train_counts, rare_ids, data_info = download_and_read(args.archive)
    if args.stage == "train":
        train_stage(split, train_counts, rare_ids, args.out, data_info)
    else:
        if not (args.out / "training_selection.json").exists():
            raise FileNotFoundError("all development-selected checkpoints must exist before fresh audit")
        audit_stage(split, train_counts, rare_ids, args.out, data_info)


if __name__ == "__main__":
    main()
