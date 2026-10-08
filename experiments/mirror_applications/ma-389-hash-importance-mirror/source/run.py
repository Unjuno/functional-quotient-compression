import argparse
import hashlib
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

VOCAB, BUCKETS, DIM, CLASSES = 128, 16, 16, 8
TRAIN_N, VAL_N, TEST_N, BATCH, UPDATES = 131072, 32768, 32768, 256, 512
LR = 3e-3
METHODS = ("independent", "fixed_hash", "native_hash", "mirror_angle")
torch.set_num_threads(1)


def hash_config(seed):
    rng = np.random.default_rng(seed + 9001)
    return int(rng.integers(1, 2**15)), int(rng.integers(1, 2**15))


def _hash(token_ids, salt):
    salt_word = np.uint32((int(salt) * 0x9E3779B1) & 0xFFFFFFFF)
    value = (np.asarray(token_ids, dtype=np.uint32) + salt_word)
    value = (value ^ (value >> np.uint32(16))) * np.uint32(0x85EBCA6B)
    value = value ^ (value >> np.uint32(13))
    return (value % BUCKETS).astype(np.int64)


def hash_indices(seed):
    salt0, salt1 = hash_config(seed)
    ids = np.arange(VOCAB, dtype=np.uint32)
    return np.stack([_hash(ids, salt0), _hash(ids, salt1)], axis=1), np.asarray([salt0, salt1], dtype=np.int32)


def indices_from_params(params):
    salt0, salt1 = np.asarray(params, dtype=np.int64).tolist()
    ids = np.arange(VOCAB, dtype=np.uint32)
    return torch.tensor(np.stack([_hash(ids, salt0), _hash(ids, salt1)], axis=1), dtype=torch.long)


def make_data(seed):
    indices, hash_params = hash_indices(seed)
    rng = np.random.default_rng(seed + 71000)
    true_tables = rng.normal(0, 0.45, (2, BUCKETS, DIM)).astype(np.float32)
    angles = rng.uniform(-np.pi, np.pi, VOCAB).astype(np.float32)
    token_embed = (np.cos(angles)[:, None] * true_tables[0, indices[:, 0]] +
                   np.sin(angles)[:, None] * true_tables[1, indices[:, 1]])
    teacher = rng.normal(0, 0.7, (CLASSES, DIM)).astype(np.float32)
    logits = token_embed @ teacher.T + rng.normal(0, 0.03, (VOCAB, CLASSES)).astype(np.float32)
    token_labels = logits.argmax(axis=1).astype(np.int64)
    _, counts = np.unique(indices, axis=0, return_counts=True)
    signature_counts = {(int(x[0]), int(x[1])): int(c) for x, c in zip(*np.unique(indices, axis=0, return_counts=True))}
    collided_token = np.asarray([signature_counts[tuple(x)] > 1 for x in indices], dtype=bool)
    out = {"hash_indices": indices, "hash_params": hash_params,
           "collision_rate": float(collided_token.mean()),
           "collision_groups": int(np.sum(counts > 1)), "token_labels": token_labels,
           "teacher_token_logits": logits.astype(np.float32), "collided_token": collided_token}
    for split, n, salt in (("train", TRAIN_N, 11), ("validation", VAL_N, 23), ("test", TEST_N, 37)):
        srng = np.random.default_rng(seed + salt)
        tokens = srng.integers(0, VOCAB, n, dtype=np.int64)
        out[split] = {"tokens": torch.tensor(tokens),
                      "labels": torch.tensor(token_labels[tokens], dtype=torch.long),
                      "collided": torch.tensor(collided_token[tokens], dtype=torch.bool)}
    return out


class HashModel(torch.nn.Module):
    def __init__(self, method, seed):
        super().__init__()
        self.method = method
        torch.manual_seed(seed + METHODS.index(method) * 79)
        if method == "independent":
            self.embedding = torch.nn.Parameter(torch.randn(VOCAB, DIM) * 0.05)
        else:
            self.table0 = torch.nn.Parameter(torch.randn(BUCKETS, DIM) * 0.05)
            self.table1 = torch.nn.Parameter(torch.randn(BUCKETS, DIM) * 0.05)
            if method == "native_hash":
                self.importance = torch.nn.Parameter(torch.ones(VOCAB, 2))
            elif method == "mirror_angle":
                self.angles = torch.nn.Parameter(torch.zeros(VOCAB))
        self.head = torch.nn.Parameter(torch.randn(CLASSES, DIM) * 0.05)
        self.bias = torch.nn.Parameter(torch.zeros(CLASSES))

    def embed(self, tokens, indices):
        if self.method == "independent":
            return self.embedding[tokens]
        i0, i1 = indices[tokens, 0], indices[tokens, 1]
        e0, e1 = self.table0[i0], self.table1[i1]
        if self.method == "fixed_hash":
            return 0.5 * e0 + 0.5 * e1
        if self.method == "native_hash":
            weights = self.importance[tokens]
            return weights[:, 0, None] * e0 + weights[:, 1, None] * e1
        angle = self.angles[tokens]
        return torch.cos(angle)[:, None] * e0 + torch.sin(angle)[:, None] * e1

    def forward(self, tokens, indices):
        return self.embed(tokens, indices) @ self.head.T + self.bias


def evaluate(model, split, indices, data):
    with torch.no_grad():
        logits = model(split["tokens"], indices)
        pred = logits.argmax(dim=1)
        correct = pred.eq(split["labels"])
        collision = split["collided"]
        return {
            "accuracy": float(correct.float().mean()),
            "nll": float(F.cross_entropy(logits, split["labels"])),
            "collided_token_accuracy": float(correct[collision].float().mean()),
            "unique_signature_accuracy": float(correct[~collision].float().mean()),
            "collision_rate": data["collision_rate"],
        }


def model_arrays(model, data, seed):
    arrays = {"meta": np.asarray([VOCAB, BUCKETS, DIM, CLASSES, METHODS.index(model.method)], dtype=np.uint16),
              "hash_params": np.asarray(data["hash_params"], dtype=np.int16)}
    for key, value in model.state_dict().items():
        arrays[key] = value.detach().cpu().numpy().astype("<f2")
    if model.method == "independent":
        arrays.pop("hash_params")
    return arrays


def deterministic_pack(arrays, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, np.ascontiguousarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buffer.getvalue())
    raw = path.read_bytes()
    return len(raw), hashlib.sha256(raw).hexdigest()


def load_serialized(method, seed, arrays):
    model = HashModel(method, seed)
    model.load_state_dict({k: torch.tensor(arrays[k], dtype=v.dtype) for k, v in model.state_dict().items()})
    model.eval()
    indices = None if method == "independent" else indices_from_params(arrays["hash_params"])
    return model, indices


def train(method, seed, data, outdir):
    model = HashModel(method, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    indices = torch.tensor(data["hash_indices"], dtype=torch.long)
    tokens, labels = data["train"]["tokens"], data["train"]["labels"]
    rng = np.random.default_rng(seed * 19 + METHODS.index(method))
    permutation = rng.permutation(len(tokens))
    start = time.perf_counter()
    for step in range(UPDATES):
        idx = torch.from_numpy(permutation[step * BATCH:(step + 1) * BATCH])
        logits = model(tokens[idx], indices)
        loss = F.cross_entropy(logits, labels[idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    train_wall = time.perf_counter() - start
    arrays = model_arrays(model, data, seed)
    path = Path(outdir) / "payloads" / f"development_{seed}_{method}.npz"
    size, digest = deterministic_pack(arrays, path)
    loaded, loaded_indices = load_serialized(method, seed, arrays)
    compute = {"lookup_macs": 0 if method == "independent" else 2 * DIM,
               "importance_macs": 0 if method in ("independent", "fixed_hash") else 2 * DIM,
               "classifier_macs": CLASSES * DIM,
               "trig_ops_per_token": 2 if method == "mirror_angle" else 0}
    return {"method": method, "serialized_bytes": size, "payload_sha256": digest,
            "optimizer_updates": UPDATES, "training_examples_seen": UPDATES * BATCH,
            "train_wall_s": train_wall,
            "validation": evaluate(loaded, data["validation"], loaded_indices, data),
            "test": evaluate(loaded, data["test"], loaded_indices, data),
            "macs_per_token_estimate": compute}


def run(seed, condition, outdir, json_path):
    data = make_data(seed)
    rows = [train(method, seed, data, outdir) for method in METHODS]
    result = {"condition": condition, "seed": seed,
              "hash_params": data["hash_params"].tolist(),
              "unique_hash_signatures": int(len(np.unique(data["hash_indices"], axis=0))),
              "collision_groups": data["collision_groups"],
              "collided_token_fraction": data["collision_rate"],
              "summaries": rows}
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    Path(json_path).write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--condition", choices=["development", "fresh"], required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--json", required=True)
    args = parser.parse_args()
    run(args.seed, args.condition, args.outdir, args.json)
