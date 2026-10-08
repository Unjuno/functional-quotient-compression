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

Q, R, VOCAB, DIM, HALF, CLASSES = 12, 12, 144, 16, 8, 8
TRAIN_N, VAL_N, TEST_N, BATCH, UPDATES = 65536, 32768, 32768, 128, 512
LR = 3e-3
METHODS = ("independent", "qr_add", "qr_multiply", "qr_concat", "mirror_factorized", "free_pair_angle")
torch.set_num_threads(1)


def pair_split(seed):
    offset = int(np.random.default_rng(seed + 991).integers(Q))
    q, r = np.meshgrid(np.arange(Q), np.arange(R), indexing="ij")
    held = ((r - q - offset) % R) < 3
    return held.reshape(-1)


def make_data(seed):
    held = pair_split(seed)
    rng = np.random.default_rng(seed + 51000)
    a_true = rng.normal(0, 0.5, (Q, HALF)).astype(np.float32)
    b_true = rng.normal(0, 0.5, (R, HALF)).astype(np.float32)
    alpha_true = rng.uniform(-1.2, 1.2, Q).astype(np.float32)
    beta_true = rng.uniform(-1.2, 1.2, R).astype(np.float32)
    teacher_head = rng.normal(0, 0.7, (CLASSES, DIM)).astype(np.float32)
    embeddings = np.zeros((VOCAB, DIM), dtype=np.float32)
    for token in range(VOCAB):
        q, r = divmod(token, R)
        angle = alpha_true[q] + beta_true[r]
        c, s = np.cos(angle), np.sin(angle)
        a, b = a_true[q], b_true[r]
        embeddings[token] = np.concatenate([c * a - s * b, s * a + c * b])
    labels = (embeddings @ teacher_head.T).argmax(axis=1).astype(np.int64)
    out = {"held_pairs": held, "teacher_labels": labels,
           "teacher_alpha": alpha_true, "teacher_beta": beta_true}
    train_pairs = np.flatnonzero(~held)
    for split, n, salt, allowed in (("train", TRAIN_N, 13, train_pairs),
                                    ("validation", VAL_N, 29, np.arange(VOCAB)),
                                    ("test", TEST_N, 47, np.arange(VOCAB))):
        srng = np.random.default_rng(seed + salt)
        tokens = srng.choice(allowed, size=n, replace=True).astype(np.int64)
        out[split] = {"tokens": torch.tensor(tokens),
                      "labels": torch.tensor(labels[tokens], dtype=torch.long),
                      "held": torch.tensor(held[tokens], dtype=torch.bool)}
    return out


class QRModel(torch.nn.Module):
    def __init__(self, method, seed):
        super().__init__()
        self.method = method
        torch.manual_seed(seed + METHODS.index(method) * 97)
        if method == "independent":
            self.embedding = torch.nn.Parameter(torch.randn(VOCAB, DIM) * 0.04)
        elif method in ("qr_add", "qr_multiply"):
            self.q_table = torch.nn.Parameter(torch.randn(Q, DIM) * 0.04)
            self.r_table = torch.nn.Parameter(torch.randn(R, DIM) * 0.04)
        else:
            self.q_table = torch.nn.Parameter(torch.randn(Q, HALF) * 0.04)
            self.r_table = torch.nn.Parameter(torch.randn(R, HALF) * 0.04)
            if method == "mirror_factorized":
                self.alpha = torch.nn.Parameter(torch.zeros(Q))
                self.beta = torch.nn.Parameter(torch.zeros(R))
            elif method == "free_pair_angle":
                self.pair_angles = torch.nn.Parameter(torch.zeros(VOCAB))
        self.head = torch.nn.Parameter(torch.randn(CLASSES, DIM) * 0.04)
        self.bias = torch.nn.Parameter(torch.zeros(CLASSES))

    def embedding_for(self, tokens):
        if self.method == "independent":
            return self.embedding[tokens]
        q, r = tokens // R, tokens % R
        a, b = self.q_table[q], self.r_table[r]
        if self.method == "qr_add":
            return a + b
        if self.method == "qr_multiply":
            return a * b
        if self.method == "qr_concat":
            return torch.cat([a, b], dim=1)
        if self.method == "mirror_factorized":
            angle = self.alpha[q] + self.beta[r]
        else:
            angle = self.pair_angles[tokens]
        c, s = torch.cos(angle)[:, None], torch.sin(angle)[:, None]
        return torch.cat([c * a - s * b, s * a + c * b], dim=1)

    def forward(self, tokens):
        return self.embedding_for(tokens) @ self.head.T + self.bias


def evaluate(model, split, data):
    with torch.no_grad():
        logits = model(split["tokens"])
        pred = logits.argmax(1)
        correct = pred.eq(split["labels"])
        seen, held = ~split["held"], split["held"]
        emb = model.embedding_for(torch.arange(VOCAB))
        rounded = emb.detach().cpu().numpy().astype("<f2")
        unique_count = int(len(np.unique(rounded, axis=0)))
        return {
            "accuracy": float(correct.float().mean()),
            "nll": float(F.cross_entropy(logits, split["labels"])),
            "seen_accuracy": float(correct[seen].float().mean()),
            "heldout_accuracy": float(correct[held].float().mean()),
            "seen_nll": float(F.cross_entropy(logits[seen], split["labels"][seen])),
            "heldout_nll": float(F.cross_entropy(logits[held], split["labels"][held])),
            "unique_embeddings_fp16": unique_count,
            "possible_embeddings": VOCAB,
            "heldout_pair_fraction": float(data["held_pairs"].mean()),
        }


def model_arrays(model):
    arrays = {"meta": np.asarray([Q, R, VOCAB, DIM, CLASSES, METHODS.index(model.method)], dtype=np.uint16)}
    for key, value in model.state_dict().items():
        arrays[key] = value.detach().cpu().numpy().astype("<f2")
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
    model = QRModel(method, seed)
    model.load_state_dict({k: torch.tensor(arrays[k], dtype=v.dtype) for k, v in model.state_dict().items()})
    model.eval()
    return model


def train(method, seed, data, outdir):
    model = QRModel(method, seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    tokens, labels = data["train"]["tokens"], data["train"]["labels"]
    rng = np.random.default_rng(seed * 31 + METHODS.index(method))
    perm = rng.permutation(len(tokens))
    start = time.perf_counter()
    for step in range(UPDATES):
        idx = torch.from_numpy(perm[step * BATCH:(step + 1) * BATCH])
        logits = model(tokens[idx])
        loss = F.cross_entropy(logits, labels[idx])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    train_wall = time.perf_counter() - start
    arrays = model_arrays(model)
    path = Path(outdir) / "payloads" / f"development_{seed}_{method}.npz"
    size, digest = deterministic_pack(arrays, path)
    loaded = load_serialized(method, seed, arrays)
    macs = {"embedding_compose": 2 * HALF if method in ("mirror_factorized", "free_pair_angle", "qr_add", "qr_multiply") else 0,
            "classifier": CLASSES * DIM,
            "angle_trig_ops": 2 if method in ("mirror_factorized", "free_pair_angle") else 0}
    return {"method": method, "serialized_bytes": size, "payload_sha256": digest,
            "optimizer_updates": UPDATES, "training_examples_seen": UPDATES * BATCH,
            "train_wall_s": train_wall,
            "validation": evaluate(loaded, data["validation"], data),
            "test": evaluate(loaded, data["test"], data),
            "macs_per_lookup_estimate": macs}


def run(seed, condition, outdir, result_path):
    data = make_data(seed)
    rows = [train(method, seed, data, outdir) for method in METHODS]
    result = {"condition": condition, "seed": seed,
              "heldout_pairs": np.flatnonzero(data["held_pairs"]).tolist(),
              "heldout_pair_fraction": float(data["held_pairs"].mean()),
              "teacher_alpha": data["teacher_alpha"].tolist(),
              "teacher_beta": data["teacher_beta"].tolist(), "summaries": rows}
    Path(result_path).parent.mkdir(parents=True, exist_ok=True)
    Path(result_path).write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--condition", choices=["development", "fresh"], required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--json", required=True)
    args = parser.parse_args()
    run(args.seed, args.condition, args.outdir, args.json)
