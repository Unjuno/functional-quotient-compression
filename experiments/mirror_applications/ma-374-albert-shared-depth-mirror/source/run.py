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

from model import DEPTH, D_MODEL, FFN_DIM, METHODS, N_HEADS, SEQ_LEN, VOCAB, TinyAlbertLM

torch.set_num_threads(1)
UPDATES, BATCH, LR = 1200, 32, 1e-3


def recurrence_coefficients(seed):
    rng = np.random.default_rng(seed + 51000)
    a = int(rng.integers(1, VOCAB))
    b = int(rng.integers(1, VOCAB))
    c = int(rng.integers(0, VOCAB))
    return a, b, c


def make_split(seed, n, salt):
    a, b, c = recurrence_coefficients(seed)
    rng = np.random.default_rng(seed + salt)
    tokens = np.zeros((n, SEQ_LEN), dtype=np.int64)
    tokens[:, :2] = rng.integers(0, VOCAB, size=(n, 2))
    for t in range(2, SEQ_LEN):
        tokens[:, t] = (a * tokens[:, t - 1] + b * tokens[:, t - 2] + c) % VOCAB
    return torch.from_numpy(tokens), {"a": a, "b": b, "c": c}


def make_data(seed):
    out = {}
    world = None
    for name, n, salt in (("train", 8192, 101), ("validation", 2048, 211), ("test", 4096, 307)):
        out[name], world = make_split(seed, n, salt)
    return out, world


def train(method, seed, data):
    torch.manual_seed(seed * 37 + METHODS.index(method))
    model = TinyAlbertLM(method)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    x = data["train"]
    start = time.perf_counter()
    rng = np.random.default_rng(seed * 100003 + 42201)
    for _ in range(UPDATES):
        ids = torch.from_numpy(rng.integers(0, len(x), size=BATCH, dtype=np.int64))
        seq = x[ids]
        inputs, targets = seq[:, :-1], seq[:, 1:]
        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), targets.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return model.eval(), time.perf_counter() - start


def deterministic_pack(arrays, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            buf = io.BytesIO()
            np.lib.format.write_array(buf, np.ascontiguousarray(arrays[name]), allow_pickle=False)
            info = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            archive.writestr(info, buf.getvalue())
    raw = path.read_bytes()
    return len(raw), hashlib.sha256(raw).hexdigest()


def payload_arrays(path):
    arrays = {}
    with zipfile.ZipFile(path, "r") as archive:
        for member in archive.namelist():
            arrays[member[:-4]] = np.lib.format.read_array(io.BytesIO(archive.read(member)), allow_pickle=False)
    return arrays


def save_payload(model, method, path):
    arrays = {}
    for key, tensor in model.state_dict().items():
        value = tensor.detach().cpu().numpy()
        arrays[key] = value.astype("<f2") if value.dtype.kind == "f" else value
    arrays["meta"] = np.asarray([VOCAB, SEQ_LEN, D_MODEL, N_HEADS, FFN_DIM, DEPTH], dtype=np.uint16)
    arrays["method_id"] = np.asarray([METHODS.index(method)], dtype=np.uint8)
    return deterministic_pack(arrays, path)


def load_model(method, arrays):
    model = TinyAlbertLM(method)
    state = {}
    for key, value in model.state_dict().items():
        tensor = torch.from_numpy(np.array(arrays[key], copy=True))
        state[key] = tensor.to(value.dtype) if value.is_floating_point() else tensor
    model.load_state_dict(state)
    return model.eval()


def macs_per_token(method):
    # QKV+output projections, two FFN linear maps, and attention score/value products.
    projections = 4 * D_MODEL * D_MODEL + 2 * D_MODEL * FFN_DIM
    attention = 2 * (SEQ_LEN - 1) * D_MODEL
    view = D_MODEL if method == "tied_mirror" else 0
    return projections + attention + view


def evaluate(model, split):
    inputs, targets = split[:, :-1], split[:, 1:]
    prefix_rows = []
    throughput = []
    for depth in range(1, DEPTH + 1):
        with torch.no_grad():
            start = time.perf_counter()
            chunks = []
            for j in range(0, len(inputs), 128):
                chunks.append(model.forward_to_depth(inputs[j:j + 128], depth))
            elapsed = time.perf_counter() - start
            logits = torch.cat(chunks)
            nll = float(F.cross_entropy(logits.reshape(-1, VOCAB), targets.reshape(-1)))
            acc = float((logits.argmax(-1) == targets).float().mean())
            throughput.append(float(len(inputs) / max(elapsed, 1e-9)))
        prefix_rows.append({"depth": depth, "nll": nll, "accuracy": acc,
                            "macs_per_token": depth * macs_per_token(model.method),
                            "examples_per_s": throughput[-1]})
    return {"depth_prefixes": prefix_rows,
            "final_nll": prefix_rows[-1]["nll"],
            "final_accuracy": prefix_rows[-1]["accuracy"],
            "mean_depth_nll": float(np.mean([r["nll"] for r in prefix_rows])),
            "mean_depth_accuracy": float(np.mean([r["accuracy"] for r in prefix_rows]))}


def run(seed, split_name, outdir, json_path):
    data, world = make_data(seed)
    rows = []
    for method in METHODS:
        model, wall = train(method, seed, data)
        payload = Path(outdir) / f"{split_name}_{seed}_{method}.npz"
        size, digest = save_payload(model, method, payload)
        loaded = load_model(method, payload_arrays(payload))
        val = evaluate(loaded, data["validation"])
        test = evaluate(loaded, data["test"])
        rows.append({"condition": split_name, "seed": seed, "method": method,
                     "serialized_bytes": size, "payload_sha256": digest,
                     "optimizer_updates": UPDATES, "training_sequences_seen": UPDATES * BATCH,
                     "training_tokens_seen": UPDATES * BATCH * (SEQ_LEN - 1),
                     "train_wall_s": wall, "validation": val, "test": test})
    result = {"condition": split_name, "seed": seed, "world_coefficients": world, "summaries": rows}
    path = Path(json_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", choices=["development", "fresh"], required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--json", required=True)
    args = parser.parse_args()
    run(args.seed, args.split, args.outdir, args.json)
