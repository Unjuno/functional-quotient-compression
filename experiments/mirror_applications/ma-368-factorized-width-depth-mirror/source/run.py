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

from model import (DEPTHS, HELD_OUT, INPUT_DIM, METHODS, NUM_CLASSES, WIDTHS,
                  ElasticSupernet, IndependentBank)

torch.set_num_threads(1)
UPDATES, BATCH, LR = 1200, 128, 1e-3
TRAIN_CONFIGS = tuple((w, d) for w in WIDTHS for d in DEPTHS if (w, d) not in HELD_OUT)
SANDWICH = ((WIDTHS[0], DEPTHS[0]), (WIDTHS[-1], DEPTHS[-1]))
SAMPLED_CONFIGS = tuple(config for config in TRAIN_CONFIGS if config not in SANDWICH)


def teacher(seed):
    gen = torch.Generator().manual_seed(seed + 81000)
    w1 = torch.randn(INPUT_DIM, 32, generator=gen) * 0.22
    b1 = torch.randn(32, generator=gen) * 0.05
    w2 = torch.randn(32, NUM_CLASSES, generator=gen) * 0.24
    b2 = torch.randn(NUM_CLASSES, generator=gen) * 0.03
    identity = torch.eye(32).unsqueeze(0).expand(3, 32, 32)
    hidden = identity + torch.randn(3, 32, 32, generator=gen) * 0.025
    biases = torch.zeros(3, 32)
    return w1, b1, hidden, biases, w2, b2


def make_data(seed):
    w1, b1, hidden, biases, w2, b2 = teacher(seed)
    result = {}
    for name, n, salt in (("train", 8192, 101), ("validation", 2048, 211), ("test", 4096, 307)):
        gen = torch.Generator().manual_seed(seed + salt)
        x = torch.randn(n, INPUT_DIM, generator=gen)
        with torch.no_grad():
            h = torch.relu(x @ w1 + b1)
            for i in range(3):
                h = torch.relu(h @ hidden[i] + biases[i])
            y = (h @ w2 + b2).argmax(-1)
        result[name] = (x, y)
    return result


def sampled_configs(seed, step):
    rng = np.random.default_rng(seed * 100003 + step)
    extra = SAMPLED_CONFIGS[int(rng.integers(0, len(SAMPLED_CONFIGS)))]
    return tuple(dict.fromkeys((*SANDWICH, extra)))


def train(method, seed, data):
    x, y = data["train"]
    if method == "independent":
        model = IndependentBank(seed * 31 + 7)
    else:
        model = ElasticSupernet(method, seed * 31 + METHODS.index(method))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    start = time.perf_counter()
    for step in range(UPDATES):
        sample_rng = np.random.default_rng(seed * 100003 + step + 47001)
        ids = torch.from_numpy(sample_rng.integers(0, len(x), size=BATCH, dtype=np.int64))
        losses = []
        if method == "independent":
            configs = tuple((w, d) for w in WIDTHS for d in DEPTHS)
        else:
            configs = sampled_configs(seed, step)
        for width, depth in configs:
            logits = model.forward_config(width, depth, x[ids])
            losses.append(F.cross_entropy(logits, y[ids]))
        loss = torch.stack(losses).mean()
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
    arrays["widths"] = np.asarray(WIDTHS, dtype=np.uint8)
    arrays["depths"] = np.asarray(DEPTHS, dtype=np.uint8)
    arrays["held_out"] = np.asarray(HELD_OUT, dtype=np.uint8)
    arrays["meta"] = np.asarray([INPUT_DIM, NUM_CLASSES, len(WIDTHS), len(DEPTHS)], dtype=np.uint16)
    arrays["method_id"] = np.asarray([METHODS.index(method)], dtype=np.uint8)
    size, digest = deterministic_pack(arrays, path)
    return size, digest


def load_model(method, arrays, seed):
    if method == "independent":
        model = IndependentBank(seed * 31 + 7)
    else:
        model = ElasticSupernet(method, seed * 31 + METHODS.index(method))
    ref = model.state_dict()
    state = {}
    for key, value in ref.items():
        tensor = torch.from_numpy(np.array(arrays[key], copy=True))
        state[key] = tensor.to(value.dtype) if value.is_floating_point() else tensor
    model.load_state_dict(state)
    return model.eval()


def macs(width, depth, method):
    total = INPUT_DIM * width + (depth - 1) * width * width + width * NUM_CLASSES
    return total + (2 * width if method == "factor_mirror" else 0)


def evaluate(model, method, split):
    x, y = split
    rows = []
    for width in WIDTHS:
        for depth in DEPTHS:
            with torch.no_grad():
                start = time.perf_counter()
                chunks = [model.forward_config(width, depth, x[j:j + 512]) for j in range(0, len(x), 512)]
                elapsed = time.perf_counter() - start
                logits = torch.cat(chunks)
                nll = float(F.cross_entropy(logits, y))
                acc = float((logits.argmax(-1) == y).float().mean())
            rows.append({"width": width, "depth": depth, "held_out": (width, depth) in HELD_OUT,
                         "nll": nll, "accuracy": acc, "macs_per_example": macs(width, depth, method),
                         "examples_per_s": float(len(x) / max(elapsed, 1e-9))})
    held = [r for r in rows if r["held_out"]]
    seen = [r for r in rows if not r["held_out"]]
    return {"configurations": rows,
            "macro_nll": float(np.mean([r["nll"] for r in rows])),
            "macro_accuracy": float(np.mean([r["accuracy"] for r in rows])),
            "held_out_macro_nll": float(np.mean([r["nll"] for r in held])),
            "held_out_macro_accuracy": float(np.mean([r["accuracy"] for r in held])),
            "seen_macro_nll": float(np.mean([r["nll"] for r in seen]))}


def run(seed, split_name, outdir, json_path):
    data = make_data(seed)
    rows = []
    for method in METHODS:
        model, wall = train(method, seed, data)
        payload = Path(outdir) / f"{split_name}_{seed}_{method}.npz"
        size, digest = save_payload(model, method, payload)
        loaded = load_model(method, payload_arrays(payload), seed)
        validation = evaluate(loaded, method, data["validation"])
        test = evaluate(loaded, method, data["test"])
        rows.append({"condition": split_name, "seed": seed, "method": method,
                     "serialized_bytes": size, "payload_sha256": digest,
                     "optimizer_updates": UPDATES, "train_examples_per_update": BATCH,
                     "active_configurations_per_update": 16 if method == "independent" else len(SANDWICH) + 1,
                     "train_wall_s": wall, "validation": validation, "test": test})
    result = {"condition": split_name, "seed": seed, "summaries": rows}
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
