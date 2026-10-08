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

from model import CLASSES, HIDDEN, INPUT_DIM, METHODS, NODES, PATHS, IndependentChildren, SharedDAG

torch.set_num_threads(1)
UPDATES, BATCH, LR = 1200, 128, 1e-3


def teacher(seed):
    g = torch.Generator().manual_seed(seed + 77000)
    stem = torch.randn(INPUT_DIM, HIDDEN, generator=g) * 0.18
    nodes = torch.randn(NODES, HIDDEN, HIDDEN, generator=g) * 0.06
    out = torch.randn(HIDDEN, CLASSES, generator=g) * 0.2
    bias = torch.randn(CLASSES, generator=g) * 0.02
    return stem, nodes, out, bias


def make_data(seed):
    stem, nodes, out, bias = teacher(seed)
    data = {}
    for name, n, salt in (("train", 8192, 101), ("validation", 2048, 211), ("test", 4096, 307)):
        g = torch.Generator().manual_seed(seed + salt)
        x = torch.randn(n, INPUT_DIM, generator=g)
        with torch.no_grad():
            h = torch.tanh(x @ stem)
            for i in range(NODES):
                h = h + torch.tanh(h @ nodes[i])
            logits = h @ out + bias
            y = logits.argmax(-1)
        data[name] = (x, y)
    return data


def train(method, seed, data):
    x, y = data["train"]
    torch.manual_seed(seed * 31 + METHODS.index(method))
    model = IndependentChildren(seed * 67 + 3) if method == "independent" else SharedDAG(method, seed * 67 + METHODS.index(method))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    rng = np.random.default_rng(seed * 100003 + 17901)
    start = time.perf_counter()
    for step in range(UPDATES):
        ids = torch.from_numpy(rng.integers(0, len(x), size=BATCH, dtype=np.int64))
        sampled_path = int(rng.integers(0, len(PATHS)))
        if method == "independent":
            losses = [F.cross_entropy(model.forward_path(path, x[ids]), y[ids]) for path in PATHS]
            loss = torch.stack(losses).mean()
        else:
            logits = model.forward_path(sampled_path, x[ids])
            loss = F.cross_entropy(logits, y[ids])
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
    arrays["paths"] = np.asarray(PATHS, dtype=np.uint8)
    arrays["meta"] = np.asarray([INPUT_DIM, HIDDEN, CLASSES, NODES], dtype=np.uint16)
    arrays["method_id"] = np.asarray([METHODS.index(method)], dtype=np.uint8)
    return deterministic_pack(arrays, path)


def load_model(method, arrays, seed):
    torch.manual_seed(seed * 31 + METHODS.index(method))
    model = IndependentChildren(seed * 67 + 3) if method == "independent" else SharedDAG(method, seed * 67 + METHODS.index(method))
    state = {}
    for key, value in model.state_dict().items():
        tensor = torch.from_numpy(np.array(arrays[key], copy=True))
        state[key] = tensor.to(value.dtype) if value.is_floating_point() else tensor
    model.load_state_dict(state)
    return model.eval()


def evaluate(model, method, split):
    x, y = split
    rows = []
    for path in PATHS:
        with torch.no_grad():
            start = time.perf_counter()
            logits = model.forward_path(path, x)
            elapsed = time.perf_counter() - start
            nll = float(F.cross_entropy(logits, y))
            acc = float((logits.argmax(-1) == y).float().mean())
        active_nodes = int(path).bit_count()
        macs = INPUT_DIM * HIDDEN + active_nodes * 2 * HIDDEN * HIDDEN + HIDDEN * CLASSES
        if method == "mirror_path":
            macs += HIDDEN
        rows.append({"path": path, "active_nodes": active_nodes, "nll": nll, "accuracy": acc,
                     "macs_per_example": macs, "examples_per_s": float(len(x) / max(elapsed, 1e-9))})
    return {"paths": rows, "mean_nll": float(np.mean([r["nll"] for r in rows])),
            "mean_accuracy": float(np.mean([r["accuracy"] for r in rows]))}


def average_ranks(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2 + 1
        i = j
    return ranks


def spearman(a, b):
    ra, rb = average_ranks(a), average_ranks(b)
    if np.std(ra) == 0 or np.std(rb) == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def run(seed, split_name, outdir, json_path):
    data = make_data(seed)
    rows = []
    for method in METHODS:
        model, wall = train(method, seed, data)
        payload = Path(outdir) / f"{split_name}_{seed}_{method}.npz"
        size, digest = save_payload(model, method, payload)
        loaded = load_model(method, payload_arrays(payload), seed)
        val = evaluate(loaded, method, data["validation"])
        test = evaluate(loaded, method, data["test"])
        rows.append({"condition": split_name, "seed": seed, "method": method,
                     "serialized_bytes": size, "payload_sha256": digest,
                     "optimizer_updates": UPDATES, "examples_seen": UPDATES * BATCH,
                     "model_example_exposures": UPDATES * BATCH * (len(PATHS) if method == "independent" else 1),
                     "examples_per_independent_child": UPDATES * BATCH if method == "independent" else None,
                     "train_wall_s": wall, "validation": val, "test": test})
    independent_test = next(r["test"]["paths"] for r in rows if r["method"] == "independent")
    target_ranking = [r["nll"] for r in independent_test]
    for row in rows:
        if row["method"] != "independent":
            row["validation_to_independent_test_spearman"] = spearman(
                [p["nll"] for p in row["validation"]["paths"]], target_ranking)
    result = {"condition": split_name, "seed": seed, "summaries": rows}
    out = Path(json_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--split", choices=["development", "fresh"], required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--json", required=True)
    args = parser.parse_args()
    run(args.seed, args.split, args.outdir, args.json)
